"""
LangGraph agent orchestration for the Autonomous Research Agent.

Flow:
  plan → retrieve → read → [hop?] → retrieve → read → write → evaluate → [retry?] → END
                                                                 ↑
                                              retry: select different config ──┘
"""

import time
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from app.agent.planner import plan_query
from app.agent.reader import read_documents
from app.agent.writer import write_answer
from app.optimization.bandit import CONFIGS, CONFIG_NAMES, compute_utility, estimate_cost
from app.optimization.evaluator import evaluate_faithfulness
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank


class AgentState(TypedDict):
    # Input
    query: str
    bandit_alpha: dict[str, float]
    bandit_beta: dict[str, float]

    # Planning output
    query_type: str
    max_hops: int

    # Retrieval tracking
    current_config: str
    first_config: str          # locked after plan; used to exclude on retry
    current_query: str         # original query or followup
    hop: int                   # number of retrieve calls made

    # Accumulated results
    all_docs: list[dict]
    evidence: list[str]
    followup_query: str | None

    # Output
    answer: str
    faithfulness: float
    cost: float
    latency: float
    utility: float
    retry_count: int

    # Internal cost tracking
    start_time: float
    total_input_chars: int
    total_output_chars: int


# ──────────────────────────────────────────────────────────────────────────────
# Node helpers
# ──────────────────────────────────────────────────────────────────────────────

def _select_config(alpha: dict, beta: dict, exclude: str | None = None) -> str:
    candidates = [c for c in CONFIG_NAMES if c != exclude]
    samples = {c: float(np.random.beta(alpha[c], beta[c])) for c in candidates}
    return max(samples, key=samples.__getitem__)


def _do_retrieve(query: str, config_name: str) -> list[dict]:
    cfg = CONFIGS[config_name]
    docs = hybrid_search(query, top_k=cfg["top_k"])
    if cfg["rerank"] and docs:
        docs = rerank(query, docs, top_k=cfg["top_k"])
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────────────────────────────────

async def plan_node(state: AgentState) -> dict[str, Any]:
    result = await plan_query(state["query"])
    config = _select_config(state["bandit_alpha"], state["bandit_beta"])
    return {
        "query_type": result["query_type"],
        "max_hops": result["max_hops"],
        "current_config": config,
        "first_config": config,
        "current_query": state["query"],
        "hop": 0,
        "all_docs": [],
        "evidence": [],
        "followup_query": None,
        "retry_count": 0,
        "start_time": time.monotonic(),
        "total_input_chars": len(state["query"]) * 2,  # plan prompt approx
        "total_output_chars": 50,
    }


async def retrieve_node(state: AgentState) -> dict[str, Any]:
    # On subsequent hops, use the followup query
    query = (
        state["followup_query"]
        if state["hop"] > 0 and state["followup_query"]
        else state["current_query"]
    )
    new_docs = _do_retrieve(query, state["current_config"])

    # Deduplicate by doc id
    existing_ids = {d["id"] for d in state["all_docs"]}
    fresh = [d for d in new_docs if d["id"] not in existing_ids]
    all_docs = state["all_docs"] + fresh

    return {
        "all_docs": all_docs,
        "followup_query": None,   # consumed
        "hop": state["hop"] + 1,
        "total_input_chars": state["total_input_chars"] + len(query) * 2,
    }


async def read_node(state: AgentState) -> dict[str, Any]:
    result = await read_documents(state["current_query"], state["all_docs"])
    prompt_len = len(state["current_query"]) + sum(len(d["text"]) for d in state["all_docs"][:10])
    answer_len = sum(len(e) for e in result.get("evidence", []))
    return {
        "evidence": state["evidence"] + result.get("evidence", []),
        "followup_query": result.get("followup_query"),
        "total_input_chars": state["total_input_chars"] + prompt_len,
        "total_output_chars": state["total_output_chars"] + answer_len,
    }


async def write_node(state: AgentState) -> dict[str, Any]:
    answer = await write_answer(
        state["current_query"], state["evidence"], state["all_docs"]
    )
    prompt_len = (
        len(state["current_query"])
        + sum(len(e) for e in state["evidence"])
        + sum(len(d["text"]) for d in state["all_docs"][:10])
    )
    return {
        "answer": answer,
        "total_input_chars": state["total_input_chars"] + prompt_len,
        "total_output_chars": state["total_output_chars"] + len(answer),
    }


async def evaluate_node(state: AgentState) -> dict[str, Any]:
    context = "\n\n".join(d["text"] for d in state["all_docs"][:10])
    eval_result = await evaluate_faithfulness(state["answer"], context)
    faithfulness = eval_result["faithfulness"]

    latency = time.monotonic() - state["start_time"]
    cost = estimate_cost(state["total_input_chars"], state["total_output_chars"])
    utility = compute_utility(faithfulness, cost, latency)

    return {
        "faithfulness": faithfulness,
        "cost": cost,
        "latency": round(latency, 2),
        "utility": utility,
    }


async def retry_node(state: AgentState) -> dict[str, Any]:
    """Select a different config and reset retrieval state for a fresh attempt."""
    new_config = _select_config(
        state["bandit_alpha"], state["bandit_beta"], exclude=state["first_config"]
    )
    return {
        "current_config": new_config,
        "current_query": state["query"],   # reset to original query
        "hop": 0,
        "all_docs": [],
        "evidence": [],
        "followup_query": None,
        "retry_count": state["retry_count"] + 1,
        # restart timing from here for retry latency tracking
        "start_time": state["start_time"],  # keep original for total latency
    }


# ──────────────────────────────────────────────────────────────────────────────
# Conditional edge functions
# ──────────────────────────────────────────────────────────────────────────────

def should_hop(state: AgentState) -> str:
    if state["followup_query"] and state["hop"] < state["max_hops"]:
        return "retrieve"
    return "write"


def should_retry(state: AgentState) -> str:
    from app.config import settings
    if (
        state["faithfulness"] < settings.faithfulness_threshold
        and state["retry_count"] < 1
    ):
        return "retry"
    return END


# ──────────────────────────────────────────────────────────────────────────────
# Build the graph
# ──────────────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("plan", plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("read", read_node)
    g.add_node("write", write_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("retry", retry_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "read")
    g.add_conditional_edges("read", should_hop, {"retrieve": "retrieve", "write": "write"})
    g.add_edge("write", "evaluate")
    g.add_conditional_edges("evaluate", should_retry, {"retry": "retry", END: END})
    g.add_edge("retry", "retrieve")

    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def run_agent(query: str, bandit_alpha: dict, bandit_beta: dict) -> AgentState:
    """
    Run the full agent pipeline and return the final state.
    The caller is responsible for updating the bandit after inspecting the result.
    """
    graph = get_graph()
    initial_state: AgentState = {
        "query": query,
        "bandit_alpha": bandit_alpha,
        "bandit_beta": bandit_beta,
        # These will be set by plan_node
        "query_type": "",
        "max_hops": 1,
        "current_config": "",
        "first_config": "",
        "current_query": query,
        "hop": 0,
        "all_docs": [],
        "evidence": [],
        "followup_query": None,
        "answer": "",
        "faithfulness": 0.0,
        "cost": 0.0,
        "latency": 0.0,
        "utility": 0.0,
        "retry_count": 0,
        "start_time": time.monotonic(),
        "total_input_chars": 0,
        "total_output_chars": 0,
    }
    final_state = await graph.ainvoke(initial_state)
    return final_state
