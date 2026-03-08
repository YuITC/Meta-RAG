"""
LangGraph agent orchestration for the Autonomous Research Agent.

Flow:
    plan → query_rewrite → retrieve → read → controller → [hop?] → query_rewrite → ...
                                                                                            ↓
                                                                                     write → claim_extract → citation_verify → evidence_graph → evaluate → [retry?] → END
"""

import time
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from app.agent.planner import plan_query
from app.agent.reader import read_documents
from app.agent.writer import write_answer
from app.config import settings
from app.optimization.bandit import CONFIGS, CONFIG_NAMES, compute_reward, estimate_cost
from app.optimization.evaluator import evaluate_answer
from app.research.controller import ResearchSignals, decide_next_action
from app.research.coverage import estimate_evidence_coverage
from app.research.evidence_graph import build_evidence_graph
from app.retrieval.hybrid import hybrid_search, hybrid_search_multi
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import rerank
from app.retrieval.guardrails import filter_retrieved_docs
from app.verification.claim_extractor import extract_claims
from app.verification.citation_verifier import verify_citations


_rewriter = QueryRewriter()


class AgentState(TypedDict):
    # Input
    query: str
    bandit_alpha: dict[str, float]
    bandit_beta: dict[str, float]
    document_ids: list[int] | None

    # Planning output
    query_type: str
    max_hops: int

    # Retrieval tracking
    current_config: str
    first_config: str          # locked after plan; used to exclude on retry
    current_query: str         # original query or followup
    query_variants: list[str]
    hop: int                   # number of retrieve calls made

    # Accumulated results
    all_docs: list[dict]
    evidence: list[str]
    followup_query: str | None
    claims: list[str]

    # Control / observability
    retrieval_diagnostics: dict[str, float]
    evidence_coverage: float
    evaluator_confidence: float
    controller_action: str
    evidence_graph: dict

    # Output
    answer: str
    abstained: bool
    faithfulness: float
    citation_precision: float
    unsupported_claim_rate: float
    evidence_alignment_score: float
    answer_completeness: float
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
    samples = {
        c: float(np.random.beta(alpha.get(c, 1.0), beta.get(c, 1.0)))
        for c in candidates
    }
    return max(samples, key=samples.__getitem__)


# ──────────────────────────────────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────────────────────────────────

async def plan_node(state: AgentState) -> dict[str, Any]:
    result = await plan_query(state["query"])
    config = _select_config(state["bandit_alpha"], state["bandit_beta"])
    return {
        "query_type": result["query_type"],
        "max_hops": min(result["max_hops"], CONFIGS[config].get("hop_limit", result["max_hops"])),
        "current_config": config,
        "first_config": config,
        "current_query": state["query"],
        "query_variants": [state["query"]],
        "hop": 0,
        "all_docs": [],
        "evidence": [],
        "followup_query": None,
        "claims": [],
        "retrieval_diagnostics": {},
        "evidence_coverage": 0.0,
        "evaluator_confidence": 0.5,
        "controller_action": "stop",
        "retry_count": 0,
        "start_time": time.monotonic(),
        "total_input_chars": len(state["query"]) * 2,  # plan prompt approx
        "total_output_chars": 50,
    }


async def query_rewrite_node(state: AgentState) -> dict[str, Any]:
    # On subsequent hops, use the followup query
    query = (
        state["followup_query"]
        if state["hop"] > 0 and state["followup_query"]
        else state["current_query"]
    )
    cfg = CONFIGS[state["current_config"]]
    query_variants = [query]
    if cfg.get("query_rewrite", False):
        if cfg.get("llm_rewrite", False):
            query_variants = await _rewriter.rewrite_with_llm(
                query,
                num_rewrites=int(cfg.get("rewrite_count", settings.default_rewrite_count)),
            )
        else:
            query_variants = _rewriter.rewrite(
                query,
                query_type=state["query_type"],
                num_rewrites=int(cfg.get("rewrite_count", settings.default_rewrite_count)),
            )

    return {
        "query_variants": query_variants,
        "total_input_chars": state["total_input_chars"] + len(query) * 2,
    }


async def retrieve_node(state: AgentState) -> dict[str, Any]:
    query = state["query_variants"][0] if state["query_variants"] else state["current_query"]
    cfg = CONFIGS[state["current_config"]]
    if len(state["query_variants"]) > 1:
        new_docs, diagnostics = hybrid_search_multi(
            state["query_variants"],
            top_k=cfg["top_k"],
            document_ids=state["document_ids"],
        )
    else:
        new_docs = hybrid_search(
            query,
            top_k=cfg["top_k"],
            document_ids=state["document_ids"],
        )
        diagnostics = {"query_coverage": 0.0, "document_diversity": 0.0, "retrieval_redundancy": 0.0, "estimated_recall_proxy": 0.0}

    if cfg["rerank"] and new_docs:
        new_docs = rerank(query, new_docs, top_k=cfg["top_k"])

    # Apply retrieval guardrails to filter unsafe/low-quality chunks
    new_docs = filter_retrieved_docs(new_docs)

    # Deduplicate by doc id
    existing_ids = {d["id"] for d in state["all_docs"]}
    fresh = [d for d in new_docs if d["id"] not in existing_ids]
    all_docs = state["all_docs"] + fresh

    return {
        "all_docs": all_docs,
        "retrieval_diagnostics": diagnostics,
        "followup_query": None,   # consumed
        "hop": state["hop"] + 1,
    }


async def read_node(state: AgentState) -> dict[str, Any]:
    result = await read_documents(state["current_query"], state["all_docs"])
    prompt_len = len(state["current_query"]) + sum(len(d["text"]) for d in state["all_docs"][:10])
    answer_len = sum(len(e) for e in result.get("evidence", []))
    evidence_coverage = estimate_evidence_coverage(state["current_query"], state["all_docs"])

    return {
        "evidence": state["evidence"] + result.get("evidence", []),
        "followup_query": result.get("followup_query"),
        "evidence_coverage": evidence_coverage,
        "total_input_chars": state["total_input_chars"] + prompt_len,
        "total_output_chars": state["total_output_chars"] + answer_len,
    }


async def controller_node(state: AgentState) -> dict[str, Any]:
    action = decide_next_action(
        ResearchSignals(
            evidence_coverage=state["evidence_coverage"],
            retrieval_diversity=state["retrieval_diagnostics"].get("document_diversity", 0.0),
            evaluator_confidence=state["evaluator_confidence"],
            estimated_recall_proxy=state["retrieval_diagnostics"].get("estimated_recall_proxy", 0.0),
            hop=state["hop"],
            max_hops=state["max_hops"],
            has_followup_query=bool(state.get("followup_query")),
        )
    )
    followup_query = state.get("followup_query")
    if action == "reformulate" and not followup_query:
        rewrites = _rewriter.rewrite(state["current_query"], state["query_type"], num_rewrites=3)
        if len(rewrites) > 1:
            followup_query = rewrites[1]
            action = "extra_hop"

    return {
        "followup_query": followup_query,
        "controller_action": action,
    }


ABSTAIN_MESSAGE = (
    "**Insufficient evidence.** The retrieved documents do not contain enough "
    "relevant information to provide a reliable answer to this query. "
    "Please try uploading more relevant documents or rephrasing your question."
)


async def write_node(state: AgentState) -> dict[str, Any]:
    # If the controller decided to abstain, produce a standard message
    if state.get("controller_action") == "abstain":
        return {
            "answer": ABSTAIN_MESSAGE,
            "abstained": True,
            "total_input_chars": state["total_input_chars"],
            "total_output_chars": state["total_output_chars"] + len(ABSTAIN_MESSAGE),
        }

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
        "abstained": False,
        "total_input_chars": state["total_input_chars"] + prompt_len,
        "total_output_chars": state["total_output_chars"] + len(answer),
    }


async def claim_extract_node(state: AgentState) -> dict[str, Any]:
    return {"claims": extract_claims(state["answer"]) }


async def citation_verify_node(state: AgentState) -> dict[str, Any]:
    metrics = verify_citations(state["answer"], state["all_docs"])
    return {
        "citation_precision": metrics["citation_precision"],
        "unsupported_claim_rate": metrics["unsupported_claim_rate"],
        "evidence_alignment_score": metrics["evidence_alignment_score"],
    }


async def evidence_graph_node(state: AgentState) -> dict[str, Any]:
    graph = build_evidence_graph(state["claims"], state["all_docs"])
    return {"evidence_graph": graph.to_dict()}


async def evaluate_node(state: AgentState) -> dict[str, Any]:
    context = "\n\n".join(d["text"] for d in state["all_docs"][:10])
    eval_result = await evaluate_answer(state["current_query"], state["answer"], context)
    faithfulness = eval_result["faithfulness"]
    answer_completeness = eval_result["answer_completeness"]
    evaluator_confidence = eval_result["confidence"]

    latency = time.monotonic() - state["start_time"]
    cost = estimate_cost(state["total_input_chars"], state["total_output_chars"])
    utility = compute_reward(
        faithfulness=faithfulness,
        citation_precision=state.get("citation_precision", 0.0),
        answer_completeness=answer_completeness,
        latency=latency,
        w1=settings.reward_w1_faithfulness,
        w2=settings.reward_w2_citation_precision,
        w3=settings.reward_w3_answer_completeness,
        w4=settings.reward_w4_latency_penalty,
    )

    return {
        "faithfulness": faithfulness,
        "answer_completeness": answer_completeness,
        "evaluator_confidence": evaluator_confidence,
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
        "query_variants": [state["query"]],
        "hop": 0,
        "all_docs": [],
        "evidence": [],
        "followup_query": None,
        "claims": [],
        "retrieval_diagnostics": {},
        "evidence_coverage": 0.0,
        "evidence_graph": {},
        "controller_action": "stop",
        "retry_count": state["retry_count"] + 1,
        # restart timing from here for retry latency tracking
        "start_time": state["start_time"],  # keep original for total latency
    }


# ──────────────────────────────────────────────────────────────────────────────
# Conditional edge functions
# ──────────────────────────────────────────────────────────────────────────────

def should_hop(state: AgentState) -> str:
    action = state.get("controller_action", "stop")
    if action == "abstain":
        return "write"  # go straight to write_node which will emit abstention
    if action in ("extra_hop", "reformulate") and state["followup_query"] and state["hop"] < state["max_hops"]:
        return "retrieve"
    return "write"


def should_retry(state: AgentState) -> str:
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
    g.add_node("query_rewrite", query_rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("read", read_node)
    g.add_node("controller", controller_node)
    g.add_node("write", write_node)
    g.add_node("claim_extract", claim_extract_node)
    g.add_node("citation_verify", citation_verify_node)
    g.add_node("evidence_graph", evidence_graph_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("retry", retry_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "query_rewrite")
    g.add_edge("query_rewrite", "retrieve")
    g.add_edge("retrieve", "read")
    g.add_edge("read", "controller")
    g.add_conditional_edges("controller", should_hop, {"retrieve": "query_rewrite", "write": "write"})
    g.add_edge("write", "claim_extract")
    g.add_edge("claim_extract", "citation_verify")
    g.add_edge("citation_verify", "evidence_graph")
    g.add_edge("evidence_graph", "evaluate")
    g.add_conditional_edges("evaluate", should_retry, {"retry": "retry", END: END})
    g.add_edge("retry", "query_rewrite")

    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def build_initial_state(
    query: str,
    bandit_alpha: dict,
    bandit_beta: dict,
    document_ids: list[int] | None = None,
) -> AgentState:
    return {
        "query": query,
        "bandit_alpha": bandit_alpha,
        "bandit_beta": bandit_beta,
        "document_ids": document_ids,
        # These will be set by plan_node
        "query_type": "",
        "max_hops": 1,
        "current_config": "",
        "first_config": "",
        "current_query": query,
        "query_variants": [query],
        "hop": 0,
        "all_docs": [],
        "evidence": [],
        "followup_query": None,
        "claims": [],
        "retrieval_diagnostics": {},
        "evidence_coverage": 0.0,
        "evaluator_confidence": 0.5,
        "controller_action": "stop",
        "evidence_graph": {},
        "answer": "",
        "abstained": False,
        "faithfulness": 0.0,
        "citation_precision": 0.0,
        "unsupported_claim_rate": 1.0,
        "evidence_alignment_score": 0.0,
        "answer_completeness": 0.0,
        "cost": 0.0,
        "latency": 0.0,
        "utility": 0.0,
        "retry_count": 0,
        "start_time": time.monotonic(),
        "total_input_chars": 0,
        "total_output_chars": 0,
    }


async def stream_agent_updates(
    query: str,
    bandit_alpha: dict,
    bandit_beta: dict,
    document_ids: list[int] | None = None,
):
    graph = get_graph()
    initial_state = build_initial_state(query, bandit_alpha, bandit_beta, document_ids)
    async for update in graph.astream(initial_state, stream_mode="updates"):
        yield update


async def run_agent(
    query: str, 
    bandit_alpha: dict, 
    bandit_beta: dict, 
    document_ids: list[int] | None = None
) -> AgentState:
    """
    Run the full agent pipeline and return the final state.
    The caller is responsible for updating the bandit after inspecting the result.
    """
    graph = get_graph()
    initial_state = build_initial_state(query, bandit_alpha, bandit_beta, document_ids)
    final_state = await graph.ainvoke(initial_state)
    return final_state
