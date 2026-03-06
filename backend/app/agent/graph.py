import time
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from app.agent.planner import Planner
from app.agent.reader import Reader
from app.agent.writer import Writer
from app.optimization.evaluator import Evaluator
from app.optimization.bandit import BanditOptimizer
from app.memory.strategy_memory import StrategyMemory
from app.retrieval.hybrid import HybridRetriever
from app.config import RETRIEVAL_CONFIGS


class AgentState(TypedDict):
    query: str
    query_type: str
    max_hops: int
    current_hop: int
    config_id: str
    retrieved_docs: list[dict]
    evidence: list[dict]
    follow_up_query: Optional[str]
    answer: str
    citations: list[dict]
    faithfulness: float
    citation_grounding: float
    cost: float
    latency: float
    utility: float
    retry_count: int
    start_time: float


def _build_graph(
    planner: Planner,
    retriever: HybridRetriever,
    reader: Reader,
    writer: Writer,
    evaluator: Evaluator,
    bandit: BanditOptimizer,
    memory: StrategyMemory,
) -> any:

    async def plan_node(state: AgentState) -> dict:
        plan = await planner.plan(state["query"])
        config_id = await bandit.select_config(state["query"])
        return {
            "query_type": plan["query_type"],
            "max_hops": plan["max_hops"],
            "current_hop": 0,
            "config_id": config_id,
            "retrieved_docs": [],
            "evidence": [],
            "follow_up_query": None,
            "cost": 0.0,
            "retry_count": 0,
            "start_time": time.time(),
        }

    async def retrieve_node(state: AgentState) -> dict:
        query = state.get("follow_up_query") or state["query"]
        new_docs = await retriever.retrieve(query, state["config_id"])
        # Merge with existing, deduplicate by id
        existing = {d["id"]: d for d in state["retrieved_docs"]}
        for d in new_docs:
            existing[d["id"]] = d
        return {
            "retrieved_docs": list(existing.values()),
            "follow_up_query": None,
        }

    async def read_node(state: AgentState) -> dict:
        result = await reader.read(
            query=state["query"],
            docs=state["retrieved_docs"],
            current_hop=state["current_hop"],
            max_hops=state["max_hops"],
        )
        return {
            "evidence": result["evidence"],
            "follow_up_query": result.get("follow_up_query"),
            "current_hop": state["current_hop"] + 1,
        }

    def should_retrieve_more(state: AgentState) -> str:
        if state.get("follow_up_query") and state["current_hop"] < state["max_hops"]:
            return "retrieve"
        return "write"

    async def write_node(state: AgentState) -> dict:
        result = await writer.write(state["query"], state["evidence"])
        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "cost": state.get("cost", 0.0) + result.get("cost", 0.0),
        }

    async def evaluate_node(state: AgentState) -> dict:
        elapsed = time.time() - state["start_time"]
        result = await evaluator.evaluate(
            query=state["query"],
            answer=state["answer"],
            evidence=state["evidence"],
            citations=state["citations"],
        )
        cost = state.get("cost", 0.0) + result.get("cost", 0.0)
        utility = (
            result["faithfulness"]
            - 0.3 * cost
            - 0.1 * (elapsed / 15.0)  # normalize latency to [0,1] range roughly
        )
        return {
            "faithfulness": result["faithfulness"],
            "citation_grounding": result["citation_grounding"],
            "latency": elapsed,
            "utility": float(utility),
            "cost": cost,
        }

    def should_retry(state: AgentState) -> str:
        from app.config import settings
        if state["faithfulness"] < settings.faithfulness_threshold and state["retry_count"] < 1:
            return "retry"
        return "update"

    async def retry_node(state: AgentState) -> dict:
        # Pick a different config from bandit
        new_config = bandit.select_different_config(state["config_id"])
        return {
            "config_id": new_config,
            "retry_count": state["retry_count"] + 1,
            "retrieved_docs": [],
            "current_hop": 0,
            "follow_up_query": None,
            "evidence": [],
        }

    async def update_bandit_node(state: AgentState) -> dict:
        await bandit.update(state["config_id"], state["utility"])
        return {}

    async def store_memory_node(state: AgentState) -> dict:
        await memory.store(
            query=state["query"],
            config_id=state["config_id"],
            utility=state["utility"],
            query_type=state["query_type"],
        )
        return {}

    workflow = StateGraph(AgentState)
    workflow.add_node("plan", plan_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("read", read_node)
    workflow.add_node("write", write_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("retry", retry_node)
    workflow.add_node("update_bandit", update_bandit_node)
    workflow.add_node("store_memory", store_memory_node)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "retrieve")
    workflow.add_edge("retrieve", "read")
    workflow.add_conditional_edges(
        "read",
        should_retrieve_more,
        {"retrieve": "retrieve", "write": "write"},
    )
    workflow.add_edge("write", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        should_retry,
        {"retry": "retry", "update": "update_bandit"},
    )
    workflow.add_edge("retry", "retrieve")
    workflow.add_edge("update_bandit", "store_memory")
    workflow.add_edge("store_memory", END)

    return workflow.compile()


class ResearchAgent:
    def __init__(self):
        self.planner = Planner()
        self.retriever = HybridRetriever()
        self.reader = Reader()
        self.writer = Writer()
        self.evaluator = Evaluator()
        self.bandit = BanditOptimizer()
        self.memory = StrategyMemory()
        self._graph = None

    async def initialize(self) -> None:
        await self.bandit.initialize()
        await self.retriever.dense.ensure_collection()
        self._graph = _build_graph(
            self.planner,
            self.retriever,
            self.reader,
            self.writer,
            self.evaluator,
            self.bandit,
            self.memory,
        )

    async def run(self, query: str) -> AgentState:
        if self._graph is None:
            await self.initialize()
        initial_state: AgentState = {
            "query": query,
            "query_type": "factual",
            "max_hops": 1,
            "current_hop": 0,
            "config_id": "A",
            "retrieved_docs": [],
            "evidence": [],
            "follow_up_query": None,
            "answer": "",
            "citations": [],
            "faithfulness": 0.0,
            "citation_grounding": 0.0,
            "cost": 0.0,
            "latency": 0.0,
            "utility": 0.0,
            "retry_count": 0,
            "start_time": time.time(),
        }
        result = await self._graph.ainvoke(initial_state)
        return result
