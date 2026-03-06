import json
import re

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
    )


def classify_rule_based(query: str) -> str:
    """Lightweight rule-based fallback (as per spec: no KMeans at runtime)."""
    q = query.lower()
    if any(w in q for w in ["compare", "difference", "vs", "versus", "contrast", "better than", "which is"]):
        return "comparative"
    if any(w in q for w in ["why", "how does", "explain", "what caused", "what led", "relationship between", "connection between"]):
        return "multi_hop"
    return "factual"


PLAN_PROMPT = """\
Classify the following research query into one of three types:
- "factual": simple fact lookup, single-document answer
- "comparative": comparing multiple things, methods, or papers
- "multi_hop": requires reasoning across multiple documents or steps

Query: {query}

Respond with JSON only:
{{"query_type": "factual|comparative|multi_hop", "max_hops": 1}}

Use max_hops=2 only for multi_hop queries. For all others use max_hops=1."""


async def plan_query(query: str) -> dict:
    """Returns {"query_type": str, "max_hops": int}"""
    prompt = PLAN_PROMPT.format(query=query)
    llm = get_llm()
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = response.content.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            if result.get("query_type") not in ("factual", "comparative", "multi_hop"):
                result["query_type"] = classify_rule_based(query)
            result["max_hops"] = 2 if result["query_type"] == "multi_hop" else 1
            return result
    except Exception:
        pass

    query_type = classify_rule_based(query)
    return {"query_type": query_type, "max_hops": 2 if query_type == "multi_hop" else 1}
