import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings

PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query planner for a research retrieval system. "
            "Classify the user query and determine retrieval depth.",
        ),
        (
            "human",
            """Classify the following query and determine retrieval strategy.

Query: {query}

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "query_type": "<factual|comparative|multi_hop>",
  "max_hops": <1 or 2>,
  "reasoning": "<brief reasoning>"
}}

Guidelines:
- factual: single-step lookup (max_hops=1)
- comparative: comparing multiple topics (max_hops=1 or 2)
- multi_hop: requires chaining multiple pieces of evidence (max_hops=2)""",
        ),
    ]
)


class Planner:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
        self.chain = PLAN_PROMPT | self.llm

    async def plan(self, query: str) -> dict:
        response = await self.chain.ainvoke({"query": query})
        text = response.content.strip()
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = {"query_type": "factual", "max_hops": 1, "reasoning": "parse error fallback"}
        return {
            "query_type": result.get("query_type", "factual"),
            "max_hops": int(result.get("max_hops", 1)),
        }
