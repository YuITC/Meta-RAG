import json
import re

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings

_llm: ChatGoogleGenerativeAI | None = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.0,
        )
    return _llm


MULTI_EVAL_PROMPT = """\
Given the answer and retrieved context below, score the following metrics from 0.0 to 1.0:
- faithfulness: fraction of statements supported by context
- answer_completeness: how fully the answer addresses the query
- confidence: confidence in the evaluation itself

Query:
{query}

Context:
{context}

Answer:
{answer}

Respond with JSON only:
{{"faithfulness": <float>, "answer_completeness": <float>, "confidence": <float>, "reasoning": "<brief explanation>"}}"""


async def evaluate_answer(query: str, answer: str, context: str) -> dict:
    """Multi-metric runtime evaluation with a single LLM call."""
    prompt = MULTI_EVAL_PROMPT.format(
        query=query[:800],
        context=context[:3000],
        answer=answer[:2200],
    )
    llm = get_llm()
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = response.content.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return {
                "faithfulness": max(0.0, min(1.0, float(result.get("faithfulness", 0.5)))),
                "answer_completeness": max(0.0, min(1.0, float(result.get("answer_completeness", 0.5)))),
                "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
                "reasoning": str(result.get("reasoning", "")),
            }
    except Exception:
        pass
    return {
        "faithfulness": 0.5,
        "answer_completeness": 0.5,
        "confidence": 0.4,
        "reasoning": "evaluation unavailable",
    }
