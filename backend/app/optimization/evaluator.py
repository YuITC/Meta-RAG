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


EVAL_PROMPT = """\
Given the answer and retrieved context below, score the faithfulness from 0.0 to 1.0.

Faithfulness = fraction of answer statements supported by context.

Context:
{context}

Answer:
{answer}

Respond with JSON only:
{{"faithfulness": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}"""


async def evaluate_faithfulness(answer: str, context: str) -> dict:
    """
    Returns {"faithfulness": float, "reasoning": str}
    Uses a single LLM call (not full RAGAS — reserved for offline evaluation).
    """
    prompt = EVAL_PROMPT.format(context=context[:3000], answer=answer[:2000])
    llm = get_llm()
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = response.content.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            result["faithfulness"] = max(0.0, min(1.0, float(result.get("faithfulness", 0.5))))
            return result
    except Exception:
        pass
    return {"faithfulness": 0.5, "reasoning": "evaluation unavailable"}
