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


READ_PROMPT = """\
You are a research reader. Given a query and retrieved documents, extract relevant evidence spans.

Query: {query}

Documents:
{context}

Respond with JSON only:
{{
  "evidence": ["<direct quote or paraphrase with citation [N]>", ...],
  "missing": "<key information still missing to answer the query, or null>",
  "followup_query": "<a targeted follow-up search query to retrieve missing info, or null>"
}}

Set "followup_query" only if important information is truly missing. Do not set it for minor details."""


async def read_documents(query: str, docs: list[dict]) -> dict:
    """
    Extract evidence spans and identify information gaps.
    Returns {"evidence": list[str], "missing": str|None, "followup_query": str|None}
    """
    context = "\n\n".join(
        [f"[{i + 1}] (Source: {d['source']})\n{d['text']}" for i, d in enumerate(docs)]
    )
    prompt = READ_PROMPT.format(query=query, context=context[:4000])
    llm = get_llm()
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = response.content.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            result.setdefault("evidence", [])
            result.setdefault("missing", None)
            result.setdefault("followup_query", None)
            return result
    except Exception:
        pass

    return {
        "evidence": [d["text"][:300] for d in docs[:3]],
        "missing": None,
        "followup_query": None,
    }
