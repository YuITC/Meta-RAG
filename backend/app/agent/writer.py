from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
    )


WRITE_PROMPT = """\
You are a research writer. Write a comprehensive, grounded answer to the query using only the provided source documents.

Query: {query}

Evidence:
{evidence}

Source documents:
{context}

Instructions:
- Write a clear, structured Markdown response
- Cite sources inline using [1], [2], etc. notation corresponding to document numbers
- Be concise but thorough
- Do not add information not present in the documents
- Include a References section at the end listing the cited sources

Answer:"""


async def write_answer(query: str, evidence: list[str], docs: list[dict]) -> str:
    """Generate a grounded Markdown answer with inline citations."""
    context = "\n\n".join(
        [f"[{i + 1}] **{d['source']}**\n{d['text']}" for i, d in enumerate(docs)]
    )
    evidence_block = "\n".join([f"- {e}" for e in evidence]) if evidence else "(no evidence extracted)"
    prompt = WRITE_PROMPT.format(
        query=query,
        evidence=evidence_block,
        context=context[:5000],
    )
    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip()
