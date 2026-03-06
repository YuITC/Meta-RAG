import json
import re
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings

WRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research writer. Generate a clear, grounded answer with explicit citations. "
            "Only make claims supported by the provided evidence.",
        ),
        (
            "human",
            """Query: {query}

Evidence:
{evidence}

Write a comprehensive answer that:
1. Directly addresses the query
2. Cites evidence using [n] notation where n is the evidence number
3. Only states facts supported by the evidence

Respond ONLY with valid JSON (no markdown):
{{
  "answer": "<your answer with [n] citation markers>",
  "citations": [
    {{"ref": 1, "doc_id": "<doc_id>", "title": "<source title or empty>", "span": "<quoted evidence span>"}}
  ]
}}""",
        ),
    ]
)


class Writer:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.1,
        )
        self.chain = WRITE_PROMPT | self.llm

    async def write(self, query: str, evidence: list[dict]) -> dict:
        if not evidence:
            return {
                "answer": "No relevant documents were found to answer this query.",
                "citations": [],
                "cost": 0.0,
            }

        evidence_text = "\n\n".join(
            f"[{i+1}] (doc_id={e.get('doc_id', 'unknown')}): {e.get('span', e.get('text', ''))}"
            for i, e in enumerate(evidence)
        )

        t0 = time.time()
        response = await self.chain.ainvoke({"query": query, "evidence": evidence_text})
        elapsed = time.time() - t0

        text = response.content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = {"answer": text, "citations": []}

        # Estimate cost based on token usage (rough approximation)
        # Gemini 2.5 Flash: ~$0.075/M input tokens, ~$0.30/M output tokens
        input_chars = len(evidence_text) + len(query)
        output_chars = len(result.get("answer", ""))
        cost = (input_chars / 4 * 0.075 + output_chars / 4 * 0.30) / 1_000_000

        return {
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "cost": cost,
        }
