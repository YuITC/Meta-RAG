import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings

EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an answer quality evaluator for a RAG system. "
            "Score answers on faithfulness and citation grounding.",
        ),
        (
            "human",
            """Evaluate the following answer against the evidence.

Query: {query}

Evidence:
{evidence}

Answer: {answer}

Citations in answer: {citations}

Score the answer on:
1. faithfulness (0.0-1.0): Are all claims in the answer supported by the evidence?
2. citation_grounding (0.0-1.0): What fraction of answer statements have valid citation support?

Respond ONLY with valid JSON (no markdown):
{{
  "faithfulness": <0.0 to 1.0>,
  "citation_grounding": <0.0 to 1.0>,
  "reasoning": "<brief explanation>"
}}""",
        ),
    ]
)


class Evaluator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
        self.chain = EVAL_PROMPT | self.llm

    async def evaluate(
        self,
        query: str,
        answer: str,
        evidence: list[dict],
        citations: list[dict],
    ) -> dict:
        if not answer or not evidence:
            return {"faithfulness": 0.0, "citation_grounding": 0.0, "cost": 0.0}

        evidence_text = "\n".join(
            f"[{i+1}] {e.get('span', e.get('text', ''))}" for i, e in enumerate(evidence)
        )
        citations_text = json.dumps(citations, indent=2) if citations else "none"

        response = await self.chain.ainvoke(
            {
                "query": query,
                "evidence": evidence_text,
                "answer": answer,
                "citations": citations_text,
            }
        )
        text = response.content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = {"faithfulness": 0.5, "citation_grounding": 0.5}

        # Estimate cost
        input_chars = len(evidence_text) + len(answer) + len(query)
        output_chars = len(text)
        cost = (input_chars / 4 * 0.075 + output_chars / 4 * 0.30) / 1_000_000

        return {
            "faithfulness": float(result.get("faithfulness", 0.5)),
            "citation_grounding": float(result.get("citation_grounding", 0.5)),
            "cost": cost,
        }
