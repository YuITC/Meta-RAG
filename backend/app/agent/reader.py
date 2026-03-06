import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings

READ_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research evidence extractor. Extract relevant evidence from retrieved documents "
            "and determine if additional retrieval is needed.",
        ),
        (
            "human",
            """Original query: {query}
Current hop: {current_hop} / {max_hops}

Retrieved documents:
{context}

Extract evidence spans that directly answer the query. If the current documents are insufficient and more hops are allowed, generate a follow-up query.

Respond ONLY with valid JSON (no markdown):
{{
  "evidence": [
    {{"doc_id": "<id>", "span": "<relevant text excerpt>", "relevance": "<why this is relevant>"}}
  ],
  "follow_up_query": "<specific follow-up question to fill gaps, or null if sufficient>",
  "coverage": "<brief assessment of how well evidence covers the query>"
}}""",
        ),
    ]
)


class Reader:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
        self.chain = READ_PROMPT | self.llm

    async def read(
        self,
        query: str,
        docs: list[dict],
        current_hop: int,
        max_hops: int,
    ) -> dict:
        context = "\n\n".join(
            f"[{i+1}] ID={d['id']}\nSource: {d.get('source','')}\n{d['text']}"
            for i, d in enumerate(docs)
        )
        response = await self.chain.ainvoke(
            {
                "query": query,
                "current_hop": current_hop,
                "max_hops": max_hops,
                "context": context,
            }
        )
        text = response.content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = {"evidence": [], "follow_up_query": None}

        follow_up = result.get("follow_up_query")
        # Only keep follow_up if not at max hops and there's a real question
        if current_hop >= max_hops or not follow_up or follow_up.lower() in ("null", "none", ""):
            follow_up = None

        return {
            "evidence": result.get("evidence", []),
            "follow_up_query": follow_up,
        }
