"""
Document ingestion pipeline.

Supported formats: PDF, HTML, Markdown, DOCX, TXT
Images (PNG, JPG, JPEG, WEBP): OCR via Gemini vision
"""
import asyncio
import hashlib
import re
import uuid
from pathlib import Path

from app.config import settings
from app.retrieval.dense import DenseRetriever, embed_texts
from app.retrieval.bm25_retrieval import BM25Retriever


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


def _chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping character-level chunks."""
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _hierarchical_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """
    Split text hierarchically: section → paragraph → chunk.
    Returns list of {section, paragraph_idx, chunk_idx, text}.
    """
    results = []
    # Split by headers (##, ###) or double newlines as section boundaries
    sections = re.split(r"\n(?=#{1,3} |\n)", text)
    for section_idx, section in enumerate(sections):
        if not section.strip():
            continue
        # Extract section title if available
        lines = section.strip().split("\n")
        title = lines[0] if lines[0].startswith("#") else f"Section {section_idx}"
        paragraphs = re.split(r"\n\n+", section)
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            chunks = _chunk_text(paragraph, chunk_size=chunk_size, overlap=overlap)
            for chunk_idx, chunk in enumerate(chunks):
                results.append(
                    {
                        "section": title,
                        "paragraph_idx": para_idx,
                        "chunk_idx": chunk_idx,
                        "text": chunk,
                    }
                )
    return results


def _parse_pdf(file_path: str) -> str:
    import fitz  # pymupdf
    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def _parse_html(file_path: str) -> str:
    from bs4 import BeautifulSoup
    with open(file_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    # Remove script and style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _parse_docx(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _parse_markdown(file_path: str) -> str:
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def _parse_txt(file_path: str) -> str:
    with open(file_path, encoding="utf-8") as f:
        return f.read()


async def _parse_image(file_path: str) -> str:
    """Use Gemini vision to extract text/caption from image."""
    import base64
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage

    with open(file_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    ext = Path(file_path).suffix.lower().lstrip(".")
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/png")

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0,
    )
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
            },
            {
                "type": "text",
                "text": "Extract all text from this image. If it contains a figure or diagram, describe it in detail. If it contains a table, format it as text.",
            },
        ]
    )
    response = await llm.ainvoke([message])
    return response.content


def _parse_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    parsers = {
        ".pdf": _parse_pdf,
        ".html": _parse_html,
        ".htm": _parse_html,
        ".docx": _parse_docx,
        ".md": _parse_markdown,
        ".txt": _parse_txt,
    }
    parser = parsers.get(ext)
    if parser:
        return parser(file_path)
    raise ValueError(f"Unsupported file format: {ext}")


class IngestionPipeline:
    def __init__(self):
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()

    async def ingest(
        self,
        file_path: str,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ) -> int:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        image_exts = {".png", ".jpg", ".jpeg", ".webp"}

        if ext in image_exts:
            text = await _parse_image(file_path)
        else:
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, _parse_file, file_path)

        # Clean text
        text = re.sub(r"\s{3,}", "\n\n", text)
        text = text.strip()

        # Hierarchical chunking
        chunk_dicts = _hierarchical_chunks(text, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunk_dicts:
            return 0

        title = path.stem
        source = str(path)
        texts = [c["text"] for c in chunk_dicts]

        # Embed all chunks
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(None, embed_texts, texts)

        # Build points for Qdrant
        doc_ids = []
        points = []
        for i, (chunk_dict, vector) in enumerate(zip(chunk_dicts, vectors)):
            # Deterministic ID based on content hash
            raw_id = f"{source}_{i}_{_hash_text(chunk_dict['text'])}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
            doc_ids.append(point_id)
            points.append(
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "text": chunk_dict["text"],
                        "title": title,
                        "source": source,
                        "section": chunk_dict["section"],
                        "chunk_index": i,
                    },
                }
            )

        await self.dense.upsert(points)
        self.bm25.add_documents(doc_ids, texts)
        return len(points)
