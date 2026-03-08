def _tokenize(text: str) -> set[str]:
    return {t for t in text.lower().split() if len(t) > 2}


def estimate_evidence_coverage(query: str, docs: list[dict]) -> float:
    """Estimate whether retrieved evidence sufficiently covers query terms."""
    if not docs:
        return 0.0

    q_terms = _tokenize(query)
    if not q_terms:
        return 0.0

    covered: set[str] = set()
    for d in docs:
        covered.update(q_terms & _tokenize(d.get("text", "")))

    return round(len(covered) / max(1, len(q_terms)), 4)
