from app.verification.claim_extractor import extract_claims


def _tokenize(text: str) -> set[str]:
    return {t for t in text.lower().split() if len(t) > 2}


def verify_citations(answer: str, docs: list[dict]) -> dict:
    """
    Claim-level support heuristic to estimate citation precision.

    Returns keys:
    - citation_precision
    - unsupported_claim_rate
    - evidence_alignment_score
    - supported_claims
    - total_claims
    """
    claims = extract_claims(answer)
    if not claims:
        return {
            "citation_precision": 0.0,
            "unsupported_claim_rate": 1.0,
            "evidence_alignment_score": 0.0,
            "supported_claims": 0,
            "total_claims": 0,
        }

    doc_terms = [_tokenize(d.get("text", "")) for d in docs]
    supported = 0

    for claim in claims:
        c_terms = _tokenize(claim)
        if not c_terms:
            continue
        matched = False
        for d_terms in doc_terms:
            overlap = len(c_terms & d_terms) / max(1, len(c_terms))
            if overlap >= 0.3:
                matched = True
                break
        if matched:
            supported += 1

    total = len(claims)
    precision = supported / max(1, total)
    unsupported_rate = 1.0 - precision
    return {
        "citation_precision": round(precision, 4),
        "unsupported_claim_rate": round(unsupported_rate, 4),
        "evidence_alignment_score": round(precision, 4),
        "supported_claims": supported,
        "total_claims": total,
    }
