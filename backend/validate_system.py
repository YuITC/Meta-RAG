"""
End-to-end system validation for the Meta-RAG Research Agent.

Tests all 10 steps:
  1. Document ingestion
  2. Retrieval validation
  3. Full pipeline validation
  4. Citation correctness
  5. Research controller behavior
  6. Abstention behavior
  7. Bandit learning loop
  8. Retrieval diagnostics
  9. Stress test
 10. Final validation report
"""

import asyncio
import json
import os
import sys
import time
import traceback

# Ensure app imports work
sys.path.insert(0, os.path.dirname(__file__))

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

report: dict = {
    "steps": {},
    "bugs": [],
    "warnings": [],
}

def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def log(msg: str, level: str = "INFO"):
    prefix = {"PASS": PASS, "FAIL": FAIL, "WARN": WARN, "INFO": INFO}.get(level, INFO)
    print(f"  {prefix} {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Document Ingestion
# ──────────────────────────────────────────────────────────────────────────────

def step1_ingestion() -> dict:
    section("STEP 1 — Document Ingestion")
    results = {}

    from app.ingestion.pipeline import parse_document, _chunk_text, ingest_document
    from app.retrieval.dense import get_qdrant_client, ensure_collection, embed
    from app.config import settings

    # 1a. Parse PDF
    pdf_path = os.path.join(os.path.dirname(__file__), "data", "test_file.pdf")
    with open(pdf_path, "rb") as f:
        content = f.read()

    text = parse_document("test_file.pdf", content)
    results["raw_text_length"] = len(text)
    log(f"Parsed PDF: {len(text)} characters extracted", "PASS" if len(text) > 500 else "FAIL")

    # 1b. Chunk
    chunks = _chunk_text(text)
    results["num_chunks"] = len(chunks)
    log(f"Chunks created: {len(chunks)}", "PASS" if len(chunks) > 5 else "FAIL")

    # Show sample chunk
    if chunks:
        log(f"Sample chunk (first 200 chars): {chunks[0][:200]!r}")
        log(f"Avg chunk size: {sum(len(c) for c in chunks)/len(chunks):.0f} chars")

    # 1c. Embed a sample
    sample_embedding = embed([chunks[0]])
    results["embedding_dim"] = len(sample_embedding[0])
    log(f"Embedding dimension: {len(sample_embedding[0])}", "PASS" if len(sample_embedding[0]) == 384 else "FAIL")

    # 1d. Ingest into Qdrant
    # First wipe existing data from this file to avoid duplicates
    from app.retrieval.dense import delete_by_source
    delete_by_source("test_file.pdf")

    num_indexed = ingest_document("test_file.pdf", content, document_id=9999)
    results["chunks_indexed"] = num_indexed
    log(f"Chunks indexed in Qdrant: {num_indexed}", "PASS" if num_indexed > 0 else "FAIL")

    # 1e. Verify in Qdrant
    client = get_qdrant_client()
    collection_info = client.get_collection(settings.qdrant_collection)
    results["total_points"] = collection_info.points_count
    log(f"Total points in Qdrant collection '{settings.qdrant_collection}': {collection_info.points_count}")

    # Verify by querying a test embedding
    test_vec = embed(["AegisUI behavioral anomaly detection"])[0]
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=test_vec,
        limit=3,
    )
    results["verification_hits"] = len(hits.points)
    if hits.points:
        log(f"Verification query returned {len(hits.points)} hits", "PASS")
        for i, p in enumerate(hits.points):
            log(f"  Hit {i+1}: score={p.score:.4f}, source={p.payload.get('source','?')}, text={p.payload.get('text','')[:100]!r}")
    else:
        log("Verification query returned NO hits", "FAIL")

    results["passed"] = num_indexed > 0 and results["verification_hits"] > 0
    report["steps"]["1_ingestion"] = results
    return results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Retrieval Validation
# ──────────────────────────────────────────────────────────────────────────────

def step2_retrieval() -> dict:
    section("STEP 2 — Retrieval Validation")
    results = {"queries": {}}

    from app.retrieval.hybrid import hybrid_search, hybrid_search_multi
    from app.retrieval.query_rewriter import QueryRewriter
    from app.retrieval.diagnostics import compute_retrieval_diagnostics

    rewriter = QueryRewriter()

    queries = {
        "factual": "What accuracy did Random Forest achieve on the AegisUI dataset?",
        "comparative": "Compare the performance of Isolation Forest, Autoencoder, and Random Forest in AegisUI",
        "multi_hop": "Why are manipulative UI attacks harder to detect than layout abuse, and what features contribute to this difference?",
    }

    for qtype, query in queries.items():
        print(f"\n  --- Query type: {qtype} ---")
        print(f"  Query: {query}")
        qr = {}

        # 2a. Query rewriting
        variants = rewriter.rewrite(query, query_type=qtype, num_rewrites=4)
        qr["query_variants"] = variants
        log(f"Query variants ({len(variants)}): {variants}", "PASS" if len(variants) > 1 else "WARN")

        # 2b. Single-query retrieval
        single_results = hybrid_search(query, top_k=5)
        qr["single_result_count"] = len(single_results)
        log(f"Single-query hybrid retrieved: {len(single_results)} docs", "PASS" if single_results else "FAIL")

        # 2c. Multi-query retrieval
        multi_results, diag = hybrid_search_multi(variants, top_k=10)
        qr["multi_result_count"] = len(multi_results)
        qr["diagnostics"] = diag
        log(f"Multi-query hybrid retrieved: {len(multi_results)} docs", "PASS" if multi_results else "FAIL")
        log(f"Diagnostics: {json.dumps(diag, indent=2)}")

        # Show retrieved chunks
        for i, doc in enumerate(multi_results[:3]):
            log(f"  Doc {i+1} [score={doc.get('score', 0):.4f}]: {doc['text'][:150]!r}")

        qr["passed"] = len(multi_results) > 0
        results["queries"][qtype] = qr

    results["passed"] = all(q["passed"] for q in results["queries"].values())
    report["steps"]["2_retrieval"] = results
    return results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Full Pipeline Validation
# ──────────────────────────────────────────────────────────────────────────────

async def step3_pipeline() -> dict:
    section("STEP 3 — Full Research Pipeline Validation")
    results = {"runs": []}

    from app.agent.graph import run_agent
    from app.agent.planner import classify_rule_based
    from app.optimization.bandit import CONFIG_NAMES

    # Use neutral priors for clean test
    alpha = {c: 1.0 for c in CONFIG_NAMES}
    beta = {c: 1.0 for c in CONFIG_NAMES}

    test_queries = [
        "What is the F1 score of Random Forest on the AegisUI dataset?",
        "Compare the detection performance of the three models used in AegisUI",
        "How does the AegisUI framework generate and validate synthetic payloads, and why was synthetic data chosen over real-world data?",
    ]

    for query in test_queries:
        print(f"\n  --- Pipeline run ---")
        print(f"  Query: {query}")
        run = {"query": query}

        try:
            t0 = time.monotonic()
            state = await run_agent(query, alpha, beta)
            elapsed = time.monotonic() - t0

            run["query_type"] = state["query_type"]
            run["config"] = state["current_config"]
            run["first_config"] = state["first_config"]
            run["hops"] = state["hop"]
            run["max_hops"] = state["max_hops"]
            run["num_docs"] = len(state["all_docs"])
            run["num_evidence"] = len(state["evidence"])
            run["num_claims"] = len(state["claims"])
            run["query_variants"] = state["query_variants"]
            run["controller_action"] = state["controller_action"]
            run["evidence_coverage"] = state["evidence_coverage"]
            run["evaluator_confidence"] = state.get("evaluator_confidence", 0)
            run["faithfulness"] = state["faithfulness"]
            run["citation_precision"] = state.get("citation_precision", 0)
            run["unsupported_claim_rate"] = state.get("unsupported_claim_rate", 1)
            run["answer_completeness"] = state.get("answer_completeness", 0)
            run["evidence_graph"] = state.get("evidence_graph", {})
            run["retrieval_diagnostics"] = state.get("retrieval_diagnostics", {})
            run["abstained"] = state.get("abstained", False)
            run["answer_preview"] = state["answer"][:500]
            run["latency"] = round(elapsed, 2)
            run["retry_count"] = state["retry_count"]

            log(f"Query type: {run['query_type']}")
            log(f"Config selected: {run['config']} (first: {run['first_config']})")
            log(f"Hops: {run['hops']} / {run['max_hops']}")
            log(f"Docs retrieved: {run['num_docs']}, Evidence spans: {run['num_evidence']}")
            log(f"Query variants: {len(run['query_variants'])}")
            log(f"Controller action: {run['controller_action']}")
            log(f"Evidence coverage: {run['evidence_coverage']:.4f}")
            log(f"Claims extracted: {run['num_claims']}")
            log(f"Faithfulness: {run['faithfulness']:.4f}")
            log(f"Citation precision: {run['citation_precision']:.4f}")
            log(f"Unsupported claim rate: {run['unsupported_claim_rate']:.4f}")
            log(f"Answer completeness: {run['answer_completeness']:.4f}")
            log(f"Evaluator confidence: {run['evaluator_confidence']:.4f}")
            log(f"Evidence graph: {json.dumps(run['evidence_graph'], indent=2)[:300]}")
            log(f"Latency: {run['latency']}s")
            log(f"Retry count: {run['retry_count']}")
            log(f"Abstained: {run['abstained']}")
            log(f"Answer preview: {run['answer_preview'][:300]!r}")

            run["passed"] = (
                run["faithfulness"] > 0
                and run["num_docs"] > 0
                and len(state["answer"]) > 50
            )
            log("Pipeline completed successfully", "PASS" if run["passed"] else "FAIL")

        except Exception as e:
            run["error"] = str(e)
            run["traceback"] = traceback.format_exc()
            run["passed"] = False
            log(f"Pipeline FAILED: {e}", "FAIL")
            print(f"  {traceback.format_exc()}")

        results["runs"].append(run)

    results["passed"] = all(r.get("passed", False) for r in results["runs"])
    report["steps"]["3_pipeline"] = results
    return results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Citation Verification
# ──────────────────────────────────────────────────────────────────────────────

def step4_citations(pipeline_results: dict) -> dict:
    section("STEP 4 — Citation Verification")
    results = {"runs": []}

    for run in pipeline_results.get("runs", []):
        if run.get("error"):
            continue

        cr = {
            "query": run["query"],
            "citation_precision": run["citation_precision"],
            "unsupported_claim_rate": run["unsupported_claim_rate"],
            "num_claims": run["num_claims"],
            "evidence_graph": run.get("evidence_graph", {}),
        }

        log(f"Query: {run['query'][:80]}...")
        log(f"  Citation precision: {cr['citation_precision']:.4f}")
        log(f"  Unsupported claim rate: {cr['unsupported_claim_rate']:.4f}")
        log(f"  Total claims: {cr['num_claims']}")

        eg = cr.get("evidence_graph", {})
        if eg:
            log(f"  Evidence graph: {eg.get('supported_claims', 0)}/{eg.get('total_claims', 0)} claims supported (coverage={eg.get('coverage_ratio', 0):.4f})")

        cr["passed"] = cr["citation_precision"] > 0 or cr["num_claims"] == 0
        log("Citation verification OK" if cr["passed"] else "Citation verification issues detected",
            "PASS" if cr["passed"] else "WARN")
        results["runs"].append(cr)

    results["passed"] = all(r["passed"] for r in results["runs"]) if results["runs"] else False
    report["steps"]["4_citations"] = results
    return results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Research Controller Behavior
# ──────────────────────────────────────────────────────────────────────────────

def step5_controller(pipeline_results: dict) -> dict:
    section("STEP 5 — Research Controller Behavior")
    results = {"runs": [], "controller_test": {}}

    from app.research.controller import ResearchSignals, decide_next_action

    # 5a. Check observed behavior from pipeline runs
    for run in pipeline_results.get("runs", []):
        if run.get("error"):
            continue
        cr = {
            "query": run["query"][:80],
            "controller_action": run["controller_action"],
            "evidence_coverage": run["evidence_coverage"],
            "hops": run["hops"],
            "max_hops": run["max_hops"],
            "retrieval_diagnostics": run.get("retrieval_diagnostics", {}),
        }
        log(f"Query: {cr['query']}...")
        log(f"  Controller decision: {cr['controller_action']}")
        log(f"  Evidence coverage: {cr['evidence_coverage']:.4f}")
        log(f"  Hops used: {cr['hops']}/{cr['max_hops']}")
        log(f"  Diagnostics: {json.dumps(cr['retrieval_diagnostics'])}")
        results["runs"].append(cr)

    # 5b. Unit-test the controller with synthetic signals
    test_cases = [
        ("Stop (high coverage)", ResearchSignals(evidence_coverage=0.8, retrieval_diversity=0.5, evaluator_confidence=0.7, estimated_recall_proxy=0.6, hop=1, max_hops=2, has_followup_query=False), "stop"),
        ("Extra hop (low coverage, followup available)", ResearchSignals(evidence_coverage=0.2, retrieval_diversity=0.3, evaluator_confidence=0.5, estimated_recall_proxy=0.4, hop=1, max_hops=3, has_followup_query=True), "extra_hop"),
        ("Reformulate (low coverage, no followup)", ResearchSignals(evidence_coverage=0.2, retrieval_diversity=0.3, evaluator_confidence=0.5, estimated_recall_proxy=0.4, hop=1, max_hops=3, has_followup_query=False), "reformulate"),
        ("Abstain (hops exhausted, very low evidence)", ResearchSignals(evidence_coverage=0.05, retrieval_diversity=0.1, evaluator_confidence=0.2, estimated_recall_proxy=0.1, hop=2, max_hops=2, has_followup_query=False), "abstain"),
        ("Stop (hops exhausted, okay coverage)", ResearchSignals(evidence_coverage=0.5, retrieval_diversity=0.5, evaluator_confidence=0.7, estimated_recall_proxy=0.5, hop=2, max_hops=2, has_followup_query=False), "stop"),
        ("Reformulate (low diversity)", ResearchSignals(evidence_coverage=0.5, retrieval_diversity=0.1, evaluator_confidence=0.7, estimated_recall_proxy=0.5, hop=1, max_hops=3, has_followup_query=False), "reformulate"),
        ("Extra hop (low confidence)", ResearchSignals(evidence_coverage=0.5, retrieval_diversity=0.5, evaluator_confidence=0.2, estimated_recall_proxy=0.5, hop=1, max_hops=3, has_followup_query=False), "extra_hop"),
        ("Reformulate (low recall proxy)", ResearchSignals(evidence_coverage=0.5, retrieval_diversity=0.5, evaluator_confidence=0.5, estimated_recall_proxy=0.1, hop=1, max_hops=3, has_followup_query=False), "reformulate"),
    ]

    controller_results = []
    for name, signals, expected in test_cases:
        actual = decide_next_action(signals)
        passed = actual == expected
        controller_results.append({"name": name, "expected": expected, "actual": actual, "passed": passed})
        log(f"{name}: expected={expected}, got={actual}", "PASS" if passed else "FAIL")

    results["controller_test"] = controller_results
    results["passed"] = all(r["passed"] for r in controller_results)
    report["steps"]["5_controller"] = results
    return results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Abstention Behavior
# ──────────────────────────────────────────────────────────────────────────────

async def step6_abstention() -> dict:
    section("STEP 6 — Abstention Behavior")
    results = {}

    from app.agent.graph import run_agent
    from app.optimization.bandit import CONFIG_NAMES

    alpha = {c: 1.0 for c in CONFIG_NAMES}
    beta = {c: 1.0 for c in CONFIG_NAMES}

    # Query about a topic completely unrelated to AegisUI paper
    abstention_queries = [
        "What is the GDP of France in 2024 and how does it compare to Germany?",
        "Explain the process of photosynthesis in C4 plants",
    ]

    abstention_results = []
    for query in abstention_queries:
        print(f"\n  Query: {query}")
        try:
            state = await run_agent(query, alpha, beta)
            ar = {
                "query": query,
                "abstained": state.get("abstained", False),
                "controller_action": state["controller_action"],
                "evidence_coverage": state["evidence_coverage"],
                "faithfulness": state["faithfulness"],
                "answer_preview": state["answer"][:300],
                "num_docs": len(state["all_docs"]),
                "retrieval_diagnostics": state.get("retrieval_diagnostics", {}),
            }
            log(f"Abstained: {ar['abstained']}")
            log(f"Controller action: {ar['controller_action']}")
            log(f"Evidence coverage: {ar['evidence_coverage']:.4f}")
            log(f"Faithfulness: {ar['faithfulness']:.4f}")
            log(f"Docs retrieved: {ar['num_docs']}")
            log(f"Answer: {ar['answer_preview'][:200]!r}")

            if ar["abstained"]:
                log("System correctly abstained from answering", "PASS")
            else:
                log("System did NOT abstain — checking if evidence was used", "WARN")
                report["warnings"].append(f"Abstention not triggered for unrelated query: {query[:60]}")

            ar["passed"] = True  # We're observing, not strictly failing
            abstention_results.append(ar)

        except Exception as e:
            log(f"Error: {e}", "FAIL")
            abstention_results.append({"query": query, "error": str(e), "passed": False})

    results["queries"] = abstention_results
    results["any_abstained"] = any(r.get("abstained", False) for r in abstention_results)
    results["passed"] = True  # Observational step
    report["steps"]["6_abstention"] = results
    return results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — Bandit Learning Loop
# ──────────────────────────────────────────────────────────────────────────────

async def step7_bandit() -> dict:
    section("STEP 7 — Bandit Learning Loop")
    results = {}

    from app.optimization.bandit import ThompsonSamplingBandit, CONFIG_NAMES, compute_reward

    # 7a. Test bandit mechanics
    bandit = ThompsonSamplingBandit()
    log(f"Initial bandit alpha: {dict(bandit.alpha)}")
    log(f"Initial bandit beta:  {dict(bandit.beta)}")

    # Simulate multiple rewards favoring config B and C
    simulated_rewards = [
        ("A", 0.3), ("B", 0.9), ("C", 0.8), ("D", 0.4),
        ("B", 0.85), ("C", 0.75), ("B", 0.95), ("A", 0.5),
        ("B", 0.88), ("C", 0.82), ("D", 0.6), ("B", 0.92),
    ]

    for config, utility in simulated_rewards:
        bandit.update(config, utility)

    log(f"After 12 updates — alpha: {dict(bandit.alpha)}")
    log(f"After 12 updates — beta:  {dict(bandit.beta)}")

    stats = bandit.stats()
    for name, s in stats.items():
        log(f"  Config {name}: win_rate={s['win_rate']:.3f}, trials={s['trials']}")

    # B should have highest win rate
    b_wr = stats["B"]["win_rate"]
    a_wr = stats["A"]["win_rate"]
    results["b_dominates_a"] = b_wr > a_wr
    log(f"B win rate ({b_wr:.3f}) > A win rate ({a_wr:.3f}): {results['b_dominates_a']}", 
        "PASS" if results["b_dominates_a"] else "WARN")

    # 7b. Test config selection bias
    selections = {"A": 0, "B": 0, "C": 0, "D": 0}
    for _ in range(200):
        choice = bandit.select_config()
        selections[choice] += 1

    log(f"Selection distribution over 200 samples: {selections}")
    results["selection_distribution"] = selections
    results["b_or_c_preferred"] = selections["B"] + selections["C"] > selections["A"] + selections["D"]
    log(f"B+C selected more than A+D: {results['b_or_c_preferred']}", 
        "PASS" if results["b_or_c_preferred"] else "WARN")

    # 7c. Test compute_reward
    reward = compute_reward(faithfulness=0.9, citation_precision=0.8, answer_completeness=0.7, latency=5.0)
    log(f"compute_reward(faith=0.9, cit=0.8, comp=0.7, lat=5.0) = {reward:.4f}")
    results["reward_positive"] = reward > 0
    log(f"Reward is positive: {results['reward_positive']}", "PASS" if results["reward_positive"] else "FAIL")

    # 7d. Test DB persistence (using real DB)
    try:
        from app.database import async_session_factory, init_db
        await init_db()
        
        from app.memory.strategy_memory import load_bandit, save_bandit

        async with async_session_factory() as session:
            # Save bandit with known state
            test_bandit = ThompsonSamplingBandit()
            test_bandit.alpha = {"A": 5.0, "B": 10.0, "C": 8.0, "D": 3.0}
            test_bandit.beta = {"A": 8.0, "B": 3.0, "C": 4.0, "D": 7.0}
            await save_bandit(session, "factual", test_bandit)

            # Reload and verify
            loaded = await load_bandit(session, "factual")
            log(f"Loaded bandit alpha: {dict(loaded.alpha)}")
            log(f"Loaded bandit beta:  {dict(loaded.beta)}")
            match = (loaded.alpha.get("B", 0) == 10.0 and loaded.beta.get("B", 0) == 3.0)
            results["db_persistence"] = match
            log(f"DB persistence verified: {match}", "PASS" if match else "FAIL")

    except Exception as e:
        log(f"DB persistence test failed: {e}", "FAIL")
        results["db_persistence"] = False
        report["bugs"].append(f"Bandit DB persistence error: {e}")

    results["passed"] = results.get("b_dominates_a", False) and results.get("reward_positive", False)
    report["steps"]["7_bandit"] = results
    return results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 — Retrieval Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def step8_diagnostics(pipeline_results: dict) -> dict:
    section("STEP 8 — Retrieval Diagnostics")
    results = {"runs": []}

    from app.retrieval.diagnostics import compute_retrieval_diagnostics

    # 8a. Check diagnostics from pipeline runs
    for run in pipeline_results.get("runs", []):
        if run.get("error"):
            continue
        diag = run.get("retrieval_diagnostics", {})
        dr = {
            "query": run["query"][:80],
            "query_coverage": diag.get("query_coverage", -1),
            "document_diversity": diag.get("document_diversity", -1),
            "retrieval_redundancy": diag.get("retrieval_redundancy", -1),
            "estimated_recall_proxy": diag.get("estimated_recall_proxy", -1),
        }
        log(f"Query: {dr['query']}...")
        log(f"  query_coverage:         {dr['query_coverage']}")
        log(f"  document_diversity:      {dr['document_diversity']}")
        log(f"  retrieval_redundancy:    {dr['retrieval_redundancy']}")
        log(f"  estimated_recall_proxy:  {dr['estimated_recall_proxy']}")

        has_all = all(v >= 0 for v in [dr["query_coverage"], dr["document_diversity"], dr["retrieval_redundancy"], dr["estimated_recall_proxy"]])
        dr["has_all_metrics"] = has_all
        log(f"  All metrics present: {has_all}", "PASS" if has_all else "FAIL")
        results["runs"].append(dr)

    # 8b. Direct unit test
    diag = compute_retrieval_diagnostics(
        "AegisUI anomaly detection",
        ["AegisUI anomaly detection", "behavioral anomaly detection UI", "AegisUI framework evaluation"],
        [
            {"text": "AegisUI detects behavioral anomalies in structured UI payloads", "source": "test1"},
            {"text": "The framework uses Random Forest for supervised detection with F1 of 0.843", "source": "test2"},
            {"text": "Isolation Forest provides unsupervised anomaly detection baseline", "source": "test3"},
        ]
    )
    log(f"Direct diagnostics test: {json.dumps(diag)}")
    results["direct_test"] = diag

    results["passed"] = all(r.get("has_all_metrics", False) for r in results["runs"]) if results["runs"] else True
    report["steps"]["8_diagnostics"] = results
    return results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 9 — Stress Test
# ──────────────────────────────────────────────────────────────────────────────

async def step9_stress() -> dict:
    section("STEP 9 — Stress Test")
    results = {"runs": []}

    from app.agent.graph import run_agent
    from app.optimization.bandit import CONFIG_NAMES

    alpha = {c: 1.0 for c in CONFIG_NAMES}
    beta = {c: 1.0 for c in CONFIG_NAMES}

    stress_queries = [
        # Factual
        "How many payloads are in the AegisUI dataset?",
        "What is the architecture of the autoencoder used in AegisUI?",
        # Analytical
        "What are the limitations of the AegisUI framework as discussed by the authors?",
        "Why does the autoencoder need no malicious labels at training time?",
        # Multi-hop
        "How do the feature extraction dimensions (structural, semantic, binding, session) collectively contribute to attack detection, and which feature group is most important?",
        # Ambiguous
        "Is AegisUI good enough for production use?",
        "What should be done next?",
    ]

    for query in stress_queries:
        print(f"\n  Query: {query[:80]}")
        try:
            t0 = time.monotonic()
            state = await run_agent(query, alpha, beta)
            elapsed = time.monotonic() - t0

            sr = {
                "query": query,
                "query_type": state["query_type"],
                "config": state["current_config"],
                "faithfulness": state["faithfulness"],
                "citation_precision": state.get("citation_precision", 0),
                "evidence_coverage": state["evidence_coverage"],
                "controller_action": state["controller_action"],
                "abstained": state.get("abstained", False),
                "hops": state["hop"],
                "latency": round(elapsed, 2),
                "answer_length": len(state["answer"]),
            }
            log(f"Type={sr['query_type']}, Config={sr['config']}, Faith={sr['faithfulness']:.3f}, "
                f"CitPrec={sr['citation_precision']:.3f}, Cov={sr['evidence_coverage']:.3f}, "
                f"Action={sr['controller_action']}, Hops={sr['hops']}, Lat={sr['latency']}s, "
                f"AnsLen={sr['answer_length']}")
            sr["passed"] = sr["answer_length"] > 20
            results["runs"].append(sr)

        except Exception as e:
            log(f"FAILED: {e}", "FAIL")
            results["runs"].append({"query": query, "error": str(e), "passed": False})
            report["bugs"].append(f"Stress test failed for query: {query[:60]} — {e}")

    results["total"] = len(results["runs"])
    results["succeeded"] = sum(1 for r in results["runs"] if r.get("passed", False))
    results["passed"] = results["succeeded"] == results["total"]
    log(f"\nStress test: {results['succeeded']}/{results['total']} queries succeeded",
        "PASS" if results["passed"] else "WARN")

    report["steps"]["9_stress"] = results
    return results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 10 — Final Validation Report
# ──────────────────────────────────────────────────────────────────────────────

def step10_report():
    section("STEP 10 — Final Validation Report")

    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │              SYSTEM VALIDATION SUMMARY                      │")
    print("  └─────────────────────────────────────────────────────────────┘\n")

    all_passed = True
    for step_name, step_data in report["steps"].items():
        passed = step_data.get("passed", False)
        status = PASS if passed else FAIL
        if not passed:
            all_passed = False
        print(f"  {status} {step_name}")

    if report["bugs"]:
        print(f"\n  Bugs discovered ({len(report['bugs'])}):")
        for bug in report["bugs"]:
            print(f"    {FAIL} {bug}")

    if report["warnings"]:
        print(f"\n  Warnings ({len(report['warnings'])}):")
        for warn in report["warnings"]:
            print(f"    {WARN} {warn}")

    # Detailed metrics summary
    pipeline_data = report["steps"].get("3_pipeline", {})
    if pipeline_data.get("runs"):
        print("\n  --- Pipeline Metrics Summary ---")
        for run in pipeline_data["runs"]:
            if "error" in run:
                continue
            print(f"  Query: {run['query'][:60]}...")
            print(f"    Faithfulness: {run.get('faithfulness', 0):.4f}")
            print(f"    Citation Precision: {run.get('citation_precision', 0):.4f}")
            print(f"    Answer Completeness: {run.get('answer_completeness', 0):.4f}")
            print(f"    Evidence Coverage: {run.get('evidence_coverage', 0):.4f}")
            print(f"    Config: {run.get('config', '?')}, Hops: {run.get('hops', 0)}")
            print(f"    Latency: {run.get('latency', 0)}s")

    abstention_data = report["steps"].get("6_abstention", {})
    if abstention_data:
        print(f"\n  --- Abstention ---")
        print(f"  Any query triggered abstention: {abstention_data.get('any_abstained', False)}")

    bandit_data = report["steps"].get("7_bandit", {})
    if bandit_data:
        print(f"\n  --- Bandit Learning ---")
        print(f"  B dominates A: {bandit_data.get('b_dominates_a', False)}")
        print(f"  DB persistence: {bandit_data.get('db_persistence', False)}")
        print(f"  Selection distribution: {bandit_data.get('selection_distribution', {})}")

    print(f"\n  {'='*60}")
    if all_passed and not report["bugs"]:
        print(f"  {PASS} FINAL VERDICT: The system behaves as a properly")
        print(f"  functioning Meta-RAG research agent when operating on")
        print(f"  real documents.")
    else:
        print(f"  {WARN} FINAL VERDICT: The system is partially functional.")
        print(f"  Some issues were detected — see details above.")
    print(f"  {'='*60}")

    return all_passed


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

async def main():
    print("\n" + "="*70)
    print("  META-RAG RESEARCH AGENT — END-TO-END SYSTEM VALIDATION")
    print("="*70)

    # Step 1: Ingestion
    ingestion_results = step1_ingestion()

    if not ingestion_results.get("passed"):
        log("Ingestion failed — cannot proceed with remaining steps", "FAIL")
        step10_report()
        return

    # Step 2: Retrieval
    step2_retrieval()

    # Step 3: Full pipeline
    pipeline_results = await step3_pipeline()

    # Step 4: Citation correctness
    step4_citations(pipeline_results)

    # Step 5: Controller behavior
    step5_controller(pipeline_results)

    # Step 6: Abstention
    await step6_abstention()

    # Step 7: Bandit
    await step7_bandit()

    # Step 8: Diagnostics
    step8_diagnostics(pipeline_results)

    # Step 9: Stress test
    await step9_stress()

    # Step 10: Final report
    step10_report()


if __name__ == "__main__":
    asyncio.run(main())
