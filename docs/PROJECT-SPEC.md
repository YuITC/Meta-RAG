# Autonomous Research Agent with Adaptive Retrieval Optimization

> **Technical Specification Document**
>
> Includes design decisions, discussion notes & implementation guidance

---

# 1. Project Overview

LLM-based Retrieval-Augmented Generation (RAG) systems typically rely on **static retrieval configurations** — fixed `top_k`, `chunk_size`, and reranking strategy. However, different query types require fundamentally different retrieval strategies:

| Query Type           | Retrieval Requirement                     |
| -------------------- | ----------------------------------------- |
| Factual lookup       | Shallow retrieval, low `top_k`            |
| Comparative analysis | Multiple relevant documents, high `top_k` |
| Multi-hop reasoning  | Iterative retrieval across hops           |

A single static configuration causes:

- Increased hallucination rate
- Weak citation grounding
- Unnecessary retrieval cost
- Higher latency

This project builds an **Autonomous Research Agent** that evaluates its own output quality and **adapts retrieval strategies over time under a fixed cost constraint**.

Instead of only answering questions, the system acts as a **meta-agent that continuously optimizes its own RAG pipeline**.

---

# 2. Core Capabilities

## 2.1 Iterative Multi-Hop Retrieval

1. Generate follow-up queries from intermediate evidence
2. Perform second-hop retrieval if necessary
3. Allows answering multi-hop reasoning questions (**max 2 hops**)

## 2.2 Hybrid Retrieval

1. Dense embedding search (`bge-small-en-v1.5`)
2. Sparse lexical search (**BM25**)
3. Dense + BM25 run **in parallel**, then merged and deduplicated
4. Cross-encoder reranking (`bge-reranker-base`, optional per config)

## 2.3 Citation-Aware Answer Generation

1. Extract evidence spans from retrieved documents
2. Generate grounded responses with explicit citations
3. Reduces hallucination and improves answer transparency

## 2.4 Self-Evaluation (Lightweight)

The system evaluates its own answers using **a single LLM call** — not full RAGAS at runtime.

RAGAS is reserved for **offline benchmarking only**.

1. Faithfulness score: fraction of answer statements supported by context
2. If `faithfulness < threshold (0.7)`, retry once with a different config
3. Both runs logged to the Bandit

> **Design Decision — Why not full RAGAS at runtime?**
>
> RAGAS internally calls the LLM **2–3 times per metric computation**.
> With retries, one query could trigger **8–10 LLM calls**, exhausting a **$10 budget** quickly.
>
> **Solution**
>
> - Lightweight single-call evaluator at runtime
> - RAGAS used **offline only**

---

## 2.5 Adaptive Retrieval Optimization (Thompson Sampling)

The system maintains **3 retrieval configurations** and selects among them using **Thompson Sampling**.

| Config | top_k | chunk_size | rerank | Intended Strategy                  |
| ------ | ----- | ---------- | ------ | ---------------------------------- |
| A      | 5     | 256        | No     | Fast & cheap — factual queries     |
| B      | 10    | 512        | No     | Broad recall — comparative queries |
| C      | 10    | 512        | Yes    | High precision — multi-hop queries |

### Utility Function

```
Utility = Faithfulness − λ₁ × Cost_norm − λ₂ × Latency_norm
```

Where

```
λ₁ = 0.3
λ₂ = 0.2
```

Priority ordering:

```
Faithfulness > Cost > Latency
```

Normalization:

```
Cost_norm = cost / 0.005
Latency_norm = latency / 15.0
```

## 2.6 Strategy Memory (Rule-Based)

Each query type has **its own Bandit instance**.

```python
bandits = {
    "factual":     ThompsonSamplingBandit(configs),
    "comparative": ThompsonSamplingBandit(configs),
    "multi_hop":   ThompsonSamplingBandit(configs),
}
```

> **Design Decision — Why not KMeans?**
>
> KMeans requires **50–100+ queries** before clusters stabilize. Demo sessions only have **20–50 queries**, producing near-random clusters.
>
> Additionally, re-clustering **invalidates Bandit history**.
>
> Therefore, **Rule-based classification at runtime** and **KMeans used offline** for analysis and visualization.

---

# 3. Data, Input & Output

## 3.1 Input

User Query: Text only

Documents:

- Supported formats: PDF, HTML, Markdown, DOCX, TXT
- Documents may contain **embedded images** (figures, tables, diagrams).
- Standalone image uploads (PNG/JPG/JPEG/WEBP) are **not supported**.

## 3.2 Data Sources

1. User uploads
2. Web scraper: top-50 trending papers from `huggingface.co/papers/trending` (backend/scraper/scaper.py)

Sources can be **combined**.

## 3.3 Document Processing Pipeline

All documents pass through **Unstructured**.

```
Document (PDF / HTML / MD / DOCX / TXT)
↓
Unstructured (element classification)
↓
Text elements → hierarchical chunking
Image elements → OCR (Tesseract)
↓
Embedding (bge-small-en-v1.5)
↓
Vector Index (Qdrant)
```

### Embedded Image Handling

- Unstructured detects images
- OCR extracts text
- OCR output indexed as normal text chunk
- No Vision LLM captioning

Trade-off:

- **Text images supported**
- **Pure visual diagrams skipped**

## 3.4 Output Format

Responses return a **structured Markdown report** including:

- Citation references `[1][2]`
- Faithfulness score
- Latency
- Cost
- Utility score

---

# 4. System Architecture

## 4.1 Agent Orchestration Layer

Workflow:

```
User Query
 → Planner
 → Retrieval Loop
 → Reader
 → Writer
 → Answer + Citations
```

### 4.1.1 Planner

Classifies query type.

Output:

```json
{
  "query_type": "multi_hop | comparative | factual",
  "max_hops": 1 | 2
}
```

### 4.1.2 Reader

Responsibilities:

- Extract evidence spans
- Identify missing information
- Generate follow-up queries

### 4.1.3 Writer

Responsibilities:

- Synthesize evidence
- Generate grounded answers with citations

## 4.2 Retrieval Layer

```
Query
   ↓
Dense Retrieval
    ‖
BM25 Retrieval
   ↓
Merge & Deduplicate
   ↓
Optional Rerank
   ↓
Top-N Context
```

Only **Config C uses reranking**.

## 4.3 Self-Improvement Layer

```
Answer
↓
Evaluator
↓
Faithfulness Score
↓
Bandit Update
↓
Retry if < 0.7
```

Retry rules:

- Retry **max once**
- Both runs logged
- Prevent **survivorship bias**

## 4.4 Memory Layer

Stores:

- `alpha` and `beta` for each config
- Full run history
  - config
  - utility
  - cost
  - latency

Offline **KMeans clustering** used for post-hoc analysis.

---

# 5. Tech Stack

| Component           | Technology           | Notes                      |
| ------------------- | -------------------- | -------------------------- |
| LLM                 | Gemini 2.5 Flash     | Planner, Writer, Evaluator |
| Agent orchestration | LangGraph            |                            |
| Dense embedding     | bge-small-en-v1.5    |                            |
| Sparse retrieval    | BM25                 |                            |
| Vector DB           | Qdrant               | Docker container           |
| Reranker            | bge-reranker-base    | Config C only              |
| Document parsing    | Unstructured         | OCR included               |
| Offline evaluation  | RAGAS                | Not runtime                |
| Backend             | FastAPI + PostgreSQL | Docker                     |
| Frontend            | Next.js + shadcn/ui  | Vercel                     |
| Package manager     | uv                   |                            |
| Deploy              | Docker Compose       |                            |

---

# 6. Thompson Sampling Implementation

## 6.1 Algorithm Choice

- Algorithm: Thompson Sampling
- Expore Logic: Bayesian posterior

## 6.2 Reward Binarization

```
success = 1  if utility >= 0.7
success = 0  otherwise
```

Loss of precision but keeps the algorithm correct.

## 6.3 Implementation Sketch

```python
class ThompsonSamplingBandit:

    def __init__(self, configs):
        self.configs = configs
        self.alpha = {c: 1 for c in configs} # Beta(1,1)
        self.beta  = {c: 1 for c in configs} # Beta(1,1)

    def select_config(self, exclude=None):
        samples = {
            c: np.random.beta(self.alpha[c], self.beta[c])
            for c in self.configs if c != exclude
        }
        return max(samples, key=samples.get)

    def update(self, config, utility, threshold=0.7):

        if utility >= threshold:
            self.alpha[config] += 1
        else:
            self.beta[config] += 1
```

---

# 7. Utility Function

## 7.1 Formula

```
Utility = Faithfulness − λ₁ × Cost_norm − λ₂ × Latency_norm
```

| Metric       | Range          | Normalization | Weight |
| ------------ | -------------- | ------------- | ------ |
| Faithfulness | 0–1            | none          | 1.0    |
| Cost         | $0.0001–$0.005 | cost/0.005    | λ₁=0.3 |
| Latency      | 3–15s          | latency/15    | λ₂=0.2 |

## 7.2 Implementation

```python
def compute_utility(faithfulness, cost, latency,
                    lambda1=0.3, lambda2=0.2):

    cost_norm    = cost / 0.005
    latency_norm = latency / 15.0

    return faithfulness \
           - lambda1 * cost_norm \
           - lambda2 * latency_norm
```

---

# 8. Self-Evaluation & Retry Logic

## 8.1 Evaluator Prompt

```text
Given the answer and retrieved context below, score the faithfulness from 0.0 to 1.0.

Faithfulness = fraction of answer statements supported by context.

Context: {context}
Answer: {answer}

Respond JSON:
{"faithfulness": float, "reasoning": str}
```

## 8.2 Retry Pipeline

```python
async def run_with_retry(query, bandit):

    config = bandit.select_config()

    answer, context, metrics = await run_pipeline(query, config)

    faithfulness = await evaluate(answer, context)

    if faithfulness < FAITHFULNESS_THRESHOLD:

        utility_fail = compute_utility(faithfulness, ...)
        bandit.update(config, utility_fail)

        fallback = bandit.select_config(exclude=config)

        answer, context, metrics = await run_pipeline(query, fallback)

        faithfulness = await evaluate(answer, context)

        utility = compute_utility(faithfulness, ...)
        bandit.update(fallback, utility)

    else:

        utility = compute_utility(faithfulness, ...)
        bandit.update(config, utility)

    return answer
```

---

# 9. Benchmark & Evaluation

## 9.1 Benchmark Datasets

| Dataset  | Objective                     | Type         |
| -------- | ----------------------------- | ------------ |
| HotpotQA | Multi-hop reasoning           | QA           |
| SciFact  | Scientific claim verification | Verification |

## 9.2 Experimental Protocol

Compare **Static RAG vs Adaptive Bandit RAG**

| Metric                | Static      | Adaptive Target |
| --------------------- | ----------- | --------------- |
| Faithfulness          | Fixed       | Higher          |
| Citation accuracy     | Fixed       | Higher          |
| Cost/query            | Fixed       | Lower           |
| Retrieval precision@k | Fixed       | Higher          |
| Utility               | Not tracked | Optimized       |

Baseline:

```
top_k=5
chunk_size=300
rerank=off
single-hop
```

---

# 10. Constraints

| Constraint     | Value | Notes           |
| -------------- | ----- | --------------- |
| Latency target | ≤15s  | end-to-end      |
| Budget         | ≤$10  | total           |
| Retry limit    | 1     | logged          |
| Max hops       | 2     | planner decides |
| Bandit configs | 3     | A,B,C           |
