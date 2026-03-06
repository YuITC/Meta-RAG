# Autonomous Research Agent with Adaptive Retrieval Optimization

---

# 1. Project Overview

## 1.1 Motivation

Large Language Model (LLM)-based **Retrieval-Augmented Generation (RAG)** systems typically rely on **static retrieval configurations**, such as fixed:

- `top_k`
- `chunk_size`
- reranking strategy

However, different query types require different retrieval strategies.

For example:

| Query Type           | Retrieval Requirement       |
| -------------------- | --------------------------- |
| Factual lookup       | shallow retrieval           |
| Comparative analysis | multiple relevant documents |
| Multi-hop reasoning  | iterative retrieval         |

Using a single static configuration introduces several problems:

- Increased hallucination rate
- Weak citation grounding
- Unnecessary retrieval cost
- Higher latency

This project proposes an **Autonomous Research Agent** capable of:

> Evaluating its own output quality and **adapting retrieval strategies over time under a fixed cost constraint**.

Instead of only answering questions, the system acts as a **meta-agent** that continuously optimizes its own RAG pipeline.

---

## 1.2 Core Capabilities

### 1. Iterative Multi-hop Retrieval

The system can perform retrieval across multiple steps.

Capabilities:

- Generate follow-up queries from intermediate evidence
- Perform second-hop retrieval if necessary

This allows the system to answer **multi-hop reasoning questions**.

---

### 2. Hybrid Retrieval

The retrieval pipeline combines:

- Dense embedding search
- Sparse lexical search (BM25)
- Optional cross-encoder reranking

This hybrid approach improves both **recall** and **precision**.

---

### 3. Citation-aware Answer Generation

The system produces answers with explicit citations.

Capabilities:

- Extract evidence spans from retrieved documents
- Generate grounded responses
- Attach citations to supporting evidence

This reduces hallucination and improves answer transparency.

---

### 4. Self-Evaluation

The system evaluates its own answers using LLM-based scoring.

Evaluation includes:

- Faithfulness
- Citation grounding
- Evidence coverage
- Cost tracking
- Latency tracking

If answer quality is below threshold, the system may retry using a different retrieval configuration.

---

### 5. Adaptive Retrieval Optimization

The system maintains multiple retrieval configurations and selects among them using a **Multi-Armed Bandit algorithm**.

Example retrieval configurations:

| Config | top_k | chunk_size | rerank |
| ------ | ----- | ---------- | ------ |
| A      | 5     | 200        | No     |
| B      | 8     | 300        | No     |
| C      | 8     | 300        | Yes    |

The system optimizes the following utility function:

[
Utility = Faithfulness - \lambda_1 \times Cost - \lambda_2 \times Latency
]

---

### 6. Strategy Memory

The system stores historical performance data:

- query embedding
- retrieval configuration used
- resulting utility score

Queries are clustered to learn **which retrieval strategy works best for each query type**.

---

## 1.3 Input

Supported input formats:

### Text

- PDF
- HTML
- Markdown
- DOCX
- TXT

### Image

- PNG
- JPG
- JPEG
- WEBP

Images may contain:

- figures
- diagrams
- tables

Processing pipeline:

```
Image
  ↓
OCR
  ↓
Caption extraction
  ↓
Index as text chunk
```

---

## 1.4 Output

The system produces:

- Structured Markdown reports
- Answers with citation references
- Cost tracking
- Latency tracking
- Utility score

Example output:

```
Answer:
Transformer models replaced RNNs due to improved parallelization
and attention mechanisms [1][2].

Latency: 8.2s
Cost: $0.002
Utility: 0.87
```

---

# 2. System Architecture

The system is divided into four main layers:

1. Agent Orchestration Layer
2. Retrieval Layer
3. Self-Improvement Layer
4. Memory Layer

---

# 2.1 Agent Orchestration Layer

The orchestration layer controls the overall reasoning workflow.

Pipeline:

```
User Query
    ↓
Planner
    ↓
Retrieval Loop (1–2 hops)
    ↓
Reader
    ↓
Writer
    ↓
Answer + Citations
```

---

### Planner

The planner determines the retrieval strategy.

Responsibilities:

- Classify query type
- Determine retrieval depth

Example output:

```json
{
  "query_type": "multi_hop",
  "max_hops": 2
}
```

Query types include:

- factual
- comparative
- multi-hop

---

### Reader

The reader processes retrieved context.

Responsibilities:

- Extract relevant evidence spans
- Identify missing information
- Generate follow-up queries if needed

Example follow-up query:

```
"What limitations of RNNs led to transformer adoption?"
```

---

### Writer

The writer generates the final answer using retrieved evidence.

Responsibilities:

- Synthesize information
- Produce grounded answers
- Attach citation references

---

# 2.2 Retrieval Layer

The retrieval layer implements hybrid search.

Pipeline:

```
Query
   ↓
Dense Retrieval (bge-small)
BM25 Retrieval
   ↓
Merge & Deduplicate
   ↓
Cross-Encoder Rerank
   ↓
Top-N Context
```

Dense and BM25 retrieval run **in parallel**.

---

### Retrieval Configurations

The system maintains several retrieval configurations.

Example:

| Config | top_k | chunk_size | rerank |
| ------ | ----- | ---------- | ------ |
| A      | 5     | 200        | No     |
| B      | 8     | 300        | No     |
| C      | 8     | 300        | Yes    |

A Multi-Armed Bandit algorithm selects the configuration with the highest expected utility.

---

# 2.3 Self-Improvement Layer

This layer evaluates answer quality and updates retrieval policies.

Pipeline:

```
Generated Answer
       ↓
Evaluator
       ↓
Metric Computation
       ↓
Bandit Optimizer
       ↓
Update Retrieval Policy
```

---

### Evaluation Metrics

The evaluator computes:

- Faithfulness
- Citation grounding ratio
- Context overlap
- Retrieval precision@k (when ground truth is available)
- Cost per query
- Latency

If:

```
Faithfulness < threshold
```

The system retries once using a different retrieval configuration.

---

# 2.4 Memory Layer

The memory layer stores historical performance.

Stored fields:

- query embedding
- retrieval configuration
- utility score
- cost
- latency

Queries are clustered using **KMeans**.

Runtime usage:

```
New Query
   ↓
Find nearest cluster
   ↓
Select best retrieval config for that cluster
```

This allows the system to learn **retrieval strategies for different query types**.

---

# 3. Tech Stack

---

# 3.1 LLM & Agent

Model used:

| Component | Model            |
| --------- | ---------------- |
| Planner   | Gemini 2.5 Flash |
| Writer    | Gemini 2.5 Flash |
| Evaluator | Gemini 2.5 Flash |

Agent orchestration:

- LangGraph

---

# 3.2 Embedding & Retrieval & Vector DB

Dense embedding:

```
bge-small-en-v1.5
```

Sparse retrieval:

```
BM25
```

Vector database:

```
Qdrant
```

Reranker:

```
bge-reranker-small
```

---

# 3.3 Evaluation & Observability

Evaluation tools:

- RAGAS (offline evaluation)

Metrics logging:

- cost
- latency
- retrieval config frequency

Logs stored as:

```
JSON logs
```

---

# 3.4 Backend

Backend framework:

```
FastAPI
```

Database:

```
PostgreSQL
```

Responsibilities:

- API serving
- query routing
- experiment logging

---

# 3.5 Frontend

Frontend deployment:

```
Vercel
```

UI requirements:

- minimal
- clean
- focused on system outputs

Display components:

- user query
- generated answer
- citations
- latency
- cost
- utility score

---

# 3.6 Deploy

Services deployed using:

```
Docker Compose
```

Containers:

- Qdrant
- PostgreSQL
- API server

---

# 3.7 Others

Package manager:

```
uv
```

Testing strategy:

### Unit tests

- retrieval pipeline
- metric computation

### Integration tests

- end-to-end query pipeline

---

# 4. Dataset & Benchmark

---

# 4.1 Domain Corpus

The domain corpus consists of **AI research papers**.

Sources:

- arXiv (AI category)
- OpenReview

Corpus size:

```
~100 research papers
```

Processing pipeline:

```
PDF (PyMuPDF)
   ↓
Text parsing
   ↓
Hierarchical chunking
   ↓
Embedding
   ↓
Vector indexing
```

Hierarchical chunking structure:

```
Section
   ↓
Paragraph
   ↓
Chunk
```

---

# 4.2 Benchmark Datasets & How to Use

| Dataset  | Objective                     | Type         |
| -------- | ----------------------------- | ------------ |
| HotpotQA | multi-hop reasoning           | QA           |
| SciFact  | scientific claim verification | verification |

Usage:

HotpotQA:

- evaluate multi-hop reasoning
- test retrieval depth strategies

SciFact:

- evaluate evidence grounding
- test citation correctness

---

# 4.3 Evaluation Metrics

Quality metrics:

- F1
- Faithfulness
- Citation accuracy
  (percentage of answer statements supported by evidence)

Retrieval metrics:

- Retrieval precision@k (when ground truth evidence available)

System metrics:

- Utility score
- Cost per query
- Latency

---

# 4.4 Experimental Protocol

The experiment compares:

```
1. Static RAG baseline
2. Adaptive Bandit RAG
```

Example baseline configuration:

```
top_k = 5
chunk_size = 300
rerank = off
single-hop retrieval
```

The experiment aims to demonstrate:

- Reduced hallucination rate
- Improved citation grounding
- Lower retrieval cost under fixed budget

---

# 5. Other Technical Requirements

Latency target:

```
≤ 15 seconds per query
```

Latency measured from:

```
user query received → final answer returned
```

Budget constraint:

```
Total project cost ≤ $10
```

Cost control strategies:

- limit self-evaluation calls
- retry at most once
- minimize LLM calls

---

# 6. Example Execution Flow

Example query:

```
Why did Transformer models replace RNNs in NLP?
```

Execution steps:

```
User Query
     ↓
Planner classifies query
     ↓
Bandit selects retrieval config
     ↓
Hybrid retrieval (dense + BM25)
     ↓
Reranker selects best context
     ↓
Reader extracts evidence
     ↓
Writer generates answer
     ↓
Evaluator scores answer
     ↓
Utility computed
     ↓
Bandit updates policy
     ↓
Memory stores result
     ↓
Return answer to user
```

The system gradually learns:

```
Which retrieval strategy works best for each query type.
```

This enables continuous **adaptive optimization of the RAG pipeline**.
