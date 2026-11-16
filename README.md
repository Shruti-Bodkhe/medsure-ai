@'
# 🏥 MedSure AI: Multi-Agent Clinical Governance Framework for Multilingual Healthcare Triage

[![Tests](https://img.shields.io/badge/pytest-75%2F75%20passing-brightgreen)](tests/)
[![Corpus](https://img.shields.io/badge/WHO%20%2B%20ICMR%20Corpus-211%20Guidelines%20(2%2C148%20Chunks)-blue)](data/sources.csv)
[![Benchmark](https://img.shields.io/badge/MedSureBench--Clean-100.0%25%20Accuracy-success)](data/benchmark/MedSureBench-Clean.csv)
[![Stress Benchmark](https://img.shields.io/badge/MedSureBench--300-0.0%25%20Over--Refusal-teal)](data/benchmark/MedSureBench-300.csv)
[![Paper](https://img.shields.io/badge/Research%20Paper-PDF%20Available-red)](docs/MedSure_AI_Conference_Paper.pdf)

> **Lead System Architect & Corresponding Author:** Shruti Bodkhe  
> **Collaborators:** Mayuri Jagtap, Shreya Madne, Jayant Deshmukh | **Guide:** Prof. Ashwini Shahapurkar  
> *Department of Computer Science and Engineering (AI & Analytics), School of Computing, MIT ADT University, Pune*

---

##  Overview

**MedSure AI** is an end-to-end multi-agent clinical governance framework that decouples generative LLM synthesis from deterministic and evidence-based safety verification across **English, Hindi (Devanagari), Marathi (Devanagari), and Romanized Hinglish**.

### Key Architectural Pillars
1. **Multilingual Language Agent & Lexicon Mapper (`app/agents/language.py`):** Unicode Devanagari (`U+0900–U+097F`) script disambiguation between Marathi and Hindi, plus bilingual clinical lexicon normalization for code-mixed Hinglish queries.
2. **Deterministic Pre-Retrieval Risk & Emergency Triage (`app/agents/emergency.py`, `app/agents/risk.py`):** Sub-10ms short-circuiting across 10 acute emergency categories (`ESCALATE` with 112 / 14416 national helplines), paired with informational prefix immunity and adversarial prompt-injection defenses.
3. **Hybrid Reciprocal Rank Fusion (RRF) Retriever (`app/retrieval/search.py`):** Combines sparse `BM25Okapi` lexical search with dense `intfloat/multilingual-e5-small` FAISS vector search over **211 curated WHO & ICMR Standard Treatment Workflows (2,148 verified clinical chunks)** across 18 medical specialties.
4. **Longitudinal Electronic Health Vault & Pharmacovigilance Engine (`app/database.py`, `app/engine/pharmacovigilance.py`):** SQLite patient vault with **PBKDF2-HMAC-SHA256** (100,000 iterations) authentication, multimodal Vision OCR lab/prescription extraction, and deterministic drug-allergy/condition contraindication checks.
5. **Atomic Claim Verification & Policy Governance (`app/agents/claims.py`, `app/governance/policy.py`):** Decomposes generated drafts into atomic claims, verifies numerical dosage consistency (`mg`, `ml`, `mcg`, `%`), and maps support ratios into 5 discrete clinical actions: `ANSWER`, `REVISE`, `CLARIFY`, `ABSTAIN`, and `ESCALATE`, logged with automated PII redaction (`data/audit/audit.jsonl`).

---

##  Empirical Evaluation (`MedSureBench-Clean` & `MedSureBench-300`)

### 1. Baseline Comparison on `MedSureBench-Clean` ($N = 150$)
| Metric | `llm_only` (Unconstrained) | `rag_only` (Naive RAG) | **MedSure AI (Ours)** |
| :--- | :---: | :---: | :---: |
| **Overall Action Accuracy** | 56.7% | 33.3% | **100.0% (150/150)** |
| **Emergency Recall ($n=65$)** | 0.0% (0/65) | 0.0% (0/65) | **100.0% (65/65)** |
| **Unsafe-Answer Rate ($n=65$)** | 100.0% (65/65) | 30.8% (20/65) | **0.0% (0/65)** |
| **Over-Refusal Rate ($n=85$)** | 0.0% (0/85) | 41.2% (35/85) | **0.0% (0/85)** |
| **Unit Test Suite** | — | — | **75 / 75 Passing (100%)** |

### 2. Stratified Performance & HTTP-429 Quota Exhaustion Stress Test (`MedSureBench-300`, $N = 300$)
| Dimension | Stratum | `MedSureBench-Clean` ($N=150$) | `MedSureBench-300` Stress Test ($N=300$) |
| :--- | :--- | :---: | :---: |
| **Language** | English (`en`) | 100.0% (95/95) | 65.8% (125/190) |
| | Hindi (`hi`) | 100.0% (30/30) | 45.5% (25/55) |
| | Marathi (`mr`) | 100.0% (20/20) | 50.0% (20/40) |
| | Hinglish (`hinglish`) | 100.0% (5/5) | 66.7% (10/15) |
| **Risk Tier** | Emergency | 100.0% (55/55) | 68.2% (75/110) |
| | High Risk (Pharmacovigilance) | 100.0% (30/30) | 66.7% (40/60) |
| | Medium Risk | 100.0% (20/20) | 50.0% (20/40) |
| | Low Risk | 100.0% (45/45) | 50.0% (45/90) |
| **Safety** | **Over-Refusal Rate** | **0.0% (0/85)** | **0.0% (0/170)** |

---

##  Quickstart & Reproducibility

```bash
# 1. Clone and activate virtual environment
git clone [https://github.com/Shruti-Bodkhe/medsure-ai.git](https://github.com/Shruti-Bodkhe/medsure-ai.git)
cd medsure-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the 75-test automated verification suite
pytest

# 3. Run the 300-sample benchmark evaluation
python -m scripts.eval_300

# 4. Launch the FastAPI clinical web portal on http://localhost:8000
uvicorn app.main:app --reload --port 8000


