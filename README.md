# 🩺 MedSure AI: A Multi-Agent Clinical Governance Framework for Multilingual Healthcare Triage

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research Prototype](https://img.shields.io/badge/Status-Conference%20Submission-success.svg)]()

> **Abstract:** MedSure AI is an open-source, multi-agent clinical governance framework engineered to provide safe, evidence-verified, and multilingual health triage across English, Hindi, Marathi, and Hinglish. By coupling deterministic emergency short-circuiting with Hybrid Reciprocal Rank Fusion (RRF) retrieval and automated claim verification, MedSure AI bridges the gap between raw Large Language Model generations and strict medical safety standards.

---

## 🏛️ System Architecture

MedSure AI implements a multi-tier governance pipeline that intercepts user queries before unverified LLM generation can occur:

```mermaid
graph TD
    A[User Query / OCR Upload] --> B[Language & Risk Assessment Agent]
    B -->|Emergency / Red Flag| C[Deterministic Short-Circuit: ESCALATE (112)]
    B -->|General / Symptom Query| D[Hybrid RRF Retrieval: BM25 + FAISS]
    D --> E[Tier-2 Evidence-Guided Clinical Synthesis]
    E --> F[Claim Verification & Contradiction Filter]
    F --> G[Response Responsiveness Judge]
    G --> H[Multilingual Tone & Dosage Preservation]
    H --> I[Governed Output + Evidence Citations]