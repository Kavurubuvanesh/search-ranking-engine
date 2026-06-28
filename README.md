<div align="center">
  
# 🔎 Enterprise Search Ranking Engine 

*A Learning-to-Rank (LTR) microservice powered by LambdaMART and FastAPI.*

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-F37021?logo=lightgbm&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

</div>

---

## ⚡ Overview
This microservice provides real-time predictive ranking for search engine results. Using the **LambdaMART** algorithm, it ingests candidate documents and their mathematical feature vectors, scoring and sorting them to optimize for search relevance.

The underlying model was trained on the **Microsoft Learning to Rank (MSLR-WEB10K)** dataset, processing over 1.2 million query-document pairs to optimize the **NDCG@10** (Normalized Discounted Cumulative Gain) metric.

## 🏗️ Architecture & Implementation
*   **Algorithm:** LambdaMART (Pairwise Learning-to-Rank)
*   **Core Frameworks:** LightGBM, scikit-learn, NumPy
*   **API Layer:** FastAPI, Uvicorn, Pydantic (Strict Schema Validation)
*   **Features:** Evaluates 136 dense mathematical features per document (including BM25, PageRank, and LMIR scores) to determine relative competitive ranking.

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone [https://github.com/Kavurubuvanesh/search-ranking-engine.git](https://github.com/Kavurubuvanesh/search-ranking-engine.git)
cd search-ranking-engine
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Boot the API Engine
```bash
cd api
uvicorn main:app --port 8001
```

### 3. Test the Endpoint
Run the included simulation script to fire candidate documents at the engine:
```bash
python test_api.py
```

---

## 📡 API Schema
**POST `/rank`**
Accepts a search query and a list of candidate documents with their 136-dimensional feature vectors, returning a strictly ordered array.

```json
{
  "query_id": "google_search_query_8472",
  "results": [
    {
      "doc_id": "url_candidate_003",
      "score": 0.7741,
      "rank": 1
    },
    {
      "doc_id": "url_candidate_001",
      "score": 0.7240,
      "rank": 2
    }
  ]
}
```