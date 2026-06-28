from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import lightgbm as lgb
import numpy as np
import os


# --- 1. Data Validation Schemas (Pydantic) ---
class Document(BaseModel):
    doc_id: str
    # Enforce exactly 136 features, matching the MSLR-WEB10K architecture
    features: list[float] = Field(..., min_length=136, max_length=136)


class RankingRequest(BaseModel):
    query_id: str
    documents: list[Document]
    top_k: int = 10  # Default to returning the top 10 results


class RankedDocument(BaseModel):
    doc_id: str
    score: float
    rank: int


class RankingResponse(BaseModel):
    query_id: str
    results: list[RankedDocument]


# --- 2. Server Lifespan & Memory Management ---
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Boot sequence: Load the saved C++ LightGBM graph into memory
    model_path = "../models/lambdamart_baseline.txt"
    if os.path.exists(model_path):
        ml_models["ranker"] = lgb.Booster(model_file=model_path)
        print("✅ LambdaMART Engine loaded successfully into API memory.")
    else:
        raise RuntimeError(f"FATAL: Model not found at {model_path}")

    yield  # Server is running

    # Shutdown sequence: Clear memory
    ml_models.clear()


# --- 3. API Initialization ---
app = FastAPI(
    title="Search Ranking API",
    description="A microservice for real-time document ranking using LambdaMART",
    lifespan=lifespan
)


# --- 4. The Core Routing Logic ---
@app.post("/rank", response_model=RankingResponse)
def rank_documents(request: RankingRequest):
    ranker = ml_models.get("ranker")
    if not ranker:
        raise HTTPException(status_code=500, detail="Ranking engine offline.")

    if not request.documents:
        return RankingResponse(query_id=request.query_id, results=[])

    try:
        # 1. Extract raw features into a highly optimized NumPy array
        feature_matrix = np.array([doc.features for doc in request.documents])

        # 2. Execute LambdaMART inference to generate relative NDCG scores
        scores = ranker.predict(feature_matrix)

        # 3. Pair the original documents with their new scores and sort them descending
        scored_docs = list(zip(request.documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 4. Construct the strictly-typed response for the top_k results
        results = []
        for rank, (doc, score) in enumerate(scored_docs[:request.top_k], start=1):
            results.append(RankedDocument(
                doc_id=doc.doc_id,
                score=float(score),
                rank=rank
            ))

        return RankingResponse(query_id=request.query_id, results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))