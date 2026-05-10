"""
PARSON API SERVER
----------------
Exposes PARSON recommender engine as HTTP API
"""

# -------------------------------
# Imports
# -------------------------------
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel

# IMPORTANT:
# Import recommender using package path
from backend.recommender import recommend_books

# -------------------------------
# Create FastAPI app
# -------------------------------

app = FastAPI(
    title="PARSON API",
    description="Explainable Book Recommendation Engine",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Request Schema
# -------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# -------------------------------
# Health Check Route
# -------------------------------

@app.get("/")
def root():
    """
    Simple health check
    """
    return {"status": "PARSON API running"}


# -------------------------------
# Recommendation Route
# -------------------------------

@app.post("/recommend")
def recommend(request: QueryRequest):
    """
    Generate book recommendations
    """
    results = recommend_books(
        user_query=request.query,
        top_k=request.top_k
    )

    return {
        "query": request.query,
        "results": results
    }
