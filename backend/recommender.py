"""
PARSON - Book Recommendation Engine
----------------------------------
This script:
1. Loads book metadata, embeddings, and FAISS index
2. Converts a user query into an embedding
3. Finds the most semantically similar books
4. Returns recommendations with basic explanations
"""
import math
import os
import numpy as np
import pandas as pd
import faiss
import joblib

from datetime import datetime
from backend.llm_helper import (
    generate_book_explanations,
    generate_fallback_response
)

def safe_float(value):
    """
    Prevent NaN / inf JSON serialization crashes.
    """

    try:
        value = float(value)

        if math.isnan(value):
            return 0.0

        if math.isinf(value):
            return 0.0

        return value

    except:
        return 0.0

def sanitize_json(obj):
    """
    Recursively sanitize dictionaries/lists
    so FastAPI JSON serialization never crashes.
    """

    if isinstance(obj, dict):
        return {
            k: sanitize_json(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            sanitize_json(v)
            for v in obj
        ]

    if isinstance(obj, (float, int, np.floating, np.integer)):
        return safe_float(obj)

    return obj
# ------------------------------ 
INTENT_MODEL_PATH = "models/intent_classifier.pkl"
intent_pipeline = None

# -------------------------------
# EMBEDDING MODEL
# -------------------------------
EMBEDDING_MODEL = None
def get_intent_pipeline():
    global intent_pipeline

    if intent_pipeline is None:
        intent_pipeline = joblib.load(INTENT_MODEL_PATH)

    return intent_pipeline


def get_embedding_model():
    global EMBEDDING_MODEL

    if EMBEDDING_MODEL is None:
        

        EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    return EMBEDDING_MODEL
# -------------------------------
# Explainability configuration
# -------------------------------

ENABLE_EXPLAINABILITY = True


# -------------------------------
# Configuration (file paths)
# -------------------------------

BOOKS_CSV_PATH = "data/books.csv"
EMBEDDINGS_PATH = "models/embeddings.npy"
FAISS_INDEX_PATH = "models/books.index"

# -------------------------------
# Ranking configuration
# -------------------------------

# Number of candidates retrieved from FAISS before re-ranking
CANDIDATE_POOL_SIZE = 60
# Hybrid scoring weights
SEMANTIC_WEIGHT = 0.7
INTENT_WEIGHT = 0.3
MAX_INTENT_ADJUSTMENT = 1.0
# -------------------------------
# Load resources
# -------------------------------
# -------------------------------
# Recommendation logic
#--------------------------------

def log_decision_trace(query, book_title, explanation):
    """
    Persist structured decision trace to disk.
    Each line = one JSON object.
    """

    import json
    from datetime import datetime

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "book_title": book_title,
        "semantic_rank": explanation.get("semantic_rank"),
        "semantic_score": explanation.get("semantic_score"),
        "detected_user_intent": explanation.get("detected_user_intent"),
        "inferred_book_form": explanation.get("inferred_book_form"),
        "intent_adjustment": explanation.get("intent_adjustment"),
        "final_score": explanation.get("final_score")
    }
    import os
    os.makedirs("logs", exist_ok=True)
    with open("logs/decision_traces.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# -------------------------------
# Helper functions
# -------------------------------
def detect_language_preference(user_query: str):
    """
    Detect explicit language preference from user query.

    Returns:
    - 'hi' if Hindi is requested
    - 'en' if English is requested
    - None if no language preference is found
    """

    query_lower = user_query.lower()

    if "hindi" in query_lower:
        return "hi"

    if "english" in query_lower:
        return "en"

    return None


def load_books_data():
    """Load book metadata from CSV."""
    return pd.read_csv(BOOKS_CSV_PATH)


def load_embeddings():
    """Load precomputed book embeddings."""
    return np.load(EMBEDDINGS_PATH)


def load_faiss_index(embedding_dimension):
    """Load FAISS index from disk."""
    index = faiss.read_index(FAISS_INDEX_PATH)
    return index

def build_intent_result(intent: str, confidence: float) -> dict:
    """
    Standard structure for detected intent with confidence.
    """
    return {
        "intent": intent,
        "confidence": round(confidence, 2)
    }
def detect_user_intents(user_query: str) -> dict:
    """
    Predict user intent using trained ML pipeline.
    Returns confidence score for predicted intent(s).
    """
    pipeline = get_intent_pipeline()
    probabilities = pipeline.predict_proba([user_query])[0]
    labels = pipeline.classes_

    intent_scores = {}

    for label, prob in zip(labels, probabilities):
        if prob >= 0.25:
            intent_scores[label] = round(float(prob), 2)

    return intent_scores

def infer_book_form_scores(book_row) -> dict:
    """
    Infer probabilistic book forms.

    Returns:
    {
        "story": 0.2,
        "learn": 0.5,
        "self_help": 0.2,
        "reference": 0.1
    }
    """

    text = (
        f"{book_row.get('title', '')} "
        f"{book_row.get('synopsis', '')} "
        f"{book_row.get('themes', '')}"
    ).lower()

    form_keywords = {
        "story": ["novel", "story", "fantasy", "fiction", "adventure", "superhero", "hero", "villain", "magic", "wizard", "dragon", "cosmic", "multiverse", "battle", "war", "comic", "marvel", "galaxy", "alien", "space", "kingdom", "quest", "sci-fi", "science fiction"],
        "learn": ["how to", "understanding", "explains", "introduction", "learn", "guide", "tutorial", "textbook", "manual", "reference", "education", "course", "training"],
        "self_help": ["self", "life", "growth", "success", "mindset", "relationship", "relationships", "motivation", "discipline", "habits", "confidence", "healing", "anxiety", "productivity", "self-improvement"],
        "reference": ["encyclopedia", "handbook", "reference", "guide"]
    }

    scores = {}
    total = 0

    for form, words in form_keywords.items():
        count = sum(1 for w in words if w in text)
        if count > 0:
            scores[form] = count
            total += count

    if total == 0:
        return {"story": 1.0}

    for form in scores:
        scores[form] = round(scores[form] / total, 2)

    return scores

def compute_intent_adjustment(user_intent: str, book_form: str) -> float:
    """
    Compute score adjustment based on compatibility
    between user intent and book form.

    Positive values boost ranking.
    Negative values penalize ranking.
    Zero means neutral.
    """

    # No intent detected → no adjustment
    if user_intent is None:
        return 0.0

    # Define intent ↔ book form compatibility
    intent_compatibility = {
        "story": {
            "boost": {"story"},
            "penalize": {"learn", "reference", "self_help"}
        },
        "learn": {
            "boost": {"learn", "reference"},
            "penalize": {"story"}
        },
        "self_help": {
            "boost": {"self_help"},
            "penalize": {"story"}
        },
        "reference": {
            "boost": {"reference", "learn"},
            "penalize": {"story", "self_help"}
        }
    }

    rules = intent_compatibility.get(user_intent)

    if rules is None:
        return 0.0

    if book_form in rules["boost"]:
        return +1.0        # Intent match bonus

    if book_form in rules["penalize"]:
        return -0.7        # Intent mismatch penalty

    return 0.0            # Neutral case
def is_intent_compatible(user_intent: str, book_form: str) -> bool:
    """
    Decide whether a book form is compatible with the user's intent.
    Incompatible books are excluded entirely.
    """

    if user_intent is None:
        return True

    incompatible_rules = {
        "self_help": {"story"},
        "learn": {"story"},
        "story": {"reference"},
        "reference": {"story", "self_help"}
    }

    disallowed_forms = incompatible_rules.get(user_intent, set())
    return book_form not in disallowed_forms

def build_score_explanation(
    semantic_rank,
    semantic_distance,
    semantic_score,
    user_intent,
    book_form,
    intent_adjustment,
    final_score,
    intent_contributions
    ):





    """
    Build a structured explanation of how the final score was computed.
    This is used for debugging and transparency.
    """

    return {
    "semantic_rank": semantic_rank + 1,
    "semantic_distance": round(safe_float(semantic_distance), 4),
    "semantic_score": round(safe_float(semantic_score), 4),
    "semantic_weight": SEMANTIC_WEIGHT,
    "detected_user_intent": user_intent,
    "inferred_book_form": book_form,
    "intent_adjustment": round(
    max(-MAX_INTENT_ADJUSTMENT,
        min(intent_adjustment, MAX_INTENT_ADJUSTMENT)
    ), 4),

    "intent_weight": INTENT_WEIGHT,
    "final_score": round(safe_float(final_score), 4),
    "intent_contributions": intent_contributions,
    }




def generate_user_friendly_explanation(explanation: dict) -> str:
    """
    Convert a technical explanation trace into a human-readable explanation.
    This is shown to end users.
    """

    intent = explanation["detected_user_intent"]
    book_form_scores = explanation["inferred_book_form"]
    intent_contributions = explanation.get("intent_contributions", {})
    semantic_score = explanation["semantic_score"]

    primary_form = max(book_form_scores, key=book_form_scores.get)

    semantic_rank = explanation["semantic_rank"]
    intent_adjustment = explanation["intent_adjustment"]

    explanation_parts = []

    # Intent explanation
    if intent:
        explanation_parts.append(
            f"It matches your interests in {intent.replace('_', ' ')}."
        )

    # Book type explanation
    explanation_parts.append(
        f"This book is primarily a {primary_form.replace('_', ' ')} book."
    )

    # Semantic relevance explanation
    if semantic_rank <= 3:
        explanation_parts.append(
            "It is one of the closest matches to your query."
        )
    else:
        explanation_parts.append(
            "Even though it was not the closest semantic match, it aligns well with your intent."
        )

    # Intent bonus explanation
    if intent_adjustment > 0:
        explanation_parts.append(
            "Its strong alignment with your intent increased its ranking."
        )
    elif intent_adjustment < 0:
        explanation_parts.append(
            "It was slightly penalized due to weaker alignment with your intent."
        )
    parts = []
    semantic_pct = max(0, min(round(semantic_score * 100), 100))
    parts.append(f"{semantic_pct}% semantic relevance")
    for intent, value in intent_contributions.items():
        intent_pct = max(0,min(round(abs(value) * 100), 100))
        parts.append(f"{intent_pct}% {intent.replace('_',' ')} intent")
    explanation_parts.insert(
        0,
        "Recommended because: " + ", ".join(parts) + "."
    )
    return " ".join(explanation_parts)


def compute_semantic_score_from_distance(distance: float) -> float:
    """
    Convert FAISS L2 distance into a normalized semantic score.
    Lower distance => higher similarity.

    Formula used:
        score = 1 / (1 + distance)

    This keeps the score in (0, 1].
    """
    distance = safe_float(distance)
    score = 1.0 / (1.0 + distance)
    return round(safe_float(score), 4)

def normalize_and_combine_scores(
    semantic_score: float,
    intent_adjustment: float
) -> float:

    semantic_score = safe_float(semantic_score)
    intent_adjustment = safe_float(intent_adjustment)

    clipped_intent = max(
        -MAX_INTENT_ADJUSTMENT,
        min(intent_adjustment, MAX_INTENT_ADJUSTMENT)
    )

    final_score = (
        SEMANTIC_WEIGHT * semantic_score
        + INTENT_WEIGHT * clipped_intent
    )

    return round(safe_float(final_score), 4)

def compute_intent_contributions(user_intents, book_form_scores):
    """
    Returns per-intent contribution values.

    Example:
    {
        "self_help": 0.21,
        "learn": 0.08
    }
    """
    contributions = {}

    for intent, intent_conf in user_intents.items():
        total = 0.0
        for form, form_conf in book_form_scores.items():
            adj = compute_intent_adjustment(intent, form)
            total += adj * intent_conf * form_conf
        total = safe_float(total)
        if total != 0:
            contributions[intent] = round(total, 4)

    return contributions

# -------------------------------
# Recommendation logic
# -------------------------------

def recommend_books(user_query, top_k=10):
    """
    Recommend books based on a natural language query.

    Parameters:
    - user_query (str): User's input text
    - top_k (int): Number of recommendations to return

    Returns:
    - List of recommended books with explanations
    """
    # Load all required resources
    books_df = load_books_data()
    embeddings = load_embeddings()
    embedding_model = get_embedding_model()


    # Step 1: Detect language preference
    preferred_language = detect_language_preference(user_query)

    # Step 2: Apply language filter if needed
    if preferred_language is not None:
        language_mask = books_df["language"] == preferred_language
        filtered_books_df = books_df[language_mask].reset_index(drop=True)
        filtered_embeddings = embeddings[language_mask.values]
    else:
        filtered_books_df = books_df
        filtered_embeddings = embeddings

    # Step 3: Build FAISS index on filtered embeddings
    dimension = filtered_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(filtered_embeddings)


    # Convert user query into embedding
    query_embedding = embedding_model.encode([user_query]).astype("float32")

    # Search for similar books
    distances, indices = faiss_index.search(
        query_embedding,
        CANDIDATE_POOL_SIZE
    )


    # Step 4: Detect user intent
    user_intents = detect_user_intents(user_query)

    INTENT_CONFIDENCE_THRESHOLD = 0.45

    user_intents = {
        intent: conf
        for intent, conf in user_intents.items()
        if conf >= INTENT_CONFIDENCE_THRESHOLD
    }



    scored_books = []

    for rank, idx in enumerate(indices[0]):
        book = filtered_books_df.iloc[idx]

        # Infer what kind of book this is
        book_form_scores = infer_book_form_scores(book)
        # Hard intent compatibility check
        if user_intents:
            compatible = False
            for intent in user_intents:
                for form in book_form_scores:
                    if is_intent_compatible(intent, form):
                        compatible = True
                        break
            if not compatible:
                continue



        # Base score from semantic similarity (FAISS rank)
        distance = distances[0][rank]
        semantic_score = compute_semantic_score_from_distance(distance)


        # Compute intent-based adjustment (boost or penalty)
        intent_contributions = compute_intent_contributions(
            user_intents,
            book_form_scores
        )
        intent_adjustment = sum(intent_contributions.values())

        # Final composite score
        final_score = normalize_and_combine_scores(
            semantic_score=semantic_score,
            intent_adjustment=intent_adjustment
        )

        score_explanation = build_score_explanation(
            semantic_rank=rank,
            semantic_distance=distance,   # ← THIS IS THE KEY LINE
            semantic_score=semantic_score,
            user_intent=", ".join(user_intents.keys()),
            book_form=book_form_scores,
            intent_adjustment=intent_adjustment,
            final_score=final_score,
            intent_contributions=intent_contributions,
        )



        scored_books.append((final_score, book, score_explanation))



    # Step 5: Sort by final score
    scored_books.sort(key=lambda x: x[0], reverse=True)
    top_books = [
        {
            "title": book[1]["title"],
            "synopsis": book[1]["synopsis"]
        }
    for book in scored_books[:top_k]
    ]

    batch_explanations = generate_book_explanations(
        user_query,
        top_books
    )
    # Step 6: Build final recommendations
    recommendations = []

    for score, book, explanation in scored_books[:top_k]:
        user_explanation = batch_explanations.get(
            str(len(recommendations) + 1),
            "A potentially relevant recommendation based on your search themes."
        )
        log_decision_trace(user_query, book["title"], explanation)
        
        recommendations.append(
            sanitize_json({
                "title": str(book.get("title", "")),
                "author": (
                    ""
                    if pd.isna(book.get("author"))
                    else str(book.get("author"))
                ),
                "language": str(book.get("language", "")),
                "summary": str(book.get("synopsis", ""))[:700],
                "thumbnail": (
                    ""
                    if pd.isna(book.get("thumbnail"))
                    else str(book.get("thumbnail"))
                ),
                "preview_link": (
                    ""
                    if pd.isna(book.get("preview_link"))
                    else str(book.get("preview_link"))
            ),
                "reason": str(user_explanation),
                "debug_explanation":
                    sanitize_json(explanation)
                    if ENABLE_EXPLAINABILITY else None
            })
        )

    return recommendations



# -------------------------------
# Test the recommender
# -------------------------------

if __name__ == "__main__":

    # List of realistic test queries
    test_queries = [
        "Suggest me Hindi books that talk about life, struggle, and personal growth.",
        "Suggest English books about relationships and human behavior.",
        "I want to understand space conceptually without equations.",
        "Suggest books about anxiety disorders",
        "Books about PTSD healing",
        "Therapy workbooks for trauma",
        "Neuroscience of emotions",
        "Cognitive behavioral therapy guide",
        "Epic dragon fantasy saga",
        "Magic wizard school novel",
        "Sword and sorcery adventure",
        "Space opera science fiction",
        "Alien invasion story"
    ]

    # Run recommendations for each test query
    for query in test_queries:
        print("\n" + "=" * 80)
        print("USER QUERY:")
        print(query)
        print("\nRECOMMENDED BOOKS:\n")

        results = recommend_books(query, top_k=3)

        for i, book in enumerate(results, start=1):
            print(f"{i}. {book['title']} by {book['author']}")
            print(f"   Why this book: {book['reason']}")
            print(f"   Preview: {book['preview_link']}\n")
            if ENABLE_EXPLAINABILITY:
                print("   Debug Explanation Trace:")
                for key, value in book["debug_explanation"].items():
                    print(f"     - {key}: {value}")
