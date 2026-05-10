"""
GENERATE EMBEDDINGS
-------------------
Creates semantic embeddings for all books
using SentenceTransformer.
"""

# ============================================
# Imports
# ============================================

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

# ============================================
# File Paths
# ============================================

BOOKS_PATH = "data/books.csv"

OUTPUT_PATH = "models/embeddings.npy"

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv(BOOKS_PATH)

print("Books loaded:", len(df))

# ============================================
# Load Embedding Model
# ============================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ============================================
# Create Semantic Text
# ============================================

semantic_texts = (
    df["title"].fillna("") + " " +
    df["themes"].fillna("") + " " +
    df["synopsis"].fillna("") + " " +
    df["synopsis"].fillna("")
).tolist()

# ============================================
# Generate Embeddings
# ============================================

print("\nGenerating embeddings...")

embeddings = model.encode(
    semantic_texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

# ============================================
# Save Embeddings
# ============================================

np.save(OUTPUT_PATH, embeddings)

print("\nEmbeddings generated successfully.")
print("Saved to:", OUTPUT_PATH)
print("Embedding shape:", embeddings.shape)