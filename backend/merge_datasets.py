"""
MERGE DATASETS
--------------
Combines original books dataset
with filtered comics dataset.
"""

# ============================================
# Imports
# ============================================

import pandas as pd

# ============================================
# File Paths
# ============================================

BOOKS_PATH = "data/books.csv"

COMICS_PATH = "data/comics_books_filtered.csv"

OUTPUT_PATH = "data/final_books_dataset.csv"

# ============================================
# Load Datasets
# ============================================

books_df = pd.read_csv(BOOKS_PATH)

comics_df = pd.read_csv(COMICS_PATH)

print("Original books:", len(books_df))
print("Filtered comics:", len(comics_df))

# ============================================
# Combine Datasets
# ============================================

combined_df = pd.concat(
    [books_df, comics_df],
    ignore_index=True
)

# ============================================
# Remove Duplicate Entries
# ============================================

combined_df = combined_df.drop_duplicates(
    subset=["title", "author"]
)

# ============================================
# Reset Index
# ============================================

combined_df = combined_df.reset_index(drop=True)

# ============================================
# Save Final Dataset
# ============================================

combined_df.to_csv(OUTPUT_PATH, index=False)

# ============================================
# Completion Message
# ============================================

print("\nDatasets merged successfully.")
print("Saved to:", OUTPUT_PATH)
print("Final total rows:", len(combined_df))