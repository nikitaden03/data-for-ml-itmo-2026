"""
Merge Russian Fragrantica corpus (reviews) with scraped Fragrantica notes/ratings.
Join key: perfume name found at start of corpus 'product' column.
"""

import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# --- Load ---
corpus = pd.read_csv("data/raw/fragrantica_corpus.csv", sep=";", encoding="utf-8-sig", on_bad_lines="skip")
scraped = pd.read_csv("data/raw/fragrantica_scraped.csv", encoding="utf-8-sig")

print(f"Corpus shape: {corpus.shape}")
print(f"Scraped shape: {scraped.shape}")

# --- Build join key ---
# corpus.product looks like: "Chanel No 5 Eau de Parfum Chanel для женщин"
# scraped.name looks like:   "Chanel No 5 Eau de Parfum"
# Strategy: for each corpus row, check which scraped name appears at the start of product

scraped_names = scraped["name"].tolist()

def find_perfume_name(product_str):
    product_lower = product_str.lower()
    best = None
    best_len = 0
    for name in scraped_names:
        if product_lower.startswith(name.lower()) and len(name) > best_len:
            best = name
            best_len = len(name)
    return best

corpus["perfume_name"] = corpus["product"].apply(find_perfume_name)

print("\nMatching stats:")
print(corpus["perfume_name"].value_counts(dropna=False))
unmatched = corpus[corpus["perfume_name"].isna()]
if len(unmatched) > 0:
    print(f"\nUnmatched products ({len(unmatched)}):")
    print(unmatched["product"].unique())

# --- Merge ---
merged = corpus.merge(
    scraped.rename(columns={"name": "perfume_name"}),
    on="perfume_name",
    how="left"
)

# Clean up columns
merged = merged.rename(columns={"text": "review_text", "date": "review_date", "user": "reviewer"})
merged = merged.drop(columns=["no", "product", "url"], errors="ignore")
merged = merged[["perfume_name", "brand", "rating", "votes",
                 "top_notes", "middle_notes", "base_notes",
                 "reviewer", "review_date", "review_text"]]

print(f"\nMerged shape: {merged.shape}")
print(f"Columns: {merged.columns.tolist()}")
print("\nSample:")
print(merged.head(3)[["perfume_name", "brand", "rating", "top_notes", "review_text"]].to_string())

# --- Save ---
merged.to_csv("data/processed/perfumes_merged.csv", index=False, encoding="utf-8-sig")
print("\nSaved to data/processed/perfumes_merged.csv")

# Summary stats
print("\n=== Summary ===")
per_perfume = merged.groupby("perfume_name").agg(
    brand=("brand", "first"),
    rating=("rating", "first"),
    votes=("votes", "first"),
    n_reviews=("review_text", "count"),
    top_notes=("top_notes", "first"),
).reset_index()
print(per_perfume.to_string())
