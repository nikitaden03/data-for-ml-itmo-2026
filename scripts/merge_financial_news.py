"""
Merge Kaggle financial news dataset with scraped news.
Unifies columns: title, body, date, source
"""
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')


def main():
    # --- Load Kaggle data ---
    kaggle = pd.read_csv('data/raw/kaggle_financial_news.csv', encoding='utf-8-sig')
    print(f"Kaggle raw shape: {kaggle.shape}")
    # Keep relevant columns, rename to unified schema
    kaggle = kaggle[['title', 'body', 'date', 'source']].copy()
    kaggle['origin'] = 'kaggle'

    # --- Load scraped data ---
    scraped = pd.read_csv('data/raw/scraped_financial_news.csv', encoding='utf-8-sig')
    print(f"Scraped raw shape: {scraped.shape}")
    scraped = scraped[['title', 'body', 'date', 'source']].copy()
    scraped['origin'] = 'scraped'

    # --- Merge ---
    merged = pd.concat([kaggle, scraped], ignore_index=True)
    print(f"After concat: {merged.shape}")

    # --- Normalize ---
    merged['title'] = merged['title'].astype(str).str.strip()
    merged['body'] = merged['body'].fillna('').astype(str).str.strip()
    merged['date'] = pd.to_datetime(merged['date'], errors='coerce')

    # --- Deduplicate by title ---
    before = len(merged)
    merged = merged.drop_duplicates(subset='title', keep='first')
    print(f"Deduplicated: {before} -> {len(merged)} rows")

    # --- Remove empty titles ---
    merged = merged[merged['title'].str.len() > 5].reset_index(drop=True)

    # --- Stats ---
    print(f"\nFinal shape: {merged.shape}")
    print(f"Sources:\n{merged['source'].value_counts().to_dict()}")
    print(f"Origins: {merged['origin'].value_counts().to_dict()}")
    print(f"Date range: {merged['date'].min()} - {merged['date'].max()}")
    print(f"Null counts:\n{merged.isnull().sum().to_dict()}")

    # --- Save ---
    output_path = 'data/processed/financial_news_merged.csv'
    import os
    os.makedirs('data/processed', exist_ok=True)
    merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved to: {output_path}")

    # --- Sample ---
    print("\nSample rows:")
    for _, row in merged.sample(3, random_state=42).iterrows():
        print(f"  [{row['source']}] {row['title'][:80]}")


if __name__ == '__main__':
    main()
