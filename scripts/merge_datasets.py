"""
Объединение датасетов:
- fragrantica.csv (Kaggle, 24k строк)
- tom_ford_creed.csv (scraped, 40 строк)
Стратегия: вертикальный CONCAT
"""

import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path

DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Загрузка ──────────────────────────────────────────────────────────────────
df_kaggle = pd.read_csv(DATA_DIR / "fragrantica.csv")
df_scraped = pd.read_csv(DATA_DIR / "tom_ford_creed.csv")

print(f"Kaggle:   {df_kaggle.shape}")
print(f"Scraped:  {df_scraped.shape}")


# ── Нормализация brand ────────────────────────────────────────────────────────
# Kaggle хранит lowercase; приведём к Title Case для обоих
def normalize_brand(s: pd.Series) -> pd.Series:
    return s.str.strip().str.title()

df_kaggle["brand"]   = normalize_brand(df_kaggle["brand"])
df_scraped["brand"]  = normalize_brand(df_scraped["brand"])


# ── Нормализация rating_value → float ────────────────────────────────────────
def to_float_rating(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(",", ".", regex=False)
         .str.extract(r"([\d.]+)")[0]
         .astype(float)
    )

df_kaggle["rating_value"]  = to_float_rating(df_kaggle["rating_value"])
df_scraped["rating_value"] = to_float_rating(df_scraped["rating_value"])


# ── Убираем Creed из Kaggle (заменяем свежескраченными) ───────────────────────
scraped_brands = df_scraped["brand"].unique().tolist()
print(f"\nScraped brands: {scraped_brands}")

df_kaggle_clean = df_kaggle[~df_kaggle["brand"].isin(scraped_brands)].copy()
print(f"Kaggle after removing scraped brands: {df_kaggle_clean.shape}")


# ── Concat ────────────────────────────────────────────────────────────────────
df = pd.concat([df_kaggle_clean, df_scraped], axis=0, ignore_index=True)

print(f"\nAfter concat: {df.shape}")
print(f"Brands (top 10):\n{df['brand'].value_counts().head(10).to_string()}")


# ── Диагностика ───────────────────────────────────────────────────────────────
print(f"\nNulls:")
print(df.isna().sum().to_string())

dups = df.duplicated(subset=["perfume", "brand"]).sum()
print(f"\nДублей (perfume+brand): {dups}")
if dups:
    df = df.drop_duplicates(subset=["perfume", "brand"], keep="last")
    print(f"После дедупликации: {df.shape}")


# ── Сохранение ────────────────────────────────────────────────────────────────
output_path = OUT_DIR / "perfumes_merged.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"\n{'='*50}")
print(f"Saved: {output_path}")
print(f"Final shape: {df.shape}")
print(f"\nTom Ford sample:")
print(df[df["brand"] == "Tom Ford"][["perfume", "rating_value", "top"]].head(3).to_string())
print(f"\nCreed sample:")
print(df[df["brand"] == "Creed"][["perfume", "rating_value", "top"]].head(3).to_string())
