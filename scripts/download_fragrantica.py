"""
Скачивание датасета: Fragrantica.com Fragrance Dataset
Источник: kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset
"""

import os
import zipfile
import pandas as pd
from pathlib import Path
import kaggle

# Настройка
DATASET_SLUG = "olgagmiufana1/fragrantica-com-fragrance-dataset"
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Скачивание
print(f"Скачиваю {DATASET_SLUG}...")
kaggle.api.dataset_download_files(DATASET_SLUG, path=DATA_DIR, unzip=True)
print("Готово!")

# Находим CSV файлы
csv_files = list(DATA_DIR.glob("*.csv"))
print(f"\nНайдены файлы: {[f.name for f in csv_files]}")

# Загружаем cleaned версию если есть, иначе первый попавшийся
target = next((f for f in csv_files if "cleaned" in f.name.lower()), csv_files[0])
print(f"\nЗагружаю: {target.name}")
for enc in ["utf-8", "latin-1", "cp1252"]:
    try:
        df = pd.read_csv(target, encoding=enc, sep=";", on_bad_lines="skip")
        print(f"Кодировка: {enc}")
        break
    except UnicodeDecodeError:
        continue

# Базовая очистка column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Сохранение
output_path = DATA_DIR / "fragrantica.csv"
df.to_csv(output_path, index=False)

# Summary
print(f"\n{'='*50}")
print(f"✅ Сохранено: {output_path}")
print(f"   Строк:    {len(df)}")
print(f"   Столбцов: {len(df.columns)}")
print(f"\nСтолбцы:")
for col in df.columns:
    print(f"  - {col}: {df[col].dtype}  (nulls: {df[col].isna().sum()})")
print(f"\nПервые 3 строки:")
print(df.head(3).to_string())
