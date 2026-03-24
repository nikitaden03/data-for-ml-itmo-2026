"""
quality_utils.py — утилиты для оценки и очистки качества данных.
Используется во всех ноутбуках пайплайна data-quality.
"""
import json
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


# ─── КОНСТАНТЫ ────────────────────────────────────────────────────────────────

REPORTS_DIR = Path("data/reports")
CLEANED_DIR = Path("data/cleaned")


# ─── ДЕТЕКТИРОВАНИЕ ПРОБЛЕМ ───────────────────────────────────────────────────

def detect_encoding_issues(series: pd.Series, threshold: int = 3) -> pd.Series:
    """Возвращает булеву маску строк с символами замены Unicode (\\ufffd / ?)."""
    replacement = "\ufffd"
    pattern = f"[{re.escape(replacement)}\\?]{{{threshold},}}"
    return series.str.contains(pattern, regex=True, na=False)


def detect_short_texts(series: pd.Series, min_len: int = 50) -> pd.Series:
    """Возвращает булеву маску слишком коротких строк."""
    return series.str.len() < min_len


def detect_duplicate_reviewers(df: pd.DataFrame, reviewer_col: str = "reviewer") -> pd.Series:
    """Возвращает булеву маску строк, где рецензент встречается более одного раза."""
    counts = df[reviewer_col].map(df[reviewer_col].value_counts())
    return counts > 1


def detect_rating_leakage(df: pd.DataFrame,
                           group_col: str = "perfume_name",
                           target_col: str = "rating") -> bool:
    """Проверяет, является ли target_col агрегатом на уровне group_col (а не отзыва)."""
    unique_per_group = df.groupby(group_col)[target_col].nunique()
    return bool((unique_per_group == 1).all())


def detect_iqr_outliers(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Возвращает булеву маску выбросов по методу IQR."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)


# ─── ОЦЕНКА КАЧЕСТВА ─────────────────────────────────────────────────────────

def compute_quality_score(df: pd.DataFrame,
                           text_col: str = "review_text",
                           reviewer_col: str = "reviewer",
                           group_col: str = "perfume_name",
                           target_col: str = "rating") -> dict:
    """
    Вычисляет агрегированный Quality Score (0–100) и возвращает словарь с деталями.

    Состав штрафов:
      - Encoding issues   : -25 за каждый % поражённых строк
      - Short texts       : -15 за каждый % коротких строк
      - Duplicate reviewers: -10 за каждый % дублированных строк
      - Rating leakage    : -10 если rating не индивидуальный
      - Missing values    : -20 за каждый % пропусков (среднее по колонкам)
    """
    n = len(df)
    issues = []
    penalty = 0.0

    # 1. Encoding
    enc_mask = detect_encoding_issues(df[text_col])
    enc_pct = enc_mask.sum() / n * 100
    if enc_mask.sum() > 0:
        p = min(enc_pct * 25, 30)
        penalty += p
        issues.append({
            "type": "encoding_corruption",
            "severity": "critical",
            "count": int(enc_mask.sum()),
            "pct": round(enc_pct, 2),
            "description": "Текст отзыва содержит символы замены Unicode (\\ufffd) — данные нечитаемы для NLP",
            "penalty": round(p, 1),
        })

    # 2. Short texts
    short_mask = detect_short_texts(df[text_col])
    # exclude already flagged encoding issues to avoid double count
    short_only = short_mask & ~enc_mask
    short_pct = short_only.sum() / n * 100
    if short_only.sum() > 0:
        p = min(short_pct * 15, 15)
        penalty += p
        issues.append({
            "type": "short_review",
            "severity": "medium",
            "count": int(short_only.sum()),
            "pct": round(short_pct, 2),
            "description": "Отзыв слишком короткий (<50 символов) — мало информации для NLP",
            "penalty": round(p, 1),
        })

    # 3. Duplicate reviewers
    dup_mask = detect_duplicate_reviewers(df, reviewer_col)
    dup_pct = dup_mask.sum() / n * 100
    if dup_mask.sum() > 0:
        p = min(dup_pct * 10, 15)
        penalty += p
        issues.append({
            "type": "duplicate_reviewer",
            "severity": "high",
            "count": int(dup_mask.sum()),
            "pct": round(dup_pct, 2),
            "description": "Один рецензент оставил несколько отзывов — риск утечки данных при train/test split",
            "penalty": round(p, 1),
        })

    # 4. Rating leakage
    if detect_rating_leakage(df, group_col, target_col):
        p = 10.0
        penalty += p
        issues.append({
            "type": "perfume_level_rating",
            "severity": "high",
            "count": df[group_col].nunique(),
            "pct": 100.0,
            "description": f"'{target_col}' — агрегат уровня парфюма, а не оценка отдельного отзыва; прямое использование как таргета приведёт к утечке",
            "penalty": round(p, 1),
        })

    # 5. Missing values
    miss_pct_mean = df.isnull().mean().mean() * 100
    if miss_pct_mean > 0:
        p = min(miss_pct_mean * 20, 20)
        penalty += p
        issues.append({
            "type": "missing_values",
            "severity": "high",
            "count": int(df.isnull().sum().sum()),
            "pct": round(miss_pct_mean, 2),
            "description": "Пропущенные значения в датасете",
            "penalty": round(p, 1),
        })

    score = max(0, round(100 - penalty, 1))

    return {
        "quality_score": score,
        "n_rows": n,
        "n_cols": len(df.columns),
        "total_penalty": round(penalty, 1),
        "issues": issues,
    }


def save_report(report: dict, path: Optional[Union[str, Path]] = None) -> Path:
    """Сохраняет отчёт в JSON."""
    if path is None:
        path = REPORTS_DIR / "quality_report.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def load_report(path: Optional[Union[str, Path]] = None) -> dict:
    """Загружает отчёт из JSON."""
    if path is None:
        path = REPORTS_DIR / "quality_report.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── СТРАТЕГИИ ОЧИСТКИ ────────────────────────────────────────────────────────

def apply_strategy_a(df: pd.DataFrame,
                      text_col: str = "review_text",
                      reviewer_col: str = "reviewer",
                      min_len: int = 50) -> pd.DataFrame:
    """
    Стратегия A — «Консервативная»:
      1. Удаляем строки с encoding-порчей.
      2. Удаляем строки с review_text < min_len символов.
      3. Для дублей рецензентов оставляем самый длинный отзыв.
    """
    df = df.copy()

    enc_mask = detect_encoding_issues(df[text_col])
    df = df[~enc_mask].reset_index(drop=True)

    short_mask = detect_short_texts(df[text_col], min_len)
    df = df[~short_mask].reset_index(drop=True)

    df["_len"] = df[text_col].str.len()
    df = (
        df.sort_values("_len", ascending=False)
          .drop_duplicates(subset=reviewer_col, keep="first")
          .drop(columns="_len")
          .reset_index(drop=True)
    )
    return df


def apply_strategy_b(df: pd.DataFrame,
                      text_col: str = "review_text",
                      reviewer_col: str = "reviewer",
                      min_len: int = 50) -> pd.DataFrame:
    """
    Стратегия B — «Мягкая»:
      1. Удаляем строки с encoding-порчей.
      2. Сохраняем короткие отзывы (не удаляем), но добавляем флаг is_short.
      3. Для дублей рецензентов оставляем все отзывы (не дедуплицируем).
    """
    df = df.copy()

    enc_mask = detect_encoding_issues(df[text_col])
    df = df[~enc_mask].reset_index(drop=True)

    df["is_short_review"] = detect_short_texts(df[text_col], min_len)
    return df


# ─── СРАВНЕНИЕ СТРАТЕГИЙ ──────────────────────────────────────────────────────

def strategy_metrics(df_orig: pd.DataFrame,
                     df_clean: pd.DataFrame,
                     text_col: str = "review_text",
                     reviewer_col: str = "reviewer",
                     group_col: str = "perfume_name",
                     target_col: str = "rating") -> dict:
    """Вычисляет метрики до/после очистки для одной стратегии."""
    report_clean = compute_quality_score(
        df_clean, text_col, reviewer_col, group_col, target_col
    )
    loss_pct = round((1 - len(df_clean) / len(df_orig)) * 100, 2)
    return {
        "rows_before": len(df_orig),
        "rows_after": len(df_clean),
        "loss_pct": loss_pct,
        "quality_score": report_clean["quality_score"],
        "issues_remaining": report_clean["issues"],
    }
