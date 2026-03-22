"""
quality_utils.py — утилиты для анализа и очистки качества данных.
Используется во всех фазах пайплайна: detect → fix → compare.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────
# DataClasses
# ─────────────────────────────────────────────

@dataclass
class IssueDetail:
    type: str          # "missing" | "duplicate" | "outlier" | "imbalance"
    severity: str      # "critical" | "high" | "medium" | "low"
    column: str        # имя столбца или "dataset"
    count: int
    percentage: float
    description: str


@dataclass
class QualityReport:
    dataset_name: str
    n_rows: int
    n_cols: int
    target_col: Optional[str]

    # сырые метрики
    missing_counts: dict = field(default_factory=dict)       # col → count
    missing_pct: dict = field(default_factory=dict)          # col → %
    n_duplicates: int = 0
    duplicate_pct: float = 0.0
    outliers_iqr: dict = field(default_factory=dict)         # col → count
    outliers_zscore: dict = field(default_factory=dict)      # col → count
    target_distribution: dict = field(default_factory=dict)  # для регрессии: basic stats
    imbalance_ratio: Optional[float] = None                  # для классификации

    # список всех проблем
    issues: list = field(default_factory=list)

    # итоговый балл
    quality_score: float = 100.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["issues"] = [asdict(i) for i in self.issues]
        return d

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "QualityReport":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        issues = [IssueDetail(**i) for i in data.pop("issues", [])]
        report = QualityReport(**data)
        report.issues = issues
        return report


# ─────────────────────────────────────────────
# Detect helpers
# ─────────────────────────────────────────────

def _severity_missing(pct: float) -> str:
    if pct > 50:
        return "critical"
    if pct > 30:
        return "high"
    if pct > 10:
        return "medium"
    return "low"


def _severity_duplicates(pct: float) -> str:
    if pct > 5:
        return "high"
    if pct > 1:
        return "medium"
    return "low"


def _severity_outliers(pct: float) -> str:
    if pct > 10:
        return "high"
    if pct > 3:
        return "medium"
    return "low"


def _penalty(severity: str) -> float:
    return {"critical": 20, "high": 10, "medium": 5, "low": 2}.get(severity, 0)


def detect_all(df: pd.DataFrame, target_col: Optional[str] = None,
               dataset_name: str = "dataset") -> QualityReport:
    """
    Полный анализ качества датасета.
    Возвращает QualityReport.
    """
    report = QualityReport(
        dataset_name=dataset_name,
        n_rows=len(df),
        n_cols=len(df.columns),
        target_col=target_col,
    )
    issues = []
    penalty_total = 0.0

    # ── 1. Пропуски ──────────────────────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    report.missing_counts = missing.to_dict()
    report.missing_pct = (missing / len(df) * 100).round(2).to_dict()

    for col, cnt in missing.items():
        pct = cnt / len(df) * 100
        sev = _severity_missing(pct)
        issues.append(IssueDetail(
            type="missing",
            severity=sev,
            column=col,
            count=int(cnt),
            percentage=round(pct, 2),
            description=f"Пропущено {cnt} значений ({pct:.1f}%)",
        ))
        penalty_total += _penalty(sev)

    # ── 2. Дубликаты ─────────────────────────────────
    n_dup = int(df.duplicated().sum())
    dup_pct = n_dup / len(df) * 100
    report.n_duplicates = n_dup
    report.duplicate_pct = round(dup_pct, 2)

    if n_dup > 0:
        sev = _severity_duplicates(dup_pct)
        issues.append(IssueDetail(
            type="duplicate",
            severity=sev,
            column="dataset",
            count=n_dup,
            percentage=round(dup_pct, 2),
            description=f"{n_dup} полных дубликатов ({dup_pct:.1f}%)",
        ))
        penalty_total += _penalty(sev)

    # ── 3. Выбросы ────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != target_col]

    iqr_counts: dict = {}
    z_counts: dict = {}

    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue

        # IQR
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_iqr = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        iqr_counts[col] = n_iqr

        # Z-score
        z = np.abs(stats.zscore(s))
        n_z = int((z > 3).sum())
        z_counts[col] = n_z

        if n_iqr > 0:
            pct = n_iqr / len(s) * 100
            sev = _severity_outliers(pct)
            issues.append(IssueDetail(
                type="outlier",
                severity=sev,
                column=col,
                count=n_iqr,
                percentage=round(pct, 2),
                description=f"IQR: {n_iqr} выбросов ({pct:.1f}%), Z-score: {n_z}",
            ))
            penalty_total += _penalty(sev) * 0.5  # выбросы штрафуем меньше

    report.outliers_iqr = iqr_counts
    report.outliers_zscore = z_counts

    # ── 4. Целевая переменная ─────────────────────────
    if target_col and target_col in df.columns:
        t = df[target_col].dropna()
        report.target_distribution = {
            "mean": round(float(t.mean()), 4),
            "std": round(float(t.std()), 4),
            "min": round(float(t.min()), 4),
            "max": round(float(t.max()), 4),
            "median": round(float(t.median()), 4),
            "skewness": round(float(t.skew()), 4),
        }

    report.issues = issues
    report.quality_score = max(0.0, round(100.0 - penalty_total, 1))
    return report


# ─────────────────────────────────────────────
# Fix helpers (Фаза 2)
# ─────────────────────────────────────────────

def apply_strategy(df: pd.DataFrame, strategy: dict,
                   target_col: Optional[str] = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Применяет стратегию очистки к датасету.
    Порядок: дубликаты → пропуски → выбросы.

    strategy = {
        "duplicates": "drop_exact",
        "missing": {"numeric": "fill_median", "categorical": "fill_mode"},
        "outliers": {"method": "clip_iqr", "factor": 1.5},
    }

    Возвращает (cleaned_df, log_messages).
    """
    df = df.copy()
    log = []
    n_orig = len(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if target_col:
        # строки с пропущенным target удаляем
        before = len(df)
        df = df.dropna(subset=[target_col])
        removed = before - len(df)
        if removed:
            log.append(f"Удалено {removed} строк с пропущенным target ({target_col})")
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col in cat_cols:
            cat_cols.remove(target_col)

    # ── Дубликаты ─────────────────────────────────────
    dup_strategy = strategy.get("duplicates", "drop_exact")
    if dup_strategy == "drop_exact":
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        log.append(f"Дубликаты (drop_exact): удалено {removed} строк ({removed/n_orig*100:.1f}%)")

    # ── Пропуски ──────────────────────────────────────
    miss_strategy = strategy.get("missing", {})
    num_method = miss_strategy.get("numeric", "fill_median") if isinstance(miss_strategy, dict) else miss_strategy
    cat_method = miss_strategy.get("categorical", "fill_mode") if isinstance(miss_strategy, dict) else "fill_mode"
    drop_col_threshold = miss_strategy.get("drop_col_threshold", None) if isinstance(miss_strategy, dict) else None
    drop_row_threshold = miss_strategy.get("drop_row_threshold", None) if isinstance(miss_strategy, dict) else None

    # удалить столбцы с пропусков > threshold
    if drop_col_threshold is not None:
        cols_to_drop = [c for c in df.columns
                        if df[c].isnull().mean() * 100 > drop_col_threshold]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            log.append(f"Удалены столбцы (>{drop_col_threshold}% пропусков): {cols_to_drop}")
            numeric_cols = [c for c in numeric_cols if c not in cols_to_drop]
            cat_cols = [c for c in cat_cols if c not in cols_to_drop]

    # удалить строки с > N пропусков
    if drop_row_threshold is not None:
        before = len(df)
        mask = df.isnull().sum(axis=1) > drop_row_threshold
        df = df[~mask]
        removed = before - len(df)
        log.append(f"Удалены строки с >({drop_row_threshold}) пропусков: {removed} строк ({removed/n_orig*100:.1f}%)")

    # числовые
    for col in [c for c in numeric_cols if c in df.columns]:
        if df[col].isnull().sum() == 0:
            continue
        cnt = df[col].isnull().sum()
        if num_method == "fill_median":
            df[col] = df[col].fillna(df[col].median())
            log.append(f"  {col}: заполнено медианой ({cnt} значений)")
        elif num_method == "fill_mean":
            df[col] = df[col].fillna(df[col].mean())
            log.append(f"  {col}: заполнено средним ({cnt} значений)")
        elif num_method == "drop_rows":
            before = len(df)
            df = df.dropna(subset=[col])
            log.append(f"  {col}: удалено {before - len(df)} строк с NaN")

    # категориальные
    for col in [c for c in cat_cols if c in df.columns]:
        if df[col].isnull().sum() == 0:
            continue
        cnt = df[col].isnull().sum()
        if cat_method == "fill_mode":
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
            df[col] = df[col].fillna(fill_val)
            log.append(f"  {col}: заполнено модой '{fill_val}' ({cnt} значений)")
        elif cat_method == "fill_unknown":
            df[col] = df[col].fillna("unknown")
            log.append(f"  {col}: заполнено 'unknown' ({cnt} значений)")
        elif cat_method == "drop_rows":
            before = len(df)
            df = df.dropna(subset=[col])
            log.append(f"  {col}: удалено {before - len(df)} строк с NaN")

    # ── Выбросы ───────────────────────────────────────
    out_strategy = strategy.get("outliers", {})
    out_method = out_strategy.get("method", "clip_iqr") if isinstance(out_strategy, dict) else out_strategy
    factor = out_strategy.get("factor", 1.5) if isinstance(out_strategy, dict) else 1.5

    for col in [c for c in numeric_cols if c in df.columns]:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr

        if out_method == "clip_iqr":
            n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n_out > 0:
                df[col] = df[col].clip(lo, hi)
                log.append(f"  {col}: clip_iqr, обрезано {n_out} выбросов")
        elif out_method == "remove_iqr":
            before = len(df)
            df = df[(df[col].isna()) | ((df[col] >= lo) & (df[col] <= hi))]
            removed = before - len(df)
            if removed:
                log.append(f"  {col}: remove_iqr, удалено {removed} строк")
        elif out_method == "clip_zscore":
            z = np.abs(stats.zscore(df[col].dropna()))
            # пересчитываем на весь df
            col_mean, col_std = df[col].mean(), df[col].std()
            if col_std > 0:
                lo_z = col_mean - 3 * col_std
                hi_z = col_mean + 3 * col_std
                n_out = int(((df[col] < lo_z) | (df[col] > hi_z)).sum())
                if n_out > 0:
                    df[col] = df[col].clip(lo_z, hi_z)
                    log.append(f"  {col}: clip_zscore, обрезано {n_out} выбросов")
        elif out_method == "remove_zscore":
            col_mean, col_std = df[col].mean(), df[col].std()
            if col_std > 0:
                mask = ((df[col] - col_mean).abs() / col_std) <= 3
                before = len(df)
                df = df[mask | df[col].isna()]
                removed = before - len(df)
                if removed:
                    log.append(f"  {col}: remove_zscore, удалено {removed} строк")

    total_removed = n_orig - len(df)
    log.append(f"\nИтого: {n_orig} → {len(df)} строк (потеря {total_removed/n_orig*100:.1f}%)")
    return df, log
