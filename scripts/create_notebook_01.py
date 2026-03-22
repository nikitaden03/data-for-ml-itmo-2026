"""Script to create 01_detect_issues.ipynb"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# ── Cell 0: Title ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "# 01 — Detect Issues: Анализ качества данных\n\n"
    "**Датасет:** data/processed/perfumes_merged.csv  \n"
    "**Задача:** Регрессия — предсказание рейтинга духов (`rating_value`)  \n"
    "**Цель:** Найти все проблемы качества данных, оценить severity, вычислить Quality Score."
))

# ── Cell 1: Imports ───────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, "..")
from scripts.quality_utils import detect_all

plt.rcParams["figure.dpi"] = 100
sns.set_theme(style="whitegrid", palette="muted")

DATA_PATH  = "../data/processed/perfumes_merged.csv"
REPORT_PATH = "../data/reports/quality_report.json"
TARGET_COL = "rating_value"

print("Imports OK")
"""))

# ── Cell 2: Load data ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
df.head(3)
"""))

cells.append(nbf.v4.new_code_cell("""\
print("Типы данных и базовая статистика:")
print(df.dtypes)
print()
print(df.describe().T.round(3))
"""))

# ── Section 1: Missing Values ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 1. Пропущенные значения (Missing Values)"))

cells.append(nbf.v4.new_code_cell("""\
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = (pd.DataFrame({"count": missing, "percent": missing_pct})
              .query("count > 0")
              .sort_values("percent", ascending=False))

print(f"Всего пропусков: {missing.sum()} из {df.size} ({missing.sum()/df.size*100:.2f}%)")
print()
print(missing_df.to_string())
"""))

cells.append(nbf.v4.new_code_cell("""\
if len(missing_df) == 0:
    print("Пропусков не обнаружено!")
else:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    ax = axes[0]
    pct_vals = missing_df["percent"].values
    colors = ["#d62728" if p > 30 else "#ff7f0e" if p > 10 else "#1f77b4"
              for p in pct_vals]
    bars = ax.barh(missing_df.index, pct_vals, color=colors)
    ax.axvline(30, color="red", ls="--", alpha=0.5, label="30% (high)")
    ax.axvline(10, color="orange", ls="--", alpha=0.5, label="10% (medium)")
    ax.set_xlabel("Пропуски (%)")
    ax.set_title("Пропуски по столбцам")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, pct_vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)

    # Heatmap паттернов
    ax2 = axes[1]
    sample_cols = missing_df.index[:10].tolist()
    sample_size = min(500, len(df))
    sns.heatmap(df[sample_cols].sample(sample_size, random_state=42).isnull(),
                cbar=False, yticklabels=False,
                cmap=["#f0f0f0", "#d62728"], ax=ax2)
    ax2.set_title(f"Паттерн пропусков (сэмпл {sample_size} строк)")

    plt.tight_layout()
    plt.savefig("../data/reports/fig_missing.png", bbox_inches="tight")
    plt.show()
    print("Сохранено: data/reports/fig_missing.png")
"""))

# ── Section 2: Duplicates ─────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 2. Дубликаты (Duplicates)"))

cells.append(nbf.v4.new_code_cell("""\
n_dup = int(df.duplicated().sum())
dup_pct = n_dup / len(df) * 100
sev = "HIGH" if dup_pct > 5 else "MEDIUM" if dup_pct > 1 else "LOW"
print(f"Полных дубликатов: {n_dup} ({dup_pct:.2f}%) — severity: {sev}")

if n_dup > 0:
    print("\\nПримеры дублирующихся строк:")
    dup_rows = df[df.duplicated(keep=False)].sort_values(list(df.columns[:3]))
    display(dup_rows.head(6))
else:
    print("Полных дубликатов не обнаружено.")

# Частичные дубликаты по ключевым полям
key_cols = [c for c in ["url", "perfume", "brand"] if c in df.columns]
if key_cols:
    n_partial = int(df.duplicated(subset=key_cols).sum())
    print(f"\\nЧастичных дубликатов по {key_cols}: {n_partial} ({n_partial/len(df)*100:.2f}%)")
"""))

# ── Section 3: Outliers ───────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 3. Выбросы (Outliers)"))

cells.append(nbf.v4.new_code_cell("""\
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Числовых столбцов: {len(numeric_cols)} — {numeric_cols}")

outlier_rows = []
for col in numeric_cols:
    s = df[col].dropna()
    if len(s) == 0:
        continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    n_iqr = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
    pct_iqr = n_iqr / len(s) * 100
    z = np.abs(stats.zscore(s))
    n_z = int((z > 3).sum())
    pct_z = n_z / len(s) * 100
    sev = "HIGH" if pct_iqr > 10 else "MEDIUM" if pct_iqr > 3 else "LOW"
    outlier_rows.append({
        "column": col, "n_iqr": n_iqr, "pct_iqr": round(pct_iqr, 2),
        "n_zscore": n_z, "pct_zscore": round(pct_z, 2),
        "mean": round(float(s.mean()), 3), "std": round(float(s.std()), 3),
        "min": round(float(s.min()), 3), "max": round(float(s.max()), 3),
        "skewness": round(float(s.skew()), 3), "severity": sev
    })

outlier_df = pd.DataFrame(outlier_rows)
print("\\nСводка по выбросам:")
print(outlier_df.to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
# Boxplots
n = len(numeric_cols)
ncols_plot = min(3, n)
nrows_plot = (n + ncols_plot - 1) // ncols_plot
fig, axes = plt.subplots(nrows_plot, ncols_plot,
                         figsize=(6*ncols_plot, 4*nrows_plot), squeeze=False)
axes_flat = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes_flat[i]
    data = df[col].dropna()
    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
               medianprops=dict(color="red", linewidth=2),
               flierprops=dict(marker="o", markersize=3, alpha=0.3))
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    n_out = int(((data < q1-1.5*iqr) | (data > q3+1.5*iqr)).sum())
    ax.set_title(f"{col}\\nIQR: {n_out} выбросов ({n_out/len(data)*100:.1f}%)", fontsize=10)

for j in range(len(numeric_cols), len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle("Boxplots числовых столбцов", fontsize=14)
plt.tight_layout()
plt.savefig("../data/reports/fig_boxplots.png", bbox_inches="tight")
plt.show()
print("Сохранено: data/reports/fig_boxplots.png")
"""))

# ── Section 4: Target distribution ───────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "## 4. Целевая переменная: `rating_value`\n\n"
    "Задача — регрессия. Анализируем распределение целевой переменной."
))

cells.append(nbf.v4.new_code_cell("""\
target = df[TARGET_COL].dropna()
print(f"Статистика {TARGET_COL}:")
print(target.describe().round(3))
print(f"\\nАсимметрия (skewness): {target.skew():.3f}")
print(f"Пропусков в target: {df[TARGET_COL].isnull().sum()}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram
axes[0].hist(target, bins=40, color="#1f77b4", edgecolor="white", alpha=0.8)
axes[0].axvline(target.mean(), color="red", ls="--", label=f"mean={target.mean():.2f}")
axes[0].axvline(target.median(), color="orange", ls="--", label=f"median={target.median():.2f}")
axes[0].set_xlabel(TARGET_COL)
axes[0].set_title("Распределение rating_value")
axes[0].legend()

# KDE
target.plot.kde(ax=axes[1], color="#1f77b4")
axes[1].set_title("KDE: rating_value")
axes[1].set_xlabel(TARGET_COL)

# QQ-plot
stats.probplot(target, dist="norm", plot=axes[2])
axes[2].set_title("Q-Q plot (нормальность)")

plt.tight_layout()
plt.savefig("../data/reports/fig_target_dist.png", bbox_inches="tight")
plt.show()
print("Сохранено: data/reports/fig_target_dist.png")
"""))

# ── Section 5: Full detect + Quality Score ────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 5. Сводный Quality Report"))

cells.append(nbf.v4.new_code_cell("""\
import os
os.makedirs("../data/reports", exist_ok=True)

report = detect_all(df, target_col=TARGET_COL, dataset_name="perfumes_merged")

print(f"Quality Score: {report.quality_score}/100")
print(f"Всего проблем: {len(report.issues)}")

severity_order = ["critical", "high", "medium", "low"]
severity_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

for sev in severity_order:
    issues = [i for i in report.issues if i.severity == sev]
    if issues:
        print(f"\\n{severity_icons[sev]} {sev.upper()} ({len(issues)})")
        for issue in issues:
            print(f"   [{issue.type}] {issue.column}: {issue.description}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Сводная таблица
issues_data = [{
    "Тип": i.type, "Severity": i.severity, "Столбец": i.column,
    "Кол-во": i.count, "Процент": f"{i.percentage:.1f}%", "Описание": i.description,
} for i in report.issues]
issues_table = pd.DataFrame(issues_data)
print("СВОДНАЯ ТАБЛИЦА ПРОБЛЕМ:")
print(issues_table.to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
# Визуализация Quality Score и распределения проблем
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Score bar
ax = axes[0]
score = report.quality_score
color = "#2ca02c" if score >= 80 else "#ff7f0e" if score >= 60 else "#d62728"
ax.barh(["Quality Score"], [score], color=color, height=0.4)
ax.barh(["Quality Score"], [100 - score], left=score, color="#eeeeee", height=0.4)
ax.set_xlim(0, 100)
ax.axvline(80, color="green", ls="--", alpha=0.5, label="80 — хорошо")
ax.axvline(60, color="orange", ls="--", alpha=0.5, label="60 — приемлемо")
ax.text(score / 2, 0, f"{score}/100", ha="center", va="center",
        fontsize=18, fontweight="bold", color="white")
ax.set_title("Quality Score")
ax.legend(fontsize=9)

# Issues by severity
ax2 = axes[1]
severity_order = ["critical", "high", "medium", "low"]
colors_map = {"critical": "#d62728", "high": "#ff7f0e", "medium": "#ffdd57", "low": "#2ca02c"}
sev_counts = {s: len([i for i in report.issues if i.severity == s])
              for s in severity_order}
sev_counts = {k: v for k, v in sev_counts.items() if v > 0}
if sev_counts:
    ax2.bar(sev_counts.keys(), sev_counts.values(),
            color=[colors_map[s] for s in sev_counts])
    for x, (k, v) in enumerate(sev_counts.items()):
        ax2.text(x, v + 0.05, str(v), ha="center", fontweight="bold")
    ax2.set_title("Проблемы по уровню severity")
    ax2.set_ylabel("Количество проблем")

plt.tight_layout()
plt.savefig("../data/reports/fig_quality_score.png", bbox_inches="tight")
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Сохраняем JSON-отчёт
report.save(REPORT_PATH)
print(f"✅ Отчёт сохранён: {REPORT_PATH}")
print(f"   Quality Score: {report.quality_score}/100")
print(f"   Проблем найдено: {len(report.issues)}")
"""))

nb.cells = cells

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/01_detect_issues.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Ноутбук создан: notebooks/01_detect_issues.ipynb")
