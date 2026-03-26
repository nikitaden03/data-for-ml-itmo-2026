"""Build 01_detect_issues.ipynb programmatically."""
import nbformat
import os

nb = nbformat.v4.new_notebook()

cells = []

# --- Markdown: title ---
cells.append(nbformat.v4.new_markdown_cell(
    "# 01 — Obnaruzhenie problem kachestva dannykh\n\n"
    "**Datasest**: `data/processed/financial_news_merged.csv`  \n"
    "**Zadacha**: binarnaya klassifikatsiya — `positive_market_impact` (0/1)"
))

# --- Cell 1: imports ---
cells.append(nbformat.v4.new_code_cell("""\
import sys, os
sys.path.insert(0, '..')
import warnings
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.quality_utils import run_full_detection, save_report, print_report_summary

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
os.makedirs('../data/reports', exist_ok=True)
print('OK')
"""))

# --- Markdown: load ---
cells.append(nbformat.v4.new_markdown_cell("## 1. Zagruzka dannykh"))

# --- Cell 2: load data ---
cells.append(nbformat.v4.new_code_cell("""\
df = pd.read_csv('../data/processed/financial_news_merged.csv', encoding='utf-8-sig')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
df.head(3)
"""))

# --- Markdown: detection ---
cells.append(nbformat.v4.new_markdown_cell("## 2. Polnaya proverka kachestva"))

# --- Cell 3: run detection ---
cells.append(nbformat.v4.new_code_cell("""\
report = run_full_detection(df, target_col='positive_market_impact')
print_report_summary(report)
"""))

# --- Cell 4: save report ---
cells.append(nbformat.v4.new_code_cell("""\
save_report(report, '../data/reports/quality_report.json')
print(f"Quality Score: {report['quality_score']}/100")
"""))

# --- Markdown: viz ---
cells.append(nbformat.v4.new_markdown_cell("## 3. Vizualizatsiya problem"))

# --- Cell 5: fig missing ---
cells.append(nbformat.v4.new_code_cell("""\
fig, ax = plt.subplots(figsize=(10, 4))
nulls = df.isnull().sum()
nulls = nulls[nulls > 0]
if len(nulls) > 0:
    nulls.plot(kind='bar', ax=ax, color='#e07070', edgecolor='white')
    ax.set_title('Missing values by column', fontsize=13)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=30)
    for i, v in enumerate(nulls.values):
        ax.text(i, v + 0.3, f'{v} ({v/len(df)*100:.1f}%)', ha='center', fontsize=9)
else:
    ax.text(0.5, 0.5, 'No missing values', ha='center', va='center', fontsize=14)
plt.tight_layout()
plt.savefig('../data/reports/fig_missing.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved fig_missing.png')
"""))

# --- Cell 6: fig quality score ---
cells.append(nbformat.v4.new_code_cell("""\
score = report['quality_score']
fig, ax = plt.subplots(figsize=(6, 3))
color = '#70b070' if score >= 70 else '#e0a050' if score >= 50 else '#e07070'
ax.barh(['Quality Score'], [score], color=color, edgecolor='white')
ax.barh(['Quality Score'], [100 - score], left=[score], color='#eeeeee', edgecolor='white')
ax.set_xlim(0, 100)
ax.set_title(f'Overall Quality Score: {score}/100', fontsize=14)
ax.text(score + 1, 0, str(score), va='center', fontsize=14, fontweight='bold')
ax.set_xlabel('Score')
plt.tight_layout()
plt.savefig('../data/reports/fig_quality_score.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved fig_quality_score.png')
"""))

# --- Cell 7: article type + body length ---
cells.append(nbformat.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

type_counts = df['article_type'].value_counts(dropna=False)
colors = ['#e07070' if str(t) == 'advertising' else '#6699cc' for t in type_counts.index]
type_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='white')
axes[0].set_title('Article types (red = noise)', fontsize=12)
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=40)
for i, v in enumerate(type_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontsize=8)

axes[1].hist(df['body'].str.len().clip(upper=2000), bins=50, color='#6699cc', edgecolor='white')
axes[1].axvline(20, color='red', linestyle='--', label='threshold = 20 chars')
axes[1].set_title('Body length distribution', fontsize=12)
axes[1].set_xlabel('Characters')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.savefig('../data/reports/fig_quality_check.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved fig_quality_check.png')
"""))

# --- Markdown: details ---
cells.append(nbformat.v4.new_markdown_cell("## 4. Detalny razbor problem"))

# --- Cell 8: no title ---
cells.append(nbformat.v4.new_code_cell("""\
no_title = df[df['title'] == 'no title']
print(f'Rows without title: {len(no_title)} ({len(no_title)/len(df)*100:.1f}%)')
print(f'Unique bodies among them: {no_title["body"].nunique()}')
print(f'Sources: {no_title["source"].value_counts().to_dict()}')
print('\\nSamples:')
for b in no_title['body'].head(3):
    print(f'  {repr(b[:100])}')
"""))

# --- Cell 9: short body ---
cells.append(nbformat.v4.new_code_cell("""\
short = df[df['body'].str.len() < 20]
print(f'Short texts (< 20 chars): {len(short)}')
print(short[['title', 'body', 'source']].head(8).to_string())
"""))

# --- Cell 10: exact duplicates ---
cells.append(nbformat.v4.new_code_cell("""\
real = df[df['title'] != 'no title']
dups = real[real.duplicated(subset=['title', 'body'], keep=False)]
print(f'Exact duplicates (title+body): {len(dups)}')
if len(dups) > 0:
    print(dups[['title', 'source']].to_string())
"""))

# --- Cell 11: class balance ---
cells.append(nbformat.v4.new_code_cell("""\
labeled = df[df['positive_market_impact'].notna()]
vc = labeled['positive_market_impact'].value_counts()
print('Class balance (labeled rows):')
for label, cnt in vc.items():
    print(f'  {int(label)}: {cnt} ({cnt/len(labeled)*100:.1f}%)')
"""))

# --- Markdown: summary table ---
cells.append(nbformat.v4.new_markdown_cell("## 5. Summary table of issues"))

# --- Cell 12: summary table ---
cells.append(nbformat.v4.new_code_cell("""\
summary = [
    {'Issue': 'Rows without labels (scraped)', 'Count': 40, 'Pct': '0.3%', 'Severity': 'Medium', 'Fix': 'Auto-label in step 3'},
    {'Issue': 'no title entries', 'Count': 11021, 'Pct': '73.3%', 'Severity': 'Medium', 'Fix': 'Use body only for classification'},
    {'Issue': 'Advertising articles', 'Count': 1213, 'Pct': '8.1%', 'Severity': 'Medium', 'Fix': 'Remove in Strategy A, keep in Strategy B'},
    {'Issue': 'Short body (< 20 chars)', 'Count': 29, 'Pct': '0.2%', 'Severity': 'Medium', 'Fix': 'Drop from both strategies'},
    {'Issue': 'Exact duplicates (title+body)', 'Count': 2, 'Pct': '0.01%', 'Severity': 'High', 'Fix': 'Keep first occurrence'},
    {'Issue': 'Emoji in titles', 'Count': 3, 'Pct': '0.02%', 'Severity': 'Low', 'Fix': 'Clean with regex'},
    {'Issue': 'sectors/tickers as strings', 'Count': 15000, 'Pct': '100%', 'Severity': 'Low', 'Fix': 'ast.literal_eval when needed'},
]
pd.DataFrame(summary)
"""))

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"}
}

path = 'notebooks/01_detect_issues.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f'Written: {path}')
