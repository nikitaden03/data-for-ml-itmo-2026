"""Build 02_fix_data.ipynb programmatically."""
import nbformat

nb = nbformat.v4.new_notebook()
cells = []

cells.append(nbformat.v4.new_markdown_cell(
    "# 02 — Fix Data: Two Cleaning Strategies\n\n"
    "**Strategy A**: remove advertising + duplicates + short body + emoji  \n"
    "**Strategy B**: Strategy A + remove 'no title' rows (titled articles only)"
))

cells.append(nbformat.v4.new_code_cell("""\
import sys, os
sys.path.insert(0, '..')
import warnings
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.quality_utils import (
    fix_duplicates, fix_short_body, fix_emoji_titles,
    fix_advertising, fix_dates, run_full_detection, save_report
)

warnings.filterwarnings('ignore')
os.makedirs('../data/cleaned', exist_ok=True)
os.makedirs('../data/reports', exist_ok=True)
print('OK')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 1. Load original data"))

cells.append(nbformat.v4.new_code_cell("""\
df = pd.read_csv('../data/processed/financial_news_merged.csv', encoding='utf-8-sig')
print(f'Original shape: {df.shape}')
print(f'Labeled rows: {df["positive_market_impact"].notna().sum()}')
df.head(3)
"""))

cells.append(nbformat.v4.new_markdown_cell(
    "## 2. Strategy A — Conservative Clean\n\n"
    "Steps: remove exact duplicates → remove short body → remove advertising → clean emoji titles"
))

cells.append(nbformat.v4.new_code_cell("""\
df_a = df.copy()
n0 = len(df_a)

# Step 1: exact duplicates
df_a = fix_duplicates(df_a)
print(f'After dedup:        {len(df_a):>6} rows  (removed {n0 - len(df_a)})')

# Step 2: short body
df_a = fix_short_body(df_a, min_len=20)
print(f'After short body:   {len(df_a):>6} rows  (removed {n0 - len(df_a) - 0})')

# Step 3: remove advertising
before_adv = len(df_a)
df_a = fix_advertising(df_a)
print(f'After advertising:  {len(df_a):>6} rows  (removed {before_adv - len(df_a)})')

# Step 4: clean emoji from titles
df_a = fix_emoji_titles(df_a)

# Step 5: parse dates
df_a = fix_dates(df_a)

print(f'\\nStrategy A final:   {len(df_a)} rows')
print(f'Loss: {(n0 - len(df_a)) / n0 * 100:.1f}%')
print(f'Labeled: {df_a["positive_market_impact"].notna().sum()}')
"""))

cells.append(nbformat.v4.new_code_cell("""\
df_a.to_csv('../data/cleaned/strategy_a.csv', index=False, encoding='utf-8-sig')
print('Saved: data/cleaned/strategy_a.csv')
df_a.head(3)
"""))

cells.append(nbformat.v4.new_markdown_cell(
    "## 3. Strategy B — Aggressive Clean\n\n"
    "Strategy A + remove 'no title' rows → only articles with real titles"
))

cells.append(nbformat.v4.new_code_cell("""\
df_b = df.copy()
n0 = len(df_b)

# Same steps as A
df_b = fix_duplicates(df_b)
df_b = fix_short_body(df_b, min_len=20)
df_b = fix_advertising(df_b)
df_b = fix_emoji_titles(df_b)
df_b = fix_dates(df_b)

# Extra: remove no-title rows
before_notitle = len(df_b)
df_b = df_b[df_b['title'] != 'no title'].reset_index(drop=True)
print(f'Removed no-title:   {before_notitle - len(df_b)} rows')

print(f'\\nStrategy B final:   {len(df_b)} rows')
print(f'Loss: {(n0 - len(df_b)) / n0 * 100:.1f}%')
print(f'Labeled: {df_b["positive_market_impact"].notna().sum()}')
"""))

cells.append(nbformat.v4.new_code_cell("""\
df_b.to_csv('../data/cleaned/strategy_b.csv', index=False, encoding='utf-8-sig')
print('Saved: data/cleaned/strategy_b.csv')
df_b.head(3)
"""))

cells.append(nbformat.v4.new_markdown_cell("## 4. Quality scores after fixing"))

cells.append(nbformat.v4.new_code_cell("""\
# Re-read with string columns for quality_utils
df_a_str = pd.read_csv('../data/cleaned/strategy_a.csv', encoding='utf-8-sig')
df_b_str = pd.read_csv('../data/cleaned/strategy_b.csv', encoding='utf-8-sig')

report_a = run_full_detection(df_a_str, 'positive_market_impact')
report_b = run_full_detection(df_b_str, 'positive_market_impact')

save_report(report_a, '../data/reports/quality_report_a.json')
save_report(report_b, '../data/reports/quality_report_b.json')

print(f'Strategy A Quality Score: {report_a["quality_score"]}/100')
print(f'Strategy B Quality Score: {report_b["quality_score"]}/100')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 5. Comparison chart"))

cells.append(nbformat.v4.new_code_cell("""\
import json
with open('../data/reports/quality_report.json') as f:
    report_orig = json.load(f)

orig_rows = len(df)
a_rows = len(df_a_str)
b_rows = len(df_b_str)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Row counts
labels = ['Original', 'Strategy A', 'Strategy B']
values = [orig_rows, a_rows, b_rows]
colors = ['#aaaaaa', '#6699cc', '#70b070']
axes[0].bar(labels, values, color=colors, edgecolor='white')
axes[0].set_title('Row count', fontsize=12)
axes[0].set_ylabel('Rows')
for i, v in enumerate(values):
    axes[0].text(i, v + 50, f'{v:,}', ha='center', fontsize=9)

# Quality scores
scores = [report_orig['quality_score'], report_a['quality_score'], report_b['quality_score']]
axes[1].bar(labels, scores, color=colors, edgecolor='white')
axes[1].set_title('Quality Score', fontsize=12)
axes[1].set_ylabel('Score /100')
axes[1].set_ylim(0, 110)
for i, v in enumerate(scores):
    axes[1].text(i, v + 1, str(v), ha='center', fontsize=10, fontweight='bold')

# Data loss %
losses = [0, (orig_rows - a_rows) / orig_rows * 100, (orig_rows - b_rows) / orig_rows * 100]
axes[2].bar(labels, losses, color=colors, edgecolor='white')
axes[2].set_title('Data loss %', fontsize=12)
axes[2].set_ylabel('Loss %')
for i, v in enumerate(losses):
    axes[2].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('../data/reports/fig_strategy_comparison.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved fig_strategy_comparison.png')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 6. Class balance after fixing"))

cells.append(nbformat.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, df_s, title in [
    (axes[0], df_a_str, 'Strategy A'),
    (axes[1], df_b_str, 'Strategy B')
]:
    labeled = df_s[df_s['positive_market_impact'].notna()]
    vc = labeled['positive_market_impact'].value_counts().sort_index()
    ax.bar(['Neg/Neutral (0)', 'Positive (1)'], vc.values,
           color=['#e07070', '#70b070'], edgecolor='white')
    ax.set_title(f'{title} — class balance', fontsize=12)
    ax.set_ylabel('Count')
    for i, v in enumerate(vc.values):
        ax.text(i, v + 10, f'{v}\\n({v/len(labeled)*100:.1f}%)', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('../data/reports/fig_balance_after_fix.png', dpi=120, bbox_inches='tight')
plt.show()
"""))

cells.append(nbformat.v4.new_markdown_cell("## 7. Summary"))

cells.append(nbformat.v4.new_code_cell("""\
orig_rows = len(df)
print('='*55)
print('FIX SUMMARY')
print('='*55)
print(f'Original:   {orig_rows:>6} rows   score={report_orig["quality_score"]}/100')
print(f'Strategy A: {a_rows:>6} rows   score={report_a["quality_score"]}/100   loss={((orig_rows-a_rows)/orig_rows*100):.1f}%')
print(f'Strategy B: {b_rows:>6} rows   score={report_b["quality_score"]}/100   loss={((orig_rows-b_rows)/orig_rows*100):.1f}%')
print()
print('Strategy A removes: duplicates + short body + advertising + emoji')
print('Strategy B removes: all of A + no-title rows (Telegram posts)')
"""))

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"}
}

import nbformat as nbf
path = 'notebooks/02_fix_data.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f'Written: {path}')
