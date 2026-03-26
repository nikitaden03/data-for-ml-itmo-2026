"""Build 03_compare_results.ipynb programmatically."""
import nbformat

nb = nbformat.v4.new_notebook()
cells = []

cells.append(nbformat.v4.new_markdown_cell(
    "# 03 — Compare Strategies: A vs B\n\n"
    "Quantitative and visual comparison of two cleaning strategies.\n"
    "Final recommendation is made based on data retention, quality score, and class balance."
))

cells.append(nbformat.v4.new_code_cell("""\
import sys, os, json
sys.path.insert(0, '..')
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
os.makedirs('../data/reports', exist_ok=True)
print('OK')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 1. Load datasets"))

cells.append(nbformat.v4.new_code_cell("""\
df_orig = pd.read_csv('../data/processed/financial_news_merged.csv', encoding='utf-8-sig')
df_a = pd.read_csv('../data/cleaned/strategy_a.csv', encoding='utf-8-sig')
df_b = pd.read_csv('../data/cleaned/strategy_b.csv', encoding='utf-8-sig')

with open('../data/reports/quality_report.json') as f:
    rep_orig = json.load(f)
with open('../data/reports/quality_report_a.json') as f:
    rep_a = json.load(f)
with open('../data/reports/quality_report_b.json') as f:
    rep_b = json.load(f)

print(f'Original:   {len(df_orig):>6} rows')
print(f'Strategy A: {len(df_a):>6} rows')
print(f'Strategy B: {len(df_b):>6} rows')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 2. Quantitative comparison"))

cells.append(nbformat.v4.new_code_cell("""\
orig_n = len(df_orig)

def labeled(df):
    return df[df['positive_market_impact'].notna()]

def balance(df):
    lab = labeled(df)
    vc = lab['positive_market_impact'].value_counts(normalize=True)
    return vc.get(1.0, 0) * 100

comparison = pd.DataFrame({
    'Metric': ['Rows', 'Labeled rows', 'Data loss %', 'Quality score',
               'Positive class %', 'Mean title length', 'Mean body length',
               'No-title rows', 'Advertising rows'],
    'Original': [
        orig_n,
        labeled(df_orig).__len__(),
        0,
        rep_orig['quality_score'],
        round(balance(df_orig), 1),
        round(df_orig['title'].str.len().mean(), 1),
        round(df_orig['body'].str.len().mean(), 1),
        (df_orig['title'] == 'no title').sum(),
        (df_orig['article_type'] == 'advertising').sum()
    ],
    'Strategy A': [
        len(df_a),
        labeled(df_a).__len__(),
        round((orig_n - len(df_a)) / orig_n * 100, 1),
        rep_a['quality_score'],
        round(balance(df_a), 1),
        round(df_a['title'].str.len().mean(), 1),
        round(df_a['body'].str.len().mean(), 1),
        (df_a['title'] == 'no title').sum(),
        (df_a.get('article_type', pd.Series()) == 'advertising').sum()
    ],
    'Strategy B': [
        len(df_b),
        labeled(df_b).__len__(),
        round((orig_n - len(df_b)) / orig_n * 100, 1),
        rep_b['quality_score'],
        round(balance(df_b), 1),
        round(df_b['title'].str.len().mean(), 1),
        round(df_b['body'].str.len().mean(), 1),
        (df_b['title'] == 'no title').sum(),
        (df_b.get('article_type', pd.Series()) == 'advertising').sum()
    ]
})
comparison
"""))

cells.append(nbformat.v4.new_markdown_cell("## 3. KS-tests: distribution comparison"))

cells.append(nbformat.v4.new_code_cell("""\
# KS test: body length distribution before/after
ks_a = stats.ks_2samp(df_orig['body'].str.len(), df_a['body'].str.len())
ks_b = stats.ks_2samp(df_orig['body'].str.len(), df_b['body'].str.len())

print('KS-test: body length distribution')
print(f'  Original vs Strategy A: statistic={ks_a.statistic:.4f}, p={ks_a.pvalue:.4f}')
print(f'  Original vs Strategy B: statistic={ks_b.statistic:.4f}, p={ks_b.pvalue:.4f}')
print()
print('Interpretation:')
print(f'  Strategy A: {"SIMILAR distribution (p>0.05)" if ks_a.pvalue > 0.05 else "DIFFERENT distribution (p<=0.05)"}')
print(f'  Strategy B: {"SIMILAR distribution (p>0.05)" if ks_b.pvalue > 0.05 else "DIFFERENT distribution (p<=0.05)"}')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 4. Visual comparison"))

cells.append(nbformat.v4.new_code_cell("""\
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

colors_map = {'Original': '#aaaaaa', 'Strategy A': '#6699cc', 'Strategy B': '#70b070'}

# 1. Row counts
row_data = {'Original': orig_n, 'Strategy A': len(df_a), 'Strategy B': len(df_b)}
axes[0,0].bar(row_data.keys(), row_data.values(),
              color=[colors_map[k] for k in row_data], edgecolor='white')
axes[0,0].set_title('Row count', fontsize=12)
for i, (k, v) in enumerate(row_data.items()):
    axes[0,0].text(i, v + 100, f'{v:,}', ha='center', fontsize=9)

# 2. Quality scores
score_data = {'Original': rep_orig['quality_score'], 'Strategy A': rep_a['quality_score'], 'Strategy B': rep_b['quality_score']}
axes[0,1].bar(score_data.keys(), score_data.values(),
              color=[colors_map[k] for k in score_data], edgecolor='white')
axes[0,1].set_title('Quality Score /100', fontsize=12)
axes[0,1].set_ylim(0, 110)
for i, (k, v) in enumerate(score_data.items()):
    axes[0,1].text(i, v + 1, str(v), ha='center', fontsize=11, fontweight='bold')

# 3. Class balance
bal_data = {'Original': balance(df_orig), 'Strategy A': balance(df_a), 'Strategy B': balance(df_b)}
axes[0,2].bar(bal_data.keys(), bal_data.values(),
              color=[colors_map[k] for k in bal_data], edgecolor='white')
axes[0,2].set_title('Positive class %', fontsize=12)
axes[0,2].set_ylim(0, 80)
axes[0,2].axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
axes[0,2].legend()
for i, (k, v) in enumerate(bal_data.items()):
    axes[0,2].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)

# 4. Body length KDE
ax = axes[1,0]
for df_s, label, color in [
    (df_orig, 'Original', '#aaaaaa'),
    (df_a, 'Strategy A', '#6699cc'),
    (df_b, 'Strategy B', '#70b070')
]:
    lens = df_s['body'].str.len().clip(upper=2000)
    lens.plot.kde(ax=ax, label=label, color=color)
ax.set_title('Body length KDE', fontsize=12)
ax.set_xlabel('Characters')
ax.legend()

# 5. Article types after fix
for ax_s, df_s, title in [(axes[1,1], df_a, 'Strategy A'), (axes[1,2], df_b, 'Strategy B')]:
    tc = df_s['article_type'].value_counts(dropna=False)
    tc.plot(kind='bar', ax=ax_s, color='#6699cc', edgecolor='white')
    ax_s.set_title(f'{title} — article types', fontsize=12)
    ax_s.tick_params(axis='x', rotation=40)
    for i, v in enumerate(tc.values):
        ax_s.text(i, v + 5, str(v), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('../data/reports/distribution_comparison.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved distribution_comparison.png')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 5. Recommendation"))

cells.append(nbformat.v4.new_code_cell("""\
print('='*55)
print('RECOMMENDATION: Strategy A')
print('='*55)
print()
print('Reasons:')
print('1. Preserves 13,799 rows vs 3,895 in B — 3.5x more data for training')
print('2. "no title" rows contain valid financial text in body field')
print('   — useful for classification even without a title')
print('3. Both strategies have the same quality score (57/100)')
print('4. Class balance remains near-ideal: ~49% positive in both')
print('5. Strategy B loses 74% of data — too aggressive for ML training')
print()
print('Recommended file: data/cleaned/strategy_a.csv')
print(f'Rows: {len(df_a):,}  |  Labeled: {labeled(df_a).__len__():,}')
"""))

cells.append(nbformat.v4.new_code_cell("""\
# Save comparison report
ks_a = stats.ks_2samp(df_orig['body'].str.len(), df_a['body'].str.len())
ks_b = stats.ks_2samp(df_orig['body'].str.len(), df_b['body'].str.len())

report = {
    'recommendation': 'strategy_a',
    'recommended_file': 'data/cleaned/strategy_a.csv',
    'reasons': [
        '13799 rows retained (vs 3895 in B) — 3.5x more training data',
        'no-title rows contain valid financial text in body',
        'Same quality score (57/100) for both strategies',
        'Near-ideal class balance preserved (~49% positive)',
        'Strategy B removes 74% of data — too aggressive'
    ],
    'strategies': {
        'original': {'rows': len(df_orig), 'quality_score': rep_orig['quality_score'], 'loss_pct': 0},
        'strategy_a': {
            'rows': len(df_a),
            'quality_score': rep_a['quality_score'],
            'loss_pct': round((len(df_orig) - len(df_a)) / len(df_orig) * 100, 1),
            'positive_pct': round(balance(df_a), 1),
            'ks_statistic': round(ks_a.statistic, 4),
            'ks_pvalue': round(ks_a.pvalue, 4)
        },
        'strategy_b': {
            'rows': len(df_b),
            'quality_score': rep_b['quality_score'],
            'loss_pct': round((len(df_orig) - len(df_b)) / len(df_orig) * 100, 1),
            'positive_pct': round(balance(df_b), 1),
            'ks_statistic': round(ks_b.statistic, 4),
            'ks_pvalue': round(ks_b.pvalue, 4)
        }
    }
}

import json
with open('../data/reports/comparison_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print('Saved: data/reports/comparison_report.json')
"""))

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"}
}

path = 'notebooks/03_compare_results.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f'Written: {path}')
