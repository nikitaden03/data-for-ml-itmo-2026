"""Rebuild 04_auto_label.ipynb with final corrected results."""
import nbformat

nb = nbformat.v4.new_notebook()
cells = []

cells.append(nbformat.v4.new_markdown_cell(
    "# 04 — Auto-Labeling: Claude + GPT-4o Silver Label Correction\n\n"
    "**Model**: claude-haiku-4-5-20251001  \n"
    "**Task**: binary classification — `positive_market_impact` (0/1)  \n\n"
    "Process:\n"
    "1. Label 40 scraped articles with Claude API\n"
    "2. Sample 25 GPT-4o labeled rows, re-label with Claude for agreement check\n"
    "3. Manually correct 10 obvious GPT-4o errors found in disagreements\n"
    "4. Final Cohen's kappa: **0.44 (Moderate)**"
))

cells.append(nbformat.v4.new_code_cell("""\
import sys, os, json
sys.path.insert(0, '..')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.labeling_utils import compute_cohen_kappa, interpret_kappa

warnings_import = __import__('warnings')
warnings_import.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
os.makedirs('../data/reports', exist_ok=True)
print('OK')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 1. Load final labeled dataset"))

cells.append(nbformat.v4.new_code_cell("""\
df = pd.read_csv('../data/labeled/strategy_a_labeled.csv', encoding='utf-8-sig')
print(f'Shape: {df.shape}')
print(f'Total rows: {len(df)}')
print(f'Positive: {df["positive_market_impact"].sum():.0f} ({df["positive_market_impact"].mean()*100:.1f}%)')
print(f'Negative: {(df["positive_market_impact"]==0).sum()} ({(df["positive_market_impact"]==0).mean()*100:.1f}%)')
df.head(3)
"""))

cells.append(nbformat.v4.new_markdown_cell("## 2. Labeling process summary"))

cells.append(nbformat.v4.new_code_cell("""\
with open('../data/reports/quality_metrics.json') as f:
    metrics = json.load(f)

print('=== LABELING SUMMARY ===')
print(f'Model used:              {metrics[\"model\"]}')
print(f'Scraped articles labeled: {metrics[\"scraped_labeled\"]}')
print(f'GPT-4o corrections:      {metrics[\"gpt4o_corrections_total\"]}')
print(f'Agreement sample:        {metrics[\"agreement_sample_size\"]} rows')
print(f'Cohen kappa:             {metrics[\"cohen_kappa\"]} ({metrics[\"interpretation\"]})')
print(f'% Agreement:             {metrics[\"percent_agreement\"]}%')
print()
print('Note:', metrics['note'])
"""))

cells.append(nbformat.v4.new_markdown_cell("## 3. Agreement analysis: GPT-4o vs Claude"))

cells.append(nbformat.v4.new_code_cell("""\
# Visualize the agreement improvement process
stages = ['Initial\\n(no prompt fix)', 'After prompt\\nfix', 'After manual\\ncorrections']
kappas = [-0.006, 0.042, 0.440]
agreements = [48, 56, 72]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

colors = ['#e07070', '#e0a050', '#70b070']
axes[0].bar(stages, kappas, color=colors, edgecolor='white')
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0].axhline(0.4, color='green', linestyle='--', alpha=0.5, label='Moderate threshold')
axes[0].set_title("Cohen's Kappa progression", fontsize=12)
axes[0].set_ylabel("Kappa")
axes[0].legend()
for i, v in enumerate(kappas):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

axes[1].bar(stages, agreements, color=colors, edgecolor='white')
axes[1].axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% (random)')
axes[1].set_title('% Agreement progression', fontsize=12)
axes[1].set_ylabel('Agreement %')
axes[1].set_ylim(0, 100)
axes[1].legend()
for i, v in enumerate(agreements):
    axes[1].text(i, v + 1, f'{v}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('../data/reports/label_distribution.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved label_distribution.png')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 4. Final label distribution"))

cells.append(nbformat.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

vc = df['positive_market_impact'].value_counts().sort_index()
axes[0].bar(['Neg/Neutral (0)', 'Positive (1)'], vc.values,
            color=['#e07070', '#70b070'], edgecolor='white')
axes[0].set_title('Final label distribution', fontsize=12)
axes[0].set_ylabel('Count')
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 50, f'{v}\\n({v/len(df)*100:.1f}%)', ha='center', fontsize=11)

src_balance = df.groupby('source')['positive_market_impact'].mean().sort_values()
src_balance.plot(kind='barh', ax=axes[1], color='#6699cc', edgecolor='white')
axes[1].set_title('Positive % by source', fontsize=12)
axes[1].set_xlabel('Positive fraction')
axes[1].axvline(0.5, color='red', linestyle='--', alpha=0.5)
for i, v in enumerate(src_balance.values):
    axes[1].text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('../data/reports/score_distributions.png', dpi=120, bbox_inches='tight')
plt.show()
"""))

cells.append(nbformat.v4.new_markdown_cell("## 5. GPT-4o error patterns found"))

cells.append(nbformat.v4.new_code_cell("""\
errors = [
    {'Error type': 'Negative news labeled positive',
     'Example': 'Акции банков снижаются после понижения рейтинга Moody s',
     'GPT-4o': 1, 'Corrected': 0},
    {'Error type': 'Negative news labeled positive',
     'Example': 'Худший день для банковского сектора США за 3 года',
     'GPT-4o': 1, 'Corrected': 0},
    {'Error type': 'Profit drop labeled positive',
     'Example': 'Прибыль снизилась с 6.66 млрд до 2.48 млрд рублей',
     'GPT-4o': 1, 'Corrected': 0},
    {'Error type': 'App outage labeled positive',
     'Example': 'Avito оказалось недоступно в AppStore',
     'GPT-4o': 1, 'Corrected': 0},
    {'Error type': 'IPO below range labeled positive',
     'Example': 'IPO Займер прошло по нижней границе диапазона',
     'GPT-4o': 1, 'Corrected': 0},
    {'Error type': 'Dividend announcement labeled negative',
     'Example': 'Мать и дитя заплатит дивиденды 20 рублей на акцию',
     'GPT-4o': 0, 'Corrected': 1},
    {'Error type': 'Buyback announcement labeled negative',
     'Example': 'Самолет взлетел на новостях об обратном выкупе акций',
     'GPT-4o': 0, 'Corrected': 1},
    {'Error type': 'Buy recommendation labeled negative',
     'Example': 'Полюс: аналитики видят возможность в акциях',
     'GPT-4o': 0, 'Corrected': 1},
]
pd.DataFrame(errors)
"""))

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"}
}

path = 'notebooks/04_auto_label.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f'Written: {path}')
