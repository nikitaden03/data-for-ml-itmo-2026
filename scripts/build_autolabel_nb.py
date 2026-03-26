"""Build notebooks/04_auto_label.ipynb programmatically."""
import nbformat

nb = nbformat.v4.new_notebook()
cells = []

cells.append(nbformat.v4.new_markdown_cell(
    "# 04 — Auto-Labeling with Claude API\n\n"
    "Labels 40 scraped news articles + re-labels 25 GPT-4o articles for quality assessment.\n\n"
    "**Model**: claude-haiku-4-5 (cost-efficient)  \n"
    "**Task**: binary classification — `positive_market_impact` (0/1)"
))

cells.append(nbformat.v4.new_code_cell("""\
import sys, os, json
sys.path.insert(0, '..')
import pandas as pd
import anthropic
from src.labeling_utils import label_batch, compute_cohen_kappa, interpret_kappa, save_quality_metrics

os.makedirs('../data/labeled', exist_ok=True)
os.makedirs('../data/reports', exist_ok=True)

# Load API key
with open('../claude-api-key.txt') as f:
    api_key = f.read().strip()

client = anthropic.Anthropic(api_key=api_key)
print('Client ready. Model: claude-haiku-4-5-20251001')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 1. Load data"))

cells.append(nbformat.v4.new_code_cell("""\
df = pd.read_csv('../data/cleaned/strategy_a.csv', encoding='utf-8-sig')
print(f'Total rows: {len(df)}')

# Split: scraped (unlabeled) vs labeled
scraped = df[df['positive_market_impact'].isna()].copy()
labeled = df[df['positive_market_impact'].notna()].copy()

print(f'Scraped (to label): {len(scraped)}')
print(f'Labeled (GPT-4o):   {len(labeled)}')
print(f'Scraped sources: {scraped["source"].value_counts().to_dict()}')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 2. Sample 25 GPT-4o rows for agreement check"))

cells.append(nbformat.v4.new_code_cell("""\
agreement_sample = labeled.sample(25, random_state=42).copy()
print(f'Agreement sample: {len(agreement_sample)} rows')
print(f'GPT-4o labels: {agreement_sample["positive_market_impact"].value_counts().to_dict()}')
agreement_sample[['title', 'body', 'source', 'positive_market_impact']].head(5)
"""))

cells.append(nbformat.v4.new_markdown_cell("## 3. Label scraped articles (40 rows)"))

cells.append(nbformat.v4.new_code_cell("""\
scraped_rows = [
    {'idx': i, 'title': row['title'], 'body': row['body']}
    for i, row in scraped.iterrows()
]

print(f'Labeling {len(scraped_rows)} scraped articles...')
scraped_results = label_batch(client, scraped_rows, delay=0.5)
errors = [r for r in scraped_results if r['error']]
print(f'Done. Errors: {len(errors)}')

# Apply labels
idx_to_label = {r['idx']: r['label'] for r in scraped_results if r['label'] is not None}
for idx, label in idx_to_label.items():
    scraped.loc[idx, 'positive_market_impact'] = label

print(f'Label distribution: {scraped["positive_market_impact"].value_counts().to_dict()}')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 4. Re-label 25 GPT-4o rows for agreement check"))

cells.append(nbformat.v4.new_code_cell("""\
agreement_rows = [
    {'idx': i, 'title': row['title'], 'body': row['body']}
    for i, row in agreement_sample.iterrows()
]

print(f'Re-labeling {len(agreement_rows)} rows for agreement check...')
agreement_results = label_batch(client, agreement_rows, delay=0.5)
errors2 = [r for r in agreement_results if r['error']]
print(f'Done. Errors: {len(errors2)}')

# Attach Claude labels to agreement sample
idx_to_claude = {r['idx']: r['label'] for r in agreement_results if r['label'] is not None}
agreement_sample['claude_label'] = agreement_sample.index.map(idx_to_claude)
agreement_sample = agreement_sample.dropna(subset=['claude_label'])
agreement_sample['claude_label'] = agreement_sample['claude_label'].astype(int)
agreement_sample['gpt4o_label'] = agreement_sample['positive_market_impact'].astype(int)

print(f'Agreement rows with labels: {len(agreement_sample)}')
print(f'Claude labels: {agreement_sample["claude_label"].value_counts().to_dict()}')
print(f'GPT-4o labels: {agreement_sample["gpt4o_label"].value_counts().to_dict()}')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 5. Compute Cohen's Kappa"))

cells.append(nbformat.v4.new_code_cell("""\
gpt4o_labels = agreement_sample['gpt4o_label'].tolist()
claude_labels = agreement_sample['claude_label'].tolist()

kappa = compute_cohen_kappa(gpt4o_labels, claude_labels)
pct_agree = sum(a == b for a, b in zip(gpt4o_labels, claude_labels)) / len(gpt4o_labels) * 100
interpretation = interpret_kappa(kappa)

print(f'Cohen kappa (GPT-4o vs Claude): {kappa:.4f}')
print(f'% Agreement: {pct_agree:.1f}%')
print(f'Interpretation: {interpretation}')
print()

# Show disagreements
disagree = agreement_sample[agreement_sample['gpt4o_label'] != agreement_sample['claude_label']]
print(f'Disagreements: {len(disagree)}/{len(agreement_sample)}')
if len(disagree) > 0:
    print(disagree[['title', 'body', 'gpt4o_label', 'claude_label']].head(5).to_string())
"""))

cells.append(nbformat.v4.new_markdown_cell("## 6. Build final labeled dataset"))

cells.append(nbformat.v4.new_code_cell("""\
# Merge: original labeled (GPT-4o) + scraped (Claude)
df_final = pd.concat([labeled, scraped], ignore_index=True)
df_final['positive_market_impact'] = df_final['positive_market_impact'].astype(int)

print(f'Final dataset shape: {df_final.shape}')
print(f'Total labeled: {df_final["positive_market_impact"].notna().sum()}')
print(f'Label distribution: {df_final["positive_market_impact"].value_counts().to_dict()}')
print(f'Balance: {df_final["positive_market_impact"].mean()*100:.1f}% positive')

df_final.to_csv('../data/labeled/strategy_a_labeled.csv', index=False, encoding='utf-8-sig')
print('Saved: data/labeled/strategy_a_labeled.csv')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 7. Save quality metrics"))

cells.append(nbformat.v4.new_code_cell("""\
metrics = {
    'model': 'claude-haiku-4-5-20251001',
    'scraped_labeled': len(scraped),
    'agreement_sample_size': len(agreement_sample),
    'cohen_kappa': round(kappa, 4),
    'percent_agreement': round(pct_agree, 1),
    'interpretation': interpretation,
    'disagreements': len(disagree),
    'final_dataset': {
        'rows': len(df_final),
        'positive': int(df_final['positive_market_impact'].sum()),
        'negative': int((df_final['positive_market_impact'] == 0).sum()),
        'positive_pct': round(df_final['positive_market_impact'].mean() * 100, 1)
    }
}
save_quality_metrics(metrics, '../data/reports/quality_metrics.json')
print(json.dumps(metrics, indent=2, ensure_ascii=False))
"""))

cells.append(nbformat.v4.new_markdown_cell("## 8. Visualize label distribution"))

cells.append(nbformat.v4.new_code_cell("""\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Final label distribution
vc = df_final['positive_market_impact'].value_counts().sort_index()
axes[0].bar(['Neg/Neutral (0)', 'Positive (1)'], vc.values,
            color=['#e07070', '#70b070'], edgecolor='white')
axes[0].set_title('Final label distribution', fontsize=12)
axes[0].set_ylabel('Count')
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 20, f'{v}\\n({v/len(df_final)*100:.1f}%)', ha='center', fontsize=10)

# Scraped labels
vc_s = scraped['positive_market_impact'].value_counts().sort_index()
axes[1].bar(['Neg/Neutral (0)', 'Positive (1)'], vc_s.values,
            color=['#e07070', '#70b070'], edgecolor='white')
axes[1].set_title(f'Scraped labels (Claude)', fontsize=12)
for i, v in enumerate(vc_s.values):
    axes[1].text(i, v + 0.2, str(v), ha='center', fontsize=11)

# Agreement confusion
from collections import Counter
combos = Counter(zip(gpt4o_labels, claude_labels))
cm = [[combos.get((i, j), 0) for j in [0, 1]] for i in [0, 1]]
sns.heatmap(cm, annot=True, fmt='d', ax=axes[2],
            xticklabels=['Claude 0', 'Claude 1'],
            yticklabels=['GPT-4o 0', 'GPT-4o 1'],
            cmap='Blues')
axes[2].set_title(f'Agreement matrix\\nkappa={kappa:.3f} ({interpretation})', fontsize=11)

plt.tight_layout()
plt.savefig('../data/reports/label_distribution.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved label_distribution.png')
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
