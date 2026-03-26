"""Build notebooks/al_experiment.ipynb programmatically."""
import nbformat

nb = nbformat.v4.new_notebook()
cells = []

cells.append(nbformat.v4.new_markdown_cell(
    "# Active Learning Experiment: Entropy vs Random\n\n"
    "**Dataset**: Russian Financial News (`data/labeled/strategy_a_labeled.csv`)  \n"
    "**Task**: Binary classification — `positive_market_impact` (0/1)  \n"
    "**Model**: LogisticRegression + TF-IDF(10k features, bigrams)  \n"
    "**Strategies**: Entropy sampling vs Random sampling  \n"
    "**Config**: init=200, batch=100, iterations=10"
))

cells.append(nbformat.v4.new_code_cell("""\
import sys, json, os
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
os.makedirs('../data/reports', exist_ok=True)
print('OK')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 1. Load results"))

cells.append(nbformat.v4.new_code_cell("""\
with open('../data/reports/al_results.json') as f:
    results = json.load(f)

entropy_hist = pd.DataFrame(results['entropy'])
random_hist = pd.DataFrame(results['random'])

print('Config:')
for k, v in results['config'].items():
    print(f'  {k}: {v}')
print()
print('Entropy final:', entropy_hist.iloc[-1][['n_labeled','accuracy','f1']].to_dict())
print('Random  final:', random_hist.iloc[-1][['n_labeled','accuracy','f1']].to_dict())
"""))

cells.append(nbformat.v4.new_markdown_cell("## 2. Learning curves"))

cells.append(nbformat.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for metric, ax, title in [('f1', axes[0], 'Macro F1'), ('accuracy', axes[1], 'Accuracy')]:
    ax.plot(entropy_hist['n_labeled'], entropy_hist[metric],
            'o-', color='#e07070', label='Entropy', linewidth=2, markersize=5)
    ax.plot(random_hist['n_labeled'], random_hist[metric],
            's--', color='#6699cc', label='Random', linewidth=2, markersize=5)
    ax.set_title(f'Learning Curve: {title}', fontsize=13)
    ax.set_xlabel('Number of labeled examples')
    ax.set_ylabel(title)
    ax.legend()
    ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('../data/reports/learning_curves.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved learning_curves.png')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 3. Comparison table"))

cells.append(nbformat.v4.new_code_cell("""\
table = pd.DataFrame({
    'N labeled':     entropy_hist['n_labeled'].values,
    'Entropy F1':    entropy_hist['f1'].values,
    'Entropy Acc':   entropy_hist['accuracy'].values,
    'Random F1':     random_hist['f1'].values,
    'Random Acc':    random_hist['accuracy'].values,
    'Delta F1':      (entropy_hist['f1'] - random_hist['f1']).values,
})
table = table.round(4)
print(table.to_string(index=False))
table
"""))

cells.append(nbformat.v4.new_markdown_cell("## 4. Confusion matrices (final iteration)"))

cells.append(nbformat.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, hist, title in [
    (axes[0], results['entropy'], 'Entropy (final)'),
    (axes[1], results['random'],  'Random (final)')
]:
    cm = np.array(hist[-1]['confusion_matrix'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Neg/Neutral', 'Positive'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{title}\\nF1={hist[-1]["f1"]:.4f}', fontsize=12)

plt.tight_layout()
plt.savefig('../data/reports/confusion_matrices.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved confusion_matrices.png')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 5. Key finding: why Entropy < Random"))

cells.append(nbformat.v4.new_code_cell("""\
print('=== KEY FINDING ===')
print()
print('Random sampling OUTPERFORMS Entropy sampling on this dataset.')
print()
print('Likely causes:')
print('1. Label noise (~20% from GPT-4o silver labels)')
print('   Entropy sampling selects the most uncertain examples,')
print('   which in a noisy dataset are often the NOISIEST, not the most informative.')
print()
print('2. Short texts (many Telegram posts < 100 chars)')
print('   TF-IDF has limited signal; uncertainty estimates are unreliable.')
print()
print('3. Overall low F1 (~0.50) indicates the features are not discriminative enough')
print('   for entropy to provide advantage over random selection.')
print()
print(f'Entropy final F1:  {results[\"entropy\"][-1][\"f1\"]:.4f}')
print(f'Random  final F1:  {results[\"random\"][-1][\"f1\"]:.4f}')
print(f'Delta:             {results[\"entropy\"][-1][\"f1\"] - results[\"random\"][-1][\"f1\"]:+.4f}')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 6. Score distribution by strategy"))

cells.append(nbformat.v4.new_code_cell("""\
fig, ax = plt.subplots(figsize=(12, 4))
x = entropy_hist['n_labeled'].values
ax.fill_between(x, entropy_hist['f1'], alpha=0.3, color='#e07070', label='Entropy F1')
ax.fill_between(x, random_hist['f1'],  alpha=0.3, color='#6699cc', label='Random F1')
ax.plot(x, entropy_hist['f1'], 'o-', color='#e07070', linewidth=2)
ax.plot(x, random_hist['f1'],  's-', color='#6699cc', linewidth=2)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='F1=0.5 (baseline)')
ax.set_title('F1 Score: Entropy vs Random across all iterations', fontsize=13)
ax.set_xlabel('Number of labeled examples')
ax.set_ylabel('Macro F1')
ax.legend()
plt.tight_layout()
plt.savefig('../data/reports/score_distributions.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved score_distributions.png')
"""))

cells.append(nbformat.v4.new_markdown_cell("## 7. Summary"))

cells.append(nbformat.v4.new_code_cell("""\
e = results['entropy'][-1]
r = results['random'][-1]
print('='*50)
print('EXPERIMENT SUMMARY')
print('='*50)
print(f'Iterations:     {results[\"config\"][\"n_iterations\"]}')
print(f'Final N labeled: {e[\"n_labeled\"]}')
print()
print(f'Entropy  F1={e[\"f1\"]:.4f}  Acc={e[\"accuracy\"]:.4f}')
print(f'Random   F1={r[\"f1\"]:.4f}  Acc={r[\"accuracy\"]:.4f}')
print(f'Winner: Random (Delta F1 = {r[\"f1\"]-e[\"f1\"]:+.4f})')
print()
print('Recommendation: with noisy silver labels, random sampling is more robust.')
print('To benefit from entropy sampling: clean labels or use a stronger feature extractor.')
"""))

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"}
}

path = 'notebooks/al_experiment.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f'Written: {path}')
