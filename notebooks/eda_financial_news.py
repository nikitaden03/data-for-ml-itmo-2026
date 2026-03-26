"""
EDA: Russian Financial News for stock market sentiment classification.
"""
import sys
import json
import warnings
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted')
FIGDIR = 'data/reports'

import os
os.makedirs(FIGDIR, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
df_news = pd.read_parquet('data/raw/RussianFinancialNews/news_collection.parquet')
df_news = df_news.reset_index(drop=True)

with open('data/raw/RussianFinancialNews/news_descriptions/news_descriptions_GPT4o.json', encoding='utf-8') as f:
    raw = json.load(f)
labels = pd.DataFrame([{'idx': int(k), **v} for k, v in raw.items()])

# Merge labels into news (only labeled rows)
df = df_news.iloc[labels['idx']].copy().reset_index(drop=True)
df = pd.concat([df.reset_index(drop=True), labels.drop(columns='idx').reset_index(drop=True)], axis=1)

# Create binary target
df['positive_market_impact'] = (df['sentiment_score'] > 0).astype(int)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['title_len'] = df['title'].str.len()
df['body_len'] = df['body'].str.len()

print(f"Labeled dataset shape: {df.shape}")
print(f"Target distribution:\n{df['positive_market_impact'].value_counts()}")

# ─── Fig 1: Target distribution ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts = df['positive_market_impact'].value_counts().sort_index()
axes[0].bar(['Негативная/Нейтральная (0)', 'Позитивная (1)'],
            counts.values, color=['#e07070', '#70b070'], edgecolor='white', linewidth=0.8)
axes[0].set_title('Распределение целевой переменной', fontsize=13)
axes[0].set_ylabel('Количество новостей')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 50, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=11)

# Sentiment score distribution
axes[1].hist(df['sentiment_score'], bins=30, color='#6699cc', edgecolor='white')
axes[1].set_title('Распределение sentiment_score (GPT-4o)', fontsize=13)
axes[1].set_xlabel('Sentiment Score')
axes[1].set_ylabel('Количество новостей')
axes[1].axvline(0, color='red', linestyle='--', alpha=0.7, label='порог = 0')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig_01_target_distribution.png', dpi=120)
plt.close()
print("Saved fig_01_target_distribution.png")

# ─── Fig 2: Source distribution ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
src_counts = df['source'].value_counts()
src_counts.plot(kind='bar', ax=ax, color='#6699cc', edgecolor='white')
ax.set_title('Распределение по источникам', fontsize=13)
ax.set_ylabel('Количество новостей')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=30)
for i, v in enumerate(src_counts.values):
    ax.text(i, v + 30, str(v), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig_02_sources.png', dpi=120)
plt.close()
print("Saved fig_02_sources.png")

# ─── Fig 3: Article type vs sentiment ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
type_sent = df.groupby('article_type')['positive_market_impact'].mean().sort_values(ascending=False)
type_sent.plot(kind='bar', ax=ax, color='#70b070', edgecolor='white')
ax.set_title('Доля позитивных новостей по типу статьи', fontsize=13)
ax.set_ylabel('Доля позитивных (positive_market_impact=1)')
ax.set_xlabel('')
ax.set_ylim(0, 1)
ax.tick_params(axis='x', rotation=30)
ax.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='50%')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig_03_article_type_sentiment.png', dpi=120)
plt.close()
print("Saved fig_03_article_type_sentiment.png")

# ─── Fig 4: Text length distribution ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for col, ax, label in [('title_len', axes[0], 'Длина заголовка (символов)'),
                        ('body_len', axes[1], 'Длина текста (символов)')]:
    for val, color, lbl in [(1, '#70b070', 'Позитивная'), (0, '#e07070', 'Негативная/Нейтральная')]:
        subset = df[df['positive_market_impact'] == val][col].clip(upper=df[col].quantile(0.99))
        ax.hist(subset, bins=40, alpha=0.6, color=color, label=lbl, density=True)
    ax.set_title(label, fontsize=12)
    ax.set_xlabel('Символов')
    ax.set_ylabel('Плотность')
    ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig_04_text_lengths.png', dpi=120)
plt.close()
print("Saved fig_04_text_lengths.png")

# ─── Fig 5: Time dynamics ─────────────────────────────────────────────────────
df_dated = df.dropna(subset=['date']).copy()
df_dated['month'] = df_dated['date'].dt.to_period('M')
monthly = df_dated.groupby(['month', 'positive_market_impact']).size().unstack(fill_value=0)
monthly.index = monthly.index.to_timestamp()

fig, ax = plt.subplots(figsize=(14, 4))
monthly.plot(ax=ax, color=['#e07070', '#70b070'])
ax.set_title('Динамика новостей по месяцам', fontsize=13)
ax.set_ylabel('Количество новостей')
ax.set_xlabel('')
ax.legend(['Негативная/Нейтральная (0)', 'Позитивная (1)'])
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig_05_time_dynamics.png', dpi=120)
plt.close()
print("Saved fig_05_time_dynamics.png")

# ─── Fig 6: Top sectors ───────────────────────────────────────────────────────
all_sectors = []
for s in df['sectors']:
    if isinstance(s, list):
        all_sectors.extend(s)
sector_counts = Counter(all_sectors).most_common(15)
sectors, counts = zip(*sector_counts) if sector_counts else ([], [])

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(list(sectors)[::-1], list(counts)[::-1], color='#6699cc', edgecolor='white')
ax.set_title('Топ-15 секторов в новостях', fontsize=13)
ax.set_xlabel('Количество упоминаний')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig_06_top_sectors.png', dpi=120)
plt.close()
print("Saved fig_06_top_sectors.png")

# ─── Summary stats ────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("EDA SUMMARY")
print("="*50)
print(f"Total labeled rows:      {len(df):,}")
print(f"Positive (impact=1):     {df['positive_market_impact'].sum():,} ({df['positive_market_impact'].mean()*100:.1f}%)")
print(f"Negative/neutral (0):    {(df['positive_market_impact']==0).sum():,} ({(df['positive_market_impact']==0).mean()*100:.1f}%)")
print(f"Mean title length:       {df['title_len'].mean():.0f} chars")
print(f"Mean body length:        {df['body_len'].mean():.0f} chars")
print(f"Date range:              {df_dated['date'].min().date()} — {df_dated['date'].max().date()}")
print(f"Sources:                 {df['source'].nunique()} unique")
print(f"Article types:           {df['article_type'].value_counts().to_dict()}")
print(f"Null body:               {(df['body']=='').sum()}")
