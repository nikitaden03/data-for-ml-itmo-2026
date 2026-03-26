# Сбор и анализ данных: Русскоязычные финансовые новости

## Задача

Бинарная классификация финансовых новостей на русском языке:
**`positive_market_impact`** = 1, если новость может позитивно повлиять на фондовый рынок, иначе 0.

---

## Что сделали

| Источник | Метод | Файл | Строк |
|---|---|---|---|
| Kaggle `kkhubiev/russian-financial-news` | `kaggle datasets download` | `data/raw/kaggle_financial_news.csv` | 91 955 |
| GPT-4o разметка (входит в Kaggle датасет) | JSON из `news_descriptions_GPT4o.json` | — | 15 000 |
| smart-lab.ru RSS | `scripts/scrape_financial_news.py` | `data/raw/scraped_financial_news.csv` | 20 |
| rbc.ru RSS | `scripts/scrape_financial_news.py` | `data/raw/scraped_financial_news.csv` | 20 |
| **Итого (merged)** | `scripts/merge_financial_news.py` | `data/processed/financial_news_merged.csv` | **15 040** |

Скрипты: `scripts/scrape_financial_news.py`, `scripts/merge_financial_news.py`
EDA: `notebooks/eda_financial_news.ipynb`

---

## Структура данных

Файл: `data/processed/financial_news_merged.csv`

| Колонка | Тип | Описание |
|---|---|---|
| `title` | str | Заголовок новости |
| `body` | str | Текст/описание новости |
| `date` | str (YYYY-MM-DD) | Дата публикации |
| `source` | str | Источник: finam, bcs, bcs_tech, rdv, smart_lab_scraped, rbc_scraped |
| `sentiment_score` | float (-1..1) | Оценка тональности от GPT-4o (NaN для 40 scraped) |
| `article_type` | str | Тип статьи: finance, advertising, opinions, technical analysis, politics, others |
| `sectors` | list[str] | Секторы экономики (Energy, Finance, IT, …) |
| `tickers` | list[str] | Тикеры упомянутых акций |
| `positive_market_impact` | int (0/1) | **Целевая переменная**: 1 если sentiment_score > 0 |

---

## Ключевые инсайты

1. **Почти идеальный баланс классов**: 49% позитивных (7 351) vs 51% негативных/нейтральных (7 649) — не нужна балансировка при обучении.
2. **Готовая GPT-4o разметка**: 15 000 новостей уже имеют `sentiment_score`, что позволяет использовать их как silver labels без дополнительной разметки.
3. **Доминирующий тип — finance**: 77.7% статей относятся к типу `finance`, остальные — реклама, мнения, теханализ, политика.
4. **Короткие заголовки**: средняя длина заголовка — 23 символа, тела — 740 символов. Для классификации можно использовать только заголовок как baseline.
5. **Технический анализ наиболее позитивен**: статьи типа `technical analysis` имеют наибольшую долю позитивных меток — аналитики чаще пишут про потенциал роста.
6. **Период данных**: 2022-05-26 — 2024-12-15, охватывает период санкций и адаптации рынка.
7. **40 свежих новостей без меток**: собраны с smart-lab.ru и rbc.ru (март 2026), будут размечены на шаге автоматической разметки.

---

## Проблемы для препроцессинга

| Проблема | Критичность | Рекомендация |
|---|---|---|
| 40 scraped новостей без меток | Средняя | Авторазметка LLM на шаге 3 |
| Emoji и спецсимволы в заголовках (smart-lab) | Низкая | Удалить при токенизации |
| Рекламные статьи (`article_type=advertising`) | Средняя | Рассмотреть исключение из обучения |
| `sectors` и `tickers` — вложенные списки | Низкая | Парсить при feature engineering |
| Нет меток для 76 955 Kaggle строк | Высокая | Active Learning на шаге 4 покрывает это |

---

## Графики EDA

| Файл | Описание |
|---|---|
| `data/reports/fig_01_target_distribution.png` | Распределение целевой переменной и sentiment_score |
| `data/reports/fig_02_sources.png` | Количество новостей по источникам |
| `data/reports/fig_03_article_type_sentiment.png` | Доля позитивных по типу статьи |
| `data/reports/fig_04_text_lengths.png` | Длина заголовков и текстов по классам |
| `data/reports/fig_05_time_dynamics.png` | Временная динамика по месяцам |
| `data/reports/fig_06_top_sectors.png` | Топ-15 секторов экономики |
