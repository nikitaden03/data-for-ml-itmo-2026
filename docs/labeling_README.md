# Auto-Labeling Pipeline: Perfume Reviews

Авто-разметка отзывов на парфюмерию для задачи бинарной классификации.

## Задача

Предсказать, понравился ли парфюм рецензенту, на основе текста отзыва и доп. фич (ноты, бренд, рейтинг).

**Финальные классы:** `liked` / `not_liked`

## Данные

| Параметр | Значение |
|----------|----------|
| Источник | Fragrantica |
| Файл | `data/cleaned/strategy_a.csv` |
| Объём | 807 отзывов |
| Язык | Русский |
| Колонки | perfume_name, brand, rating, votes, top/middle/base_notes, reviewer, review_date, review_text |

## Процесс разметки

### Фаза 1: Авто-разметка

**Модель:** [`seara/rubert-tiny2-russian-sentiment`](https://huggingface.co/seara/rubert-tiny2-russian-sentiment) (~112 MB, CPU)

Изначально использовались 3 класса: `liked` / `mixed` / `not_liked`.

| Класс | Кол-во | % |
|-------|--------|---|
| liked | 464 | 57.5% |
| mixed | 296 | 36.7% |
| not_liked | 47 | 5.8% |

Средний confidence: **0.76** | Uncertain (<0.5): **44 записи (5.5%)**

### Фаза 2: Спецификация

Документ: [`docs/annotation_spec.md`](annotation_spec.md)

- Определения классов с примерами высокого confidence
- Граничные случаи (ambivalent reviews, технические вопросы, «купил и продал»)
- Частые ошибки разметки

### Фаза 3: Проверка качества

Запущен второй аннотатор — [`blanchefort/rubert-base-cased-sentiment`](https://huggingface.co/blanchefort/rubert-base-cased-sentiment) (~700 MB).

**Проблема:** класс `mixed` давал низкое согласие между моделями (f1=0.46).

**Решение:** объединили `mixed` + `not_liked` → бинарная задача.

| Метрика | 3 класса | 2 класса (итог) |
|---------|----------|-----------------|
| Cohen's κ | 0.28 | **0.38** |
| % Agreement | 56.5% | **68.9%** |

Финальное распределение:

| Класс | Кол-во | % |
|-------|--------|---|
| liked | 464 | 57.5% |
| not_liked | 343 | 42.5% |

### Фаза 4: Экспорт

Данные готовы для загрузки в Label Studio.

## Артефакты

```
data/
  cleaned/strategy_a.csv          — исходные данные
  labeled/strategy_a_labeled.csv  — разметка (label_binary: liked/not_liked)
  export/labelstudio_tasks.json   — 807 задач для Label Studio
  export/labelstudio_config.xml   — XML интерфейс разметки
  reports/quality_metrics.json    — метрики качества
  reports/label_distribution.png  — распределение меток
  reports/quality_check.png       — confusion matrix + confidence

docs/
  annotation_spec.md              — спецификация для разметчиков
  labeling_README.md              — этот файл

notebooks/
  04_auto_label.ipynb             — авто-разметка
  05_check_quality.ipynb          — проверка качества

src/
  labeling_utils.py               — утилиты (load, label, export, metrics)
```

## Воспроизведение

```bash
pip install transformers torch scikit-learn jupyter

# Авто-разметка
jupyter nbconvert --to notebook --execute --inplace notebooks/04_auto_label.ipynb

# Проверка качества
jupyter nbconvert --to notebook --execute --inplace notebooks/05_check_quality.ipynb
```

## Следующие шаги

1. Загрузить `data/export/labelstudio_tasks.json` в Label Studio
2. Использовать `docs/annotation_spec.md` для инструктажа разметчиков
3. Отправить **44 uncertain записи** (confidence < 0.5) на ручную проверку
4. После ручной разметки пересчитать κ с ground truth

## Ключевые решения

| Решение | Причина |
|---------|---------|
| rubert-tiny2 для авто-разметки | 112 MB, CPU, хорошее качество на русском |
| Бинарная задача вместо 3 классов | `mixed` не воспроизводим между моделями (κ=0.28→0.38) |
| Keyword heuristic как baseline | Быстрая проверка без скачивания второй модели |
| ML-vs-ML проверка (700 MB модель) | Более честная оценка agreement |
