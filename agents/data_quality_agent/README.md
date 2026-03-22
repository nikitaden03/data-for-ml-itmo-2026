# Data Quality Agent

Пайплайн анализа и очистки данных для датасета духов Fragrantica.
Запускается через скилл `/data-quality-pipeline`.

## Что делали

**Датасет:** `data/processed/perfumes_merged.csv` — 23 800 строк, 18 столбцов
**Задача:** регрессия, целевая переменная `rating_value`

---

### Фаза 1 — Detect Issues (`notebooks/01_detect_issues.ipynb`)

Анализировали качество исходного датасета:

| Проблема | Severity | Детали |
|---|---|---|
| `perfumer2` — 94.5% missing | CRITICAL | 22 483 пропуска из 23 800 |
| `rating_count` — выбросы IQR | HIGH | 3 035 выбросов (12.8%), макс = 31 179 |
| `year` — выбросы | MEDIUM | 1 669 записей до 1996, 5 записей после 2024 |
| `year` — пропуски | LOW | 2 009 строк (8.4%) |
| `mainaccord1-5` — пропуски | LOW | от 0.2% до 4.2% |
| `gender`, `country`, `top/middle/base` | LOW | < 100 строк каждый |

**Quality Score исходного датасета: 52.5/100**

Артефакты: `data/reports/quality_report.json`, `data/reports/fig_*.png`

---

### Решения по каждой проблеме (обсуждались с пользователем итеративно)

| Столбец | Решение | Обоснование |
|---|---|---|
| `perfumer2` | drop_col | 94.5% пустоты — не признак, а шум |
| `rating_count` | log_transform → `log_rating_count` | Реальные популярные духи, не ошибки; лог убирает скошенность |
| `year` выбросы | clip [1920, 2024] | Старые духи реальны, 5 записей с 2025 — ошибки |
| `year` пропуски | оставить NaN | Tree-based модели обрабатывают нативно |
| `mainaccord1-3` | drop_rows (150 строк) | Случайные пропуски у обязательных полей |
| `mainaccord4-5` | оставить NaN | Структурный пропуск: у духа просто нет 4-5 аккорда |
| `gender`, `country`, `top`, `middle`, `base` | fill `'unknown'` | Незначительные пропуски |

---

### Фаза 2 — Fix Data (`notebooks/02_fix_data.ipynb`)

Применялись две стратегии:

**Strategy A — «Консервативная»**
- Все решения выше применяются как есть
- `year` NaN — оставить
- `log_rating_count` — без дополнительного clip
- Потеря: **0.63%** (150 строк)
- Quality Score: **85.5/100**

**Strategy B — «Агрессивная»**
- То же, плюс:
- `year` NaN — drop_rows (−2 009 строк)
- `log_rating_count` — clip_iqr после трансформации
- Потеря: **8.99%** (2 140 строк)
- Quality Score: **88.5/100**

Артефакты: `data/cleaned/strategy_a.csv`, `data/cleaned/strategy_b.csv`, `data/reports/fix_report.json`

---

### Фаза 3 — Compare Results (`notebooks/03_compare_results.ipynb`)

Количественное сравнение, KDE/boxplot/heatmap до-после, KS-тесты.

| Метрика | Original | Strategy A | Strategy B |
|---|---|---|---|
| Строк | 23 800 | 23 650 | 21 660 |
| Пропуски (%) | ~27% | ~3% | ~1% |
| Выбросы IQR | высокие | снижены | снижены |
| Quality Score | 52.5 | **85.5** | 88.5 |
| Потеря данных | — | **0.63%** | 8.99% |

**KS-тесты:**
- Strategy A: p=1.0 по всем числовым столбцам — распределения не изменились
- Strategy B: `rating_count` p=0.0001 — clip после log сдвинул распределение

**Рекомендованная стратегия: Strategy A**

Причины:
1. Потеря 0.63% vs 8.99% при разнице в Quality Score всего 3 балла
2. KS-тесты подтвердили: A не исказила ни одно распределение, B изменила `rating_count`
3. NaN в `year`/`mainaccord4-5` несут информацию и не требуют обработки для LightGBM/XGBoost

Рекомендации при обучении модели:
- Использовать `log_rating_count` вместо `rating_count`
- Для tree-based моделей (LightGBM, XGBoost) — NaN обрабатываются нативно
- Для линейных моделей — добавить `SimpleImputer(strategy='median')` для `year`
- `mainaccord4-5` кодировать с NaN как отдельной категорией

---

## Структура файлов

```
scripts/
  quality_utils.py          # dataclasses и функции: detect_all, apply_strategy

notebooks/
  01_detect_issues.ipynb    # Фаза 1: анализ качества
  02_fix_data.ipynb         # Фаза 2: очистка двумя стратегиями
  03_compare_results.ipynb  # Фаза 3: сравнение и вердикт

data/
  processed/
    perfumes_merged.csv     # исходный датасет (не изменялся)
  cleaned/
    strategy_a.csv          # 23 650 строк — рекомендованный
    strategy_b.csv          # 21 660 строк — альтернатива
  reports/
    quality_report.json     # отчёт Фазы 1
    fix_report.json         # отчёт Фазы 2
    comparison_report.json  # финальный отчёт с вердиктом
    fig_missing.png
    fig_boxplots.png
    fig_target_dist.png
    fig_quality_score.png
    fig_strategy_comparison.png
    fig_compare_missing.png
    fig_compare_kde.png
    fig_compare_boxplot.png
    fig_compare_corr.png

agents/data_quality_agent/
  detect-issues.md          # скилл: детектив
  fix-data.md               # скилл: хирург
  compare-results.md        # скилл: аргумент
  data-quality-pipeline.md  # оркестрирующий скилл
  README.md                 # этот файл
```

## Как воспроизвести

```bash
# Установить зависимости
pip install pandas numpy scipy matplotlib seaborn nbformat ipykernel

# Зарегистрировать venv как Jupyter-ядро
python -m ipykernel install --user --name=data-ml-venv

# Выполнить ноутбуки последовательно
jupyter nbconvert --execute --inplace notebooks/01_detect_issues.ipynb --ExecutePreprocessor.kernel_name=data-ml-venv
jupyter nbconvert --execute --inplace notebooks/02_fix_data.ipynb     --ExecutePreprocessor.kernel_name=data-ml-venv
jupyter nbconvert --execute --inplace notebooks/03_compare_results.ipynb --ExecutePreprocessor.kernel_name=data-ml-venv
```
