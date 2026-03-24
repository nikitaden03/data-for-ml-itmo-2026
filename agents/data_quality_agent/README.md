# Data Quality Agent — Run Log

Пайплайн анализа и очистки данных для датасета отзывов на парфюмы.
Запускается через скилл `/data-quality-pipeline`.

---

## 1. Датасет и задача

| Параметр | Значение |
|---|---|
| Файл | `data/processed/perfumes_merged.csv` |
| Shape | 931 строк × 10 столбцов |
| Целевая переменная | sentiment (будет размечен отдельно) |
| ML-задача | Классификация сентимента по тексту отзыва + числовым признакам |

**Колонки:** `perfume_name`, `brand`, `rating`, `votes`, `top_notes`, `middle_notes`, `base_notes`, `reviewer`, `review_date`, `review_text`

Уникальных парфюмов: 8, уникальных рецензентов: 845.

---

## 2. Фаза 1 — Найденные проблемы

Quality Score исходного датасета: **30/100**

| Тип проблемы | Severity | Строк | % | Описание |
|---|---|---|---|---|
| `encoding_corruption` | 🔴 CRITICAL | 21 | 2.3% | Текст отзыва содержит символы замены Unicode `\ufffd` — нечитаем для NLP |
| `duplicate_reviewer` | 🟠 HIGH | 151 | 16.2% | 65 рецензентов с несколькими отзывами — риск data leakage при train/test split |
| `perfume_level_rating` | 🟠 HIGH | 8 парф. | 100% | `rating` и `votes` — агрегаты уровня парфюма, не оценка конкретного отзыва |
| `short_review` | 🟡 MEDIUM | 27 | 2.9% | Отзыв < 50 символов — мало информации для NLP |

Пропусков в колонках нет. Полных дубликатов строк нет.

---

## 3. Решения по каждой проблеме

| Проблема | Решение | Обоснование |
|---|---|---|
| `encoding_corruption` | Удалить строки | Текст полностью нечитаем — использование в NLP невозможно |
| `short_review` | Удалить строки | Отзывы < 50 символов не содержат достаточно информации для классификации сентимента |
| `duplicate_reviewer` | **Strategy A:** дедуплицировать (оставить самый длинный отзыв) / **Strategy B:** оставить с флагом | Стиль письма рецензента — сильный предиктор сентимента; дубли создают data leakage |
| `perfume_level_rating` | Использовать как признак, не как таргет | Нельзя предсказывать как сентимент — это голосование всех пользователей, а не оценка данного отзыва |

---

## 4. Фаза 2 — Результаты стратегий

**Strategy A — Консервативная:**
- Удалить encoding-повреждённые отзывы
- Удалить слишком короткие отзывы (< 50 символов)
- Дедуплицировать рецензентов: для каждого оставить самый длинный отзыв

**Strategy B — Мягкая:**
- Удалить encoding-повреждённые отзывы
- Удалить слишком короткие отзывы (< 50 символов)
- Дублирующихся рецензентов оставить, пометить флагом `is_duplicate_reviewer`

| Метрика | Исходный | Strategy A | Strategy B |
|---|---|---|---|
| Строк | 931 | 807 | 883 |
| Потеря данных | — | 13.3% | 5.2% |
| Quality Score | 30/100 | **90/100** | 75/100 |
| Уникальных рецензентов | 845 | 807 | 845 |

---

## 5. Фаза 3 — Вердикт

**Рекомендованная стратегия: Strategy A** (`data/cleaned/strategy_a.csv`)

**Причины выбора:**
1. Quality Score 90/100 vs 75/100 у Strategy B — принципиальная разница
2. Дедупликация рецензентов устраняет риск data leakage: стиль письма конкретного человека — сильный предиктор сентимента, и утечка стиля между train/test завышает метрики
3. Потеря 13.3% данных приемлема — остаётся 807 отзывов, покрывающих все 8 парфюмов

**KS-тесты (длины отзывов):**

| Сравнение | Statistic | p-value | Вывод |
|---|---|---|---|
| Исходный vs Strategy A | 0.0327 | 0.726 | H0 не отвергается — распределение не изменилось |
| Исходный vs Strategy B | 0.0292 | 0.819 | H0 не отвергается — распределение не изменилось |
| Strategy A vs Strategy B | 0.0129 | 1.0 | Стратегии статистически неразличимы по длинам |

**Рекомендации при обучении модели:**
- Использовать `data/cleaned/strategy_a.csv`
- Стратификацию при train/test split делать по `perfume_name` — каждый парфюм должен быть в обеих выборках
- `rating` и `votes` использовать только как числовые признаки (не как таргет)
- Таргет (сентимент) размечать отдельно — автоматически или вручную

---

## 6. Структура файлов

```
scripts/
  quality_utils.py              # функции: detect_*, compute_quality_score, apply_strategy_*

notebooks/
  01_detect_issues.ipynb        # Фаза 1: анализ качества, quality_report.json
  02_fix_data.ipynb             # Фаза 2: очистка двумя стратегиями
  03_compare_results.ipynb      # Фаза 3: KS-тесты, визуализация, вердикт

data/
  processed/
    perfumes_merged.csv         # исходный датасет (не изменялся)
  cleaned/
    strategy_a.csv              # 807 строк — рекомендованный
    strategy_b.csv              # 883 строки — альтернатива
  reports/
    quality_report.json         # отчёт Фазы 1
    fix_report.json             # отчёт Фазы 2
    comparison_report.json      # финальный отчёт с вердиктом
    fig_missing.png             # обзор проблем по типам
    fig_quality_score.png       # Quality Score исходного датасета
    fig_strategy_comparison.png # сравнение до/после очистки
    fig_compare_kde.png         # KDE, boxplot, bar по парфюмам

agents/data_quality_agent/
  README.md                     # этот файл
```

---

## 7. Как воспроизвести

```bash
# Выполнить ноутбуки последовательно
jupyter nbconvert --to notebook --execute --output-dir=notebooks notebooks/01_detect_issues.ipynb
jupyter nbconvert --to notebook --execute --output-dir=notebooks notebooks/02_fix_data.ipynb
jupyter nbconvert --to notebook --execute --output-dir=notebooks notebooks/03_compare_results.ipynb
```
