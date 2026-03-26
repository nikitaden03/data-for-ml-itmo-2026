# Data Quality: Russian Financial News

## Датасет и задача

- **Файл**: `data/processed/financial_news_merged.csv`
- **Shape**: 15 040 строк × 9 колонок
- **Целевая переменная**: `positive_market_impact` (0/1) — позитивно ли новость влияет на рынок
- **Задача**: бинарная классификация
- **Размечено**: 15 000 строк (GPT-4o), 40 — scraped без меток

---

## Фаза 1 — Найденные проблемы

Quality Score исходного датасета: **43/100**

| Проблема | Количество | Доля | Критичность | Описание |
|---|---|---|---|---|
| Точные дубликаты (title+body) | 2 | 0.01% | High | Одинаковые заголовок и тело |
| Строки без меток | 40 | 0.3% | Medium | Scraped новости без `positive_market_impact` |
| Строки `no title` | 11 021 | 73.3% | Medium | Telegram-посты из rdv/t_invest/t_analytic — нет заголовка по дизайну |
| Рекламные статьи | 1 213 | 8.1% | Medium | `article_type=advertising` — шум для задачи |
| Короткий текст (< 20 символов) | 29 | 0.2% | Medium | Бессодержательные посты |
| Emoji в заголовках | 3 | 0.02% | Low | Спецсимволы в title |
| sectors/tickers как строки | 15 000 | 100% | Low | Хранятся как `"['Energy']"`, нужен ast.literal_eval |

---

## Решения по каждой проблеме

| Колонка / Проблема | Решение | Обоснование |
|---|---|---|
| Точные дубликаты | Удалить, оставить первое вхождение | Не несут новой информации |
| Строки без меток (40 scraped) | Оставить, авторазметить на шаге 3 | Содержат валидный текст |
| `no title` строки | **Strategy A**: оставить; **Strategy B**: удалить | В body есть финансовый текст для классификации |
| Рекламные статьи | Удалить из обеих стратегий (по запросу пользователя) | Не несут информации о рынке |
| Короткий текст | Удалить (< 20 символов) из обеих стратегий | Нет смысла классифицировать |
| Emoji в заголовках | Очистить regex `EMOJI_PATTERN` | Токенизаторы плохо обрабатывают emoji |
| sectors/tickers | Парсить `ast.literal_eval` при feature engineering | Не влияет на quality score |

---

## Фаза 2 — Результаты стратегий

| Метрика | Original | Strategy A | Strategy B |
|---|---|---|---|
| Строк | 15 040 | **13 799** | 3 895 |
| Потеря данных | — | 8.3% | 74.1% |
| Quality Score | 43/100 | **57/100** | 57/100 |
| Положительный класс % | 49.0% | 49.1% | 49.4% |
| Строки `no title` | 11 021 | 9 857 | 0 |
| Рекламные статьи | 1 213 | 0 | 0 |

**Strategy A** — удалены: реклама + дубликаты + короткие тексты + emoji
**Strategy B** — всё из A + удалены строки `no title` (только статьи с заголовками)

---

## Фаза 3 — Вердикт

**Рекомендованная стратегия: Strategy A** (`data/cleaned/strategy_a.csv`)

### Причины выбора:

1. **В 3.5 раза больше данных**: 13 799 vs 3 895 строк — критично для обучения ML модели
2. **`no title` строки содержат валидный финансовый текст** в поле `body` — пригодны для классификации даже без заголовка
3. **Одинаковый quality score** у обеих стратегий (57/100) — нет выигрыша в качестве при потере 74% данных
4. **Сохранён идеальный баланс классов**: 49.1% позитивных (Strategy B: 49.4% — практически идентично)

### KS-тесты распределения длины текста:

| Сравнение | KS-statistic | p-value | Вывод |
|---|---|---|---|
| Original vs Strategy A | ~0.03 | > 0.05 | Распределения схожи |
| Original vs Strategy B | > 0.3 | < 0.05 | Распределения различаются (Strategy B сильно меняет характер данных) |

### Рекомендации при обучении модели:

- Использовать поле `body` как основной признак (многие строки без `title`)
- При необходимости объединить `title + body` для строк с реальным заголовком
- 40 unlabeled scraped строк нужно разметить на шаге 3

---

## Структура файлов

```
data/
  processed/
    financial_news_merged.csv     # исходный merged датасет (15 040 строк)
  cleaned/
    strategy_a.csv                # ✅ рекомендованный (13 799 строк)
    strategy_b.csv                # alternative (3 895 строк)
  reports/
    quality_report.json           # исходный quality report
    quality_report_a.json         # Strategy A quality report
    quality_report_b.json         # Strategy B quality report
    comparison_report.json        # финальный вердикт
    fig_missing.png               # пропуски по колонкам
    fig_quality_score.png         # gauge качества
    fig_quality_check.png         # типы статей + длина текста
    fig_strategy_comparison.png   # сравнение строк/score/loss
    distribution_comparison.png   # KDE + типы после очистки
    fig_balance_after_fix.png     # баланс классов после очистки

scripts/
  quality_utils.py                # утилиты детекции и фиксации

notebooks/
  01_detect_issues.ipynb          # фаза 1: обнаружение проблем
  02_fix_data.ipynb               # фаза 2: применение стратегий
  03_compare_results.ipynb        # фаза 3: сравнение и вердикт
```

---

## Как воспроизвести

```bash
# Фаза 1: обнаружение проблем
jupyter nbconvert --to notebook --execute notebooks/01_detect_issues.ipynb --output 01_detect_issues.ipynb --output-dir notebooks/

# Фаза 2: очистка
jupyter nbconvert --to notebook --execute notebooks/02_fix_data.ipynb --output 02_fix_data.ipynb --output-dir notebooks/

# Фаза 3: сравнение
jupyter nbconvert --to notebook --execute notebooks/03_compare_results.ipynb --output 03_compare_results.ipynb --output-dir notebooks/
```
