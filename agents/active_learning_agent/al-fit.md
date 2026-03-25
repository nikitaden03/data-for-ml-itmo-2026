---
description: Обучает базовую модель на текущем размеченном подмножестве данных
---

# Скилл: Fit (Обучение модели)

Ты обучаешь модель на текущем labeled pool в рамках Active Learning цикла.
Модель должна быть лёгкой, быстрой и поддерживать predict_proba
(нужно для стратегий отбора).

## Вход

- `labeled_df` — текущий размеченный датасет (может быть маленьким, от 50 записей)
- Целевая переменная (target column)
- (опционально) Тип модели — по умолчанию выбери сам
- (опционально) Гиперпараметры

## Выбор модели

Модель ОБЯЗАТЕЛЬНО должна поддерживать `predict_proba()` —
без этого entropy и margin стратегии не работают.

Приоритет выбора:
1. **LogisticRegression** — дефолт, быстрая, стабильная на малых данных,
   есть predict_proba из коробки
2. **RandomForestClassifier** — если данные нелинейные,
   predict_proba через усреднение деревьев
3. **SVM с CalibratedClassifierCV** — если хочется SVM,
   но нужна калибровка для predict_proba
4. **GradientBoosting** — если данных побольше (>500)

Для Active Learning важно: модель должна **переобучаться быстро**
(секунды, не минуты), потому что fit вызывается на каждой итерации цикла.

**НЕ используй** нейросети/трансформеры — слишком долго для AL-цикла.

## Что делать

### Шаг 1: Подготовка данных

- Раздели features и target
- Если есть категориальные признаки — закодируй
  (LabelEncoder / OneHotEncoder)
- Если есть текст — TF-IDF векторизация
- Стандартизация числовых признаков (StandardScaler)
- Собери всё в sklearn Pipeline чтобы переиспользовать

Важно: пайплайн предобработки ДОЛЖЕН быть одинаковым
на всех итерациях AL. Создай его один раз и переиспользуй.

### Шаг 2: Обучение

- Обучи на labeled_df
- Замерь время обучения
- Логируй: размер обучающей выборки, время, модель

### Шаг 3: Базовая проверка

- predict_proba на labeled_df — убедись что работает
- Распределение предсказанных вероятностей — не вырожденное ли
  (всё 0.5 = модель не учится)

## Что создать

### Файл: `agents/al_agent.py`

Создай (или дополни) основной файл агента. На этом шаге — класс модели:

Структура `al_agent.py`:

```
class ActiveLearningAgent:
    - __init__(self, task, target_col, model_type, random_state)
    - fit(self, labeled_df) → fitted model
    - query(self, pool_df, strategy, n_samples) → indices
    - evaluate(self, test_df) → Metrics
    - report(self, history) → LearningCurve
    - run_cycle(self, initial_labeled, pool, test, ...) → full results
```

Функция `fit` должна:
- Принимать DataFrame
- Создавать/обновлять sklearn Pipeline (vectorizer + scaler + model)
- Возвращать self (для chaining)
- Сохранять модель в self.model_ и self.pipeline_

## Вывод

После обучения покажи:
```
🎯 Model fitted
   Model: {type}
   Train size: {n} samples
   Classes: {list}
   Train time: {seconds}s
   predict_proba check: ✅
```

## Правила

- Pipeline создаётся ОДИН РАЗ, модель внутри переобучается
- Всегда `random_state` для воспроизводимости
- Всегда проверяй что predict_proba не вырожденное
- Если labeled_df < 10 записей на класс — предупреди
  что модель будет нестабильной
- Логируй каждый fit: iteration, n_samples, train_time
- Не делай train/val split внутри fit — это задача evaluate