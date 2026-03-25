---
description: Оценивает текущую модель на тестовом наборе — accuracy, F1, confusion matrix
---

# Скилл: Evaluate (Оценка модели)

Оцениваешь качество текущей модели на фиксированном тестовом наборе.
Метрики сохраняются в историю для построения learning curve.

## Вход

- Обученная модель (self.pipeline_)
- `test_df` — фиксированный тестовый набор (НЕ МЕНЯЕТСЯ между итерациями)
- Целевая переменная
- Номер итерации и размер текущего labeled set

## Какие метрики считать

### Обязательные
- **Accuracy** — общая доля правильных ответов
- **F1 macro** — среднее F1 по классам (учитывает дисбаланс)
- **F1 per class** — F1 для каждого класса отдельно

### Дополнительные
- **Precision / Recall per class**
- **Confusion matrix**
- **Classification report** (sklearn)
- **Время inference** на тестовом наборе

## Что делать

### Шаг 1: Предсказание на тесте

- Примени pipeline к test_df
- Получи predictions и predict_proba

### Шаг 2: Метрики

Посчитай всё через sklearn.metrics:
- accuracy_score
- f1_score(average='macro')
- f1_score(average=None) — per class
- classification_report
- confusion_matrix

### Шаг 3: Сохрани в историю

Каждый вызов evaluate добавляет запись:
```python
{
    "iteration": int,
    "n_labeled": int,
    "strategy": str,
    "accuracy": float,
    "f1_macro": float,
    "f1_per_class": dict,
    "precision_macro": float,
    "recall_macro": float,
    "eval_time": float,
    "timestamp": str,
}
```

История хранится в self.history_ — список словарей.

## Функция в al_agent.py

`evaluate(self, test_df) -> dict`

Возвращает словарь метрик. Автоматически добавляет в self.history_.

## Вывод

```
📊 Evaluation (iteration {i}, n_labeled={n})
   Accuracy:  {acc:.4f}
   F1 macro:  {f1:.4f}
   
   Per class:
   {class_1}: F1={f1:.3f} P={p:.3f} R={r:.3f}
   {class_2}: F1={f1:.3f} P={p:.3f} R={r:.3f}
   
   Δ from previous: accuracy {+/-}{diff:.4f}, F1 {+/-}{diff:.4f}
```

## Правила

- Тестовый набор ФИКСИРОВАННЫЙ — один и тот же на всех итерациях
- Тестовый набор НЕ ПЕРЕСЕКАЕТСЯ с labeled pool и unlabeled pool
- Всегда показывай дельту с предыдущей итерацией
- Если метрики упали — предупреди (модель может деградировать
  при неудачном отборе)
- Сохраняй ВСЕ метрики в историю, даже если показываешь не все
- history_ должен быть сериализуемый в JSON