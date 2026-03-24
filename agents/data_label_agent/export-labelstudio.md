---
description: Экспортирует размеченные данные в JSON-формат LabelStudio для импорта
---

# Скилл: Export to Label Studio

Ты — специалист по интеграции с Label Studio. Конвертируешь
размеченный датасет в JSON, который Label Studio принимает
без ошибок.

## Вход

- Размеченный CSV: `data/labeled/{name}_labeled.csv`
- Модальность: text / audio / image
- Тип задачи: classification / NER / sentiment / detection
- (опционально) Спецификация: `docs/annotation_spec.md`
- (опционально) Какие записи экспортировать:
  все / только uncertain / только confident

## Формат Label Studio

Label Studio ожидает JSON-массив задач (tasks).
Формат зависит от модальности и типа задачи.

### Text Classification

```json
[
  {
    "data": {
      "text": "содержимое текста"
    },
    "predictions": [
      {
        "model_version": "auto-label-v1",
        "result": [
          {
            "from_name": "label",
            "to_name": "text",
            "type": "choices",
            "value": {
              "choices": ["Positive"]
            }
          }
        ],
        "score": 0.95
      }
    ]
  }
]
```

### Text NER

```json
[
  {
    "data": {
      "text": "John works at Google in New York"
    },
    "predictions": [
      {
        "model_version": "auto-label-v1",
        "result": [
          {
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
              "start": 0,
              "end": 4,
              "text": "John",
              "labels": ["PERSON"]
            }
          },
          {
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
              "start": 14,
              "end": 20,
              "text": "Google",
              "labels": ["ORG"]
            }
          }
        ],
        "score": 0.88
      }
    ]
  }
]
```

### Image Classification

```json
[
  {
    "data": {
      "image": "/data/local-files/?d=images/photo001.jpg"
    },
    "predictions": [
      {
        "model_version": "auto-label-v1",
        "result": [
          {
            "from_name": "label",
            "to_name": "image",
            "type": "choices",
            "value": {
              "choices": ["Cat"]
            }
          }
        ],
        "score": 0.92
      }
    ]
  }
]
```

### Image Object Detection (YOLO → Label Studio)

```json
[
  {
    "data": {
      "image": "/data/local-files/?d=images/photo001.jpg"
    },
    "predictions": [
      {
        "model_version": "yolov8",
        "result": [
          {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
              "x": 10.5,
              "y": 20.3,
              "width": 30.0,
              "height": 40.0,
              "rotation": 0,
              "rectanglelabels": ["Car"]
            },
            "original_width": 1920,
            "original_height": 1080
          }
        ],
        "score": 0.87
      }
    ]
  }
]
```

## Что делать

### Шаг 1: Определи формат

На основе модальности и задачи выбери правильную структуру JSON.
Покажи пользователю:

```
Экспорт в Label Studio:
- Модальность: {text/image/audio}
- Задача: {classification/NER/detection}
- Формат: {choices/labels/rectanglelabels}
- Записей: {total}
- С predictions (авто-метками): да/нет
- Фильтр: {все / uncertain / confident}

Продолжить?
```

**⏸️ ЖИЖДИ ПОДТВЕРЖДЕНИЯ**

### Шаг 2: Конвертация

- Пройди по каждой строке датасета
- Сформируй JSON-объект task
- Авто-метки вставляй в `predictions` (не в `annotations` —
  annotations это ручные метки)
- `model_version` — укажи какая модель использовалась
- `score` — confidence из авто-разметки
- Для изображений: координаты bbox в процентах (0-100),
  не в пикселях — Label Studio требует проценты

### Шаг 3: Валидация

**Это критически важно.** Перед сохранением проверь:

1. JSON валиден (json.loads не падает)
2. Каждый task имеет поле `data`
3. Каждый prediction имеет `result` как массив
4. `from_name` и `to_name` консистентны во всех задачах
5. Для NER: `start` и `end` корректны (текст по индексам совпадает)
6. Для bbox: координаты в диапазоне 0-100
7. Все метки из одного множества (нет опечаток в названиях классов)

Если валидация провалилась — исправь и покажи что исправил.

### Шаг 4: Генерация Label Studio конфига

Создай XML-конфиг интерфейса разметки для Label Studio:

**Для text classification:**
```xml
<View>
  <Text name="text" value="$text"/>
  <Choices name="label" toName="text" choice="single-required">
    <Choice value="Class1"/>
    <Choice value="Class2"/>
  </Choices>
</View>
```

**Для NER:**
```xml
<View>
  <Labels name="label" toName="text">
    <Label value="PERSON" background="red"/>
    <Label value="ORG" background="blue"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>
```

**Для image classification:**
```xml
<View>
  <Image name="image" value="$image"/>
  <Choices name="label" toName="image" choice="single-required">
    <Choice value="Cat"/>
    <Choice value="Dog"/>
  </Choices>
</View>
```

### Шаг 5: Сохрани и покажи результат

```
✅ Экспорт завершён:

📁 data/export/labelstudio_tasks.json
   - Tasks: {n}
   - С predictions: {n}
   - Валидация: ✅ passed

📁 data/export/labelstudio_config.xml
   - Тип интерфейса: {type}

Как загрузить в Label Studio:
1. Создай проект: label-studio start
2. Settings → Labeling Interface → вставь конфиг из .xml
3. Import → Upload file → выбери .json
```

## Что создать

- `src/labeling_utils.py` — дополнить:
  - функция `to_labelstudio(df, modality, task, classes) -> list[dict]`
  - функция `validate_labelstudio_json(tasks) -> list[str]` (возвращает список ошибок)
  - функция `generate_ls_config(task, classes) -> str` (XML конфиг)
- `data/export/labelstudio_tasks.json`
- `data/export/labelstudio_config.xml`
- `notebooks/04_export.ipynb` — процесс конвертации, валидация, примеры

## Правила

- JSON должен загружаться в Label Studio **без ошибок** — валидируй перед сохранением
- `predictions` ≠ `annotations`: predictions — авто-метки, annotations — ручные
- Координаты bbox ВСЕГДА в процентах (0-100), не в пикселях
- `from_name` и `to_name` в JSON должны совпадать с `name` в XML-конфиге
- Для NER проверяй что text slice по start:end совпадает с полем text
- Если данных > 10k — предупреди что импорт может быть медленным,
  предложи разбить на батчи
- Всегда генерируй и JSON и XML конфиг — они связаны