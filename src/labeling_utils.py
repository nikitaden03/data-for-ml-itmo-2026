"""
Labeling utilities for Russian Financial News.
Uses Claude API for auto-labeling.
"""
import time
import json
import anthropic


SYSTEM_PROMPT = """Ты эксперт по российскому фондовому рынку. Определи, может ли новость ПОЗИТИВНО повлиять на котировки акций (инвесторы захотят купить).

Отвечай ТОЛЬКО одной цифрой без пояснений: 1 или 0.

ПРАВИЛА:
1 (позитивная) — новость ЯВНО хорошая для акций:
  - рост прибыли, выручки, операционных показателей
  - объявление дивидендов или buyback акций
  - M&A, новые контракты, расширение бизнеса
  - снижение долга, улучшение рейтинга
  - позитивный прогноз аналитика с целевой ценой выше рынка

0 (не позитивная) — всё остальное:
  - убытки, падение прибыли, проблемы
  - падение рынка или акций ("снижаются", "распродажа", "худший день")
  - нейтральные сводки без явного позитива
  - геополитика, санкции, неопределённость
  - технический анализ без прогноза роста
  - реклама, опросы, общие обзоры

ПРИМЕРЫ:
Текст: "Газпром увеличил дивиденды до 30 руб. на акцию" → 1
Текст: "Сургут преф: возможна 30% дивдоходность в следующие 12 месяцев" → 1
Текст: "Совет директоров одобрил обратный выкуп акций" → 1
Текст: "Новатэк отчитался: производство СПГ выросло на 5%" → 1
Текст: "Худший день для банковского сектора США за 3 года" → 0
Текст: "Американские акции снижаются второй день подряд" → 0
Текст: "Avito оказалось недоступно в AppStore" → 0
Текст: "Главное к открытию: индекс ЖиС 43, спокойствие" → 0
Текст: "Делистинг Polymetal с Мосбиржи" → 0
"""


def label_single(client: anthropic.Anthropic, title: str, body: str, model: str = "claude-haiku-4-5-20251001") -> dict:
    """Label a single news item. Returns dict with label and confidence."""
    text = f"Заголовок: {title}\n\nТекст: {body[:600]}"

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=10,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}]
        )
        raw = msg.content[0].text.strip()
        label = 1 if "1" in raw else 0
        return {"label": label, "raw": raw, "error": None}
    except Exception as e:
        return {"label": None, "raw": None, "error": str(e)}


def label_batch(client: anthropic.Anthropic, rows: list[dict], model: str = "claude-haiku-4-5-20251001", delay: float = 0.3) -> list[dict]:
    """Label a batch of rows. Each row must have 'title' and 'body'."""
    results = []
    for i, row in enumerate(rows):
        result = label_single(client, row.get("title", ""), row.get("body", ""), model)
        result["idx"] = row.get("idx", i)
        results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  Labeled {i+1}/{len(rows)}...")
        time.sleep(delay)
    return results


def compute_cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Compute Cohen's kappa between two label lists."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 0.0

    # Observed agreement
    agree = sum(a == b for a, b in zip(labels_a, labels_b))
    p_o = agree / n

    # Expected agreement
    classes = sorted(set(labels_a) | set(labels_b))
    p_e = 0.0
    for c in classes:
        p_a = labels_a.count(c) / n
        p_b = labels_b.count(c) / n
        p_e += p_a * p_b

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def interpret_kappa(kappa: float) -> str:
    if kappa < 0:
        return "Poor (worse than chance)"
    elif kappa < 0.2:
        return "Slight"
    elif kappa < 0.4:
        return "Fair"
    elif kappa < 0.6:
        return "Moderate"
    elif kappa < 0.8:
        return "Substantial"
    else:
        return "Almost perfect"


def save_quality_metrics(metrics: dict, path: str = "data/reports/quality_metrics.json"):
    """Save quality metrics to JSON."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def to_labelstudio_tasks(df, text_col: str = "body", label_col: str = "positive_market_impact") -> list[dict]:
    """Convert dataframe to Label Studio task format."""
    tasks = []
    for i, row in df.iterrows():
        task = {
            "id": i,
            "data": {
                "text": str(row.get(text_col, "")),
                "title": str(row.get("title", "")),
                "source": str(row.get("source", "")),
                "date": str(row.get("date", "")),
            }
        }
        label = row.get(label_col)
        if label is not None and str(label) not in ("nan", "None", ""):
            task["annotations"] = [{
                "result": [{
                    "from_name": "label",
                    "to_name": "text",
                    "type": "choices",
                    "value": {"choices": ["Positive" if int(float(label)) == 1 else "Negative"]}
                }]
            }]
        tasks.append(task)
    return tasks


LABELSTUDIO_CONFIG = """<View>
  <Header value="Financial News: Market Impact Classification"/>
  <Text name="text" value="$text"/>
  <Header value="Title: $title | Source: $source | Date: $date"/>
  <Choices name="label" toName="text" choice="single">
    <Choice value="Positive" hint="News may POSITIVELY affect stock market"/>
    <Choice value="Negative" hint="Neutral, negative, or irrelevant news"/>
  </Choices>
</View>"""
