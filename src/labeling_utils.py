"""
Labeling utilities for perfume review classification.
Task: predict whether a reviewer liked a perfume.
Label mapping: positive→liked, neutral→mixed, negative→not_liked
"""

import json
import csv
from pathlib import Path
from typing import Optional


LABEL_MAP = {
    "positive": "liked",
    "neutral": "mixed",
    "negative": "not_liked",
}

LABEL_MAP_REVERSE = {v: k for k, v in LABEL_MAP.items()}


def load_data(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def map_sentiment_to_label(sentiment: str) -> str:
    return LABEL_MAP.get(sentiment.lower(), "mixed")


def run_auto_label(rows: list[dict], classifier, text_col: str = "review_text", batch_size: int = 32) -> list[dict]:
    """Run sentiment classifier on all rows, add label + confidence."""
    texts = [r[text_col] for r in rows]
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Truncate to 512 tokens max (model limit)
        preds = classifier(batch, truncation=True, max_length=512)
        results.extend(preds)

    labeled = []
    for row, pred in zip(rows, results):
        row = dict(row)
        sentiment = pred["label"].lower()
        row["sentiment_raw"] = sentiment
        row["label"] = map_sentiment_to_label(sentiment)
        row["confidence"] = round(pred["score"], 4)
        labeled.append(row)

    return labeled


def save_labeled(rows: list[dict], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_distribution(rows: list[dict]) -> dict:
    from collections import Counter
    counts = Counter(r["label"] for r in rows)
    total = len(rows)
    return {label: {"count": c, "pct": round(c / total * 100, 1)} for label, c in counts.items()}


def uncertain_records(rows: list[dict], threshold: float = 0.5) -> list[dict]:
    return [r for r in rows if float(r["confidence"]) < threshold]


def quality_summary(rows: list[dict], threshold: float = 0.5) -> dict:
    uncertain = uncertain_records(rows, threshold)
    confidences = [float(r["confidence"]) for r in rows]
    return {
        "total": len(rows),
        "labeled": len(rows),
        "mean_confidence": round(sum(confidences) / len(confidences), 4),
        "uncertain_count": len(uncertain),
        "uncertain_pct": round(len(uncertain) / len(rows) * 100, 1),
        "distribution": label_distribution(rows),
    }


def save_quality_report(summary: dict, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def export_labelstudio(rows: list[dict], path: str, text_col: str = "review_text") -> None:
    """Export to Label Studio JSON format."""
    tasks = []
    for i, row in enumerate(rows):
        task = {
            "id": i + 1,
            "data": {
                "text": row[text_col],
                "perfume_name": row.get("perfume_name", ""),
                "brand": row.get("brand", ""),
                "rating": row.get("rating", ""),
                "top_notes": row.get("top_notes", ""),
                "middle_notes": row.get("middle_notes", ""),
                "base_notes": row.get("base_notes", ""),
            },
            "predictions": [{
                "result": [{
                    "from_name": "label",
                    "to_name": "text",
                    "type": "choices",
                    "value": {"choices": [row["label"]]},
                }],
                "score": float(row["confidence"]),
                "model_version": "rubert-tiny2-sentiment-v1",
            }],
        }
        tasks.append(task)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def export_labelstudio_config(labels: list[str], path: str) -> None:
    """Generate Label Studio XML config."""
    choices = "\n    ".join(f'<Choice value="{l}"/>' for l in labels)
    xml = f"""<View>
  <Text name="text" value="$text"/>
  <Header value="Perfume: $perfume_name | Brand: $brand | Rating: $rating"/>
  <Header value="Notes: $top_notes / $middle_notes / $base_notes"/>
  <Choices name="label" toName="text" choice="single" showInline="true">
    {choices}
  </Choices>
</View>"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
