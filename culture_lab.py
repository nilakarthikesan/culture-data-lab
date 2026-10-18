"""Synthetic annotation disagreement and time-respecting forecast baseline lab.

Python 3.10+, standard library only. No accounts, network, model APIs, or scraping.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

LABELS = ("relevant", "not_relevant", "unsure")
FIELDS = ("item_id", "period", "available_at", "panel", "topic", "annotator_id", "label", "rights_basis")
RIGHTS = {"synthetic", "consented", "public_licensed"}


def validate(rows: list[dict]) -> list[dict]:
    seen, item_meta = set(), {}
    for index, row in enumerate(rows, 2):
        if any(not str(row.get(key, "")).strip() for key in FIELDS):
            raise ValueError(f"row {index}: missing required field")
        when, available = date.fromisoformat(row["period"]), date.fromisoformat(row["available_at"])
        # Later cutoffs and sorting use strings. Require one canonical date form,
        # rather than accepting ISO week/basic forms on newer Python versions.
        if when.isoformat() != row["period"] or available.isoformat() != row["available_at"]:
            raise ValueError(f"row {index}: dates must use canonical YYYY-MM-DD format")
        if available < when:
            raise ValueError(f"row {index}: annotation available before observation period")
        if row["label"] not in LABELS or row["rights_basis"] not in RIGHTS:
            raise ValueError(f"row {index}: unsupported label or rights basis")
        key = (row["item_id"], row["annotator_id"])
        if key in seen:
            raise ValueError(f"row {index}: duplicate item/annotator pair")
        seen.add(key)
        meta = tuple(row[k] for k in ("period", "panel", "topic"))
        if row["item_id"] in item_meta and item_meta[row["item_id"]] != meta:
            raise ValueError(f"row {index}: inconsistent item metadata")
        item_meta[row["item_id"]] = meta
    return rows


def distribution(labels: list[str]) -> dict[str, float]:
    if not labels:
        raise ValueError("distribution requires at least one annotation")
    if any(label not in LABELS for label in labels):
        raise ValueError("unsupported annotation label")
    counts = Counter(labels)
    return {label: counts[label] / len(labels) for label in LABELS}


def item_summaries(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["item_id"]].append(row)
    result = []
    for item_id, ratings in sorted(grouped.items()):
        labels = [r["label"] for r in ratings]
        counts, n = Counter(labels), len(labels)
        probs = distribution(labels)
        # Ordered distinct-rater pairs; undefined for singleton items.
        agreement = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1)) if n > 1 else None
        result.append({
            "item_id": item_id, **{k: ratings[0][k] for k in ("period", "panel", "topic")},
            "n_annotations": n, "label_distribution": probs,
            "entropy_bits": -sum(p * math.log2(p) for p in probs.values() if p),
            "pairwise_agreement": agreement,
            "plurality_labels": [l for l in LABELS if counts[l] == max(counts.values())],
        })
    return result


def average_distribution(items: list[dict]) -> dict[str, float]:
    # Every creative item has equal weight regardless of its annotation count.
    return {label: sum(i["label_distribution"][label] for i in items) / len(items) for label in LABELS}


def period_summaries(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for item in item_summaries(rows):
        groups[(item["panel"], item["topic"], item["period"])].append(item)
    result, previous = [], {}
    for (panel, topic, period), items in sorted(groups.items()):
        probs = average_distribution(items)
        old = previous.get((panel, topic))
        tv = sum(abs(probs[l] - old[l]) for l in LABELS) / 2 if old else None
        result.append({"period": period, "panel": panel, "topic": topic,
                       "n_items": len(items), "distribution": probs,
                       "total_variation_from_previous_observed_period": tv})
        previous[(panel, topic)] = probs
    return result


def forecast(history: list[dict], method: str, decay: float = 0.5) -> dict[str, float]:
    if method not in {"uniform", "expanding", "hard_vote", "persistence", "ewma"}:
        raise ValueError("unknown forecast method")
    if not 0 < decay <= 1:
        raise ValueError("decay must be in (0, 1]")
    if not history or method == "uniform":
        return {label: 1 / len(LABELS) for label in LABELS}
    if method == "hard_vote":
        flattened = []
        for item in history:
            probabilities = item["label_distribution"]
            winners = [label for label in LABELS if probabilities[label] == max(probabilities.values())]
            flattened.append({"label_distribution": {label: (1 / len(winners) if label in winners else 0) for label in LABELS}})
        return average_distribution(flattened)
    by_period = defaultdict(list)
    for item in history:
        by_period[item["period"]].append(item)
    points = [average_distribution(by_period[p]) for p in sorted(by_period)]
    if method == "expanding":
        return average_distribution(history)
    if method == "persistence":
        return points[-1]
    estimate = points[0]
    for point in points[1:]:
        estimate = {l: decay * point[l] + (1 - decay) * estimate[l] for l in LABELS}
    return estimate


def soft_brier(prediction: dict, target: dict) -> float:
    """Squared distance to the empirical label distribution (range 0..2).

    This is NOT expected categorical Brier against a sampled individual label;
    the latter adds intrinsic annotator variance, common across models per item.
    """
    return sum((prediction[l] - target[l]) ** 2 for l in LABELS)


def backtest(rows: list[dict], minimum_history_periods: int = 2) -> dict:
    # Keep direct library calls under the same chronology/data contract as CLI.
    validate(rows)
    if minimum_history_periods < 1:
        raise ValueError("minimum history must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["panel"], row["topic"])].append(row)
    records = []
    for (panel, topic), group in sorted(groups.items()):
        for cutoff in sorted({r["period"] for r in group}):
            # Event and availability timestamps must BOTH precede prediction.
            past = [r for r in group if r["period"] < cutoff and r["available_at"] < cutoff]
            history = item_summaries(past)
            if len({i["period"] for i in history}) < minimum_history_periods:
                continue
            targets = item_summaries([r for r in group if r["period"] == cutoff])
            for method in ("uniform", "expanding", "hard_vote", "persistence", "ewma"):
                pred = forecast(history, method)
                records.append({"period": cutoff, "panel": panel, "topic": topic,
                                "method": method, "prediction": pred,
                                "n_history_items": len(history), "n_target_items": len(targets),
                                "soft_brier": sum(soft_brier(pred, i["label_distribution"]) for i in targets) / len(targets)})
    scores = defaultdict(list)
    for record in records:
        scores[record["method"]].append(record["soft_brier"])
    return {"metric": "mean item-level soft-label Brier, then equal-weight panel/topic/period folds",
            "scores": {m: sum(v) / len(v) for m, v in scores.items()}, "folds": records}


def demo_rows(seed: int = 7, weeks: int = 12) -> list[dict]:
    rng, rows = random.Random(seed), []
    for week in range(weeks):
        period = date(2026, 1, 5) + timedelta(weeks=week)
        for panel in ("workshop_A", "workshop_B"):
            for topic in ("repair_stories", "shared_gardens"):
                # Deliberately changing synthetic response distribution; no real cultural claims.
                relevant = 0.15 + week * 0.045 + (0.15 if panel == "workshop_B" else 0)
                if topic == "shared_gardens":
                    relevant = 0.7 - week * 0.035
                for item in range(5):
                    item_id = f"{week}-{panel}-{topic}-{item}"
                    for annotator in range(4):
                        draw = rng.random()
                        label = "relevant" if draw < relevant else "unsure" if draw < relevant + 0.15 else "not_relevant"
                        delay = 10 if item == 0 and annotator == 0 else 1
                        rows.append(dict(zip(FIELDS, [item_id, period.isoformat(), (period + timedelta(days=delay)).isoformat(),
                                                     panel, topic, f"rater_{annotator}", label, "synthetic"])))
    return rows


def analyze(rows: list[dict]) -> dict:
    validate(rows)
    return {"data_status": "synthetic demonstration or explicitly rights-tagged input; not population representative",
            "n_annotations": len(rows), "items": item_summaries(rows),
            "periods": period_summaries(rows), "backtest": backtest(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="CSV matching the documented schema; omit for synthetic demo")
    parser.add_argument("--output", type=Path, default=Path("results/demo_report.json"))
    parser.add_argument("--export-demo", type=Path, help="write reproducible synthetic CSV")
    args = parser.parse_args()
    if args.input:
        with args.input.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = demo_rows()
    report = analyze(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.export_demo:
        args.export_demo.parent.mkdir(parents=True, exist_ok=True)
        with args.export_demo.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(demo_rows())
    print(json.dumps({"output": str(args.output), "n_annotations": len(rows),
                      "n_items": len(report["items"]), "scores": report["backtest"]["scores"]}, indent=2))


if __name__ == "__main__":
    main()
