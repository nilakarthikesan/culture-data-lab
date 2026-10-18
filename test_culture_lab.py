import unittest
from culture_lab import (analyze, backtest, demo_rows, distribution, forecast,
                         item_summaries, period_summaries, validate)


def row(item, period, label, annotator="a", available=None):
    return {"item_id": item, "period": period, "available_at": available or period,
            "panel": "panel", "topic": "topic", "annotator_id": annotator,
            "label": label, "rights_basis": "synthetic"}


class CultureTests(unittest.TestCase):
    def test_disagreement_preserves_tie_and_missing_agreement(self):
        items = item_summaries([row("x", "2026-01-01", "relevant"),
                                row("x", "2026-01-01", "not_relevant", "b"),
                                row("y", "2026-01-01", "unsure")])
        self.assertEqual(items[0]["plurality_labels"], ["relevant", "not_relevant"])
        self.assertEqual(items[0]["entropy_bits"], 1)
        self.assertEqual(items[0]["pairwise_agreement"], 0)
        self.assertIsNone(items[1]["pairwise_agreement"])

    def test_item_weight_does_not_grow_with_more_raters(self):
        rows = [row("x", "2026-01-01", "relevant", str(i)) for i in range(10)]
        rows += [row("y", "2026-01-01", "not_relevant")]
        self.assertEqual(period_summaries(rows)[0]["distribution"]["relevant"], 0.5)

    def test_majority_aggregation_removes_minority_probability(self):
        items = item_summaries([row("x", "2026-01-01", "relevant", str(i)) for i in range(3)] +
                               [row("x", "2026-01-01", "not_relevant", "b")])
        self.assertEqual(forecast(items, "expanding")["not_relevant"], 0.25)
        self.assertEqual(forecast(items, "hard_vote")["not_relevant"], 0)

    def test_late_labels_cannot_leak_into_forecast(self):
        rows = [row("old", "2026-01-01", "not_relevant"),
                row("late", "2026-01-02", "relevant", available="2026-03-01"),
                row("target", "2026-01-10", "relevant")]
        folds = backtest(rows, minimum_history_periods=1)["folds"]
        pred = next(f for f in folds if f["period"] == "2026-01-10" and f["method"] == "persistence")
        self.assertEqual(pred["prediction"]["not_relevant"], 1)
        self.assertEqual(pred["n_history_items"], 1)

    def test_future_outcomes_do_not_change_earlier_predictions(self):
        rows = demo_rows(weeks=5)
        before = backtest(rows)
        modified = [dict(r, label="unsure") if r["period"] == "2026-02-02" else r for r in rows]
        after = backtest(modified)
        self.assertEqual([f["prediction"] for f in before["folds"]],
                         [f["prediction"] for f in after["folds"]])

    def test_reject_duplicate_and_unlicensed(self):
        first = row("x", "2026-01-01", "relevant")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate([first, first])
        with self.assertRaisesRegex(ValueError, "rights"):
            validate([dict(first, rights_basis="scraped_without_permission")])

    def test_backtest_validates_dates_and_labels_at_public_boundary(self):
        first = row("x", "2026-01-01", "relevant")
        # Basic ISO dates are parseable on some supported Python versions, but
        # cannot safely be mixed with YYYY-MM-DD in lexicographic cutoffs.
        with self.assertRaises(ValueError):
            backtest([dict(first, period="20260101")])
        with self.assertRaises(ValueError):
            backtest([dict(first, label="unsupported")])
        with self.assertRaises(ValueError):
            distribution(["relevant", "unsupported"])

    def test_probability_conservation_and_demo_reproducibility(self):
        rows = validate(demo_rows())
        self.assertEqual(rows, demo_rows())
        result = analyze(rows)
        self.assertEqual(result["n_annotations"], 960)
        for fold in result["backtest"]["folds"]:
            self.assertAlmostEqual(sum(fold["prediction"].values()), 1)
            self.assertTrue(0 <= fold["soft_brier"] <= 2)


if __name__ == "__main__":
    unittest.main()
