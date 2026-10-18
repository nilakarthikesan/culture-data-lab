# Culture Data Lab

A CPU-only research prototype for **preserving disagreement in creative-content annotation and testing forecasts under time shift**. The runnable demo uses entirely synthetic annotations of two invented workshop panels responding to two invented creative themes. It measures an operational response distribution, not a universal representation of a culture.

## Run

Requires Python 3.10+; no dependencies, API keys, GPUs, accounts, or network calls.

```sh
python3 -m unittest -v
python3 culture_lab.py --export-demo data/demo_annotations.csv
python3 culture_lab.py --input data/demo_annotations.csv --output results/reloaded_report.json
```

The script exports a JSON report containing item-level label distributions, entropy in bits, distinct-rater pairwise agreement, observed-period distribution changes, and chronological forecast folds. A singleton item's pairwise agreement is undefined (`null`); tied plurality labels are all retained. It does not compute a chance-corrected agreement coefficient or infer demographic traits.

## Tractable research questions

1. Does retaining the full distribution of subjective labels improve next-period distribution forecasts compared with flattening each item to its plurality label?
2. Does a recent-history baseline outperform a historical mean when the operational response distribution changes?
3. Which errors come from ambiguous content, annotation instructions, panel sampling, or time shift? The prototype reports disagreement but does not identify its cause.

The five forecast baselines are uniform, an expanding mean of item distributions, an expanding mean of item plurality labels (`hard_vote`, sharing ties), previous-period persistence, and a fixed-0.5 exponentially weighted average. Compare `expanding` against `hard_vote` to isolate aggregation, and `expanding` against `ewma` to test recency. No learned model or creative-content embedding is present yet.

## Data contract and timing

CSV fields: `item_id, period, available_at, panel, topic, annotator_id, label, rights_basis`.

- `period` is the ISO date of an observation batch; every observation in that batch is to be forecast before that date.
- `available_at` is when that annotation became usable. It must be at or after `period`.
- Both dates must use canonical `YYYY-MM-DD`; mixed ISO week/basic forms are rejected so chronological string cutoffs remain valid. Direct backtest calls enforce the same validation as the CLI.
- Labels are `relevant`, `not_relevant`, and `unsure`. In a future study, define “relevant” in the written instructions, for example “I would voluntarily explore more of this creative theme during the next week.” These are responses, not objective content truth.
- `panel` is an explicitly recruited panel or distribution channel, not an inferred nationality, race, religion, or psychological category. Demo names `workshop_A/B` have no human identity.
- `rights_basis` must be `synthetic`, `consented`, or `public_licensed`. A tag is an assertion by the uploader, not proof of legal rights or consent. Do not upload another person's copyrighted work or private conversations merely because a URL is public.

For each forecast, both observation `period` and annotation `available_at` must precede the target cutoff. Late annotations are excluded from training. Final target labels are used retrospectively only for scoring. Items have equal weight regardless of their rater counts; folds have equal weight across panel/topic/period. This prevents heavily annotated items from dominating. Production work should also require fixed observation windows and report label-finalization delays.

The metric is the squared distance between predicted and empirical three-class distributions: sum over labels of `(prediction - observed_distribution)^2`, range 0–2, lower is better. It is called **soft-label Brier** here. It omits the intrinsic variance term that appears when averaging categorical Brier over individual raters. Do not compare its absolute score to a conventional individual-label Brier score. Adjacent-period total variation describes observed change; it is not a causal detector of cultural change.

## Demo results and interpretation

`data/demo_annotations.csv` has 960 annotations, 240 creative items, 12 weekly batches, 2 panels and 2 themes. The synthetic generator deliberately changes responses over time and includes delayed labels. Scores in `results/demo_report.json` describe this toy process only. A recency advantage here is a pipeline sanity check, not evidence that future real culture is predictable.

The test suite checks label-distribution preservation, ties, equal item weighting, late-label exclusion, future-target non-leakage, duplicate/rights validation, reproducibility, and probability bounds.

## Proposed real pilot

Recruit a small opt-in creative panel; compensate consistently. Begin with 30–50 original or explicitly licensed items per week for 6 weeks, at least 3 independent raters per item. Collect responses before showing aggregate answers. Keep an `unsure` response and an optional free-text rationale; do not force apparent consensus. Separate willingness to engage from observed engagement, and measure paid exposure/distribution so promotion is not mislabeled as preference.

Preregister label instructions, prediction cutoff, forecast horizon, baseline choice, sample exclusions and a locked future test window. Tune on earlier weeks only. Repeated raters/items require clustered uncertainty estimates; this minimal prototype does not supply them. A subsequent study should use rolling windows and independent-panel/creator holdouts, evaluate calibration and coverage, and publish failures alongside successes. A convenience sample does not justify population-level claims. Limit forecasting to aggregate, opt-in product/creative feedback; do not score individual vulnerabilities or target persuasion.

**Go/no-go:** continue only if panel recruitment and content rights are feasible, annotators understand the construct, and a future locked window beats a prespecified simple baseline by a useful margin with uncertainty reported. Stop or revise the label construct if disagreement tracks instructions more than content or if results vanish under panel/creator holdouts.

## Primary literature and what it contributes

- Davani, Díaz & Prabhakaran (TACL 2022), [Dealing with Disagreements](https://aclanthology.org/2022.tacl-1.6/): multi-annotator modeling illustrates why aggregating subjective labels can discard useful variation. This repo retains labels; it does not reproduce their model.
- Prabhakaran, Davani & Díaz (LAW-DMR 2021), [On Releasing Annotator-Level Labels and Information](https://aclanthology.org/2021.law-1.14/): motivates annotator-level transparency. Release requires explicit participant consent and careful minimization; public pseudonyms alone do not prevent reidentification.
- Ni et al. (EACL March 2026), [Can Reasoning Help Large Language Models Capture Human Annotator Disagreement?](https://aclanthology.org/2026.eacl-long.3/): their tested reasoning settings do not reliably replace human disagreement distributions. Relevant caution for substituting synthetic annotators for consented participants.
- Belay et al. (TACL 2026), [Beyond Majority Voting](https://aclanthology.org/2026.tacl-1.61/): agreement-based perspective clustering is a possible future extension, not implemented here. Clusters should not be assigned sensitive cultural identities by inference.
- Yao et al. (2022/NeurIPS benchmark), [Wild-Time](https://arxiv.org/abs/2211.14238): supports time-separated and streaming evaluation. Its datasets and reported results do not establish performance for this creative-content task.
- Zaghouani (C3NLP 2026), [Annotation as Cultural Interpretation](https://aclanthology.org/2026.c3nlp-1.1/): conceptual framing for culturally situated annotation; distinguish a position paper from an experimental result.

These are related foundations and recent examples checked on 2026-10-18, not a systematic review.

## Contribution roles and credit

Open roles: annotation protocol reviewer; content-rights/creator partnership lead; chronological evaluation engineer; panel researcher; independent methods reviewer. All are proposed and opt-in. Define a bounded contribution and shared deliverable before work. Credit follows actual contributions (for example a CRediT statement); authorship is not promised in exchange for participation. Personal collaborator assessments are deliberately stored outside this project.
