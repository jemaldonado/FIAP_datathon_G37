# Bank Marketing — Janiobachmann Variant

| Field    | Value |
|----------|-------|
| Author   | janiobachmann |
| URL      | https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset |
| Version  | 1 |
| License  | CC BY-SA 4.0 |
| Role     | **Comparison** — schema validation and conversion rate benchmarking |

## Description

Alternative version of the UCI Bank Marketing dataset. Used to cross-validate conversion rate distributions and feature importance against the primary dataset.

## Leakage Decision

Same `duration` leakage issue as primary — dropped from any processed representation.

## Download

```bash
scripts/download_data.sh
```

Or manually: download from the URL above and place the CSV in this directory.
