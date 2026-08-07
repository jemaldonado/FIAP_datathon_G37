# Bank Marketing — Primary Dataset

| Field    | Value |
|----------|-------|
| Author   | henriqueyamahata |
| URL      | https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing |
| Version  | 1 |
| License  | CC BY-SA 4.0 |
| Role     | **Primary** — base for processed layer and bandit simulation |

## Description

Direct marketing campaigns (phone calls) of a Portuguese banking institution. Target variable `y` indicates whether the client subscribed to a term deposit.

## Columns

| Column       | Type    | Notes |
|--------------|---------|-------|
| age          | int     | |
| job          | string  | |
| marital      | string  | |
| education    | string  | |
| default      | string  | Credit in default |
| housing      | string  | Housing loan |
| loan         | string  | Personal loan |
| contact      | string  | Contact communication type |
| month        | string  | Last contact month |
| day_of_week  | string  | Last contact day of week |
| duration     | int     | **DROPPED — leakage** (only known after call) |
| campaign     | int     | Number of contacts this campaign |
| pdays        | int     | Days since previous campaign contact |
| previous     | int     | Previous campaign contacts |
| poutcome     | string  | Previous campaign outcome |
| emp.var.rate | float   | Employment variation rate (macro) |
| cons.price.idx | float | Consumer price index (macro) |
| cons.conf.idx | float  | Consumer confidence index (macro) |
| euribor3m    | float   | Euribor 3-month rate (macro) |
| nr.employed  | float   | Number of employees (macro) |
| y            | string  | **Target** — term deposit subscription (yes/no) |

## Leakage Decision

`duration` is dropped entirely from `data/processed/`. It is only observable after the call concludes and would not be available at decision time in a real deployment.

## Download

```bash
# Requires KAGGLE_USERNAME and KAGGLE_KEY in .env
scripts/download_data.sh
```

Or manually: download `bank-additional-full.csv` from the URL above and place it in this directory.
