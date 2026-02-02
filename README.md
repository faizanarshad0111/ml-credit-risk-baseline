# ML Credit Risk Baseline

Basic credit risk prediction using logistic regression.

## Setup
```bash
pip install -r requirements.txt
```

## Usage

Train model:
```bash
python -m src.train --data data/credit_sample.csv
```

Run bootstrap for confidence intervals:
```bash
python -m src.bootstrap_eval --data data/credit_sample.csv
```
