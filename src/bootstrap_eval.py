import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from src.train import build_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="defaulted")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target])
    y = df[args.target].astype(int)

    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(exclude=["number"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    scores = []
    n = len(X_train)

    for _ in range(args.iters):
        idx = rng.integers(0, n, size=n)
        model = build_pipeline(num_cols, cat_cols, 1.0, args.seed)
        model.fit(X_train.iloc[idx], y_train.iloc[idx])
        scores.append(f1_score(y_test, model.predict(X_test)))

    scores = np.array(scores)
    print("Bootstrap mean F1:", scores.mean())
    print("95% CI:", np.percentile(scores, [2.5, 97.5]))


if __name__ == "__main__":
    main()
