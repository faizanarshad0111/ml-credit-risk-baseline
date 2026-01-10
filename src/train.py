import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(num_cols, cat_cols, C, seed):
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])

    clf = LogisticRegression(max_iter=3000, C=C, random_state=seed)
    return Pipeline([("prep", preprocess), ("clf", clf)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="defaulted")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--model-out", default="artifacts/model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target])
    y = df[args.target].astype(int)

    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(exclude=["number"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    model = build_pipeline(num_cols, cat_cols, args.C, args.seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("F1:", f1_score(y_test, preds))

    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(model, args.model_out)


if __name__ == "__main__":
    main()
