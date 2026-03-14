#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


TARGET_CANDIDATES = [
    "Посещаемость_категория",
    "Доля_заполняемости_зала",
    "Посещаемость",
]


def load_data(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {ext}. Use .xlsx, .xls or .csv")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def coerce_numeric_columns(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
            )
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if numeric.notna().mean() >= threshold:
                df[col] = numeric
    return df


def pick_target(df: pd.DataFrame, target_arg: Optional[str]) -> str:
    if target_arg:
        if target_arg not in df.columns:
            raise ValueError(
                f"Target '{target_arg}' not found. Available columns: {list(df.columns)}"
            )
        return target_arg
    for cand in TARGET_CANDIDATES:
        if cand in df.columns:
            return cand
    raise ValueError(
        "Target column not found. Use --target to specify. "
        f"Available columns: {list(df.columns)}"
    )


def infer_task(y: pd.Series, target_name: str) -> str:
    name = target_name.lower()
    if "доля" in name or "заполняем" in name:
        return "regression"
    if pd.api.types.is_float_dtype(y):
        return "regression"
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 10:
        return "regression"
    return "classification"


def build_preprocess(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor, categorical_cols, numeric_cols


def run_eda(df: pd.DataFrame, target: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df.describe(include="all").to_csv(out_dir / "eda_summary.csv")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        print(f"EDA skipped: matplotlib/seaborn not available ({exc})")
        return

    target_path = out_dir / f"target_{target}.png"
    plt.figure(figsize=(6, 4))
    if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
        sns.histplot(df[target], kde=True)
    else:
        sns.countplot(x=df[target])
        plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(target_path, dpi=160)
    plt.close()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.tight_layout()
        plt.savefig(out_dir / f"num_{col}.png", dpi=160)
        plt.close()

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        plt.figure(figsize=(6, 4))
        df[col].value_counts().head(20).plot(kind="bar")
        plt.title(col)
        plt.tight_layout()
        plt.savefig(out_dir / f"cat_{col}.png", dpi=160)
        plt.close()


def build_synthetic_sample(
    X: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]
) -> pd.DataFrame:
    sample = {}
    for col in categorical_cols:
        mode = X[col].mode(dropna=True)
        sample[col] = mode.iloc[0] if not mode.empty else None
    for col in numeric_cols:
        sample[col] = float(X[col].median())
    return pd.DataFrame([sample], columns=X.columns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a model to predict event attendance/popularity."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ai_event_dataset.xlsx",
        help="Path to dataset (.xlsx/.xls/.csv).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name. If omitted, inferred.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size fraction.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help="If >0, run cross-validation with given folds.",
    )
    parser.add_argument(
        "--no-eda",
        action="store_true",
        help="Skip EDA report/plots.",
    )
    parser.add_argument(
        "--eda-dir",
        type=str,
        default="reports",
        help="Directory to save EDA outputs.",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Optional path to save trained pipeline with joblib.",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    df = load_data(data_path)
    df = normalize_columns(df)
    target = pick_target(df, args.target)

    df = df.dropna(subset=[target]).reset_index(drop=True)
    X = df.drop(columns=[target])
    X = coerce_numeric_columns(X)
    y = df[target].copy()

    task = infer_task(y, target)

    if not args.no_eda:
        run_eda(df, target, Path(args.eda_dir))

    preprocessor, categorical_cols, numeric_cols = build_preprocess(X)

    label_encoder = None
    if task == "classification":
        if y.dtype == object or y.dtype.name == "category":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.astype(str))
        model = RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        )
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    stratify = y if task == "classification" and len(np.unique(y)) > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=None
        )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print(f"Task: {task}")
    if task == "classification":
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-macro: {f1:.4f}")
        print("Classification report:")
        if label_encoder is not None:
            print(
                classification_report(
                    y_test, preds, target_names=label_encoder.classes_
                )
            )
        else:
            print(classification_report(y_test, preds))
    else:
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

    if args.cv and args.cv > 1:
        scoring = "f1_macro" if task == "classification" else "r2"
        scores = cross_val_score(pipeline, X, y, cv=args.cv, scoring=scoring)
        print(f"CV ({args.cv} folds) {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")

    synthetic = build_synthetic_sample(X, categorical_cols, numeric_cols)
    pred_syn = pipeline.predict(synthetic)[0]
    if task == "classification" and label_encoder is not None:
        pred_syn = label_encoder.inverse_transform([pred_syn])[0]
    print("Synthetic sample prediction:", pred_syn)

    if args.save_model:
        try:
            import joblib
        except Exception as exc:
            raise SystemExit(f"joblib not available: {exc}")
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)
        print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
