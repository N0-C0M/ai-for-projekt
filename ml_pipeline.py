#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


TARGET_CANDIDATES = [
    "attendance_score",
    "attendance",
    "attendance_category",
    "Посещаемость_категория",
    "Доля_заполняемости_зала",
    "Посещаемость",
]


MODEL_CHOICES = ["rf", "gb", "hgb", "linear"]


@dataclass
class TrainResult:
    task: str
    target: str
    pipeline: Pipeline
    label_encoder: Optional[LabelEncoder]
    best_params: Optional[Dict[str, Any]]
    metrics: Dict[str, float]
    report: Optional[Dict[str, Any]]
    report_text: Optional[str]
    confusion: Optional[np.ndarray]
    cv_metrics: Optional[Dict[str, float]]
    feature_importance: Optional[pd.DataFrame]
    synthetic_sample: pd.DataFrame
    synthetic_prediction: Any
    categorical_cols: List[str]
    numeric_cols: List[str]
    feature_columns: List[str]


def load_data(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"Неподдерживаемый формат файла: {ext}. Используйте .xlsx, .xls или .csv"
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df.dropna(axis=1, how="all")


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
                f"Целевая колонка '{target_arg}' не найдена. Доступные колонки: {list(df.columns)}"
            )
        return target_arg
    for cand in TARGET_CANDIDATES:
        if cand in df.columns:
            return cand
    raise ValueError(
        "Целевая колонка не найдена. Укажите --target. "
        f"Доступные колонки: {list(df.columns)}"
    )


def infer_task(y: pd.Series, target_name: str) -> str:
    name = target_name.lower()
    if "доля" in name or "заполняем" in name:
        return "regression"
    if "share" in name or "ratio" in name or "fill" in name:
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
        print(f"EDA пропущен: matplotlib/seaborn недоступны ({exc})")
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


def build_model(task: str, model_name: str, random_state: int):
    if model_name == "rf":
        if task == "classification":
            return RandomForestClassifier(
                n_estimators=400, random_state=random_state, class_weight="balanced"
            )
        return RandomForestRegressor(n_estimators=400, random_state=random_state)
    if model_name == "gb":
        if task == "classification":
            return GradientBoostingClassifier(random_state=random_state)
        return GradientBoostingRegressor(random_state=random_state)
    if model_name == "hgb":
        if task == "classification":
            return HistGradientBoostingClassifier(random_state=random_state)
        return HistGradientBoostingRegressor(random_state=random_state)
    if model_name == "linear":
        if task == "classification":
            return LogisticRegression(max_iter=2000, class_weight="balanced")
        return Ridge(random_state=random_state)
    raise ValueError(f"Неизвестная модель: {model_name}. Выберите из {MODEL_CHOICES}")


def get_param_distributions(task: str, model_name: str) -> Dict[str, List[Any]]:
    if model_name == "rf":
        return {
            "model__n_estimators": [200, 400, 700],
            "model__max_depth": [None, 6, 10, 18],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        }
    if model_name == "gb":
        return {
            "model__n_estimators": [150, 300, 500],
            "model__learning_rate": [0.03, 0.07, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
        }
    if model_name == "hgb":
        return {
            "model__max_iter": [150, 300, 600],
            "model__learning_rate": [0.03, 0.07, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
        }
    return {}


def extract_feature_importance(
    pipeline: Pipeline, top_n: int = 20
) -> Optional[pd.DataFrame]:
    model = pipeline.named_steps.get("model")
    if model is None:
        return None
    if not hasattr(model, "feature_importances_") and not hasattr(model, "coef_"):
        return None

    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        coef = model.coef_
        if coef.ndim == 2:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False)
    if top_n:
        df = df.head(top_n)
    return df.reset_index(drop=True)


def _compute_metrics(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: Optional[LabelEncoder],
) -> Dict[str, Any]:
    metrics: Dict[str, float] = {}
    report: Optional[Dict[str, Any]] = None
    report_text: Optional[str] = None
    confusion = None

    if task == "classification":
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")

        labels = np.unique(y_true)
        target_names = None
        if label_encoder is not None:
            try:
                target_names = label_encoder.inverse_transform(labels)
            except Exception:
                target_names = None

        report = classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, output_dict=True
        )
        report_text = classification_report(
            y_true, y_pred, labels=labels, target_names=target_names
        )
        confusion = confusion_matrix(y_true, y_pred, labels=labels)
    else:
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["rmse"] = mean_squared_error(y_true, y_pred, squared=False)
        metrics["r2"] = r2_score(y_true, y_pred)

    return {
        "metrics": metrics,
        "report": report,
        "report_text": report_text,
        "confusion": confusion,
    }


def _compute_cv_metrics(
    task: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int
) -> Dict[str, float]:
    if task == "classification":
        scoring = ["accuracy", "f1_macro"]
    else:
        scoring = ["r2", "neg_mean_absolute_error"]

    cv_res = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
    cv_metrics: Dict[str, float] = {}
    for key, values in cv_res.items():
        if key.startswith("test_"):
            name = key.replace("test_", "")
            mean = float(np.mean(values))
            std = float(np.std(values))
            if name == "neg_mean_absolute_error":
                name = "mae"
                mean = -mean
            cv_metrics[f"{name}_mean"] = mean
            cv_metrics[f"{name}_std"] = std
    return cv_metrics


def train_and_evaluate(
    df: pd.DataFrame,
    target: str,
    model_name: str = "rf",
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int = 0,
    tune: bool = False,
    tune_iter: int = 25,
    tune_cv: int = 3,
    run_eda_reports: bool = False,
    eda_dir: Optional[Path] = None,
) -> TrainResult:
    df = normalize_columns(df)
    df = drop_empty_columns(df)
    df = df.drop_duplicates().reset_index(drop=True)

    df = df.dropna(subset=[target]).reset_index(drop=True)
    X = df.drop(columns=[target])
    X = coerce_numeric_columns(X)
    y = df[target].copy()

    task = infer_task(y, target)

    if run_eda_reports:
        run_eda(df, target, eda_dir or Path("reports"))

    preprocessor, categorical_cols, numeric_cols = build_preprocess(X)

    label_encoder = None
    if task == "classification":
        if y.dtype == object or y.dtype.name == "category":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.astype(str))

    model = build_model(task, model_name, random_state)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    stratify = y if task == "classification" and len(np.unique(y)) > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )

    best_params = None
    if tune:
        param_dist = get_param_distributions(task, model_name)
        if param_dist:
            scoring = "f1_macro" if task == "classification" else "r2"
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=tune_iter,
                cv=tune_cv,
                scoring=scoring,
                random_state=random_state,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            pipeline = search.best_estimator_
            best_params = search.best_params_
        else:
            pipeline.fit(X_train, y_train)
    else:
        pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    metric_payload = _compute_metrics(task, y_test, preds, label_encoder)
    metrics = metric_payload["metrics"]
    report = metric_payload["report"]
    report_text = metric_payload["report_text"]
    confusion = metric_payload["confusion"]

    cv_metrics = None
    if cv and cv > 1:
        cv_metrics = _compute_cv_metrics(task, pipeline, X, y, cv)

    synthetic_sample = build_synthetic_sample(X, categorical_cols, numeric_cols)
    pred_syn = pipeline.predict(synthetic_sample)[0]
    if task == "classification" and label_encoder is not None:
        pred_syn = label_encoder.inverse_transform([pred_syn])[0]

    feature_importance = extract_feature_importance(pipeline, top_n=20)

    return TrainResult(
        task=task,
        target=target,
        pipeline=pipeline,
        label_encoder=label_encoder,
        best_params=best_params,
        metrics=metrics,
        report=report,
        report_text=report_text,
        confusion=confusion,
        cv_metrics=cv_metrics,
        feature_importance=feature_importance,
        synthetic_sample=synthetic_sample,
        synthetic_prediction=pred_syn,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        feature_columns=list(X.columns),
    )


def save_reports(result: TrainResult, report_dir: Path) -> None:
    import json

    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result.metrics, f, ensure_ascii=False, indent=2, default=_json_default)

    if result.cv_metrics:
        cv_path = report_dir / "cv_metrics.json"
        with cv_path.open("w", encoding="utf-8") as f:
            json.dump(result.cv_metrics, f, ensure_ascii=False, indent=2, default=_json_default)

    if result.best_params:
        best_path = report_dir / "best_params.json"
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(result.best_params, f, ensure_ascii=False, indent=2, default=_json_default)

    if result.report:
        report_path = report_dir / "classification_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(result.report, f, ensure_ascii=False, indent=2, default=_json_default)

    if result.confusion is not None:
        cm_path = report_dir / "confusion_matrix.csv"
        pd.DataFrame(result.confusion).to_csv(cm_path, index=False)

    if result.feature_importance is not None:
        fi_path = report_dir / "feature_importance.csv"
        result.feature_importance.to_csv(fi_path, index=False)


def build_llm_context(
    df: pd.DataFrame,
    result: TrainResult,
    top_n: int = 10,
    compact: bool = False,
) -> Dict[str, Any]:
    missing = df.isna().mean().sort_values(ascending=False).head(10)
    top_features = []
    if result.feature_importance is not None:
        top_features = result.feature_importance.head(top_n).to_dict(orient="records")

    columns = list(df.columns)
    payload: Dict[str, Any] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "task": result.task,
        "target": result.target,
        "metrics": result.metrics,
        "cv_metrics": result.cv_metrics,
        "best_params": result.best_params,
        "top_features": top_features,
        "synthetic_prediction": result.synthetic_prediction,
    }

    if compact:
        payload["columns_sample"] = columns[:10]
        payload["columns_total"] = len(columns)
        payload["missing_top"] = missing.head(3).to_dict()
    else:
        payload["columns"] = columns
        payload["missing_top"] = missing.to_dict()

    return payload


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, (set,)):
        return list(obj)
    return str(obj)
