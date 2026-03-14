#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from llm_client import build_qa_messages, build_summary_messages, chat
from ml_pipeline import (
    MODEL_CHOICES,
    build_llm_context,
    load_data,
    normalize_columns,
    pick_target,
    save_reports,
    train_and_evaluate,
)


def load_uploaded(files) -> pd.DataFrame:
    if not files:
        raise ValueError("Файлы для загрузки не выбраны.")
    if not isinstance(files, list):
        files = [files]
    frames = []
    for file in files:
        if file.name.lower().endswith(".csv"):
            frames.append(pd.read_csv(file))
        else:
            frames.append(pd.read_excel(file))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def get_data(uploaded, local_path: str) -> pd.DataFrame:
    if uploaded:
        return load_uploaded(uploaded)
    if not local_path:
        raise FileNotFoundError("Датасет не найден. Загрузите файл или укажите корректный путь.")
    parts = [p.strip() for p in local_path.split(";") if p.strip()]
    if len(parts) > 1:
        frames = [load_data(Path(p)) for p in parts if Path(p).exists()]
        if frames:
            return pd.concat(frames, ignore_index=True)
    path = Path(local_path)
    if path.exists():
        return load_data(path)
    raise FileNotFoundError("Датасет не найден. Загрузите файл или укажите корректный путь.")


def save_training_progress(result, model_name: str, params: dict, files: list) -> None:
    base_dir = Path("training_runs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    last_dir = base_dir / "last"
    run_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)

    save_reports(result, run_dir)
    save_reports(result, last_dir)

    joblib.dump(result.pipeline, run_dir / "model.joblib")
    joblib.dump(result.pipeline, last_dir / "model.joblib")

    summary = {
        "timestamp": timestamp,
        "task": result.task,
        "target": result.target,
        "metrics": result.metrics,
        "cv_metrics": result.cv_metrics,
        "best_params": result.best_params,
        "synthetic_prediction": result.synthetic_prediction,
        "model": model_name,
        "train_params": params,
        "feature_columns": result.feature_columns,
        "categorical_cols": result.categorical_cols,
        "numeric_cols": result.numeric_cols,
        "files": files,
    }

    if result.feature_importance is not None:
        summary["feature_importance"] = result.feature_importance.to_dict(orient="records")

    summary_json = json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default)
    (run_dir / "summary.json").write_text(summary_json, encoding="utf-8")
    (last_dir / "summary.json").write_text(summary_json, encoding="utf-8")


def _json_default(obj):
    try:
        import numpy as np
        import pandas as pd
    except Exception:
        np = None
        pd = None

    if np is not None:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()

    if pd is not None:
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        if isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()

    if isinstance(obj, (set,)):
        return list(obj)
    return str(obj)


st.set_page_config(page_title="Прогноз посещаемости мероприятий", layout="wide")
st.title("Прогноз посещаемости мероприятий")

with st.sidebar:
    st.header("Данные")
    uploaded = st.file_uploader(
        "Загрузить датасеты (.xlsx/.csv)", type=["xlsx", "xls", "csv"], accept_multiple_files=True
    )
    local_path = st.text_input(
        "Или локальный путь (можно несколько через ;)",
        value="ai_event_dataset.xlsx",
    )

    st.header("Модель")
    model_name = st.selectbox("Модель", MODEL_CHOICES, index=0)
    test_size = st.slider("Доля тестовой выборки", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    cv = st.slider("Фолды CV (0 = выкл.)", min_value=0, max_value=10, value=0, step=1)
    tune = st.checkbox("Тюнинг гиперпараметров", value=False)
    tune_iter = st.slider("Итераций тюнинга", min_value=10, max_value=60, value=25, step=5)
    tune_cv = st.slider("Фолды для тюнинга", min_value=2, max_value=5, value=3, step=1)
    seed = st.number_input("Случайное зерно", min_value=0, value=42, step=1)

    llm_enabled = True
    llm_provider = "ollama"
    llm_model = None
    llm_base_url = None

try:
    df = get_data(uploaded, local_path)
except Exception as exc:
    st.warning(str(exc))
    st.stop()

df = normalize_columns(df)

st.subheader("Предпросмотр данных")
st.dataframe(df.head(10), use_container_width=True)

with st.expander("Базовая статистика"):
    st.write("Строк:", df.shape[0])
    st.write("Столбцов:", df.shape[1])
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

target_options = ["(авто)"] + list(df.columns)
target_choice = st.selectbox("Целевая колонка", target_options, index=0)
if target_choice == "(авто)":
    try:
        target = pick_target(df, None)
        st.info(f"Автоопределённая цель: {target}")
    except Exception as exc:
        st.warning(str(exc))
        st.stop()
else:
    target = target_choice

if st.button("Обучить модель"):
    with st.spinner("Обучение..."):
        result = train_and_evaluate(
            df,
            target=target,
            model_name=model_name,
            test_size=test_size,
            random_state=int(seed),
            cv=int(cv),
            tune=tune,
            tune_iter=int(tune_iter),
            tune_cv=int(tune_cv),
            run_eda_reports=False,
        )
        st.session_state["result"] = result
        try:
            if uploaded:
                files = [f.name for f in uploaded]
            else:
                files = [p.strip() for p in local_path.split(";") if p.strip()]
            params = {
                "model": model_name,
                "test_size": float(test_size),
                "cv": int(cv),
                "tune": bool(tune),
                "tune_iter": int(tune_iter),
                "tune_cv": int(tune_cv),
                "seed": int(seed),
            }
            save_training_progress(result, model_name, params, files)
        except Exception as exc:
            st.warning(f"Не удалось сохранить прогресс обучения: {exc}")

result = st.session_state.get("result")
if result is not None:
    st.subheader("Метрики")
    st.json(result.metrics)

    if result.cv_metrics:
        st.subheader("Кросс-валидация")
        st.json(result.cv_metrics)

    if result.best_params:
        st.subheader("Лучшие гиперпараметры")
        st.json(result.best_params)

    if result.report:
        st.subheader("Отчёт классификации")
        st.dataframe(pd.DataFrame(result.report).transpose(), use_container_width=True)

    if result.confusion is not None:
        st.subheader("Матрица ошибок")
        st.dataframe(pd.DataFrame(result.confusion), use_container_width=True)

    if result.feature_importance is not None:
        st.subheader("Важность признаков (топ)")
        fi = result.feature_importance.set_index("feature")
        st.bar_chart(fi["importance"])

    st.subheader("Прогноз для синтетического примера")
    st.write(result.synthetic_prediction)

    st.subheader("Прогноз пользовательского события")
    with st.form("predict_form"):
        inputs = {}
        for col in result.categorical_cols:
            options = df[col].dropna().unique().tolist()
            options = sorted(options) if options else []
            default = result.synthetic_sample[col].iloc[0]
            if len(options) > 50:
                inputs[col] = st.text_input(col, value=str(default) if default is not None else "")
            else:
                idx = options.index(default) if default in options else 0
                inputs[col] = st.selectbox(col, options, index=idx) if options else ""
        for col in result.numeric_cols:
            default = result.synthetic_sample[col].iloc[0]
            try:
                default = float(default)
            except Exception:
                default = 0.0
            inputs[col] = st.number_input(col, value=default)
        submitted = st.form_submit_button("Предсказать")
    if submitted:
        input_df = pd.DataFrame([inputs], columns=result.feature_columns)
        pred = result.pipeline.predict(input_df)[0]
        if result.task == "classification" and result.label_encoder is not None:
            pred = result.label_encoder.inverse_transform([pred])[0]
        st.success(f"Прогноз: {pred}")

    if llm_enabled:
        st.subheader("LLM-анализ")
        context = build_llm_context(df, result, top_n=10)
        question = st.text_area("Задайте вопрос о данных или результатах", value="")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Сгенерировать сводку"):
                with st.spinner("Обращаемся к локальной LLM..."):
                    messages = build_summary_messages(context)
                    summary = chat(
                        messages,
                        provider=llm_provider,
                        model=llm_model or None,
                        base_url=llm_base_url or None,
                    )
                    st.write(summary)
        with col2:
            if st.button("Задать вопрос") and question.strip():
                with st.spinner("Обращаемся к локальной LLM..."):
                    messages = build_qa_messages(context, question.strip())
                    answer = chat(
                        messages,
                        provider=llm_provider,
                        model=llm_model or None,
                        base_url=llm_base_url or None,
                    )
                    st.write(answer)
