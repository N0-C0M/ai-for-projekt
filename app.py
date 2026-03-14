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


def _find_feature(columns, needles):
    for col in columns:
        col_lower = str(col).lower()
        for needle in needles:
            if needle in col_lower:
                return col
    return None


def _duration_to_bin(value):
    try:
        hours = float(value)
    except (TypeError, ValueError):
        return None
    if hours <= 2:
        return "0-2"
    if hours <= 4:
        return "2-4"
    if hours <= 6:
        return "4-6"
    if hours <= 8:
        return "6-8"
    return "8+"


def _categorical_options(df: pd.DataFrame, col: str, default):
    default_value = "" if pd.isna(default) else str(default)
    if col in df.columns:
        options = df[col].dropna().astype(str).unique().tolist()
        options = sorted(options) if options else []
        return options, default_value

    col_lower = str(col).lower()
    if "duration_bin" in col_lower or "duratetion_bin" in col_lower:
        return ["0-2", "2-4", "4-6", "6-8", "8+"], default_value
    if "temp_bin" in col_lower:
        return ["< -5", "-5..0", "0..10", "10..20", "20..30", "30+"], default_value
    if "is_weekend" in col_lower:
        return ["0", "1"], default_value
    return [], default_value


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
st.dataframe(df, use_container_width=True)
st.caption("Показаны все строки. Обучение использует полный датасет.")

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
if result is None:
    st.subheader("Прогноз пользовательского события")
    st.info("Сначала обучите модель, чтобы рассчитать прогноз.")
    st.button("Рассчитать прогноз", disabled=True)
else:
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
    duration_col = _find_feature(
        result.feature_columns,
        ["duration_hours", "duration", "продолжительность"],
    )
    duration_bin_col = _find_feature(
        result.feature_columns,
        ["duration_bin", "duratetion_bin", "duration bin", "duratetion bin"],
    )
    duration_bin_options = []
    if duration_bin_col:
        if duration_bin_col in df.columns:
            duration_bin_options = (
                df[duration_bin_col].dropna().astype(str).unique().tolist()
            )
            duration_bin_options = (
                sorted(duration_bin_options) if duration_bin_options else []
            )
        if not duration_bin_options:
            duration_bin_options = ["0-2", "2-4", "4-6", "6-8", "8+"]

    if duration_col and duration_bin_col:
        duration_key = f"pred_{duration_col}"
        duration_bin_key = f"pred_{duration_bin_col}"
        if st.button("Рассчитать duration_bin"):
            raw_value = st.session_state.get(
                duration_key,
                result.synthetic_sample[duration_col].iloc[0]
                if duration_col in result.synthetic_sample.columns
                else None,
            )
            try:
                hours = float(str(raw_value).replace(",", "."))
            except (TypeError, ValueError):
                st.warning(f"Некорректное число в поле: {duration_col}")
            else:
                bin_label = _duration_to_bin(hours)
                if not bin_label:
                    st.warning("Не удалось рассчитать duration_bin.")
                elif duration_bin_options and bin_label not in duration_bin_options:
                    st.warning("duration_bin вне диапазона доступных значений.")
                else:
                    st.session_state[duration_bin_key] = bin_label

    with st.form("predict_form"):
        inputs = {}
        for col in result.categorical_cols:
            default = result.synthetic_sample[col].iloc[0]
            options, default_value = _categorical_options(df, col, default)
            key = f"pred_{col}"
            if duration_bin_col and col == duration_bin_col:
                options = [str(opt) for opt in options]
                default_value = "" if pd.isna(default_value) else str(default_value)
            if len(options) > 50:
                inputs[col] = st.text_input(
                    col,
                    value=default_value,
                    key=key,
                )
            else:
                if options:
                    idx = options.index(default_value) if default_value in options else 0
                    inputs[col] = st.selectbox(col, options, index=idx, key=key)
                else:
                    inputs[col] = st.text_input(
                        col,
                        value=default_value,
                        key=key,
                    )
        for col in result.numeric_cols:
            default = result.synthetic_sample[col].iloc[0]
            try:
                default = float(default)
            except Exception:
                default = 0.0
            inputs[col] = st.number_input(col, value=default, key=f"pred_{col}")
        submitted = st.form_submit_button("Рассчитать прогноз")
    if submitted:
        input_df = pd.DataFrame([inputs], columns=result.feature_columns)
        pred = result.pipeline.predict(input_df)[0]
        if result.task == "classification" and result.label_encoder is not None:
            pred = result.label_encoder.inverse_transform([pred])[0]
        st.success(f"Прогноз: {pred}")

    if llm_enabled:
        st.subheader("LLM-анализ")
        context = build_llm_context(df, result, top_n=5, compact=True)
        question = st.text_area("Задайте вопрос о данных или результатах", value="")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Сгенерировать сводку"):
                with st.spinner("Обращаемся к локальной LLM..."):
                    messages = build_summary_messages(context)
                    try:
                        summary = chat(
                            messages,
                            provider=llm_provider,
                            model=llm_model or None,
                            base_url=llm_base_url or None,
                        )
                        st.write(summary)
                    except Exception as exc:
                        st.error(f"Ошибка LLM-сводки: {exc}")
        with col2:
            if st.button("Задать вопрос") and question.strip():
                with st.spinner("Обращаемся к локальной LLM..."):
                    messages = build_qa_messages(context, question.strip())
                    try:
                        answer = chat(
                            messages,
                            provider=llm_provider,
                            model=llm_model or None,
                            base_url=llm_base_url or None,
                        )
                        st.write(answer)
                    except Exception as exc:
                        st.error(f"Ошибка LLM-ответа: {exc}")
