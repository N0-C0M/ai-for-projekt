#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import joblib
import pandas as pd

from llm_client import build_qa_messages, build_summary_messages
from ml_pipeline import (
    MODEL_CHOICES,
    build_llm_context,
    load_data,
    normalize_columns,
    pick_target,
    save_reports,
    train_and_evaluate,
)
from ollama_handler import OllamaClient, OllamaError

MODEL_LABELS = {
    "rf": "Случайный лес (rf)",
    "gb": "Градиентный бустинг (gb)",
    "hgb": "Гист. градиентный бустинг (hgb)",
    "linear": "Линейная модель (linear)",
}
MODEL_LABELS_LIST = [MODEL_LABELS[m] for m in MODEL_CHOICES]
LABEL_TO_MODEL = {v: k for k, v in MODEL_LABELS.items()}
ATTENDANCE_LABELS = {0: "Низкая", 1: "Средняя", 2: "Высокая"}
DURATION_BIN_LABELS = ["0-2", "2-4", "4-6", "6-8", "8+"]
TEMP_BIN_LABELS = ["< -5", "-5..0", "0..10", "10..20", "20..30", "30+"]
WEEKEND_LABELS = ["0", "1"]


class DesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Прогноз посещаемости мероприятий")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        self.df = None
        self.result = None
        self.prediction_inputs = {}
        self.loaded_files = []
        self.last_model_name = None
        self.last_train_params = {}

        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "").strip()
        self.ollama_timeout = self._parse_int(os.getenv("OLLAMA_TIMEOUT", "60"), 60)
        self.ollama_models = []

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_left(left)
        self._build_right(right)

        self.status_var = tk.StringVar(value="Готово")
        status = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))

    def _build_left(self, parent: ttk.Frame) -> None:
        data_frame = ttk.LabelFrame(parent, text="Данные")
        data_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        data_frame.columnconfigure(1, weight=1)

        self.file_path_var = tk.StringVar(value="ai_event_dataset.xlsx")
        ttk.Label(data_frame, text="Файл").grid(row=0, column=0, sticky="w")
        ttk.Entry(data_frame, textvariable=self.file_path_var, width=40).grid(
            row=0, column=1, sticky="ew", padx=5
        )
        ttk.Button(data_frame, text="Выбрать...", command=self._choose_file).grid(
            row=0, column=2, padx=5
        )
        ttk.Label(data_frame, text="Доп. файлы для обучения").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        self.files_listbox = tk.Listbox(data_frame, height=4)
        self.files_listbox.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5)
        files_buttons = ttk.Frame(data_frame)
        files_buttons.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(files_buttons, text="Добавить файлы...", command=self._choose_files).pack(
            side="left", padx=5
        )
        ttk.Button(files_buttons, text="Очистить список", command=self._clear_files).pack(
            side="left", padx=5
        )
        self.load_button = ttk.Button(data_frame, text="Загрузить", command=self._load_data)
        self.load_button.grid(row=4, column=1, sticky="ew", pady=6)

        ttk.Label(data_frame, text="Целевая колонка").grid(row=5, column=0, sticky="w")
        self.target_var = tk.StringVar(value="(авто)")
        self.target_combo = ttk.Combobox(
            data_frame, textvariable=self.target_var, values=["(авто)"], state="readonly"
        )
        self.target_combo.grid(row=5, column=1, columnspan=2, sticky="ew", pady=(0, 5))

        model_frame = ttk.LabelFrame(parent, text="Модель")
        model_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)

        ttk.Label(model_frame, text="Тип модели").grid(row=0, column=0, sticky="w")
        self.model_label_var = tk.StringVar(value=MODEL_LABELS_LIST[0])
        ttk.Combobox(
            model_frame,
            textvariable=self.model_label_var,
            values=MODEL_LABELS_LIST,
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(model_frame, text="Доля тестовой выборки").grid(row=1, column=0, sticky="w")
        self.test_size_var = tk.StringVar(value="0.2")
        ttk.Combobox(
            model_frame,
            textvariable=self.test_size_var,
            values=["0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4"],
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(model_frame, text="Фолды CV (0 = выкл.)").grid(row=2, column=0, sticky="w")
        self.cv_var = tk.StringVar(value="0")
        ttk.Combobox(
            model_frame,
            textvariable=self.cv_var,
            values=[str(i) for i in range(0, 11)],
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", pady=2)

        self.tune_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(model_frame, text="Тюнинг гиперпараметров", variable=self.tune_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=2
        )

        ttk.Label(model_frame, text="Итераций тюнинга").grid(row=4, column=0, sticky="w")
        self.tune_iter_var = tk.StringVar(value="25")
        ttk.Combobox(
            model_frame,
            textvariable=self.tune_iter_var,
            values=[str(i) for i in range(10, 61, 5)],
            state="readonly",
        ).grid(row=4, column=1, sticky="ew", pady=2)

        ttk.Label(model_frame, text="Фолды для тюнинга").grid(row=5, column=0, sticky="w")
        self.tune_cv_var = tk.StringVar(value="3")
        ttk.Combobox(
            model_frame,
            textvariable=self.tune_cv_var,
            values=["2", "3", "4", "5"],
            state="readonly",
        ).grid(row=5, column=1, sticky="ew", pady=2)

        ttk.Label(model_frame, text="Случайное зерно").grid(row=6, column=0, sticky="w")
        self.seed_var = tk.StringVar(value="42")
        ttk.Spinbox(model_frame, from_=0, to=10_000, textvariable=self.seed_var).grid(
            row=6, column=1, sticky="ew", pady=2
        )

        self.train_button = ttk.Button(model_frame, text="Обучить модель", command=self._train_model)
        self.train_button.grid(row=7, column=0, columnspan=2, sticky="ew", pady=6)

    def _build_right(self, parent: ttk.Frame) -> None:
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.data_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.predict_tab = ttk.Frame(self.notebook)
        self.llm_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.data_tab, text="Данные")
        self.notebook.add(self.results_tab, text="Результаты")
        self.notebook.add(self.predict_tab, text="Прогноз")
        self.notebook.add(self.llm_tab, text="LLM")

        self.data_text = self._build_text_area(self.data_tab)
        self.results_text = self._build_text_area(self.results_tab)

        self._build_prediction_tab()
        self._build_llm_tab()

    def _build_text_area(self, parent: ttk.Frame) -> tk.Text:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        text = tk.Text(frame, wrap="none", height=20)
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        text.configure(state="disabled")
        return text

    def _build_prediction_tab(self) -> None:
        self.prediction_result_var = tk.StringVar(value="")
        ttk.Label(self.predict_tab, textvariable=self.prediction_result_var).pack(
            anchor="w", padx=10, pady=6
        )

        self.prediction_actions = ttk.Frame(self.predict_tab)
        self.prediction_actions.pack(fill="x", padx=10, pady=(0, 6))
        self.prediction_actions.columnconfigure(0, weight=1)
        self.prediction_actions.columnconfigure(1, weight=1)

        self.predict_button = ttk.Button(
            self.prediction_actions,
            text="Рассчитать прогноз",
            command=self._predict_custom,
            state="disabled",
        )
        self.predict_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.duration_bin_button = ttk.Button(
            self.prediction_actions,
            text="Рассчитать duration_bin",
            command=self._fill_duration_bin,
            state="disabled",
        )
        self.duration_bin_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        container = ttk.Frame(self.predict_tab)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, borderwidth=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.prediction_frame = ttk.Frame(canvas)

        self.prediction_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.prediction_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_llm_tab(self) -> None:
        summary_frame = ttk.LabelFrame(self.llm_tab, text="Сводка")
        summary_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.summary_button = ttk.Button(
            summary_frame, text="Сгенерировать сводку", command=self._run_summary
        )
        self.summary_button.pack(anchor="w", padx=4, pady=4)
        self.summary_text = self._build_text_area(summary_frame)

        qa_frame = ttk.LabelFrame(self.llm_tab, text="Вопрос")
        qa_frame.pack(fill="both", expand=True, padx=6, pady=6)

        ttk.Label(qa_frame, text="Вопрос").pack(anchor="w", padx=4, pady=(4, 0))
        self.question_text = tk.Text(qa_frame, height=3, wrap="word")
        self.question_text.pack(fill="x", padx=4, pady=4)

        self.answer_button = ttk.Button(qa_frame, text="Получить ответ", command=self._run_answer)
        self.answer_button.pack(anchor="w", padx=4, pady=2)

        self.answer_text = self._build_text_area(qa_frame)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state="disabled")

    def _choose_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Выбор файла",
            filetypes=[("Данные", "*.xlsx *.xls *.csv"), ("Все файлы", "*.*")],
        )
        if file_path:
            self.file_path_var.set(file_path)

    def _choose_files(self) -> None:
        file_paths = filedialog.askopenfilenames(
            title="Выбор файлов для обучения",
            filetypes=[("Данные", "*.xlsx *.xls *.csv"), ("Все файлы", "*.*")],
        )
        if not file_paths:
            return
        existing = set(self.files_listbox.get(0, tk.END))
        for path in file_paths:
            if path and path not in existing:
                self.files_listbox.insert(tk.END, path)
                existing.add(path)
        if file_paths:
            self.file_path_var.set("")

    def _clear_files(self) -> None:
        self.files_listbox.delete(0, tk.END)

    def _collect_file_paths(self) -> list:
        list_paths = [p for p in self.files_listbox.get(0, tk.END) if p]
        if list_paths:
            return list_paths
        path_str = self.file_path_var.get().strip()
        return [path_str] if path_str else []

    def _load_data(self) -> None:
        paths = self._collect_file_paths()
        if not paths:
            messagebox.showwarning("Данные", "Укажите файл или добавьте файлы для обучения.")
            return

        def worker() -> None:
            try:
                dfs = []
                for path in paths:
                    dfs.append(load_data(Path(path)))
                if not dfs:
                    raise ValueError("Файлы для загрузки не выбраны.")
                df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
                df = normalize_columns(df)
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Ошибка загрузки", str(exc)))
                return

            self.df = df
            self.loaded_files = paths
            self.after(0, self._after_load)

        threading.Thread(target=worker, daemon=True).start()
        self._set_status("Загрузка данных...")

    def _after_load(self) -> None:
        if self.df is None:
            return
        target_options = ["(авто)"] + list(self.df.columns)
        self.target_combo.configure(values=target_options)
        self.target_var.set("(авто)")

        info_lines = []
        if self.loaded_files:
            info_lines.append("Файлы:")
            info_lines.extend(self.loaded_files)
            info_lines.append("")

        preview_rows = min(50, self.df.shape[0])
        info_lines.extend([
            f"Строк: {self.df.shape[0]}",
            f"Столбцов: {self.df.shape[1]}",
            "",
            "Колонки:",
            ", ".join(map(str, self.df.columns)),
            "",
            f"Первые {preview_rows} строк (для просмотра, обучение использует все данные):",
            self.df.head(preview_rows).to_string(index=False),
        ])
        self._set_text(self.data_text, "\n".join(info_lines))
        self._set_status("Данные загружены")

    def _train_model(self) -> None:
        if self.df is None:
            messagebox.showwarning("Модель", "Сначала загрузите данные.")
            return

        target_choice = self.target_var.get()
        if target_choice == "(авто)":
            try:
                target = pick_target(self.df, None)
            except Exception as exc:
                messagebox.showerror("Целевая колонка", str(exc))
                return
        else:
            target = target_choice

        model_label = self.model_label_var.get()
        model_name = LABEL_TO_MODEL.get(model_label, MODEL_CHOICES[0])
        test_size = self._parse_float(self.test_size_var.get(), 0.2)
        cv = self._parse_int(self.cv_var.get(), 0)
        tune = bool(self.tune_var.get())
        tune_iter = self._parse_int(self.tune_iter_var.get(), 25)
        tune_cv = self._parse_int(self.tune_cv_var.get(), 3)
        seed = self._parse_int(self.seed_var.get(), 42)

        self.last_model_name = model_name
        self.last_train_params = {
            "model": model_name,
            "test_size": test_size,
            "cv": cv,
            "tune": tune,
            "tune_iter": tune_iter,
            "tune_cv": tune_cv,
            "seed": seed,
        }

        def worker() -> None:
            try:
                result = train_and_evaluate(
                    self.df,
                    target=target,
                    model_name=model_name,
                    test_size=test_size,
                    random_state=seed,
                    cv=cv,
                    tune=tune,
                    tune_iter=tune_iter,
                    tune_cv=tune_cv,
                    run_eda_reports=False,
                )
            except Exception as exc:
                self.after(0, lambda: self._train_failed(str(exc)))
                return

            self.result = result
            self.after(0, self._after_train)

        self.train_button.configure(state="disabled")
        threading.Thread(target=worker, daemon=True).start()
        self._set_status("Обучение модели...")

    def _after_train(self) -> None:
        self.train_button.configure(state="normal")
        if self.result is None:
            return

        lines = []
        task_label = "классификация" if self.result.task == "classification" else "регрессия"
        lines.append(f"Задача: {task_label}")
        lines.append(f"Целевая колонка: {self.result.target}")
        lines.append("")

        if self.result.task == "classification":
            lines.append(f"Точность: {self.result.metrics['accuracy']:.4f}")
            lines.append(f"F1-macro: {self.result.metrics['f1_macro']:.4f}")
            if self.result.report_text:
                lines.append("")
                lines.append("Отчёт классификации:")
                lines.append(self.result.report_text)
        else:
            lines.append(f"MAE: {self.result.metrics['mae']:.4f}")
            lines.append(f"RMSE: {self.result.metrics['rmse']:.4f}")
            lines.append(f"R2: {self.result.metrics['r2']:.4f}")

        if self.result.cv_metrics:
            lines.append("")
            lines.append("Метрики кросс-валидации:")
            for key, value in self.result.cv_metrics.items():
                lines.append(f"{key}: {value:.4f}")

        if self.result.best_params:
            lines.append("")
            lines.append("Лучшие гиперпараметры:")
            for key, value in self.result.best_params.items():
                lines.append(f"{key}: {value}")

        if self.result.confusion is not None:
            lines.append("")
            lines.append("Матрица ошибок:")
            lines.append(pd.DataFrame(self.result.confusion).to_string(index=False, header=False))

        if self.result.feature_importance is not None:
            lines.append("")
            lines.append("Важность признаков (топ):")
            lines.append(self.result.feature_importance.to_string(index=False))

        lines.append("")
        lines.append("Прогноз для синтетического примера:")
        syn_proba = self._predict_proba(self.result.synthetic_sample)
        syn_labels = self._get_class_labels()
        lines.append(self._format_prediction(self.result.synthetic_prediction, syn_proba, syn_labels))

        self._set_text(self.results_text, "\n".join(lines))
        self._build_prediction_form()
        try:
            self._save_training_progress()
            self._set_status("Обучение завершено, прогресс сохранён")
        except Exception as exc:
            self._set_status("Обучение завершено")
            messagebox.showwarning("Сохранение прогресса", f"Не удалось сохранить прогресс: {exc}")

    def _train_failed(self, message: str) -> None:
        self.train_button.configure(state="normal")
        messagebox.showerror("Ошибка обучения", message)
        self._set_status("Ошибка обучения")

    def _save_training_progress(self) -> None:
        if self.result is None:
            return

        base_dir = Path("training_runs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / timestamp
        last_dir = base_dir / "last"
        run_dir.mkdir(parents=True, exist_ok=True)
        last_dir.mkdir(parents=True, exist_ok=True)

        save_reports(self.result, run_dir)
        save_reports(self.result, last_dir)

        joblib.dump(self.result.pipeline, run_dir / "model.joblib")
        joblib.dump(self.result.pipeline, last_dir / "model.joblib")

        summary = {
            "timestamp": timestamp,
            "task": self.result.task,
            "target": self.result.target,
            "metrics": self.result.metrics,
            "cv_metrics": self.result.cv_metrics,
            "best_params": self.result.best_params,
            "synthetic_prediction": self.result.synthetic_prediction,
            "model": self.last_model_name,
            "train_params": self.last_train_params,
            "feature_columns": self.result.feature_columns,
            "categorical_cols": self.result.categorical_cols,
            "numeric_cols": self.result.numeric_cols,
        }

        if self.result.feature_importance is not None:
            summary["feature_importance"] = self.result.feature_importance.to_dict(
                orient="records"
            )

        if self.loaded_files:
            summary["files"] = list(self.loaded_files)

        summary_json = json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default)
        (run_dir / "summary.json").write_text(summary_json, encoding="utf-8")
        (last_dir / "summary.json").write_text(summary_json, encoding="utf-8")

    def _build_prediction_form(self) -> None:
        for child in self.prediction_frame.winfo_children():
            child.destroy()

        self.prediction_inputs = {}
        if self.result is None or self.df is None:
            self._update_prediction_buttons()
            return

        for row, col in enumerate(self.result.feature_columns):
            ttk.Label(self.prediction_frame, text=str(col)).grid(row=row, column=0, sticky="w", pady=2)
            default = self.result.synthetic_sample[col].iloc[0]
            if col in self.result.categorical_cols:
                options, default_value = self._get_categorical_options(col, default)
                var = tk.StringVar(value=default_value)
                if options and len(options) <= 50:
                    widget = ttk.Combobox(
                        self.prediction_frame, textvariable=var, values=options, state="readonly"
                    )
                else:
                    widget = ttk.Entry(self.prediction_frame, textvariable=var)
            else:
                var = tk.StringVar(value=str(default) if default is not None else "")
                widget = ttk.Entry(self.prediction_frame, textvariable=var)

            widget.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            self.prediction_frame.columnconfigure(1, weight=1)
            self.prediction_inputs[col] = var
        self._update_prediction_buttons()

    def _predict_custom(self) -> None:
        if self.result is None:
            return

        inputs = {}
        for col in self.result.feature_columns:
            value = self.prediction_inputs[col].get().strip()
            if col in self.result.numeric_cols:
                try:
                    inputs[col] = float(value.replace(",", "."))
                except ValueError:
                    messagebox.showerror("Ошибка ввода", f"Некорректное число в поле: {col}")
                    return
            else:
                inputs[col] = value

        input_df = pd.DataFrame([inputs], columns=self.result.feature_columns)
        pred = self.result.pipeline.predict(input_df)[0]
        if self.result.task == "classification" and self.result.label_encoder is not None:
            pred = self.result.label_encoder.inverse_transform([pred])[0]

        proba = self._predict_proba(input_df)
        class_labels = self._get_class_labels()
        self.prediction_result_var.set(self._format_prediction(pred, proba, class_labels))

    def _fill_duration_bin(self) -> None:
        if self.result is None:
            return
        duration_col, duration_bin_col = self._get_duration_columns()
        if not duration_col or not duration_bin_col:
            messagebox.showwarning("Прогноз", "Колонки duration/duration_bin не найдены.")
            return
        duration_var = self.prediction_inputs.get(duration_col)
        duration_bin_var = self.prediction_inputs.get(duration_bin_col)
        if duration_var is None or duration_bin_var is None:
            messagebox.showwarning("Прогноз", "Не найдено поле ввода для duration/duration_bin.")
            return
        raw_value = duration_var.get().strip()
        try:
            hours = float(raw_value.replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка ввода", f"Некорректное число в поле: {duration_col}")
            return
        bin_label = self._duration_to_bin(hours)
        if not bin_label:
            messagebox.showwarning("Прогноз", "Не удалось рассчитать duration_bin.")
            return
        duration_bin_var.set(bin_label)
        self._set_status("duration_bin рассчитан")

    def _run_summary(self) -> None:
        if not self._ensure_llm_ready():
            return

        host = self.ollama_host
        model = self._resolve_llm_model()
        timeout = self.ollama_timeout
        context = build_llm_context(self.df, self.result, top_n=5, compact=True)
        messages = build_summary_messages(context)

        def worker() -> None:
            try:
                client = OllamaClient(host=host, timeout=timeout)
                summary = client.chat(messages, model=model, stream=False)
            except Exception as exc:
                error_text = str(exc)
                self.after(0, lambda msg=error_text: messagebox.showerror("LLM", msg))
                return

            self.after(0, lambda: self._set_text(self.summary_text, summary))
            self.after(0, lambda: self._set_status("Сводка готова"))

        threading.Thread(target=worker, daemon=True).start()
        self._set_status("Генерация сводки...")

    def _run_answer(self) -> None:
        if not self._ensure_llm_ready():
            return

        question = self.question_text.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("LLM", "Введите вопрос.")
            return

        host = self.ollama_host
        model = self._resolve_llm_model()
        timeout = self.ollama_timeout
        context = build_llm_context(self.df, self.result, top_n=5, compact=True)
        messages = build_qa_messages(context, question)

        def worker() -> None:
            try:
                client = OllamaClient(host=host, timeout=timeout)
                answer = client.chat(messages, model=model, stream=False)
            except Exception as exc:
                error_text = str(exc)
                self.after(0, lambda msg=error_text: messagebox.showerror("LLM", msg))
                return

            self.after(0, lambda: self._set_text(self.answer_text, answer))
            self.after(0, lambda: self._set_status("Ответ готов"))

        threading.Thread(target=worker, daemon=True).start()
        self._set_status("Получение ответа...")

    def _ensure_llm_ready(self) -> bool:
        if self.df is None or self.result is None:
            messagebox.showwarning("LLM", "Сначала загрузите данные и обучите модель.")
            return False
        try:
            model = self._resolve_llm_model()
        except OllamaError as exc:
            messagebox.showerror("LLM", str(exc))
            return False
        if not model:
            messagebox.showwarning("LLM", "Модель Ollama не найдена. Установите модель в Ollama.")
            return False
        return True

    def _resolve_llm_model(self) -> str:
        if self.ollama_model:
            return self.ollama_model
        if not self.ollama_models:
            client = OllamaClient(host=self.ollama_host, timeout=self.ollama_timeout)
            models = client.list_models()
            self.ollama_models = [m.name for m in models if m.name]
        if self.ollama_models:
            self.ollama_model = self.ollama_models[0]
        return self.ollama_model

    @staticmethod
    def _parse_int(value: str, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_float(value: str, default: float) -> float:
        try:
            return float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            return default

    def _get_categorical_options(self, col: str, default) -> tuple[list, str]:
        default_value = "" if pd.isna(default) else str(default)
        if self.df is not None and col in self.df.columns:
            options = self.df[col].dropna().astype(str).unique().tolist()
            options = sorted(options) if options else []
            return options, default_value

        col_lower = str(col).lower()
        if "duration_bin" in col_lower or "duratetion_bin" in col_lower:
            return DURATION_BIN_LABELS, default_value
        if "temp_bin" in col_lower:
            return TEMP_BIN_LABELS, default_value
        if "is_weekend" in col_lower:
            return WEEKEND_LABELS, default_value
        return [], default_value

    @staticmethod
    def _find_feature(columns: list, needles: list[str]) -> str | None:
        for col in columns:
            col_lower = str(col).lower()
            for needle in needles:
                if needle in col_lower:
                    return col
        return None

    def _get_duration_columns(self) -> tuple[str | None, str | None]:
        if self.result is None:
            return None, None
        duration_col = self._find_feature(
            self.result.feature_columns,
            ["duration_hours", "duration", "продолжительность"],
        )
        duration_bin_col = self._find_feature(
            self.result.feature_columns,
            ["duration_bin", "duratetion_bin", "duration bin", "duratetion bin"],
        )
        return duration_col, duration_bin_col

    @staticmethod
    def _duration_to_bin(value: float) -> str | None:
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

    def _update_prediction_buttons(self) -> None:
        if not hasattr(self, "predict_button"):
            return
        ready = self.result is not None and self.df is not None and bool(self.prediction_inputs)
        self.predict_button.configure(state="normal" if ready else "disabled")

        duration_col, duration_bin_col = self._get_duration_columns()
        duration_ready = ready and bool(duration_col and duration_bin_col)
        if hasattr(self, "duration_bin_button"):
            self.duration_bin_button.configure(
                state="normal" if duration_ready else "disabled"
            )

    def _get_class_labels(self) -> list | None:
        if self.result is None or self.result.task != "classification":
            return None
        model = self.result.pipeline.named_steps.get("model")
        if model is None or not hasattr(model, "classes_"):
            return None
        labels = list(model.classes_)
        if self.result.label_encoder is not None:
            try:
                labels = self.result.label_encoder.inverse_transform(labels)
            except Exception:
                labels = [str(label) for label in labels]
        return list(labels)

    def _predict_proba(self, input_df: pd.DataFrame):
        if self.result is None or self.result.task != "classification":
            return None
        try:
            proba = self.result.pipeline.predict_proba(input_df)[0]
        except Exception:
            return None
        return proba

    def _get_attendance_label_map(self) -> dict | None:
        if self.result is None or self.result.task != "classification":
            return None
        target = str(self.result.target).lower()
        if not any(k in target for k in ("attendance", "посещ", "популяр", "category", "катег", "score")):
            return None
        if self.result.report:
            class_keys = [
                k
                for k in self.result.report.keys()
                if k not in ("accuracy", "macro avg", "weighted avg")
            ]
            if not {"0", "1", "2"}.issubset(set(map(str, class_keys))):
                return None
        return ATTENDANCE_LABELS

    def _label_with_attendance(self, label) -> str:
        label_text = str(label)
        label_map = self._get_attendance_label_map()
        if not label_map:
            return label_text
        try:
            value = int(label)
        except (TypeError, ValueError):
            return label_text
        if value in label_map:
            return f"{label_map[value]} ({value})"
        return label_text

    def _format_prediction(self, pred, proba=None, class_labels=None) -> str:
        if self.result is None:
            return f"Прогноз: {pred}"

        if self.result.task == "classification":
            label_text = self._label_with_attendance(pred)
            lines = [f"Прогноз: {label_text}"]
            if proba is not None and class_labels is not None:
                try:
                    pairs = list(zip(class_labels, proba))
                    pairs.sort(key=lambda item: item[1], reverse=True)
                    if pairs:
                        top_label, top_prob = pairs[0]
                        lines.append(f"Уверенность: {top_prob * 100:.1f}%")
                        top_parts = []
                        for label, prob in pairs[:3]:
                            top_parts.append(f"{self._label_with_attendance(label)}: {prob * 100:.1f}%")
                        lines.append("Распределение: " + ", ".join(top_parts))
                except Exception:
                    pass
            return "\n".join(lines)

        try:
            pred_value = float(pred)
        except (TypeError, ValueError):
            return f"Прогноз: {pred}"

        target = str(self.result.target).lower()
        if any(k in target for k in ("share", "ratio", "fill", "доля", "заполняем")) and 0 <= pred_value <= 1:
            return f"Прогноз: {pred_value:.4f} (~{pred_value * 100:.1f}%)"
        if 0 <= pred_value <= 1:
            return f"Прогноз: {pred_value:.4f} (~{pred_value * 100:.1f}%)"
        return f"Прогноз: {pred_value:.4f}"


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


if __name__ == "__main__":
    app = DesktopApp()
    app.mainloop()
