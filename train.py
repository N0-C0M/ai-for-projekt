#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from llm_client import build_summary_messages, chat
from ml_pipeline import (
    MODEL_CHOICES,
    build_llm_context,
    compare_models_cv,
    load_data,
    normalize_columns,
    pick_target,
    save_reports,
    train_and_evaluate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обучение модели для прогноза посещаемости/популярности событий."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ai_event_dataset.xlsx",
        help="Путь к датасету (.xlsx/.xls/.csv).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Имя целевой колонки. Если не указано, определяется автоматически.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=MODEL_CHOICES,
        help="Тип модели: rf, gb, hgb, linear.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Доля тестовой выборки.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help="Если >0, запустить кросс-валидацию с заданным числом фолдов.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Запустить случайный поиск гиперпараметров.",
    )
    parser.add_argument(
        "--tune-iter",
        type=int,
        default=25,
        help="Число итераций случайного поиска.",
    )
    parser.add_argument(
        "--tune-cv",
        type=int,
        default=3,
        help="Число фолдов для тюнинга гиперпараметров.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Случайное зерно.",
    )
    parser.add_argument(
        "--no-eda",
        action="store_true",
        help="Пропустить EDA-отчёт и графики.",
    )
    parser.add_argument(
        "--eda-dir",
        type=str,
        default="reports",
        help="Папка для сохранения EDA-артефактов.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Папка для сохранения метрик и важности признаков.",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Не сохранять метрики и важность признаков.",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Путь для сохранения обученного pipeline (joblib).",
    )
    parser.add_argument(
        "--llm-summary",
        action="store_true",
        help="Сформировать локальную LLM-сводку результатов.",
    )
    parser.add_argument(
        "--llm-question",
        type=str,
        default=None,
        help="Дополнительный вопрос для локальной LLM.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        help="Провайдер LLM: ollama или openai.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Имя модели LLM.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="Базовый URL LLM (для ollama/OpenAI-совместимых серверов).",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="Опциональный API-ключ для OpenAI-совместимых серверов.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=60,
        help="Таймаут LLM-запроса (сек).",
    )
    parser.add_argument(
        "--llm-output",
        type=str,
        default="reports/llm_summary.txt",
        help="Путь для сохранения LLM-сводки.",
    )
    parser.add_argument(
        "--llm-top-features",
        type=int,
        default=10,
        help="Сколько топ-признаков включать в LLM-контекст.",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    df = load_data(data_path)
    df = normalize_columns(df)
    target = pick_target(df, args.target)

    result = train_and_evaluate(
        df,
        target=target,
        model_name=args.model,
        test_size=args.test_size,
        random_state=args.seed,
        cv=args.cv,
        tune=args.tune,
        tune_iter=args.tune_iter,
        tune_cv=args.tune_cv,
        run_eda_reports=not args.no_eda,
        eda_dir=Path(args.eda_dir),
    )

    print(f"Задача: {result.task}")
    if result.task == "classification":
        print(f"Точность: {result.metrics['accuracy']:.4f}")
        print(f"F1-macro: {result.metrics['f1_macro']:.4f}")
        if result.report_text:
            print("Отчёт классификации:")
            print(result.report_text)
    else:
        print(f"MAE: {result.metrics['mae']:.4f}")
        print(f"RMSE: {result.metrics['rmse']:.4f}")
        print(f"R2: {result.metrics['r2']:.4f}")

    if result.cv_metrics:
        print("Метрики кросс-валидации:")
        for key, value in result.cv_metrics.items():
            print(f"{key}: {value:.4f}")

    if result.best_params:
        print("Лучшие гиперпараметры:")
        for key, value in result.best_params.items():
            print(f"{key}: {value}")

    print("Прогноз для синтетического примера:", result.synthetic_prediction)

    if not args.no_reports:
        save_reports(result, Path(args.report_dir))

        if args.cv and args.cv > 1:
            compare_models = [args.model] + [m for m in MODEL_CHOICES if m != args.model]
            compare_models = compare_models[:3]
            try:
                comparison = compare_models_cv(
                    df,
                    target=target,
                    model_names=compare_models,
                    cv=args.cv,
                    random_state=args.seed,
                )
                if not comparison.empty:
                    report_dir = Path(args.report_dir)
                    report_dir.mkdir(parents=True, exist_ok=True)
                    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
                    comparison.to_json(
                        report_dir / "model_comparison.json",
                        orient="records",
                        force_ascii=False,
                        indent=2,
                    )
                    print(f"Сравнение моделей сохранено в {report_dir / 'model_comparison.csv'}")
            except Exception as exc:
                print(f"Не удалось выполнить сравнение моделей: {exc}")

    if args.save_model:
        try:
            import joblib
        except Exception as exc:
            raise SystemExit(f"joblib not available: {exc}")
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.pipeline, save_path)
        print(f"Модель сохранена в: {save_path}")

    if args.llm_summary:
        top_n = min(args.llm_top_features, 5)
        context = build_llm_context(df, result, top_n=top_n, compact=True)
        messages = build_summary_messages(context, question=args.llm_question)
        try:
            summary = chat(
                messages,
                provider=args.llm_provider,
                model=args.llm_model,
                base_url=args.llm_base_url,
                api_key=args.llm_api_key,
                timeout=args.llm_timeout,
            )
            print("LLM-сводка:")
            print(summary)
            output_path = Path(args.llm_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(summary, encoding="utf-8")
        except Exception as exc:
            print(f"Ошибка LLM-сводки: {exc}")


if __name__ == "__main__":
    main()
