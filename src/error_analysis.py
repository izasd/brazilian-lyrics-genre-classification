from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .config import (
    ANALYSIS_DIR,
    DEFAULT_PROCESSED_DATA,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    FIGURES_DIR,
    MODELS_DIR,
    ensure_project_dirs,
)
from .io_utils import require_package, safe_slug, write_json


def resolve_model_path(model: str) -> Path:
    path = Path(model)
    if path.exists():
        return path
    if not model.endswith(".joblib"):
        model = f"{model}.joblib"
    path = MODELS_DIR / model
    if not path.exists():
        raise SystemExit(f"Modelo nao encontrado: {path}")
    return path


def load_test_split(data_path: Path, test_size: float, random_state: int):
    pd = require_package("pandas")
    require_package("sklearn", "scikit-learn")
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)
    missing = {"lyrics", "genre"}.difference(df.columns)
    if missing:
        raise SystemExit(f"CSV precisa conter lyrics e genre. Faltando: {sorted(missing)}")
    df = df.dropna(subset=["lyrics", "genre"]).copy()
    df["lyrics"] = df["lyrics"].astype(str)
    df["genre"] = df["genre"].astype(str)
    _, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["genre"],
        random_state=random_state,
    )
    return test_df


def build_error_tables(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    require_package("sklearn", "scikit-learn")
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    total_errors = int(matrix.sum() - matrix.trace())

    directed: list[dict[str, Any]] = []
    class_summary: list[dict[str, Any]] = []
    for true_idx, true_genre in enumerate(labels):
        row_total = int(matrix[true_idx].sum())
        correct = int(matrix[true_idx, true_idx])
        errors = row_total - correct

        wrong_predictions = [
            (labels[pred_idx], int(matrix[true_idx, pred_idx]))
            for pred_idx in range(len(labels))
            if pred_idx != true_idx and matrix[true_idx, pred_idx] > 0
        ]
        wrong_predictions.sort(key=lambda item: (-item[1], item[0]))
        top_predicted_genre, top_count = wrong_predictions[0] if wrong_predictions else ("", 0)

        class_summary.append(
            {
                "genre": true_genre,
                "total": row_total,
                "correct": correct,
                "errors": errors,
                "recall": correct / row_total if row_total else 0.0,
                "most_confused_with": top_predicted_genre,
                "most_confused_count": top_count,
                "most_confused_rate": top_count / row_total if row_total else 0.0,
            }
        )

        for pred_idx, predicted_genre in enumerate(labels):
            if pred_idx == true_idx:
                continue
            count = int(matrix[true_idx, pred_idx])
            if count == 0:
                continue
            directed.append(
                {
                    "true_genre": true_genre,
                    "predicted_genre": predicted_genre,
                    "count": count,
                    "rate_within_true_genre": count / row_total if row_total else 0.0,
                    "share_of_all_errors": count / total_errors if total_errors else 0.0,
                }
            )

    directed.sort(
        key=lambda row: (
            -int(row["count"]),
            str(row["true_genre"]),
            str(row["predicted_genre"]),
        )
    )

    pairs: list[dict[str, Any]] = []
    for left_idx, left_genre in enumerate(labels):
        for right_idx in range(left_idx + 1, len(labels)):
            right_genre = labels[right_idx]
            left_to_right = int(matrix[left_idx, right_idx])
            right_to_left = int(matrix[right_idx, left_idx])
            total = left_to_right + right_to_left
            if total == 0:
                continue
            pairs.append(
                {
                    "genre_a": left_genre,
                    "genre_b": right_genre,
                    "a_predicted_as_b": left_to_right,
                    "b_predicted_as_a": right_to_left,
                    "total_confusions": total,
                    "share_of_all_errors": total / total_errors if total_errors else 0.0,
                }
            )
    pairs.sort(
        key=lambda row: (
            -int(row["total_confusions"]),
            str(row["genre_a"]),
            str(row["genre_b"]),
        )
    )

    return {
        "matrix": matrix,
        "total_samples": int(matrix.sum()),
        "correct_predictions": int(matrix.trace()),
        "total_errors": total_errors,
        "directed_confusions": directed,
        "confusion_pairs": pairs,
        "class_summary": class_summary,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_normalized_matrix(matrix, labels: list[str], output_path: Path, title: str) -> None:
    require_package("matplotlib")
    require_package("numpy")
    import matplotlib.pyplot as plt
    import numpy as np

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Proporcao")
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Genero previsto",
        ylabel="Genero real",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = normalized.max() / 2 if normalized.size else 0
    for row_idx in range(normalized.shape[0]):
        for col_idx in range(normalized.shape[1]):
            value = normalized[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.0%}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze_errors(
    data_path: Path,
    model_path: Path,
    *,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    joblib = require_package("joblib")
    ensure_project_dirs()

    test_df = load_test_split(data_path, test_size, random_state)
    model = joblib.load(model_path)
    predictions = list(model.predict(test_df["lyrics"]))
    y_true = list(test_df["genre"])
    labels = sorted(set(y_true) | set(predictions))
    tables = build_error_tables(y_true, predictions, labels)

    model_slug = safe_slug(model_path.stem)
    directed_path = ANALYSIS_DIR / f"confusoes_direcionais_{model_slug}.csv"
    pairs_path = ANALYSIS_DIR / f"pares_confundidos_{model_slug}.csv"
    summary_path = ANALYSIS_DIR / f"resumo_erros_por_genero_{model_slug}.csv"
    report_path = ANALYSIS_DIR / f"relatorio_erros_{model_slug}.json"
    figure_path = FIGURES_DIR / f"confusion_matrix_normalized_{model_slug}.png"

    write_csv(directed_path, tables["directed_confusions"])
    write_csv(pairs_path, tables["confusion_pairs"])
    write_csv(summary_path, tables["class_summary"])
    save_normalized_matrix(
        tables["matrix"],
        labels,
        figure_path,
        title=f"Matriz de confusao normalizada - {model_path.stem}",
    )

    report = {
        "model": model_path.stem,
        "model_path": str(model_path),
        "data_path": str(data_path),
        "test_size": test_size,
        "random_state": random_state,
        "total_samples": tables["total_samples"],
        "correct_predictions": tables["correct_predictions"],
        "total_errors": tables["total_errors"],
        "accuracy": tables["correct_predictions"] / tables["total_samples"],
        "directed_confusions": tables["directed_confusions"],
        "confusion_pairs": tables["confusion_pairs"],
        "top_directed_confusions": tables["directed_confusions"][:10],
        "top_confusion_pairs": tables["confusion_pairs"][:10],
        "class_summary": tables["class_summary"],
        "outputs": {
            "directed_confusions": str(directed_path),
            "confusion_pairs": str(pairs_path),
            "class_summary": str(summary_path),
            "normalized_matrix": str(figure_path),
            "report": str(report_path),
        },
    }
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisa os erros e pares de generos confundidos.")
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_DATA)
    parser.add_argument("--model", default="baseline_tfidf_logreg_word_1_2_cv5")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    report = analyze_errors(
        args.data,
        model_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"Modelo analisado: {report['model']}")
    print(f"Total de erros: {report['total_errors']} de {report['total_samples']}")
    print("Principais pares confundidos:")
    for row in report["confusion_pairs"][: args.top]:
        print(
            f"  {row['genre_a']} <-> {row['genre_b']}: "
            f"{row['total_confusions']} erros "
            f"({row['share_of_all_errors']:.2%} do total)"
        )
    print("Saidas:")
    for name, path in report["outputs"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
