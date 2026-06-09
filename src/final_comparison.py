from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .config import ANALYSIS_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, ensure_project_dirs
from .io_utils import read_json, require_package, write_json


MODEL_SPECS = [
    {
        "model": "baseline_tfidf_logreg_word_1_2_final",
        "label": "TF-IDF palavras 1-2",
        "family": "classico",
        "cv_f1_macro": 0.2668,
        "cv_accuracy": 0.2778,
    },
    {
        "model": "baseline_tfidf_logreg_word_char_final",
        "label": "TF-IDF palavras + caracteres",
        "family": "classico",
        "cv_f1_macro": 0.2637,
        "cv_accuracy": 0.2705,
    },
    {
        "model": "cnn_pytorch",
        "label": "CNN",
        "family": "neural",
        "cv_f1_macro": None,
        "cv_accuracy": None,
    },
    {
        "model": "lstm_pytorch",
        "label": "BiLSTM aleatoria",
        "family": "neural",
        "cv_f1_macro": None,
        "cv_accuracy": None,
    },
    {
        "model": "lstm_pytorch_nilc_glove_100d_tuned",
        "label": "BiLSTM + GloVe ajustada",
        "family": "neural",
        "cv_f1_macro": None,
        "cv_accuracy": None,
    },
    {
        "model": "bertimbau_base_frozen",
        "label": "BERTimbau congelado",
        "family": "transformer",
        "cv_f1_macro": None,
        "cv_accuracy": None,
    },
]


def artifact_size_bytes(model_name: str, metrics: dict[str, Any]) -> int:
    if "model_size_bytes" in metrics:
        return int(metrics["model_size_bytes"])

    model_path = Path(metrics["model_path"])
    if model_path.is_dir():
        return sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file())

    total = model_path.stat().st_size
    for suffix in ("_vocabulary.json", "_labels.json"):
        companion = MODELS_DIR / f"{model_name}{suffix}"
        if companion.exists():
            total += companion.stat().st_size
    return total


def load_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in MODEL_SPECS:
        metrics_path = METRICS_DIR / f"{spec['model']}_metrics.json"
        if not metrics_path.exists():
            raise SystemExit(
                f"Metricas ausentes para {spec['model']}: {metrics_path}. "
                "Execute os treinamentos finalistas antes da consolidacao."
            )
        metrics = read_json(metrics_path)
        size_bytes = artifact_size_bytes(spec["model"], metrics)
        rows.append(
            {
                **spec,
                "accuracy": float(metrics["accuracy"]),
                "precision_macro": float(metrics["precision_macro"]),
                "recall_macro": float(metrics["recall_macro"]),
                "f1_macro": float(metrics["f1_macro"]),
                "f1_weighted": float(metrics["f1_weighted"]),
                "training_seconds": float(metrics["training_seconds"]),
                "training_minutes": float(metrics["training_seconds"]) / 60,
                "artifact_size_bytes": size_bytes,
                "artifact_size_mb": size_bytes / (1024 * 1024),
                "device": metrics.get("device", "cpu"),
                "test_size": int(metrics["test_size"]),
            }
        )

    best_f1 = max(row["f1_macro"] for row in rows)
    fastest = min(row["training_seconds"] for row in rows)
    smallest = min(row["artifact_size_bytes"] for row in rows)
    for row in rows:
        row["f1_gap_to_best"] = best_f1 - row["f1_macro"]
        row["time_vs_fastest"] = row["training_seconds"] / fastest
        row["size_vs_smallest"] = row["artifact_size_bytes"] / smallest
    return sorted(rows, key=lambda row: row["f1_macro"], reverse=True)


def save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "label",
        "family",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "cv_accuracy",
        "cv_f1_macro",
        "training_seconds",
        "training_minutes",
        "artifact_size_bytes",
        "artifact_size_mb",
        "f1_gap_to_best",
        "time_vs_fastest",
        "size_vs_smallest",
        "device",
        "test_size",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(rows: list[dict[str, Any]], output_path: Path) -> None:
    require_package("matplotlib")
    import matplotlib.pyplot as plt

    ordered = list(reversed(rows))
    labels = [row["label"] for row in ordered]
    colors = [
        {"classico": "#2b6cb0", "neural": "#dd6b20", "transformer": "#6b46c1"}[
            row["family"]
        ]
        for row in ordered
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    axes[0].barh(labels, [row["f1_macro"] for row in ordered], color=colors)
    axes[0].set_title("F1 macro no teste")
    axes[0].set_xlim(0, 0.32)
    for index, row in enumerate(ordered):
        axes[0].text(row["f1_macro"] + 0.004, index, f"{row['f1_macro']:.4f}", va="center")

    axes[1].barh(labels, [row["training_minutes"] for row in ordered], color=colors)
    axes[1].set_title("Tempo de treinamento (min, escala log)")
    axes[1].set_xscale("log")
    axes[1].set_xlim(
        min(row["training_minutes"] for row in ordered) / 1.3,
        max(row["training_minutes"] for row in ordered) * 1.5,
    )
    for index, row in enumerate(ordered):
        axes[1].text(
            row["training_minutes"] * 1.08,
            index,
            f"{row['training_minutes']:.1f}",
            va="center",
        )

    axes[2].barh(labels, [row["artifact_size_mb"] for row in ordered], color=colors)
    axes[2].set_title("Tamanho para inferencia (MB, escala log)")
    axes[2].set_xscale("log")
    axes[2].set_xlim(
        min(row["artifact_size_mb"] for row in ordered) / 1.3,
        max(row["artifact_size_mb"] for row in ordered) * 1.5,
    )
    for index, row in enumerate(ordered):
        axes[2].text(
            row["artifact_size_mb"] * 1.08,
            index,
            f"{row['artifact_size_mb']:.1f}",
            va="center",
        )

    fig.suptitle("Comparacao final dos modelos")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["model"]: row for row in rows}
    principal = by_name["baseline_tfidf_logreg_word_1_2_final"]
    best_test = max(rows, key=lambda row: row["f1_macro"])
    best_neural = max(
        (row for row in rows if row["family"] != "classico"),
        key=lambda row: row["f1_macro"],
    )
    return {
        "recommended_primary_model": principal["model"],
        "recommended_primary_label": principal["label"],
        "reason": (
            "Maior F1 macro medio na validacao cruzada entre os finalistas, "
            "tempo baixo, artefato pequeno e resultado de teste proximo ao melhor."
        ),
        "best_external_test_model": best_test["model"],
        "best_external_test_label": best_test["label"],
        "best_external_test_f1_macro": best_test["f1_macro"],
        "best_neural_model": best_neural["model"],
        "best_neural_label": best_neural["label"],
        "best_neural_f1_macro": best_neural["f1_macro"],
        "primary_test_f1_macro": principal["f1_macro"],
        "primary_cv_f1_macro": principal["cv_f1_macro"],
        "methodological_note": (
            "O modelo combinado lidera no teste externo, mas o TF-IDF de palavras "
            "lidera na media da validacao cruzada e e recomendado como modelo principal."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolida a comparacao final dos modelos.")
    parser.add_argument(
        "--output",
        type=Path,
        default=METRICS_DIR / "comparacao_final_modelos.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    rows = load_rows()
    save_csv(rows, args.output)
    save_figure(rows, FIGURES_DIR / "comparacao_final_modelos.png")
    recommendation = build_recommendation(rows)
    write_json(ANALYSIS_DIR / "recomendacao_modelo_final.json", recommendation)

    print(args.output.read_text(encoding="utf-8"))
    print(f"Modelo principal recomendado: {recommendation['recommended_primary_label']}")
    print(f"Melhor resultado no teste: {recommendation['best_external_test_label']}")
    print(f"Melhor modelo neural: {recommendation['best_neural_label']}")


if __name__ == "__main__":
    main()
