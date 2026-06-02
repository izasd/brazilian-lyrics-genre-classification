from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .config import METRICS_DIR, RESULTS_DIR, ensure_project_dirs
from .io_utils import require_package, write_json


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    require_package("sklearn", "scikit-learn")
    from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def save_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    output_path: Path,
    *,
    title: str,
) -> None:
    require_package("matplotlib")
    require_package("sklearn", "scikit-learn")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), max(6, len(labels) * 0.65)))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def update_model_comparison(model_name: str, metrics: dict[str, Any], output_path: Path | None = None) -> None:
    ensure_project_dirs()
    output_path = output_path or METRICS_DIR / "comparacao_modelos.csv"
    row = {
        "model": model_name,
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_weighted": metrics.get("f1_weighted"),
    }
    rows: list[dict[str, Any]] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows = [existing for existing in rows if existing.get("model") != model_name]
    rows.append(row)
    rows.sort(key=lambda item: str(item.get("model", "")))
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_all() -> None:
    comparison = METRICS_DIR / "comparacao_modelos.csv"
    if not comparison.exists():
        raise SystemExit("Nenhuma comparacao encontrada. Treine ao menos um modelo primeiro.")
    print(comparison.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Utilitarios de avaliacao.")
    parser.add_argument("--all", action="store_true", help="Mostra a comparacao de modelos salva.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all:
        summarize_all()
    else:
        raise SystemExit("Use --all para mostrar a comparacao de modelos.")


if __name__ == "__main__":
    main()
