from __future__ import annotations

import argparse
from pathlib import Path

from .config import MODELS_DIR
from .io_utils import require_package
from .preprocessing import clean_lyrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediz genero para uma nova letra.")
    parser.add_argument("--model", default="baseline_tfidf_logreg", help="Nome do modelo ou caminho .joblib.")
    parser.add_argument("--text", help="Texto da letra.")
    parser.add_argument("--file", type=Path, help="Arquivo com a letra.")
    return parser.parse_args()


def resolve_model_path(model: str) -> Path:
    path = Path(model)
    if path.exists():
        return path
    if model == "best":
        candidates = sorted(MODELS_DIR.glob("baseline_tfidf_*.joblib"))
        if not candidates:
            raise SystemExit("Nenhum modelo encontrado em models/. Treine um modelo primeiro.")
        return candidates[0]
    if not model.endswith(".joblib"):
        model = f"{model}.joblib"
    path = MODELS_DIR / model
    if not path.exists():
        raise SystemExit(f"Modelo nao encontrado: {path}")
    return path


def main() -> None:
    joblib = require_package("joblib")

    args = parse_args()
    if not args.text and not args.file:
        raise SystemExit("Informe --text ou --file.")
    text = args.text if args.text else args.file.read_text(encoding="utf-8")
    text = clean_lyrics(text, lowercase=False)
    model_path = resolve_model_path(args.model)
    model = joblib.load(model_path)
    predicted = model.predict([text])[0]
    print(f"Modelo: {model_path}")
    print(f"Genero previsto: {predicted}")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        classes = list(model.classes_)
        print("Probabilidades:")
        for label, prob in sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True):
            print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
