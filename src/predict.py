from __future__ import annotations

import argparse
from pathlib import Path

from .config import ANALYSIS_DIR, MODELS_DIR
from .io_utils import read_json, require_package
from .neural_models import build_torch_model, encode_text
from .preprocessing import clean_lyrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediz genero para uma nova letra.")
    parser.add_argument(
        "--model",
        default="best",
        help="Nome do modelo, caminho do artefato ou 'best' para usar a recomendacao final.",
    )
    parser.add_argument("--text", help="Texto da letra.")
    parser.add_argument("--file", type=Path, help="Arquivo com a letra.")
    return parser.parse_args()


def resolve_model_path(model: str) -> Path:
    path = Path(model)
    if path.exists():
        return path
    if model == "best":
        recommendation_path = ANALYSIS_DIR / "recomendacao_modelo_final.json"
        if recommendation_path.exists():
            recommended = read_json(recommendation_path)["recommended_primary_model"]
            directory = MODELS_DIR / recommended
            if directory.is_dir():
                return directory
            for extension in (".joblib", ".pt"):
                candidate = MODELS_DIR / f"{recommended}{extension}"
                if candidate.exists():
                    return candidate
        raise SystemExit(
            "Recomendacao final nao encontrada. Execute `python -m src.final_comparison`."
        )
    if Path(model).suffix:
        candidate = MODELS_DIR / model
        if candidate.exists():
            return candidate
    else:
        candidate = MODELS_DIR / model
        if candidate.is_dir():
            return candidate
        for extension in (".joblib", ".pt"):
            candidate = MODELS_DIR / f"{model}{extension}"
            if candidate.exists():
                return candidate
    raise SystemExit(f"Modelo nao encontrado: {model}")


def predict_pytorch(model_path: Path, text: str) -> None:
    torch = require_package("torch")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    vocabulary_path = MODELS_DIR / f"{model_path.stem}_vocabulary.json"
    labels_path = MODELS_DIR / f"{model_path.stem}_labels.json"
    if not vocabulary_path.exists() or not labels_path.exists():
        raise SystemExit("Arquivos de vocabulario ou rotulos do modelo neural nao encontrados.")

    vocabulary = read_json(vocabulary_path)["vocabulary"]
    labels = read_json(labels_path)["labels"]
    model = build_torch_model(
        checkpoint["model_kind"],
        vocab_size=checkpoint["vocab_size"],
        num_classes=checkpoint["num_classes"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    encoded = encode_text(text, vocabulary, checkpoint["max_len"])
    inputs = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0].tolist()
    predicted_id = max(range(len(probabilities)), key=probabilities.__getitem__)

    print(f"Modelo: {model_path}")
    print(f"Genero previsto: {labels[predicted_id]}")
    print("Probabilidades:")
    for label, probability in sorted(
        zip(labels, probabilities),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {label}: {probability:.4f}")


def predict_transformer(model_path: Path, text: str) -> None:
    torch = require_package("torch")
    transformers = require_package("transformers")
    config_path = model_path / "training_config.json"
    if not config_path.exists():
        raise SystemExit("Configuracao de treinamento do Transformer nao encontrada.")

    training_config = read_json(config_path)
    labels = training_config["labels"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=training_config["max_len"],
        return_tensors="pt",
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(**encoded).logits, dim=1)[0].tolist()
    predicted_id = max(range(len(probabilities)), key=probabilities.__getitem__)

    print(f"Modelo: {model_path}")
    print(f"Genero previsto: {labels[predicted_id]}")
    print("Probabilidades:")
    for label, probability in sorted(
        zip(labels, probabilities),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {label}: {probability:.4f}")


def main() -> None:
    args = parse_args()
    if not args.text and not args.file:
        raise SystemExit("Informe --text ou --file.")
    text = args.text if args.text else args.file.read_text(encoding="utf-8")
    text = clean_lyrics(text, lowercase=False)
    model_path = resolve_model_path(args.model)
    if model_path.is_dir():
        predict_transformer(model_path, text)
        return
    if model_path.suffix == ".pt":
        predict_pytorch(model_path, text)
        return

    joblib = require_package("joblib")
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
