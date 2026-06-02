from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_PROCESSED_DATA, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from .neural_models import train_keras_text_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina CNN textual em Keras.")
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_DATA)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-words", type=int, default=50000)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_keras_text_model(
        args.data,
        model_kind="cnn",
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_words=args.max_words,
        max_len=args.max_len,
        embedding_dim=args.embedding_dim,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Modelo salvo em: {metrics['model_path']}")
    print(f"Acuracia: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
