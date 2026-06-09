from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_PROCESSED_DATA, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from .neural_models import train_torch_text_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina LSTM textual bidirecional em PyTorch.")
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_DATA)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-words", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_torch_text_model(
        args.data,
        model_kind="lstm",
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_words=args.max_words,
        max_len=args.max_len,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device_name=args.device,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Modelo salvo em: {metrics['model_path']}")
    print(f"Acuracia: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
