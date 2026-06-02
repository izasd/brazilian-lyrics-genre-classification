from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_PROCESSED_DATA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina Transformer/BERTimbau para classificacao textual.")
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_DATA)
    parser.add_argument("--model-name", default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    raise SystemExit(
        "Modulo BERTimbau preparado como etapa opcional. Nesta primeira entrega, o pipeline executavel "
        "prioriza o baseline TF-IDF. Implemente fine-tuning com Hugging Face Transformers quando houver "
        "ambiente com memoria/GPU suficiente."
    )


if __name__ == "__main__":
    main()

