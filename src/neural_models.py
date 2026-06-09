from __future__ import annotations

import copy
import re
import time
from collections import Counter
from pathlib import Path
from typing import Literal

from .config import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    ensure_project_dirs,
)
from .evaluate import compute_metrics, save_confusion_matrix, update_model_comparison
from .io_utils import require_package, write_json


TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"


def load_dataframe(path: Path):
    pd = require_package("pandas")

    df = pd.read_csv(path)
    missing = {"lyrics", "genre"}.difference(df.columns)
    if missing:
        raise SystemExit(f"CSV precisa conter colunas lyrics e genre. Faltando: {sorted(missing)}")
    df = df.dropna(subset=["lyrics", "genre"]).copy()
    df["lyrics"] = df["lyrics"].astype(str)
    df["genre"] = df["genre"].astype(str)
    return df


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocabulary(texts: list[str], max_words: int) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    vocabulary = {PAD_TOKEN: 0, OOV_TOKEN: 1}
    for token, _ in counts.most_common(max(0, max_words - len(vocabulary))):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(text: str, vocabulary: dict[str, int], max_len: int) -> list[int]:
    oov_id = vocabulary[OOV_TOKEN]
    encoded = [vocabulary.get(token, oov_id) for token in tokenize(text)[:max_len]]
    if len(encoded) < max_len:
        encoded.extend([vocabulary[PAD_TOKEN]] * (max_len - len(encoded)))
    return encoded


def resolve_device(requested: str):
    torch = require_package("torch")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA foi solicitada, mas nao esta disponivel neste computador.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def build_torch_model(
    model_kind: Literal["cnn", "lstm"],
    *,
    vocab_size: int,
    num_classes: int,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
):
    torch = require_package("torch")
    nn = torch.nn

    if model_kind == "cnn":

        class TextCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.convolutions = nn.ModuleList(
                    [nn.Conv1d(embedding_dim, hidden_dim, kernel_size=size) for size in (3, 4, 5)]
                )
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(hidden_dim * len(self.convolutions), num_classes)

            def forward(self, inputs):
                embedded = self.embedding(inputs).transpose(1, 2)
                pooled = [
                    torch.relu(convolution(embedded)).amax(dim=2)
                    for convolution in self.convolutions
                ]
                features = torch.cat(pooled, dim=1)
                return self.classifier(self.dropout(features))

        return TextCNN()

    if model_kind == "lstm":

        class TextBiLSTM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.lstm = nn.LSTM(
                    embedding_dim,
                    hidden_dim,
                    batch_first=True,
                    bidirectional=True,
                )
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(hidden_dim * 2, num_classes)

            def forward(self, inputs):
                embedded = self.embedding(inputs)
                _, (hidden, _) = self.lstm(embedded)
                features = torch.cat((hidden[-2], hidden[-1]), dim=1)
                return self.classifier(self.dropout(features))

        return TextBiLSTM()

    raise SystemExit(f"Modelo neural invalido: {model_kind}")


def make_data_loader(features, targets, *, batch_size: int, shuffle: bool, random_state: int):
    torch = require_package("torch")
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(features, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )
    generator = torch.Generator()
    generator.manual_seed(random_state)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def evaluate_model(model, data_loader, criterion, device):
    torch = require_package("torch")
    model.eval()
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            total_loss += float(criterion(logits, labels).item()) * len(labels)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    average_loss = total_loss / max(1, len(targets))
    accuracy = sum(pred == true for pred, true in zip(predictions, targets)) / max(1, len(targets))
    return average_loss, accuracy, targets, predictions


def save_training_history(history: list[dict[str, float]], output_path: Path, title: str) -> None:
    require_package("matplotlib")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Treino")
    axes[0].plot(epochs, [row["valid_loss"] for row in history], marker="o", label="Validacao")
    axes[0].set_title("Perda")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Treino")
    axes[1].plot(epochs, [row["valid_accuracy"] for row in history], marker="o", label="Validacao")
    axes[1].set_title("Acuracia")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def train_torch_text_model(
    data_path: Path,
    *,
    model_kind: Literal["cnn", "lstm"],
    epochs: int,
    batch_size: int,
    max_words: int,
    max_len: int,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    patience: int,
    device_name: str,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    np = require_package("numpy")
    torch = require_package("torch")
    require_package("sklearn", "scikit-learn")
    from sklearn.model_selection import train_test_split

    ensure_project_dirs()
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    device = resolve_device(device_name)
    df = load_dataframe(data_path)
    labels = sorted(df["genre"].unique())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    targets = df["genre"].map(label_to_id).to_numpy()

    train_texts, test_texts, y_train, y_test = train_test_split(
        df["lyrics"].tolist(),
        targets,
        test_size=test_size,
        stratify=targets,
        random_state=random_state,
    )
    train_texts, valid_texts, y_train, y_valid = train_test_split(
        train_texts,
        y_train,
        test_size=0.1,
        stratify=y_train,
        random_state=random_state,
    )

    vocabulary = build_vocabulary(train_texts, max_words)
    x_train = [encode_text(text, vocabulary, max_len) for text in train_texts]
    x_valid = [encode_text(text, vocabulary, max_len) for text in valid_texts]
    x_test = [encode_text(text, vocabulary, max_len) for text in test_texts]

    train_loader = make_data_loader(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        random_state=random_state,
    )
    valid_loader = make_data_loader(
        x_valid,
        y_valid,
        batch_size=batch_size,
        shuffle=False,
        random_state=random_state,
    )
    test_loader = make_data_loader(
        x_test,
        y_test,
        batch_size=batch_size,
        shuffle=False,
        random_state=random_state,
    )

    model = build_torch_model(
        model_kind,
        vocab_size=len(vocabulary),
        num_classes=len(labels),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    started_at = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_examples = 0

        for inputs, batch_targets in train_loader:
            inputs = inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            train_loss_total += float(loss.item()) * len(batch_targets)
            train_correct += int((logits.argmax(dim=1) == batch_targets).sum().item())
            train_examples += len(batch_targets)

        valid_loss, valid_accuracy, _, _ = evaluate_model(model, valid_loader, criterion, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss_total / max(1, train_examples),
            "train_accuracy": train_correct / max(1, train_examples),
            "valid_loss": valid_loss,
            "valid_accuracy": valid_accuracy,
        }
        history.append(row)
        print(
            f"Epoca {epoch:02d}/{epochs} - "
            f"loss: {row['train_loss']:.4f} - acc: {row['train_accuracy']:.4f} - "
            f"val_loss: {valid_loss:.4f} - val_acc: {valid_accuracy:.4f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping apos {epoch} epocas.")
                break

    training_seconds = time.perf_counter() - started_at
    model.load_state_dict(best_state)
    _, _, test_target_ids, test_pred_ids = evaluate_model(model, test_loader, criterion, device)
    y_true = [id_to_label[idx] for idx in test_target_ids]
    y_pred = [id_to_label[idx] for idx in test_pred_ids]
    metrics = compute_metrics(y_true, y_pred)

    model_name = f"{model_kind}_pytorch"
    metrics.update(
        {
            "model": model_name,
            "framework": "pytorch",
            "torch_version": torch.__version__,
            "device": str(device),
            "train_size": len(x_train),
            "valid_size": len(x_valid),
            "test_size": len(x_test),
            "vocab_size": len(vocabulary),
            "max_words": max_words,
            "max_len": max_len,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "epochs_trained": len(history),
            "patience": patience,
            "training_seconds": training_seconds,
            "best_valid_loss": best_valid_loss,
            "history": history,
            "prediction_distribution": dict(Counter(y_pred)),
            "class_distribution": {
                str(key): int(value) for key, value in df["genre"].value_counts().to_dict().items()
            },
        }
    )

    model_path = MODELS_DIR / f"{model_name}.pt"
    vocabulary_path = MODELS_DIR / f"{model_name}_vocabulary.json"
    labels_path = MODELS_DIR / f"{model_name}_labels.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kind": model_kind,
            "vocab_size": len(vocabulary),
            "num_classes": len(labels),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "max_len": max_len,
        },
        model_path,
    )
    write_json(vocabulary_path, {"vocabulary": vocabulary})
    write_json(labels_path, {"labels": labels, "label_to_id": label_to_id})
    metrics["model_path"] = str(model_path)
    metrics["vocabulary_path"] = str(vocabulary_path)
    metrics["labels_path"] = str(labels_path)

    write_json(METRICS_DIR / f"{model_name}_metrics.json", metrics)
    save_confusion_matrix(
        y_true,
        y_pred,
        labels,
        FIGURES_DIR / f"confusion_matrix_{model_name}.png",
        title=f"Matriz de confusao - {model_kind.upper()} (PyTorch)",
    )
    save_training_history(
        history,
        FIGURES_DIR / f"training_history_{model_name}.png",
        title=f"Historico de treinamento - {model_kind.upper()}",
    )
    update_model_comparison(model_name, metrics)
    return metrics
