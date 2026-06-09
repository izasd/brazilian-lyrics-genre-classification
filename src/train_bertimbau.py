from __future__ import annotations

import argparse
import copy
import time
from collections import Counter
from pathlib import Path

from .config import (
    DEFAULT_PROCESSED_DATA,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    ensure_project_dirs,
)
from .evaluate import compute_metrics, save_confusion_matrix, update_model_comparison
from .io_utils import require_package, write_json
from .neural_models import load_dataframe, resolve_device, save_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina BERTimbau para classificacao de generos musicais."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_DATA)
    parser.add_argument("--model-name", default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--experiment-name", default="bertimbau_base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--trainable-encoder-layers",
        type=int,
        default=-1,
        help="-1 treina todo o encoder; 0 congela; N treina somente as ultimas N camadas.",
    )
    parser.add_argument(
        "--cache-frozen-features",
        action="store_true",
        help="Calcula o encoder congelado uma vez e treina somente o classificador.",
    )
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


def stratified_limit(texts, targets, max_samples: int | None, random_state: int):
    if max_samples is None or max_samples >= len(texts):
        return list(texts), targets
    require_package("sklearn", "scikit-learn")
    from sklearn.model_selection import train_test_split

    limited_texts, _, limited_targets, _ = train_test_split(
        list(texts),
        targets,
        train_size=max_samples,
        stratify=targets,
        random_state=random_state,
    )
    return limited_texts, limited_targets


def make_transformer_loader(
    texts,
    targets,
    tokenizer,
    *,
    max_len: int,
    batch_size: int,
    shuffle: bool,
    random_state: int,
):
    torch = require_package("torch")
    encoded = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    dataset = torch.utils.data.TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
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


def configure_trainable_layers(model, trainable_encoder_layers: int) -> dict[str, int]:
    if trainable_encoder_layers < -1:
        raise SystemExit("--trainable-encoder-layers precisa ser -1, 0 ou um inteiro positivo.")

    encoder = model.base_model
    layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    if trainable_encoder_layers > 0 and layers is None:
        raise SystemExit("Nao foi possivel localizar as camadas do encoder BERT.")
    if layers is not None and trainable_encoder_layers > len(layers):
        raise SystemExit(
            f"O modelo possui {len(layers)} camadas; "
            f"nao e possivel liberar {trainable_encoder_layers}."
        )

    if trainable_encoder_layers >= 0:
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        if trainable_encoder_layers > 0:
            for layer in layers[-trainable_encoder_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }


def evaluate_transformer(model, data_loader, device, id_to_label):
    torch = require_package("torch")
    model.eval()
    total_loss = 0.0
    target_ids: list[int] = []
    prediction_ids: list[int] = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += float(outputs.loss.item()) * len(labels)
            prediction_ids.extend(outputs.logits.argmax(dim=1).cpu().tolist())
            target_ids.extend(labels.cpu().tolist())

    y_true = [id_to_label[index] for index in target_ids]
    y_pred = [id_to_label[index] for index in prediction_ids]
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, len(target_ids))
    return metrics, y_true, y_pred


def extract_frozen_features(model, data_loader, device):
    torch = require_package("torch")
    model.base_model.eval()
    features = []
    targets = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            outputs = model.base_model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                return_dict=True,
            )
            pooled = outputs.pooler_output
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            features.append(pooled.cpu())
            targets.append(labels)

    return torch.cat(features), torch.cat(targets)


def make_feature_loader(
    features,
    targets,
    *,
    batch_size: int,
    shuffle: bool,
    random_state: int,
):
    torch = require_package("torch")
    dataset = torch.utils.data.TensorDataset(features, targets)
    generator = torch.Generator()
    generator.manual_seed(random_state)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def evaluate_frozen_classifier(model, data_loader, device, id_to_label):
    torch = require_package("torch")
    model.classifier.eval()
    total_loss = 0.0
    target_ids: list[int] = []
    prediction_ids: list[int] = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model.classifier(features)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_loss += float(loss.item()) * len(labels)
            prediction_ids.extend(logits.argmax(dim=1).cpu().tolist())
            target_ids.extend(labels.cpu().tolist())

    y_true = [id_to_label[index] for index in target_ids]
    y_pred = [id_to_label[index] for index in prediction_ids]
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, len(target_ids))
    return metrics, y_true, y_pred


def train_bertimbau(args: argparse.Namespace) -> dict[str, object]:
    np = require_package("numpy")
    torch = require_package("torch")
    transformers = require_package("transformers")
    require_package("sklearn", "scikit-learn")
    from sklearn.model_selection import train_test_split

    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps precisa ser maior que zero.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size precisa ser maior que zero.")
    if args.cache_frozen_features and args.trainable_encoder_layers != 0:
        raise SystemExit(
            "--cache-frozen-features exige --trainable-encoder-layers 0."
        )

    ensure_project_dirs()
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)

    device = resolve_device(args.device)
    df = load_dataframe(args.data)
    labels = sorted(df["genre"].unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    targets = df["genre"].map(label_to_id).to_numpy()

    train_texts, test_texts, y_train, y_test = train_test_split(
        df["lyrics"].tolist(),
        targets,
        test_size=args.test_size,
        stratify=targets,
        random_state=args.random_state,
    )
    train_texts, valid_texts, y_train, y_valid = train_test_split(
        train_texts,
        y_train,
        test_size=0.1,
        stratify=y_train,
        random_state=args.random_state,
    )
    train_texts, y_train = stratified_limit(
        train_texts, y_train, args.max_train_samples, args.random_state
    )
    valid_texts, y_valid = stratified_limit(
        valid_texts, y_valid, args.max_valid_samples, args.random_state
    )
    test_texts, y_test = stratified_limit(
        test_texts, y_test, args.max_test_samples, args.random_state
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        do_lower_case=False,
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    parameter_counts = configure_trainable_layers(model, args.trainable_encoder_layers)
    model = model.to(device)

    train_loader = make_transformer_loader(
        train_texts,
        y_train,
        tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        shuffle=True,
        random_state=args.random_state,
    )
    valid_loader = make_transformer_loader(
        valid_texts,
        y_valid,
        tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        shuffle=False,
        random_state=args.random_state,
    )
    test_loader = make_transformer_loader(
        test_texts,
        y_test,
        tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        shuffle=False,
        random_state=args.random_state,
    )

    feature_extraction_seconds = 0.0
    started_at = time.perf_counter()
    if args.cache_frozen_features:
        feature_started_at = time.perf_counter()
        train_features, train_feature_targets = extract_frozen_features(
            model, train_loader, device
        )
        valid_features, valid_feature_targets = extract_frozen_features(
            model, valid_loader, device
        )
        test_features, test_feature_targets = extract_frozen_features(
            model, test_loader, device
        )
        feature_extraction_seconds = time.perf_counter() - feature_started_at
        train_loader = make_feature_loader(
            train_features,
            train_feature_targets,
            batch_size=args.classifier_batch_size,
            shuffle=True,
            random_state=args.random_state,
        )
        valid_loader = make_feature_loader(
            valid_features,
            valid_feature_targets,
            batch_size=args.classifier_batch_size,
            shuffle=False,
            random_state=args.random_state,
        )
        test_loader = make_feature_loader(
            test_features,
            test_feature_targets,
            batch_size=args.classifier_batch_size,
            shuffle=False,
            random_state=args.random_state,
        )
        print(
            "Vetores congelados calculados em "
            f"{feature_extraction_seconds:.1f} segundos."
        )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = max(
        1,
        (len(train_loader) + args.gradient_accumulation_steps - 1)
        // args.gradient_accumulation_steps,
    )
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    history: list[dict[str, float]] = []
    best_f1_macro = float("-inf")
    best_epoch = 0
    best_state: dict[str, object] = {}
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        if args.cache_frozen_features:
            model.classifier.train()
        else:
            model.train()
        optimizer.zero_grad()
        train_loss_total = 0.0
        train_correct = 0
        train_examples = 0

        for step, batch in enumerate(train_loader, start=1):
            if args.cache_frozen_features:
                features, batch_targets = batch
                batch_targets = batch_targets.to(device)
                logits = model.classifier(model.dropout(features.to(device)))
                batch_loss = torch.nn.functional.cross_entropy(logits, batch_targets)
            else:
                input_ids, attention_mask, batch_targets = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                batch_targets = batch_targets.to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_targets,
                )
                logits = outputs.logits
                batch_loss = outputs.loss
            loss = batch_loss / args.gradient_accumulation_steps
            loss.backward()

            should_update = (
                step % args.gradient_accumulation_steps == 0 or step == len(train_loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss_total += float(batch_loss.item()) * len(batch_targets)
            train_correct += int(
                (logits.argmax(dim=1) == batch_targets).sum().item()
            )
            train_examples += len(batch_targets)

        if args.cache_frozen_features:
            valid_metrics, _, _ = evaluate_frozen_classifier(
                model,
                valid_loader,
                device,
                id_to_label,
            )
        else:
            valid_metrics, _, _ = evaluate_transformer(
                model,
                valid_loader,
                device,
                id_to_label,
            )
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss_total / max(1, train_examples),
            "train_accuracy": train_correct / max(1, train_examples),
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_f1_macro": valid_metrics["f1_macro"],
        }
        history.append(row)
        print(
            f"Epoca {epoch:02d}/{args.epochs} - "
            f"loss: {row['train_loss']:.4f} - acc: {row['train_accuracy']:.4f} - "
            f"val_loss: {row['valid_loss']:.4f} - "
            f"val_acc: {row['valid_accuracy']:.4f} - "
            f"val_f1_macro: {row['valid_f1_macro']:.4f}"
        )

        if row["valid_f1_macro"] > best_f1_macro:
            best_f1_macro = row["valid_f1_macro"]
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping apos {epoch} epocas.")
                break

    training_seconds = time.perf_counter() - started_at
    model.load_state_dict(best_state, strict=False)
    if args.cache_frozen_features:
        test_metrics, y_true, y_pred = evaluate_frozen_classifier(
            model,
            test_loader,
            device,
            id_to_label,
        )
    else:
        test_metrics, y_true, y_pred = evaluate_transformer(
            model,
            test_loader,
            device,
            id_to_label,
        )
    test_loss = test_metrics.pop("loss")

    model_dir = MODELS_DIR / args.experiment_name
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    write_json(
        model_dir / "training_config.json",
        {
            "model": args.experiment_name,
            "base_model": args.model_name,
            "labels": labels,
            "label_to_id": label_to_id,
            "max_len": args.max_len,
        },
    )

    metrics = {
        **test_metrics,
        "model": args.experiment_name,
        "framework": "pytorch_transformers",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "base_model": args.model_name,
        "device": str(device),
        "train_size": len(train_texts),
        "valid_size": len(valid_texts),
        "test_size": len(test_texts),
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "classifier_batch_size": args.classifier_batch_size,
        "effective_batch_size": (
            args.classifier_batch_size if args.cache_frozen_features else args.batch_size
        )
        * args.gradient_accumulation_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "epochs_requested": args.epochs,
        "epochs_trained": len(history),
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "gradient_clip": args.gradient_clip,
        "trainable_encoder_layers": args.trainable_encoder_layers,
        "cached_frozen_features": args.cache_frozen_features,
        **parameter_counts,
        "training_seconds": training_seconds,
        "feature_extraction_seconds": feature_extraction_seconds,
        "best_epoch": best_epoch,
        "best_valid_f1_macro": best_f1_macro,
        "test_loss": test_loss,
        "history": history,
        "prediction_distribution": dict(Counter(y_pred)),
        "model_path": str(model_dir),
        "class_distribution": {
            str(key): int(value) for key, value in df["genre"].value_counts().to_dict().items()
        },
    }
    write_json(METRICS_DIR / f"{args.experiment_name}_metrics.json", metrics)
    save_confusion_matrix(
        y_true,
        y_pred,
        labels,
        FIGURES_DIR / f"confusion_matrix_{args.experiment_name}.png",
        title="Matriz de confusao - BERTimbau",
    )
    save_training_history(
        history,
        FIGURES_DIR / f"training_history_{args.experiment_name}.png",
        title="Historico de treinamento - BERTimbau",
    )
    update_model_comparison(args.experiment_name, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    metrics = train_bertimbau(args)
    print(f"Modelo salvo em: {metrics['model_path']}")
    print(f"Acuracia: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
