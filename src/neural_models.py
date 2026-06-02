from __future__ import annotations

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


def build_cnn_model(vocab_size: int, max_len: int, num_classes: int, embedding_dim: int):
    require_package("tensorflow")
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(max_len,)),
            layers.Embedding(vocab_size, embedding_dim),
            layers.Conv1D(128, 5, activation="relu"),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation="relu"),
            layers.GlobalMaxPooling1D(),
            layers.Dropout(0.4),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm_model(vocab_size: int, max_len: int, num_classes: int, embedding_dim: int):
    require_package("tensorflow")
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(max_len,)),
            layers.Embedding(vocab_size, embedding_dim),
            layers.Bidirectional(layers.LSTM(96, dropout=0.3, recurrent_dropout=0.0)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_keras_text_model(
    data_path: Path,
    *,
    model_kind: Literal["cnn", "lstm"],
    epochs: int,
    batch_size: int,
    max_words: int,
    max_len: int,
    embedding_dim: int,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    np = require_package("numpy")
    require_package("sklearn", "scikit-learn")
    require_package("tensorflow")
    from sklearn.model_selection import train_test_split
    from tensorflow import keras
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    ensure_project_dirs()
    np.random.seed(random_state)
    keras.utils.set_random_seed(random_state)

    df = load_dataframe(data_path)
    labels = sorted(df["genre"].unique())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    y = df["genre"].map(label_to_id).to_numpy()

    train_texts, test_texts, y_train, y_test = train_test_split(
        df["lyrics"].tolist(),
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    train_texts, valid_texts, y_train, y_valid = train_test_split(
        train_texts,
        y_train,
        test_size=0.1,
        stratify=y_train,
        random_state=random_state,
    )

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    x_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=max_len, padding="post", truncating="post")
    x_valid = pad_sequences(tokenizer.texts_to_sequences(valid_texts), maxlen=max_len, padding="post", truncating="post")
    x_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=max_len, padding="post", truncating="post")

    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    if model_kind == "cnn":
        model = build_cnn_model(vocab_size, max_len, len(labels), embedding_dim)
    elif model_kind == "lstm":
        model = build_lstm_model(vocab_size, max_len, len(labels), embedding_dim)
    else:
        raise SystemExit(f"Modelo neural invalido: {model_kind}")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    probabilities = model.predict(x_test, batch_size=batch_size, verbose=0)
    pred_ids = probabilities.argmax(axis=1)
    y_true = [id_to_label[int(idx)] for idx in y_test]
    y_pred = [id_to_label[int(idx)] for idx in pred_ids]
    metrics = compute_metrics(y_true, y_pred)
    metrics.update(
        {
            "model": model_kind,
            "train_size": int(len(x_train)),
            "valid_size": int(len(x_valid)),
            "test_size": int(len(x_test)),
            "max_words": max_words,
            "max_len": max_len,
            "embedding_dim": embedding_dim,
            "epochs_requested": epochs,
            "epochs_trained": len(history.history.get("loss", [])),
            "class_distribution": {str(k): int(v) for k, v in df["genre"].value_counts().to_dict().items()},
        }
    )

    model_path = MODELS_DIR / f"{model_kind}.keras"
    tokenizer_path = MODELS_DIR / f"{model_kind}_tokenizer.json"
    labels_path = MODELS_DIR / f"{model_kind}_labels.json"
    model.save(model_path)
    tokenizer_path.write_text(tokenizer.to_json(), encoding="utf-8")
    write_json(labels_path, {"labels": labels, "label_to_id": label_to_id})
    metrics["model_path"] = str(model_path)
    metrics["tokenizer_path"] = str(tokenizer_path)
    metrics["labels_path"] = str(labels_path)

    write_json(METRICS_DIR / f"{model_kind}_metrics.json", metrics)
    save_confusion_matrix(
        y_true,
        y_pred,
        labels,
        FIGURES_DIR / f"confusion_matrix_{model_kind}.png",
        title=f"Matriz de confusao - {model_kind.upper()}",
    )
    update_model_comparison(model_kind, metrics)
    return metrics
