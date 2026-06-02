from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, pstdev

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


def load_dataframe(path: Path):
    pd = require_package("pandas")

    df = pd.read_csv(path)
    expected = {"lyrics", "genre"}
    missing = expected.difference(df.columns)
    if missing:
        raise SystemExit(f"CSV precisa conter colunas {expected}. Faltando: {sorted(missing)}")
    df = df.dropna(subset=["lyrics", "genre"]).copy()
    df["lyrics"] = df["lyrics"].astype(str)
    df["genre"] = df["genre"].astype(str)
    return df


def build_pipeline(classifier: str):
    require_package("sklearn", "scikit-learn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
    )
    if classifier == "logreg":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=DEFAULT_RANDOM_STATE,
        )
    elif classifier == "linearsvc":
        model = LinearSVC(class_weight="balanced", random_state=DEFAULT_RANDOM_STATE)
    elif classifier == "nb":
        model = MultinomialNB()
    else:
        raise SystemExit(f"Classificador invalido: {classifier}")

    return Pipeline([("tfidf", vectorizer), ("classifier", model)])


def cross_validate_if_possible(df, classifier: str, folds: int, random_state: int) -> dict[str, float | int | None]:
    require_package("sklearn", "scikit-learn")
    from sklearn.model_selection import StratifiedKFold

    min_class_size = int(df["genre"].value_counts().min())
    if folds < 2 or min_class_size < 2:
        return {"folds_requested": folds, "folds_used": None, "reason": "classes insuficientes para k-fold"}
    folds_used = min(folds, min_class_size)
    splitter = StratifiedKFold(n_splits=folds_used, shuffle=True, random_state=random_state)
    f1_scores: list[float] = []
    accuracy_scores: list[float] = []

    for train_idx, valid_idx in splitter.split(df["lyrics"], df["genre"]):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        pipeline = build_pipeline(classifier)
        pipeline.fit(train_df["lyrics"], train_df["genre"])
        preds = pipeline.predict(valid_df["lyrics"])
        metrics = compute_metrics(list(valid_df["genre"]), list(preds))
        f1_scores.append(metrics["f1_macro"])
        accuracy_scores.append(metrics["accuracy"])

    return {
        "folds_requested": folds,
        "folds_used": folds_used,
        "accuracy_mean": mean(accuracy_scores),
        "accuracy_std": pstdev(accuracy_scores) if len(accuracy_scores) > 1 else 0.0,
        "f1_macro_mean": mean(f1_scores),
        "f1_macro_std": pstdev(f1_scores) if len(f1_scores) > 1 else 0.0,
    }


def train_baseline(
    data_path: Path,
    *,
    classifier: str,
    test_size: float,
    folds: int,
    no_cv: bool,
    random_state: int,
) -> dict[str, object]:
    joblib = require_package("joblib")
    require_package("sklearn", "scikit-learn")
    from sklearn.model_selection import train_test_split

    ensure_project_dirs()
    df = load_dataframe(data_path)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["genre"],
        random_state=random_state,
    )

    cv_report = None if no_cv else cross_validate_if_possible(train_df, classifier, folds, random_state)

    pipeline = build_pipeline(classifier)
    pipeline.fit(train_df["lyrics"], train_df["genre"])
    predictions = list(pipeline.predict(test_df["lyrics"]))
    y_true = list(test_df["genre"])
    labels = sorted(df["genre"].unique())
    metrics = compute_metrics(y_true, predictions)
    metrics["model"] = f"baseline_tfidf_{classifier}"
    metrics["train_size"] = int(len(train_df))
    metrics["test_size"] = int(len(test_df))
    metrics["class_distribution"] = {str(k): int(v) for k, v in df["genre"].value_counts().to_dict().items()}
    metrics["cross_validation"] = cv_report

    model_path = MODELS_DIR / f"baseline_tfidf_{classifier}.joblib"
    joblib.dump(pipeline, model_path)
    metrics["model_path"] = str(model_path)

    metrics_path = METRICS_DIR / f"baseline_tfidf_{classifier}_metrics.json"
    write_json(metrics_path, metrics)
    save_confusion_matrix(
        y_true,
        predictions,
        labels,
        FIGURES_DIR / f"confusion_matrix_baseline_tfidf_{classifier}.png",
        title=f"Matriz de confusao - baseline TF-IDF ({classifier})",
    )
    update_model_comparison(f"baseline_tfidf_{classifier}", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina baseline TF-IDF.")
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_DATA)
    parser.add_argument("--classifier", choices=["logreg", "linearsvc", "nb"], default="logreg")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-cv", action="store_true", help="Pula validacao cruzada para execucao mais rapida.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_baseline(
        args.data,
        classifier=args.classifier,
        test_size=args.test_size,
        folds=args.folds,
        no_cv=args.no_cv,
        random_state=args.random_state,
    )
    print(f"Modelo salvo em: {metrics['model_path']}")
    print(f"Acuracia: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")
    print(f"F1 weighted: {metrics['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
