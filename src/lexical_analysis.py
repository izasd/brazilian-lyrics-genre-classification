from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_PROCESSED_DATA, FIGURES_DIR, ANALYSIS_DIR, ensure_project_dirs
from .io_utils import require_package, safe_slug


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


def save_class_distribution(df) -> None:
    require_package("matplotlib")
    import matplotlib.pyplot as plt

    counts = df["genre"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    counts.plot(kind="barh", ax=ax, color="#4472c4")
    ax.set_title("Distribuicao de classes")
    ax.set_xlabel("Quantidade de letras")
    ax.set_ylabel("Genero")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)


def generate_top_terms(df, top_n: int) -> None:
    pd = require_package("pandas")
    require_package("sklearn", "scikit-learn")
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    rows = []
    freq_rows = []
    for genre, group in df.groupby("genre"):
        texts = group["lyrics"].tolist()

        count_vectorizer = CountVectorizer(max_features=10000, min_df=2)
        counts = count_vectorizer.fit_transform(texts).sum(axis=0).A1
        count_terms = count_vectorizer.get_feature_names_out()
        for term, score in sorted(zip(count_terms, counts), key=lambda item: item[1], reverse=True)[:top_n]:
            freq_rows.append({"genre": genre, "term": term, "score": int(score), "metric": "frequency"})

        tfidf_vectorizer = TfidfVectorizer(max_features=10000, min_df=2, sublinear_tf=True)
        scores = tfidf_vectorizer.fit_transform(texts).sum(axis=0).A1
        tfidf_terms = tfidf_vectorizer.get_feature_names_out()
        top_terms = sorted(zip(tfidf_terms, scores), key=lambda item: item[1], reverse=True)[:top_n]
        for term, score in top_terms:
            rows.append({"genre": genre, "term": term, "score": float(score), "metric": "tfidf"})

        save_top_terms_chart(genre, top_terms)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ANALYSIS_DIR / "top_tfidf_terms_by_genre.csv", index=False)
    pd.DataFrame(freq_rows).to_csv(ANALYSIS_DIR / "top_frequency_terms_by_genre.csv", index=False)


def save_top_terms_chart(genre: str, terms: list[tuple[str, float]]) -> None:
    require_package("matplotlib")
    import matplotlib.pyplot as plt

    if not terms:
        return
    terms = list(reversed(terms))
    labels = [term for term, _ in terms]
    scores = [score for _, score in terms]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, scores, color="#70ad47")
    ax.set_title(f"Top termos TF-IDF - {genre}")
    ax.set_xlabel("Soma TF-IDF")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"top_terms_{safe_slug(genre)}.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera analise lexical por genero.")
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_DATA)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    ensure_project_dirs()
    args = parse_args()
    df = load_dataframe(args.data)
    save_class_distribution(df)
    generate_top_terms(df, args.top_n)
    print(f"Analises salvas em: {ANALYSIS_DIR}")
    print(f"Figuras salvas em: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
