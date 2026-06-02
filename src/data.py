from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from .config import DEFAULT_PROCESSED_DATA, DEFAULT_RAW_DATA, METRICS_DIR, ensure_project_dirs
from .io_utils import write_json
from .preprocessing import clean_lyrics, normalize_label, word_count


DEFAULT_EXCLUDED_GENRES = {"axe", "funk carioca", "pagode", "trilha sonora"}


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    *,
    lyrics_col: str = "lyrics",
    genre_col: str = "genre",
    min_words: int = 5,
    lowercase: bool = False,
    remove_section_markers: bool = True,
    excluded_genres: set[str] | None = None,
) -> dict[str, object]:
    ensure_project_dirs()
    excluded_genres = excluded_genres if excluded_genres is not None else DEFAULT_EXCLUDED_GENRES

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("CSV sem cabecalho.")
        if lyrics_col not in reader.fieldnames or genre_col not in reader.fieldnames:
            raise SystemExit(
                f"Colunas obrigatorias nao encontradas. Esperado: {lyrics_col!r}, {genre_col!r}. "
                f"Encontrado: {reader.fieldnames}"
            )

        rows: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        stats = Counter()
        class_counts = Counter()

        for row in reader:
            stats["read"] += 1
            lyrics = clean_lyrics(
                row.get(lyrics_col, ""),
                lowercase=lowercase,
                remove_section_markers=remove_section_markers,
            )
            genre = clean_lyrics(row.get(genre_col, ""), lowercase=True, remove_section_markers=False)
            genre_key = normalize_label(genre)

            if not lyrics or not genre:
                stats["missing"] += 1
                continue
            if word_count(lyrics) < min_words:
                stats["too_short"] += 1
                continue
            if genre_key in excluded_genres:
                stats[f"excluded_genre:{genre_key}"] += 1
                continue

            dedupe_key = (lyrics, genre_key)
            if dedupe_key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(dedupe_key)

            rows.append({"lyrics": lyrics, "genre": genre})
            class_counts[genre] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lyrics", "genre"])
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "rows_read": stats["read"],
        "rows_written": len(rows),
        "missing_removed": stats["missing"],
        "too_short_removed": stats["too_short"],
        "duplicates_removed": stats["duplicates"],
        "excluded_genres": {
            key.replace("excluded_genre:", ""): value
            for key, value in stats.items()
            if key.startswith("excluded_genre:")
        },
        "class_distribution": dict(class_counts),
    }
    write_json(METRICS_DIR / "data_preparation_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara CSV de letras e generos para treino.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROCESSED_DATA)
    parser.add_argument("--lyrics-col", default="lyrics")
    parser.add_argument("--genre-col", default="genre")
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--keep-section-markers", action="store_true")
    parser.add_argument(
        "--exclude-genres",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDED_GENRES),
        help="Generos a remover, comparados sem acento e em minusculo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_dataset(
        args.input,
        args.output,
        lyrics_col=args.lyrics_col,
        genre_col=args.genre_col,
        min_words=args.min_words,
        lowercase=args.lowercase,
        remove_section_markers=not args.keep_section_markers,
        excluded_genres={normalize_label(g) for g in args.exclude_genres},
    )
    print(f"Arquivo processado salvo em: {report['output']}")
    print(f"Linhas lidas: {report['rows_read']}")
    print(f"Linhas gravadas: {report['rows_written']}")
    print("Distribuicao por genero:")
    for genre, count in sorted(report["class_distribution"].items(), key=lambda item: (-item[1], item[0])):
        print(f"  {genre}: {count}")


if __name__ == "__main__":
    main()
