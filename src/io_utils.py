from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_package(import_name: str, package_name: str | None = None) -> Any:
    try:
        module = __import__(import_name)
    except ModuleNotFoundError as exc:
        package = package_name or import_name
        raise SystemExit(
            f"Dependencia ausente: {package}. Instale com `pip install -r requirements.txt`."
        ) from exc
    return module


def safe_slug(value: str) -> str:
    keep = []
    for ch in value.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_", "/"}:
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "classe"

