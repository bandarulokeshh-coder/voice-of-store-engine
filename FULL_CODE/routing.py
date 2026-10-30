from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from preprocessing import normalize_text

ROOT = Path(__file__).resolve().parents[0]
TAXONOMY_PATH = ROOT / "config" / "issue_taxonomy.json"


@lru_cache(maxsize=4)
def load_taxonomy(path: str | Path = TAXONOMY_PATH) -> dict[str, Any]:
    taxonomy_path = Path(path)
    with taxonomy_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=512)
def get_department_for_issue(
    issue_category: str, text: str | None = None, path: str | Path = TAXONOMY_PATH
) -> str:
    taxonomy = load_taxonomy(path)
    default_department = taxonomy.get("default_department", "Store Operations")
    category = normalize_text(issue_category)
    feedback = normalize_text(text or "")
    for item in taxonomy.get("categories", []):
        item_category = normalize_text(item.get("issue_category", ""))
        if item_category == category:
            return item.get("department", default_department)

    # Keyword-based fallback in case a future upload uses a slightly different label.
    for item in taxonomy.get("categories", []):
        keywords = [normalize_text(keyword) for keyword in item.get("keywords", [])]
        if any(keyword and keyword in feedback for keyword in keywords):
            return item.get("department", default_department)

    return default_department


@lru_cache(maxsize=512)
def get_suggested_action(issue_category: str, path: str | Path = TAXONOMY_PATH) -> str:
    taxonomy = load_taxonomy(path)
    category = normalize_text(issue_category)
    for item in taxonomy.get("categories", []):
        if normalize_text(item.get("issue_category", "")) == category:
            return item.get(
                "suggested_action", "Review the issue with the owning team."
            )
    return "Review the issue with the owning team."


@lru_cache(maxsize=512)
def get_default_severity(issue_category: str, path: str | Path = TAXONOMY_PATH) -> str:
    taxonomy = load_taxonomy(path)
    category = normalize_text(issue_category)
    for item in taxonomy.get("categories", []):
        if normalize_text(item.get("issue_category", "")) == category:
            return item.get("severity_default", "MEDIUM")
    return "MEDIUM"


@lru_cache(maxsize=512)
def category_keywords(path: str | Path = TAXONOMY_PATH) -> dict[str, list[str]]:
    taxonomy = load_taxonomy(path)
    mapping: dict[str, list[str]] = {}
    for item in taxonomy.get("categories", []):
        mapping[item["issue_category"]] = [
            normalize_text(keyword) for keyword in item.get("keywords", [])
        ]
    return mapping
