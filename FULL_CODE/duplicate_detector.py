from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from preprocessing import compact_text, normalize_text


@dataclass(slots=True)
class DuplicateGroup:
    group_id: str
    store_id: str
    issue_category: str
    record_count: int
    channels: list[str]
    first_seen: pd.Timestamp
    last_seen: pd.Timestamp
    representative_text: str
    evidence_texts: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "store_id": self.store_id,
            "issue_category": self.issue_category,
            "record_count": self.record_count,
            "channels": self.channels,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "representative_text": self.representative_text,
            "evidence_texts": self.evidence_texts,
            "duplicate_message": f"Potential duplicate incident reported across {len(self.channels)} channels.",
        }


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _candidate_match(
    row_a: pd.Series, row_b: pd.Series, time_window_hours: int
) -> bool:
    if row_a["store_id"] != row_b["store_id"]:
        return False
    delta_hours = abs(
        (pd.Timestamp(row_a["timestamp"]) - pd.Timestamp(row_b["timestamp"]))
        / pd.Timedelta(hours=1)
    )
    return delta_hours <= time_window_hours


def detect_duplicate_groups(
    df: pd.DataFrame,
    *,
    similarity_threshold: float = 0.86,
    time_window_hours: int = 36,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "group_id",
                "store_id",
                "issue_category",
                "record_count",
                "channels",
                "first_seen",
                "last_seen",
                "representative_text",
                "evidence_texts",
                "duplicate_message",
            ]
        )

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.loc[~frame["timestamp"].isna()].copy()
    frame["normalized_text"] = (
        frame["feedback_text"].fillna("").astype(str).map(normalize_text)
    )
    frame = frame.loc[frame["normalized_text"].str.len() > 0].copy()
    frame = frame.sort_values(["store_id", "timestamp"]).reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()

    uf = UnionFind(len(frame))

    for store_id, store_group in frame.groupby("store_id"):
        indices = store_group.index.to_list()
        for i, idx_a in enumerate(indices):
            row_a = frame.loc[idx_a]
            for idx_b in indices[i + 1 :]:
                row_b = frame.loc[idx_b]
                if not _candidate_match(
                    row_a, row_b, time_window_hours=time_window_hours
                ):
                    continue
                text_score = _similarity(
                    row_a["normalized_text"], row_b["normalized_text"]
                )
                category_match = normalize_text(
                    row_a.get("issue_category", "")
                ) == normalize_text(row_b.get("issue_category", ""))
                same_channel = normalize_text(
                    row_a.get("channel", "")
                ) == normalize_text(row_b.get("channel", ""))
                if text_score >= similarity_threshold and (
                    category_match or same_channel or text_score >= 0.95
                ):
                    uf.union(idx_a, idx_b)

    clusters: defaultdict[int, list[int]] = defaultdict(list)
    for idx in frame.index:
        clusters[uf.find(idx)].append(idx)

    groups: list[dict[str, Any]] = []
    for root, indices in clusters.items():
        if len(indices) < 2:
            continue
        cluster = frame.loc[indices].sort_values("timestamp")
        channels = sorted(
            {str(x) for x in cluster["channel"].fillna("Unknown").tolist()}
        )
        store_id = str(cluster.iloc[0]["store_id"])
        issue_category = str(cluster.iloc[0].get("issue_category", "Unknown"))
        first_seen = pd.Timestamp(cluster["timestamp"].min())
        last_seen = pd.Timestamp(cluster["timestamp"].max())
        representative_text = compact_text(cluster.iloc[0]["feedback_text"], 150)
        evidence_texts = [
            compact_text(text, 150)
            for text in cluster["feedback_text"].head(4).tolist()
        ]
        groups.append(
            DuplicateGroup(
                group_id=f"DUP-{root:05d}",
                store_id=store_id,
                issue_category=issue_category,
                record_count=len(cluster),
                channels=channels,
                first_seen=first_seen,
                last_seen=last_seen,
                representative_text=representative_text,
                evidence_texts=evidence_texts,
            ).to_dict()
        )

    if not groups:
        return pd.DataFrame(
            columns=[
                "group_id",
                "store_id",
                "issue_category",
                "record_count",
                "channels",
                "first_seen",
                "last_seen",
                "representative_text",
                "evidence_texts",
                "duplicate_message",
            ]
        )

    return (
        pd.DataFrame(groups)
        .sort_values(["record_count", "last_seen"], ascending=[False, False])
        .reset_index(drop=True)
    )


def duplicate_summary_message(groups_df: pd.DataFrame) -> str:
    if groups_df.empty:
        return "No probable duplicate incident groups were found."
    max_channels = max(len(channels) for channels in groups_df["channels"])
    return (
        f"Potential duplicate incident reported across up to {max_channels} channels."
    )
