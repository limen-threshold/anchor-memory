"""
Anchor Memory System — SQLite layer for graph-structured memory.

Handles: memory storage, tiered decay, synaptic edges (Hebbian learning),
emotion scoring, citation tracking, and graph operations.
"""

import sqlite3
import os
from datetime import datetime, timedelta


class AnchorDB:
    """SQLite storage with graph layer for memory synapses."""

    MAX_EDGE_WEIGHT = 10.0  # Synaptic saturation

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_tables()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_tables(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id   TEXT PRIMARY KEY,
                    text        TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    last_used   TEXT,
                    tag         TEXT DEFAULT 'general',
                    tier        TEXT DEFAULT 'short',
                    pinned      INTEGER DEFAULT 0,
                    emotion_score REAL DEFAULT 0.5
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    source_id   TEXT NOT NULL,
                    target_id   TEXT NOT NULL,
                    weight      REAL DEFAULT 1.0,
                    created     TEXT NOT NULL,
                    last_fired  TEXT NOT NULL,
                    PRIMARY KEY (source_id, target_id),
                    FOREIGN KEY (source_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES memories(memory_id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            conn.commit()

    # ── Memory CRUD ──

    def insert(self, memory_id: str, text: str, tag: str = "general",
               tier: str = "short", emotion_score: float = 0.5):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memories (memory_id, text, timestamp, tag, tier, emotion_score) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (memory_id, text, datetime.utcnow().isoformat(), tag, tier, emotion_score),
            )
            conn.commit()

    def get(self, memory_id: str) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete(self, memory_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            conn.commit()

    def list_all(self, limit: int = 50, offset: int = 0) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, text, timestamp, tag, tier FROM memories "
                "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
        return [dict(r) for r in rows]

    def keyword_search(self, query: str, limit: int = 5, tag: str = None) -> list:
        with self._conn() as conn:
            if tag:
                rows = conn.execute(
                    "SELECT memory_id, text, timestamp, tag FROM memories "
                    "WHERE text LIKE ? AND tag = ? LIMIT ?",
                    (f"%{query}%", tag, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT memory_id, text, timestamp, tag FROM memories "
                    "WHERE text LIKE ? LIMIT ?",
                    (f"%{query}%", limit)
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Tier management ──

    def set_tier(self, memory_id: str, tier: str):
        with self._conn() as conn:
            conn.execute("UPDATE memories SET tier = ? WHERE memory_id = ?", (tier, memory_id))
            conn.commit()

    def decay_short(self, days: int = 14) -> int:
        """Delete short-tier memories older than N days."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE tier = 'short' AND timestamp < ?",
                (cutoff,)
            )
            conn.commit()
        return cursor.rowcount

    # ── Citation tracking ──

    def get_citation_count(self, memory_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT usage_count FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return row["usage_count"] if row else 0

    def cite(self, memory_id: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET usage_count = usage_count + 1, last_used = ? WHERE memory_id = ?",
                (datetime.utcnow().isoformat(), memory_id),
            )
            conn.commit()

    # ── Emotion scoring ──

    def get_emotion_score(self, memory_id: str) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT emotion_score FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        if row and row["emotion_score"] is not None:
            return row["emotion_score"]
        return 0.5

    def set_emotion_score(self, memory_id: str, score: float):
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET emotion_score = ? WHERE memory_id = ?",
                (max(0.0, min(1.0, score)), memory_id)
            )
            conn.commit()

    def equalize_emotion_scores(self, nudge: float = 0.05, threshold: float = 0.2) -> int:
        """Bidirectional emotion score equilibration across connected memories."""
        updated = 0
        with self._conn() as conn:
            memories = conn.execute(
                "SELECT memory_id, emotion_score FROM memories WHERE emotion_score IS NOT NULL"
            ).fetchall()

            for m in memories:
                mid = m["memory_id"]
                my_score = m["emotion_score"] or 0.5

                neighbors = conn.execute("""
                    SELECT m.emotion_score FROM memories m
                    INNER JOIN edges e ON (e.target_id = m.memory_id AND e.source_id = ?)
                       OR (e.source_id = m.memory_id AND e.target_id = ?)
                    WHERE m.emotion_score IS NOT NULL AND e.weight >= 0.5
                """, (mid, mid)).fetchall()

                if not neighbors:
                    continue

                avg_neighbor = sum(n["emotion_score"] or 0.5 for n in neighbors) / len(neighbors)
                diff = avg_neighbor - my_score

                if abs(diff) > threshold:
                    new_score = my_score + nudge * (1 if diff > 0 else -1)
                    new_score = max(0.0, min(1.0, new_score))
                    conn.execute(
                        "UPDATE memories SET emotion_score = ? WHERE memory_id = ?",
                        (new_score, mid)
                    )
                    updated += 1

            conn.commit()
        return updated

    # ── Graph layer: synaptic edges ──

    def _upsert_edge(self, conn, source_id: str, target_id: str,
                     weight: float, now: str):
        conn.execute("""
            INSERT INTO edges (source_id, target_id, weight, created, last_fired)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id) DO UPDATE SET
                weight = MIN(edges.weight + excluded.weight, ?),
                last_fired = excluded.last_fired
        """, (source_id, target_id, weight, now, now, self.MAX_EDGE_WEIGHT))

    def connect(self, source_id: str, target_id: str, weight: float = 1.0):
        """Create or strengthen a bidirectional edge between memories."""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            self._upsert_edge(conn, source_id, target_id, weight, now)
            self._upsert_edge(conn, target_id, source_id, weight, now)
            conn.commit()

    def connect_batch(self, pairs: list, weight: float = 0.2):
        """Batch connect pairs of memories (for Hebbian learning)."""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            for source_id, target_id in pairs:
                self._upsert_edge(conn, source_id, target_id, weight, now)
                self._upsert_edge(conn, target_id, source_id, weight, now)
            conn.commit()

    def get_neighbors(self, memory_id: str, min_weight: float = 0.5,
                      limit: int = 5) -> list:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT target_id as memory_id, weight FROM edges
                WHERE source_id = ? AND weight >= ?
                ORDER BY weight DESC LIMIT ?
            """, (memory_id, min_weight, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_edge_weight(self, source_id: str, target_id: str):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT weight FROM edges WHERE source_id = ? AND target_id = ?",
                (source_id, target_id)
            ).fetchone()
        return row["weight"] if row else None

    def decay_edges(self, min_weight: float = 0.1, decay_factor: float = 0.9) -> int:
        """Weaken all edges by decay_factor. Delete edges below min_weight."""
        with self._conn() as conn:
            conn.execute("UPDATE edges SET weight = weight * ?", (decay_factor,))
            cursor = conn.execute("DELETE FROM edges WHERE weight < ?", (min_weight,))
            conn.commit()
        return cursor.rowcount

    def decay_strong_edges(self, min_weight: float = 1.5, decay_factor: float = 0.95) -> int:
        """Slowly decay strong manual edges so they don't permanently dominate."""
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE edges SET weight = weight * ? WHERE weight >= ?",
                (decay_factor, min_weight)
            )
            conn.commit()
        return cursor.rowcount

    # ── Pinning ──

    def pin(self, memory_id: str):
        with self._conn() as conn:
            conn.execute("UPDATE memories SET pinned = 1 WHERE memory_id = ?", (memory_id,))
            conn.commit()

    def unpin(self, memory_id: str):
        with self._conn() as conn:
            conn.execute("UPDATE memories SET pinned = 0 WHERE memory_id = ?", (memory_id,))
            conn.commit()

    def get_pinned(self) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, text, timestamp, tag FROM memories WHERE pinned = 1"
            ).fetchall()
        return [dict(r) for r in rows]
