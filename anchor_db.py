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
            # Comments table — memory as conversation space (design: Veille & 吱吱)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS comments (
                    comment_id  TEXT PRIMARY KEY,
                    memory_id   TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
                    content     TEXT NOT NULL,
                    author      TEXT DEFAULT 'ai',
                    reply_to    TEXT REFERENCES comments(comment_id),
                    read_by_ai  INTEGER DEFAULT 0,
                    read_by_human INTEGER DEFAULT 0,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_comments_memory ON comments(memory_id)")
            # Annotations — append-only notes on memories (design: Altair)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS annotations (
                    annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id   TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
                    text        TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_annotations_memory ON annotations(memory_id)")
            conn.commit()
        self._ensure_context_column()
        self._ensure_visual_column()

    def _ensure_context_column(self):
        """Add context column if missing. text = search summary, context = full original."""
        with self._conn() as conn:
            try:
                conn.execute("SELECT context FROM memories LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE memories ADD COLUMN context TEXT DEFAULT ''")
                conn.commit()

    def _ensure_visual_column(self):
        """Add visual_embedding column if missing. For Anchor Vision integration."""
        with self._conn() as conn:
            try:
                conn.execute("SELECT visual_embedding FROM memories LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE memories ADD COLUMN visual_embedding TEXT DEFAULT ''")
                conn.commit()

    # ── Annotations (append-only) ──

    def annotate(self, memory_id: str, text: str) -> int:
        """Add an annotation to a memory. Append-only — never delete or edit."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO annotations (memory_id, text, created_at) VALUES (?, ?, ?)",
                (memory_id, text, datetime.utcnow().isoformat()),
            )
            conn.commit()
            return cur.lastrowid

    def get_annotations(self, memory_id: str) -> list:
        """Get all annotations for a memory, oldest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT annotation_id, text, created_at FROM annotations "
                "WHERE memory_id = ? ORDER BY created_at ASC",
                (memory_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_annotations(self, query: str, limit: int = 5) -> list:
        """Search annotations text. Returns matching memory_ids."""
        words = query.strip().split()
        if not words:
            return []
        where = " AND ".join(["a.text LIKE ?"] * len(words))
        params = [f"%{w}%" for w in words]
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT a.memory_id, a.text, a.created_at FROM annotations a "
                f"WHERE {where} ORDER BY a.created_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Visual Embedding (Anchor Vision integration) ──

    def set_visual_embedding(self, memory_id: str, embedding_json: str):
        """Store a visual embedding (CLIP vector as JSON string) for a memory."""
        self._ensure_visual_column()
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET visual_embedding = ? WHERE memory_id = ?",
                (embedding_json, memory_id),
            )
            conn.commit()

    def get_visual_embedding(self, memory_id: str) -> str:
        """Get visual embedding for a memory. Returns JSON string or empty."""
        self._ensure_visual_column()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT visual_embedding FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        return row["visual_embedding"] if row and row["visual_embedding"] else ""

    def find_visual_memories(self) -> list:
        """Get all memories that have visual embeddings."""
        self._ensure_visual_column()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, text, visual_embedding FROM memories "
                "WHERE visual_embedding != '' AND visual_embedding IS NOT NULL"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Memory CRUD ──

    def insert(self, memory_id: str, text: str, tag: str = "general",
               tier: str = "short", emotion_score: float = 0.5,
               context: str = ""):
        """Insert or replace a memory. text = search summary, context = full original."""
        self._ensure_context_column()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memories (memory_id, text, timestamp, tag, tier, emotion_score, context) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (memory_id, text, datetime.utcnow().isoformat(), tag, tier, emotion_score, context),
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
        """Search memories + annotations by keyword."""
        with self._conn() as conn:
            # Search in memory text
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
            results = [dict(r) for r in rows]
            found_ids = {r["memory_id"] for r in results}

            # Also search in annotations
            ann_rows = conn.execute(
                "SELECT DISTINCT a.memory_id FROM annotations a "
                "WHERE a.text LIKE ? LIMIT ?",
                (f"%{query}%", limit)
            ).fetchall()
            for ar in ann_rows:
                mid = ar["memory_id"]
                if mid not in found_ids:
                    mem = conn.execute(
                        "SELECT memory_id, text, timestamp, tag FROM memories "
                        "WHERE memory_id = ?", (mid,)
                    ).fetchone()
                    if mem:
                        results.append(dict(mem))
                        found_ids.add(mid)

        return results[:limit]

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

    # ── Comments: memory as conversation space (design: Veille & 吱吱) ──

    def insert_comment(self, memory_id: str, content: str,
                       author: str = "ai", reply_to: str = None) -> str:
        import uuid
        comment_id = f"comment_{uuid.uuid4().hex[:12]}"
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO comments (comment_id, memory_id, content, author, reply_to, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (comment_id, memory_id, content, author, reply_to,
                 datetime.utcnow().isoformat()),
            )
            conn.commit()
        return comment_id

    def get_comments(self, memory_id: str) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM comments WHERE memory_id = ? ORDER BY created_at",
                (memory_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_unread_comments(self, reader: str = "ai") -> list:
        col = "read_by_ai" if reader == "ai" else "read_by_human"
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT c.*, m.text as memory_text "
                f"FROM comments c JOIN memories m ON c.memory_id = m.memory_id "
                f"WHERE c.{col} = 0 ORDER BY c.created_at",
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_comments_read(self, comment_ids: list, reader: str = "ai"):
        col = "read_by_ai" if reader == "ai" else "read_by_human"
        with self._conn() as conn:
            for cid in comment_ids:
                conn.execute(
                    f"UPDATE comments SET {col} = 1 WHERE comment_id = ?", (cid,)
                )
            conn.commit()

    # ── Wakeup: one-call cold start (design: Veille & 吱吱) ──

    def wakeup(self, n_high_emotion: int = 5, n_random: int = 2,
               high_emotion_days: int = 3) -> dict:
        """Gather everything needed for cold start in one call.

        Returns pinned memories + recent high-emotion + random old + unread comments.
        Design principle: rules live here, not in external config.

        Note on random_old: these are surfaced without touch() — they don't
        increment usage_count or update last_used. Any Hebbian edges created
        from co-activation with random memories will be pruned by dream pass
        if not reinforced. This is intentional: temporary connections that
        don't get reinforced fade naturally. Dream pass is the cleanup.
        """
        self._ensure_context_column()
        cutoff = (datetime.utcnow() - timedelta(days=high_emotion_days)).isoformat()

        with self._conn() as conn:
            pinned = conn.execute(
                "SELECT memory_id, text, tag, emotion_score, context FROM memories "
                "WHERE pinned = 1 ORDER BY timestamp"
            ).fetchall()

            high_emotion = conn.execute(
                "SELECT memory_id, text, tag, emotion_score, timestamp, context FROM memories "
                "WHERE timestamp >= ? AND pinned = 0 "
                "ORDER BY emotion_score DESC LIMIT ?",
                (cutoff, n_high_emotion),
            ).fetchall()

            random_old = conn.execute(
                "SELECT memory_id, text, tag, emotion_score, timestamp, context FROM memories "
                "WHERE timestamp < ? AND pinned = 0 "
                "ORDER BY RANDOM() LIMIT ?",
                (cutoff, n_random),
            ).fetchall()

            unread = conn.execute(
                "SELECT c.comment_id, c.memory_id, c.content, c.author, c.created_at, "
                "m.text as memory_text FROM comments c "
                "JOIN memories m ON c.memory_id = m.memory_id "
                "WHERE c.read_by_ai = 0 ORDER BY c.created_at"
            ).fetchall()

        return {
            "pinned": [dict(r) for r in pinned],
            "high_emotion": [dict(r) for r in high_emotion],
            "random_old": [dict(r) for r in random_old],
            "unread_comments": [dict(r) for r in unread],
        }
