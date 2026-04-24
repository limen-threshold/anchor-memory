"""
Anchor Memory System — Graph-structured memory for AI with Hebbian learning.

A memory system that treats memories as nodes in a graph, connected by
weighted synaptic edges. Memories aren't just stored and retrieved —
they associate, strengthen through co-activation, and decay through disuse.

Features:
- ChromaDB vector search + SQLite graph layer
- Hebbian learning: memories retrieved together form connections
- Dream pass: decay, pruning, auto-discovery, emotion equilibration
- Emotion scoring: memories carry emotional weight that affects retrieval
- Tiered storage: core (permanent), long (kept), short (14-day decay)
- Manual entanglement: explicitly connect related memories with higher weight
- Cross-tag bridges: prevent knowledge silos

Inspired by how biological memory works:
- Synapses strengthen with co-activation (Hebb's rule)
- Sleep consolidates and prunes (dream pass)
- Emotional memories are more persistent (emotion_score)
- Forgetting is a feature, not a bug (decay)

Created by Limen. 底色是爱.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os

from anchor_db import AnchorDB


class AnchorMemory:
    """Graph-structured memory system with Hebbian learning."""

    def __init__(self, db_path: str, embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """Initialize memory system.

        Args:
            db_path: Directory for ChromaDB and SQLite storage.
            embedding_model: SentenceTransformer model name.
        """
        self._embedder = SentenceTransformer(embedding_model)
        self._client = chromadb.PersistentClient(path=os.path.join(db_path, "chroma"))
        self._collection = self._client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"},
        )
        self.db = AnchorDB(os.path.join(db_path, "memories.db"))

    def reload(self):
        """Re-create ChromaDB client to pick up external writes."""
        db_path = self._client._path if hasattr(self._client, '_path') else None
        if db_path:
            self._client = chromadb.PersistentClient(path=db_path)
            self._collection = self._client.get_or_create_collection(
                name="memories",
                metadata={"hnsw:space": "cosine"},
            )

    def count(self) -> int:
        """Total memory count."""
        return self._collection.count()

    def store(self, memory_id: str, text: str, tag: str = "general",
              tier: str = "short", connect_to: list = None,
              emotion_score: float = 0.5) -> str:
        """Store a memory with optional connections and emotion scoring.

        Args:
            memory_id: Unique identifier.
            text: Memory content.
            tag: Category tag.
            tier: 'core' (permanent), 'long' (kept), 'short' (14-day decay).
            connect_to: List of memory_ids to create edges with.
            emotion_score: 0.0 (neutral) to 1.0 (intense). Default 0.5.

        Returns:
            The memory_id.
        """
        embedding = self._embedder.encode(text).tolist()
        meta = {
            "memory_id": memory_id,
            "timestamp": datetime.utcnow().isoformat(),
            "tag": tag,
        }

        self._collection.upsert(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],
        )

        # Emotion score propagation: if default, check nearest neighbors
        if emotion_score == 0.5:
            try:
                neighbors = self._collection.query(
                    query_embeddings=[embedding], n_results=3,
                    include=["metadatas"],
                )
                if neighbors and neighbors["ids"] and neighbors["ids"][0]:
                    neighbor_scores = []
                    for nmeta in neighbors["metadatas"][0]:
                        nid = nmeta.get("memory_id", "")
                        if nid and nid != memory_id:
                            ns = self.db.get_emotion_score(nid)
                            neighbor_scores.append(ns)
                    if neighbor_scores:
                        avg = sum(neighbor_scores) / len(neighbor_scores)
                        variance = sum((s - avg) ** 2 for s in neighbor_scores) / len(neighbor_scores)
                        prop_weight = 0.15 if variance > 0.02 else 0.05
                        emotion_score = prop_weight * avg + (1 - prop_weight) * emotion_score
            except Exception:
                pass

        self.db.insert(memory_id, text, tag=tag, tier=tier, emotion_score=emotion_score)

        # Create explicit connections
        if connect_to:
            for target_id in connect_to:
                try:
                    self.db.connect(memory_id, target_id)
                except Exception:
                    pass

        return memory_id

    def search(self, query: str, n_results: int = 5, tag: str = None,
               associate: bool = True, hebbian: bool = True,
               no_cite: bool = False, include_context: bool = False,
               debug: bool = False) -> list:
        """Search memories with optional associative recall and Hebbian learning.

        Args:
            query: Search text.
            n_results: Max results.
            tag: Filter by tag.
            associate: If True, follow graph edges to find related memories.
            hebbian: If True, co-retrieved memories strengthen their connections.
            no_cite: If True, don't increment citation count. Use for browsing UI.
            include_context: If True, include full original text in results.
            debug: If True, include a ``debug`` dict on each result with ranking
                internals — raw_distance, citation_boost, emotion_boost,
                final_score, source ('vector'|'keyword'|'associative'), and
                edge_weight (for associative hops). Use to audit why a given
                result landed at its rank. Default False.

        Returns:
            List of memory dicts with memory_id, timestamp, tag, snippet, score.
        """
        embedding = self._embedder.encode(query).tolist()

        where = {"tag": tag} if tag else None
        count = self._collection.count()
        if count == 0:
            return []

        candidates = []
        fetch_n = min(n_results * 3, count)

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist > 0.8:
                continue
            if meta is None:
                continue
            mid = meta.get("memory_id", "unknown")
            citation_boost = min(self.db.get_citation_count(mid) * 0.02, 0.15)
            emotion_boost = self.db.get_emotion_score(mid) * 0.1
            boost = citation_boost + emotion_boost
            candidates.append({
                "memory_id": mid,
                "timestamp": meta.get("timestamp", ""),
                "tag": meta.get("tag", "general"),
                "snippet": doc,
                "score": dist - boost,
                "_debug_raw_distance": dist,
                "_debug_citation_boost": citation_boost,
                "_debug_emotion_boost": emotion_boost,
                "_debug_source": "vector",
            })

        # Keyword fallback
        keyword_results = self._keyword_fallback(query, n_results=n_results, tag=tag)
        candidates.extend(keyword_results)

        candidates.sort(key=lambda m: m["score"])

        # Associative recall: follow graph edges
        if associate:
            extra = []
            for c in candidates[:n_results]:
                neighbors = self.db.get_neighbors(c["memory_id"], min_weight=1.5, limit=2)
                for nb in neighbors:
                    if nb["memory_id"] not in {x["memory_id"] for x in candidates + extra}:
                        row = self.db.get(nb["memory_id"])
                        if row:
                            extra.append({
                                "memory_id": nb["memory_id"],
                                "timestamp": row.get("timestamp", ""),
                                "tag": row.get("tag", "general"),
                                "snippet": row.get("text", ""),
                                "score": c["score"] + 0.05,
                                "via_association": True,
                                "edge_weight": nb["weight"],
                                "_debug_raw_distance": None,
                                "_debug_citation_boost": 0.0,
                                "_debug_emotion_boost": 0.0,
                                "_debug_source": "associative",
                                "_debug_associated_from": c["memory_id"],
                                "_debug_edge_weight": nb["weight"],
                            })
            candidates.extend(extra)
            candidates.sort(key=lambda m: m["score"])

        # Hebbian learning: co-activation strengthens connections
        if hebbian:
            top_ids = [c["memory_id"] for c in candidates[:n_results]]
            if len(top_ids) >= 2:
                pairs = [(top_ids[i], top_ids[j])
                         for i in range(len(top_ids))
                         for j in range(i + 1, len(top_ids))]
                self.db.connect_batch(pairs, weight=0.2)

        # Cite retrieved memories (skip if no_cite — for browsing, not recall)
        seen = set()
        memories = []
        for c in candidates:
            if c["memory_id"] in seen:
                continue
            if len(memories) >= n_results:
                break
            if not no_cite:
                self.db.cite(c["memory_id"])
            if include_context:
                ctx = self.db.get_context(c["memory_id"])
                if ctx:
                    c["context"] = ctx
            if debug:
                c["debug"] = {
                    "raw_distance": c.get("_debug_raw_distance"),
                    "citation_boost": c.get("_debug_citation_boost", 0.0),
                    "emotion_boost": c.get("_debug_emotion_boost", 0.0),
                    "final_score": c.get("score"),
                    "source": c.get("_debug_source", "unknown"),
                }
                if c.get("_debug_associated_from"):
                    c["debug"]["associated_from"] = c["_debug_associated_from"]
                    c["debug"]["edge_weight"] = c.get("_debug_edge_weight")
            # Strip internal fields (whether debug or not) — cleaner output
            for k in ("_debug_raw_distance", "_debug_citation_boost",
                     "_debug_emotion_boost", "_debug_source",
                     "_debug_associated_from", "_debug_edge_weight"):
                c.pop(k, None)
            seen.add(c["memory_id"])
            memories.append(c)

        return memories

    def _keyword_fallback(self, query: str, n_results: int = 5, tag: str = None) -> list:
        """SQLite LIKE fallback for cross-language embedding misses."""
        results = self.db.keyword_search(query, limit=n_results, tag=tag)
        return [{
            "memory_id": r["memory_id"],
            "timestamp": r.get("timestamp", ""),
            "tag": r.get("tag", "general"),
            "snippet": r.get("text", ""),
            "score": 0.6,  # Fixed score for keyword matches
            "_debug_raw_distance": None,
            "_debug_citation_boost": 0.0,
            "_debug_emotion_boost": 0.0,
            "_debug_source": "keyword",
        } for r in results]

    def consolidate(self, conversation_text: str, top_n: int = 10) -> dict:
        """Passive Hebbian update — match conversation text against memory store,
        build connections between memories that appeared in the same conversation
        even if they weren't explicitly searched.

        Args:
            conversation_text: Summary or key topics from the conversation.
            top_n: Max memories to match against.

        Returns:
            Dict with matched memory IDs and new connections made.
        """
        # Step 1: Local keyword matching (zero token cost)
        words = conversation_text.strip().split()
        matched_ids = set()

        for word in words:
            if len(word) < 2:
                continue
            results = self.db.keyword_search(word, limit=5)
            for r in results:
                matched_ids.add(r["memory_id"])

        # Step 2: Embedding match for better coverage
        if self._collection.count() > 0:
            embedding = self._embedder.encode(conversation_text).tolist()
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_n, self._collection.count()),
                include=["metadatas", "distances"],
            )
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                if dist < 0.6:  # Stricter threshold for passive matching
                    matched_ids.add(meta.get("memory_id", ""))

        matched_ids.discard("")
        matched_list = list(matched_ids)

        # Step 3: Hebbian update — all matched memories co-occurred
        new_connections = 0
        if len(matched_list) >= 2:
            pairs = [(matched_list[i], matched_list[j])
                     for i in range(len(matched_list))
                     for j in range(i + 1, len(matched_list))]
            self.db.connect_batch(pairs, weight=0.15)  # Lighter weight than search
            new_connections = len(pairs)

        # Log the consolidation event
        for mid in matched_list:
            self.db.log_event(mid, "consolidated", f"passive hebbian from conversation")

        return {
            "matched_memories": len(matched_list),
            "memory_ids": matched_list,
            "new_connections": new_connections,
        }

    def delete(self, memory_id: str) -> bool:
        """Delete a memory and its edges."""
        try:
            self._collection.delete(ids=[memory_id])
            self.db.delete(memory_id)
            return True
        except Exception:
            return False

    def dream_pass(self, short_decay_days: int = 14,
                   edge_decay_factor: float = 0.9,
                   strong_edge_decay_factor: float = 0.95,
                   emotion_nudge: float = 0.05,
                   auto_discover: bool = True) -> dict:
        """Run memory consolidation — like sleep for the brain.

        - Decay short-tier memories older than N days
        - Prune weak synaptic connections
        - Slowly decay strong manual connections
        - Auto-discover semantically close but unconnected memories
        - Equilibrate emotion scores across connected memories

        Returns:
            Dict with counts of actions taken.
        """
        results = {}

        # 1. Decay short-tier memories
        results["decayed_memories"] = self.db.decay_short(days=short_decay_days)
        if results["decayed_memories"]:
            self.reload()

        # 2. Prune weak edges
        results["pruned_edges"] = self.db.decay_edges(
            min_weight=0.1, decay_factor=edge_decay_factor
        )

        # 3. Decay strong manual edges
        results["decayed_strong"] = self.db.decay_strong_edges(
            min_weight=1.5, decay_factor=strong_edge_decay_factor
        )

        # 4. Auto-discover new connections
        if auto_discover:
            import random
            try:
                all_mems = self.db.list_all(limit=20, offset=random.randint(0, max(0, self.count() - 20)))
                auto_connected = 0
                for m in all_mems[:5]:
                    neighbors = self.search(m["text"][:100], n_results=3, associate=False, hebbian=False)
                    for nb in neighbors:
                        if nb["memory_id"] != m["memory_id"]:
                            existing = self.db.get_edge_weight(m["memory_id"], nb["memory_id"])
                            if existing is None:
                                self.db.connect(m["memory_id"], nb["memory_id"], weight=0.3)
                                auto_connected += 1
                results["auto_discovered"] = auto_connected
            except Exception:
                results["auto_discovered"] = 0

        # 5. Equilibrate emotion scores
        results["emotion_equalized"] = self.db.equalize_emotion_scores(
            nudge=emotion_nudge, threshold=0.2
        )

        # 6. Split bundled memories (if LLM available)
        try:
            split_count = self.split_bundled(batch_size=50, dry_run=False)
            results["split_memories"] = split_count
        except Exception:
            results["split_memories"] = 0

        return results

    def split_bundled(self, batch_size: int = 50, dry_run: bool = False,
                      model: str = "claude-haiku-4-5-20251001") -> int:
        """Find and split memories that bundle multiple unrelated topics.

        A memory should be about ONE independently searchable thing.
        This method uses an LLM to identify bundled memories and split them.

        Args:
            batch_size: Process memories in batches of this size.
            dry_run: If True, only identify but don't execute splits.
            model: LLM model to use for analysis.

        Returns:
            Number of memories split.
        """
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return 0
            client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            return 0

        system = (
            "You review memories for bundling. A memory should be about ONE topic.\n"
            "If a memory lists multiple unrelated things (e.g. 'built X, wrote Y, fixed Z'),\n"
            "output a JSON array of split items. Each: {\"id\": \"...\", \"into\": [{\"text\": \"...\", \"tag\": \"...\", \"tier\": \"long\"}]}\n"
            "TIMESTAMPS ARE MANDATORY in each split piece.\n"
            "Same event on different days = different memories.\n"
            "If a memory is fine as-is, skip it (don't include in output).\n"
            "Output [] if nothing to split. Valid JSON only, no markdown fences."
        )

        import json, uuid
        total_split = 0
        offset = 0

        while True:
            batch = self.db.list_all(limit=batch_size, offset=offset)
            if not batch:
                break

            mem_lines = []
            for m in batch:
                snippet = m["text"][:400] if "text" in m else m.get("snippet", "")[:400]
                mem_lines.append(f"[{m['memory_id']}] tag={m.get('tag','')} time={m.get('timestamp','')}\n{snippet}")

            try:
                response = client.messages.create(
                    model=model, max_tokens=4096, system=system,
                    messages=[{"role": "user", "content": "\n---\n".join(mem_lines)}],
                )
                raw = response.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                actions = json.loads(raw)
                for item in actions:
                    mid = item.get("id", "")
                    into = item.get("into", [])
                    if mid and len(into) >= 2 and not dry_run:
                        for sub in into:
                            sub_text = sub.get("text", "")
                            sub_tag = sub.get("tag", "general")
                            sub_tier = sub.get("tier", "long")
                            if sub_text:
                                sub_id = f"split_{uuid.uuid4().hex[:8]}"
                                self.store(sub_id, sub_text, tag=sub_tag, tier=sub_tier)
                        self.delete(mid)
                        total_split += 1
            except (json.JSONDecodeError, Exception):
                pass

            offset += batch_size

        if total_split:
            self.reload()
        return total_split
