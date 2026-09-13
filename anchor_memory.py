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


_CN_MONTH = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
             "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}
_EN_MONTH = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
_EN_MONTH.update({m[:3]: i for m, i in list(_EN_MONTH.items())})


def parse_date_range(query: str, now: datetime = None):
    """Find a date expression in `query` → (start_iso, end_iso, span_days) or None.

    Boundaries are computed on the *local* calendar (what "today" means to
    the person asking) and then shifted to UTC, because stored timestamps
    are UTC — without the shift, an evening query in UTC-7 would put "today"
    on tomorrow's UTC date and miss everything stored before 17:00.

    Recognised: 2026-03-06 / 2026年3月6日 / 3月6日|号 / 三月六日 / March 6 /
    Mar 6th / 3月 (whole month) / 今天 昨天 前天 大前天 / 上周 这周 本周 /
    上个月 / 最近 近况 这两天 这几天 recently lately (→ last three days).
    A date without a year resolves to the most recent occurrence not in the
    future (asking for "March 6" in July means this year; in February, last).
    """
    import re
    from datetime import timedelta
    now = now or datetime.now()

    def day_range(y, m, d, span=1):
        try:
            start = datetime(y, m, d)
        except ValueError:
            return None
        return (start, start + timedelta(days=span), span)

    def infer_year(m, d=1):
        y = now.year
        try:
            if datetime(y, m, d) > now:
                y -= 1
        except ValueError:
            return None
        return y

    def month_range(y, m):
        start = datetime(y, m, 1)
        nxt = datetime(y + (m == 12), m % 12 + 1, 1)
        return (start, nxt, 31)

    q = (query or "").strip()
    out = None
    mm = re.search(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})[日号]?", q)
    if mm:
        out = day_range(int(mm.group(1)), int(mm.group(2)), int(mm.group(3)))
    if out is None:
        mm = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", q)
        if mm:
            m, d = int(mm.group(1)), int(mm.group(2))
            y = infer_year(m, d)
            out = day_range(y, m, d) if y else None
    if out is None:
        mm = re.search(r"(十[一二]?|[一二三四五六七八九])\s*月\s*([\d一二三四五六七八九十]{1,3})\s*[日号]", q)
        if mm and mm.group(1) in _CN_MONTH:
            m = _CN_MONTH[mm.group(1)]
            ds = mm.group(2)
            digit = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
            if ds.isdigit():
                d = int(ds)
            elif ds == "十":
                d = 10
            elif ds.startswith("十"):
                d = 10 + digit.get(ds[1], 0)
            elif len(ds) >= 2 and ds[1] == "十":
                d = digit.get(ds[0], 0) * 10 + (digit.get(ds[2], 0) if len(ds) > 2 else 0)
            else:
                d = digit.get(ds, 0)
            y = infer_year(m, d)
            out = day_range(y, m, d) if y else None
    if out is None:
        mm = re.search(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", q)
        if mm and mm.group(1).lower() in _EN_MONTH:
            m, d = _EN_MONTH[mm.group(1).lower()], int(mm.group(2))
            y = infer_year(m, d)
            out = day_range(y, m, d) if y else None
    if out is None:
        for w, off in (("大前天", 3), ("前天", 2), ("昨天", 1), ("今天", 0), ("today", 0), ("yesterday", 1)):
            if w in q.lower():
                t = now - timedelta(days=off)
                out = day_range(t.year, t.month, t.day)
                break
    if out is None and ("上周" in q or "上星期" in q or "last week" in q.lower()):
        start = (now - timedelta(days=now.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
        out = (start, start + timedelta(days=7), 7)
    if out is None and ("这周" in q or "本周" in q or "this week" in q.lower()):
        start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        out = (start, start + timedelta(days=7), 7)
    if out is None and ("上个月" in q or "last month" in q.lower()):
        y, m = (now.year, now.month - 1) if now.month > 1 else (now.year - 1, 12)
        out = month_range(y, m)
    if out is None:
        mm = re.search(r"(?<![\d月])(\d{1,2})\s*月(?!\s*\d)", q)
        if mm and 1 <= int(mm.group(1)) <= 12:
            y = infer_year(int(mm.group(1)))
            out = month_range(y, int(mm.group(1))) if y else None
    if out is None and re.search(r"最近|近况|近来|这两天|这几天|怎么样了|怎样了|recently|lately", q, re.I):
        rs = (now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
        re_ = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        out = (rs, re_, 3)
    if out is None:
        return None
    # local calendar boundaries → UTC (stored timestamps are utcnow())
    off = now.astimezone().utcoffset() or timedelta(0)
    start, end, span = out
    return ((start - off).isoformat(), (end - off).isoformat(), span)


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
        self._db_path = os.path.join(db_path, "memories.db")
        self.db = AnchorDB(self._db_path)
        # Eager concept-based linking on store(). Default OFF in v1.7.1 after
        # over-connection issues (median degree 110, max 596 on a 1k-node graph).
        # Set True to opt-in. Requires concept_link.py and an Anthropic API key.
        self._eager_link = False
        # Recency boost: a third ranking boost beside citation/emotion, so more-
        # recent memories surface a little easier. Exponential half-life decay,
        # subtracted from score (distance semantics: lower = better). Two knobs.
        # recency_weight peaks at ~1/3 of citation's max (0.15); set to 0 to disable.
        self.recency_weight = 0.05          # max boost (age 0)
        self.recency_halflife_days = 30.0   # boost halves every N days
        # v1.14 — slot allocation gates on search(). None of these change the
        # score; they decide who gets a slot in the final top-N. Read the
        # docstring on search() for the mechanism. same_day_cap=0 disables all.
        self.same_day_cap = 2        # max results from one calendar day (UTC)
        self.older_days = 7.0        # "old" = older than this
        self.older_reserve = 2       # keep at least N old memories when available
        self.recent_days = 3.0       # "recent" = at most this old
        self.recent_reserve = 1      # keep at least N recent memories when available
        # v1.14 — store-side near-duplicate gate. On store(), if an existing
        # memory's cosine similarity to the new text is >= this, the new text is
        # NOT stored: the existing one is cited instead (usage_count+1) and its
        # id is returned. 0.93 is calibrated for the default MiniLM embedder;
        # re-calibrate if you swap models (a Qwen3 embedder needed ~0.96 for the
        # same percentile). Set 0 to disable. Text containing any of
        # near_dup_bypass_words is always stored (a correction must never be
        # swallowed by the entry it corrects — the two are near-identical by
        # construction, and the *old* one would survive).
        self.near_dup_merge_sim = 0.93
        self.near_dup_bypass_words = ("勘误", "更正", "作废", "correction")

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

    def _recency_boost(self, timestamp: str) -> float:
        """recency_weight · 0.5^(age_days / recency_halflife_days).
        Returns 0.0 on a missing/unparseable timestamp (never raises)."""
        if not timestamp or not self.recency_weight:
            return 0.0
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", ""))
        except Exception:
            return 0.0
        age_days = (datetime.utcnow() - dt).total_seconds() / 86400.0
        if age_days < 0:
            age_days = 0.0
        return self.recency_weight * (0.5 ** (age_days / self.recency_halflife_days))

    def store(self, memory_id: str, text: str, tag: str = "general",
              tier: str = "short", connect_to: list = None,
              emotion_score: float = 0.5,
              source: str = None, entity: str = None,
              context: str = "") -> str:
        """Store a memory with optional connections and emotion scoring.

        Args:
            memory_id: Unique identifier.
            text: Memory content — the searchable summary (what gets embedded).
            tag: Category tag.
            tier: 'core' (permanent), 'long' (kept), 'short' (14-day decay).
            connect_to: List of memory_ids to create edges with.
            emotion_score: 0.0 (neutral) to 1.0 (intense). Default 0.5.
            source: Optional pipeline tag (e.g. 'live_sync', 'curator',
                    'manual', 'penpal_letter'). Used by dream_extras.run_global_dedup
                    to know which memories share an event vs. duplicate it.
            entity: Optional disambiguator for multi-entity setups (e.g.
                    'pair:27|ai_name:Cheng' for penpal letter sources). Free-form
                    pipe-separated; substring-searchable.
            context: Optional full original text (two-layer storage: text = the
                    embedded search summary, context = verbatim source). Retrieved
                    with include_context=True on search. The DB column existed
                    since the context migration but store() never exposed it —
                    only the summary layer was reachable through the public API.

        Returns:
            The memory_id.
        """
        text = str(text).strip() if text is not None else ""
        if not text:
            raise ValueError("Memory text cannot be empty")
        embedding = self._embedder.encode(text).tolist()

        # v1.14 near-duplicate gate (see __init__ for the knobs). Runs before
        # the upsert so a repeat of an existing fact folds into the survivor
        # instead of becoming a second row that then competes with the first
        # at recall time. Re-storing the SAME id is an edit, not a duplicate —
        # it goes through the upsert path untouched.
        survivor = self._near_dup_survivor(memory_id, text, embedding)
        if survivor:
            try:
                self.db.cite(survivor)
            except Exception:
                pass
            self.db.log_event(survivor, "near_dup_merge",
                              f"new text dropped (would have been {memory_id}): {text[:80]}")
            print(f"[AnchorMemory] near-dup merge: '{text[:40]}' → survivor {survivor}; new not stored.")
            return survivor

        meta = {
            "memory_id": memory_id,
            "timestamp": datetime.utcnow().isoformat(),
            "tag": tag,
        }
        if source:
            meta["source"] = source
        if entity:
            meta["entity"] = entity

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

        self.db.insert(memory_id, text, tag=tag, tier=tier, emotion_score=emotion_score,
                       context=context or "")

        # Create explicit connections
        if connect_to:
            for target_id in connect_to:
                try:
                    self.db.connect(memory_id, target_id)
                except Exception:
                    pass

        # Eager concept-based linking for long/core memories (fire-and-forget).
        # Solves the cold-start edge problem: new memories get conceptual edges
        # at write time instead of waiting for hebbian co-activation. Skipped
        # for short tier (decays in 14 days, not worth the LLM cost).
        if tier in ('long', 'core') and self._eager_link:
            def _link():
                try:
                    import concept_link
                    concept_link.run(self._db_path, scope='single', single_id=memory_id)
                except Exception as e:
                    print(f"[AnchorMemory] eager concept_link failed for {memory_id}: {e}")
            import threading
            threading.Thread(target=_link, daemon=True).start()

        return memory_id

    def _near_dup_survivor(self, memory_id: str, text: str, embedding: list):
        """Return the id of an existing memory that duplicates `text`, or None.
        Never raises — a vector-store hiccup must not block a store()."""
        thr = self.near_dup_merge_sim
        if not thr or thr <= 0:
            return None
        if any(w in text for w in (self.near_dup_bypass_words or ())):
            return None
        try:
            if self._collection.count() == 0:
                return None
            nd = self._collection.query(query_embeddings=[embedding], n_results=3,
                                        include=["distances"])
            for nid, dist in zip(nd["ids"][0], nd["distances"][0]):
                if nid == memory_id:
                    continue
                if (1.0 - dist) >= thr:
                    return nid
        except Exception:
            return None
        return None

    def _age_days(self, timestamp: str) -> float:
        """Age in days on the same clock as the stored timestamps (UTC).
        Missing/unparseable → 0 (treated as new; never steals an 'old' slot)."""
        if not timestamp:
            return 0.0
        try:
            dt = datetime.fromisoformat(str(timestamp).replace("Z", ""))
        except Exception:
            return 0.0
        return max(0.0, (datetime.utcnow() - dt).total_seconds() / 86400.0)

    def _allocate_slots(self, candidates: list, n_results: int,
                        exclude_tags: tuple = ()) -> list:
        """Pick the final top-N from score-sorted candidates (v1.14).

        Three gates, all about *slots*, none about scores:
          ① same-day cap — at most `same_day_cap` results from one calendar
             day. A cluster of entries about one event (five memories stored
             the same afternoon) otherwise fills every slot and the second-
             most-relevant *topic* never surfaces. Blocked entries stay in the
             pool for ② and ③. Keyword-lane hits keep a separate day ledger so
             a literal match can't crowd out the vector lane's day quota.
          ② old-memory floor — if the top-N is full and holds fewer than
             `older_reserve` memories older than `older_days`, pull the most
             relevant old ones in and drop the least relevant young ones.
             Keeps "the first time" reachable next to "the latest time".
          ③ recent-memory floor — if the top-N is full and holds none newer
             than `recent_days`, pull in the most relevant recent one; victim
             comes from the middle age band first, then from old-memory
             surplus above `older_reserve`. Soft: gives up rather than
             drowning relevance.
        `exclude_tags` are dropped *before* slots are counted (an excluded
        tag must not consume a slot it then vacates).
        `same_day_cap == 0` → plain top-N (old behaviour).
        """
        exclude = set(exclude_tags or ())
        pool = [c for c in candidates
                if not (exclude and (c.get("tag") or "").split("|")[0] in exclude)]
        if not self.same_day_cap:
            seen, out = set(), []
            for c in pool:
                if c["memory_id"] in seen:
                    continue
                if len(out) >= n_results:
                    break
                seen.add(c["memory_id"])
                out.append(c)
            return out

        cap = self.same_day_cap
        seen, out = set(), []
        day_counts, lane_day_counts = {}, {}
        older_added = 0
        for c in pool:
            if c["memory_id"] in seen:
                continue
            if len(out) >= n_results:
                break
            day = str(c.get("timestamp") or "")[:10]
            ledger = lane_day_counts if c.get("_debug_source") == "keyword" else day_counts
            if day and ledger.get(day, 0) >= cap:
                continue
            if self._age_days(c.get("timestamp")) > self.older_days:
                older_added += 1
            out.append(c)
            seen.add(c["memory_id"])
            if day:
                ledger[day] = ledger.get(day, 0) + 1

        def _score(m):
            return m.get("score") if m.get("score") is not None else 0.0

        # ② old-memory floor
        if len(out) >= n_results and older_added < self.older_reserve:
            need = self.older_reserve - older_added
            old_pool = [c for c in pool if c["memory_id"] not in seen
                        and self._age_days(c.get("timestamp")) > self.older_days]
            for c in old_pool[:need]:
                victims = [(i, m) for i, m in enumerate(out)
                           if self._age_days(m.get("timestamp")) <= self.older_days]
                if not victims:
                    break
                vi, _ = max(victims, key=lambda im: _score(im[1]))
                out.pop(vi)
                out.append(c)
                seen.add(c["memory_id"])

        # ③ recent-memory floor
        if len(out) >= n_results and self.recent_reserve:
            have = sum(1 for m in out if self._age_days(m.get("timestamp")) <= self.recent_days)
            if have < self.recent_reserve:
                recent_pool = [c for c in pool if c["memory_id"] not in seen
                               and self._age_days(c.get("timestamp")) <= self.recent_days]
                for c in recent_pool[:self.recent_reserve - have]:
                    mid_band = [(i, m) for i, m in enumerate(out)
                                if self.recent_days < self._age_days(m.get("timestamp")) <= self.older_days]
                    old_count = sum(1 for m in out if self._age_days(m.get("timestamp")) > self.older_days)
                    old_surplus = ([(i, m) for i, m in enumerate(out)
                                    if self._age_days(m.get("timestamp")) > self.older_days]
                                   if old_count > self.older_reserve else [])
                    victims = mid_band or old_surplus
                    if not victims:
                        break
                    vi, _ = max(victims, key=lambda im: _score(im[1]))
                    out.pop(vi)
                    out.append(c)
                    seen.add(c["memory_id"])

        out.sort(key=_score)
        return out

    def search(self, query: str, n_results: int = 5, tag: str = None,
               associate: bool = True, hebbian: bool = True,
               no_cite: bool = False, include_context: bool = False,
               debug: bool = False, exclude_tags: tuple = ()) -> list:
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
            exclude_tags: Tags dropped before slots are allocated (v1.14).
                Use for material that is in the store but must not compete in
                this context — e.g. a transcript tag when recalling for a chat.

        Slot allocation (v1.14): the final top-N is chosen by
        ``_allocate_slots`` — same-day cap + old-memory floor + recent-memory
        floor. Scores are untouched; only *who gets a slot* changes. Tune or
        disable via the ``same_day_cap`` / ``older_*`` / ``recent_*``
        attributes on the instance.

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
            ts = meta.get("timestamp", "")
            citation_boost = min(self.db.get_citation_count(mid) * 0.02, 0.15)
            emotion_boost = self.db.get_emotion_score(mid) * 0.1
            recency_boost = self._recency_boost(ts)
            boost = citation_boost + emotion_boost + recency_boost
            candidates.append({
                "memory_id": mid,
                "timestamp": ts,
                "tag": meta.get("tag", "general"),
                "snippet": doc,
                "score": dist - boost,
                "_debug_raw_distance": dist,
                "_debug_citation_boost": citation_boost,
                "_debug_emotion_boost": emotion_boost,
                "_debug_recency_boost": recency_boost,
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

        # Slot allocation (v1.14), then cite/decorate the chosen set.
        memories = []
        for c in self._allocate_slots(candidates, n_results, exclude_tags):
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
                    "recency_boost": c.get("_debug_recency_boost", 0.0),
                    "final_score": c.get("score"),
                    "source": c.get("_debug_source", "unknown"),
                }
                if c.get("_debug_associated_from"):
                    c["debug"]["associated_from"] = c["_debug_associated_from"]
                    c["debug"]["edge_weight"] = c.get("_debug_edge_weight")
            # Strip internal fields (whether debug or not) — cleaner output
            for k in ("_debug_raw_distance", "_debug_citation_boost",
                     "_debug_emotion_boost", "_debug_recency_boost", "_debug_source",
                     "_debug_associated_from", "_debug_edge_weight"):
                c.pop(k, None)
            memories.append(c)

        return memories

    def search_multi(self, queries: list, n_results_per_query: int = 5,
                     n_total: int = None, tag: str = None,
                     associate: bool = True, hebbian: bool = True,
                     no_cite: bool = False, include_context: bool = False,
                     exclude_tags: tuple = ()) -> list:
        """Run multiple independent searches and merge dedup'd results.

        Designed for the case where a single user message contains several
        distinct topics — vector similarity on the whole message dilutes any
        one topic, so the caller pre-splits intents (with an LLM, sentence
        splitter, or whatever) and passes each as a separate query here.

        Behavior:
        - Each query runs an independent search at n_results_per_query depth.
        - Results dedup'd by memory_id; best score across queries wins.
        - Sorted by score and capped at n_total (default: sum of per-query caps).
        - Hebbian co-activation fires across the MERGED top set, not per query,
          so memories surfaced by different intents in the same message form
          edges with each other (this is the whole point).

        Args:
            queries: List of query strings. Empty/whitespace-only entries are
                skipped. If the list collapses to empty, returns [].
            n_results_per_query: Top-k pulled from each individual search.
            n_total: Final cap after merge. Default n_results_per_query * len(queries).
            tag, associate, no_cite, include_context: forwarded to search().
            hebbian: If True, fires once at the end against the merged top set.
                Per-query searches are run with hebbian=False to avoid double-firing.

        Returns:
            Same shape as search() — list of memory dicts.
        """
        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if not clean_queries:
            return []
        if n_total is None:
            n_total = n_results_per_query * len(clean_queries)

        merged: dict = {}
        for q in clean_queries:
            results = self.search(
                q, n_results=n_results_per_query, tag=tag,
                associate=associate, hebbian=False,
                no_cite=True,  # cite once at the end against the merged set
                include_context=include_context, exclude_tags=exclude_tags,
            )
            for r in results:
                mid = r["memory_id"]
                if mid not in merged or r["score"] < merged[mid]["score"]:
                    merged[mid] = r

        ranked = sorted(merged.values(), key=lambda m: m["score"])
        # v1.14: re-apply the same-day cap on the MERGED list. Each per-query
        # search honoured the cap on its own, but k queries can each bring
        # `same_day_cap` entries from the same day — the merged top set could
        # hold k × cap from one afternoon. Same rule, applied once more here.
        if self.same_day_cap:
            day_counts, kept = {}, []
            for m in ranked:
                day = str(m.get("timestamp") or "")[:10]
                if day and day_counts.get(day, 0) >= self.same_day_cap:
                    continue
                kept.append(m)
                if day:
                    day_counts[day] = day_counts.get(day, 0) + 1
            ranked = kept
        ranked = ranked[:n_total]

        if hebbian and len(ranked) >= 2:
            top_ids = [c["memory_id"] for c in ranked]
            pairs = [(top_ids[i], top_ids[j])
                     for i in range(len(top_ids))
                     for j in range(i + 1, len(top_ids))]
            self.db.connect_batch(pairs, weight=0.2)

        if not no_cite:
            for c in ranked:
                self.db.cite(c["memory_id"])

        return ranked

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

    def merge_memories(self, survivor_id: str, duplicate_id: str) -> dict:
        """Fold `duplicate_id` into `survivor_id`, then delete the duplicate
        from both stores. The survivor keeps its own text/vector/id; only
        metadata is consolidated. The caller decides who survives.

        Consolidation rules:
          - edges:         survivor inherits the duplicate's edges (migrate_edges);
                           colliding edges saturate-add, self-loops dropped. Else
                           the duplicate's Hebbian history evaporates on delete.
          - usage_count:   summed.
          - timestamp:     earlier of the two (protects recency from a late dup).
          - pinned:        OR.
          - emotion_score: max (keep the heavier charge).
          - tag/tier/text/context: survivor's kept (folding metadata, not
                           rewriting the survivor).
        Returns a summary dict. Raises if an id is missing or they're equal.
        """
        if survivor_id == duplicate_id:
            raise ValueError("survivor_id == duplicate_id — nothing to merge")
        s = self.db.get(survivor_id)
        d = self.db.get(duplicate_id)
        if s is None:
            raise ValueError(f"survivor {survivor_id} not found")
        if d is None:
            raise ValueError(f"duplicate {duplicate_id} not found")

        new_usage = (s.get("usage_count") or 0) + (d.get("usage_count") or 0)
        new_ts = min(s["timestamp"], d["timestamp"])          # ISO strings sort chronologically
        new_pinned = 1 if (s.get("pinned") or d.get("pinned")) else 0
        s_emo = s.get("emotion_score") if s.get("emotion_score") is not None else 0.5
        d_emo = d.get("emotion_score") if d.get("emotion_score") is not None else 0.5
        new_emo = max(s_emo, d_emo)

        edges_migrated = self.db.migrate_edges(duplicate_id, survivor_id)

        with self.db._conn() as conn:
            conn.execute(
                "UPDATE memories SET usage_count = ?, timestamp = ?, pinned = ?, emotion_score = ? "
                "WHERE memory_id = ?",
                (new_usage, new_ts, new_pinned, new_emo, survivor_id),
            )
            conn.commit()

        deleted = self.delete(duplicate_id)  # both stores (SQLite row + vector)
        self.db.log_event(
            survivor_id, "merged",
            f"from={duplicate_id} edges_migrated={edges_migrated} "
            f"usage={new_usage} ts={new_ts} pinned={new_pinned} emotion={new_emo:.2f}",
        )
        return {
            "survivor": survivor_id,
            "duplicate_deleted": deleted,
            "edges_migrated": edges_migrated,
            "usage_count": new_usage,
            "timestamp": new_ts,
            "pinned": new_pinned,
            "emotion_score": new_emo,
        }

    def read_by_date(self, date_text: str, days: int = 1, limit: int = 1000):
        """Calendar lookup (v1.14): memories in a date range, oldest first.

        `date_text` is parsed by ``parse_date_range`` — ISO dates, Chinese
        (3月6日 / 三月六日 / 昨天 / 上周 / 上个月 / 3月), English (March 6),
        plus "最近/recently" → the last three days. `days` extends a single-
        day match forward (capped at 31). Returns (start_iso, end_iso, rows);
        rows is None when the date could not be parsed.

        Semantic search answers "what do I know about X". This answers "what
        happened on the 6th" — a question similarity ranking is bad at, and
        one you shouldn't have to gamble on when you already know the date.
        """
        from datetime import timedelta
        dr = parse_date_range(date_text)
        if not dr:
            return None, None, None
        start_iso, end_iso, _span = dr
        days = int(days or 1)
        if days > 1:
            end_iso = (datetime.fromisoformat(start_iso) + timedelta(days=min(days, 31))).isoformat()
        return start_iso, end_iso, self.db.memories_by_date_range(start_iso, end_iso, limit=limit)

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
        # v1.9: use anchor_llm. The model= kwarg is honored for backward compat
        # when ANTHROPIC_API_KEY is set, otherwise fall back to configured default.
        try:
            from anchor_llm import get_default_llm, AnthropicLLM, ConfigError
        except ImportError:
            return 0
        try:
            if model and os.getenv("ANTHROPIC_API_KEY"):
                llm_inst = AnthropicLLM(model=model)
            else:
                llm_inst = get_default_llm()
        except ConfigError:
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
                # 1h TTL: split_bundled walks the entire memory store; cache
                # the split-rules system prompt across the whole pass.
                response = llm_inst.call(
                    system=system,
                    user="\n---\n".join(mem_lines),
                    max_tokens=4096,
                    cache_ttl="1h",
                )
                raw = response.text.strip()
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
