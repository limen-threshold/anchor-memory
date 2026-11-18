"""anchor_newspaper.py — the little bird: a newspaper fetched from the AI's own interest file (v1.30).

    python3 anchor_newspaper.py --db-path ./anchor_data --fly            # one trip now
    python3 anchor_newspaper.py --db-path ./anchor_data --fly --dry      # show what it would bring, write nothing
    python3 anchor_newspaper.py --db-path ./anchor_data --status         # how many papers are in the basket
    python3 anchor_newspaper.py --db-path ./anchor_data --expire         # move old papers out

Everything lives in ``<db_path>/newspaper/``:

    interests.md             written and edited by the AI (write_interests) — the only thing the bird reads
    2026-11-17_0612.md       one paper per trip; every item stamped with source, published date, fetched date
    expired/                 papers older than EXPIRE_DAYS; deleted after PURGE_DAYS
    .seen.jsonl              titles and links already brought home (never brought twice)

A trip: read the interest file → decide a few directions → fetch candidates from the pools → drop what
was seen before → pick at most ``max`` items → write the paper. The bird never reads the memory store,
the transcript or the vault: what the AI wants to read about is what it wrote in its own file.

Pools (none needs a key): arXiv, the Met Museum collection, and any RSS/Atom feed listed in the
interest file. Web search (Tavily) joins when TAVILY_API_KEY is set, and is silently absent otherwise.

Two small editor calls make the paper better and are optional: a scout that turns the interest file
into search words (plus one unfamiliar direction), and a picker that keeps only what is really
related. They use anchor_llm (ANCHOR_LLM / ~/.anchor/config.yaml) or, failing that, the presence
upstream (ANCHOR_UPSTREAM_*). No model at all → the ``topic:`` lines of the interest file are searched
literally and the first few results are brought back unpicked. Still a paper, just a rougher one.

Interest file lines the bird understands (everything else is free text for the scout to read):

    topic: continual learning; episodic memory      a direction, searched literally when there is no model
    feed: https://example.org/rss  Some label        an RSS/Atom feed to check every trip
    pools: arxiv, met, tavily                       which pools to visit (default: all available)
    max: 3                                          items per paper (default 3, at most 6)
    novelty: off                                    no unfamiliar direction (default on, needs a model)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EXPIRE_DAYS, PURGE_DAYS = 5, 7
MAX_FETCH_BYTES = 2_000_000
UA = {"User-Agent": "anchor-memory-newspaper/1.30"}
INTERESTS = "interests.md"
INTERESTS_MAX = 20_000
_PAPER = re.compile(r"^(\d{4}-\d{2}-\d{2})_\d{4}\.md$")
POOLS = ("arxiv", "met", "tavily")
POOL_LABEL = {"arxiv": "arXiv", "met": "The Met collection", "tavily": "web search"}

TEMPLATE = """# Interests

What I would like the bird to bring back. This file is mine; the bird reads nothing else.
Free text is fine. Lines in these forms are also understood:

topic: (a direction, e.g. "tide pools; marine biology")
feed: (an RSS/Atom address, then a label)

More of:
Less of:
"""

TOOL_DEFS = [
    {"name": "list_newspapers",
     "description": ("Papers the bird brought back and that have not expired yet (newest first), with item counts. "
                     "Reading them is up to you."),
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "read_newspaper",
     "description": "Read one paper from the basket. name = a file name from list_newspapers; omit it for the latest.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}}},
    {"name": "read_interests",
     "description": "Read your interest file — the only thing the bird looks at when it decides what to fetch.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "write_interests",
     "description": ("Edit your interest file. mode=append adds a line or a paragraph (e.g. 'more of: …', 'less of: …', "
                     "'topic: …', 'feed: <rss address> <label>'); mode=overwrite replaces the file (the previous version "
                     "is kept beside it). The next trip reads it."),
     "inputSchema": {"type": "object", "properties": {
         "content": {"type": "string"},
         "mode": {"type": "string", "enum": ["append", "overwrite"], "default": "append"}},
         "required": ["content"]}},
]
TOOL_NAMES = {t["name"] for t in TOOL_DEFS}


# ── basket ────────────────────────────────────────────────────────────────────

def basket_dir(db_path: str) -> str:
    return os.path.join(db_path or ".", "newspaper")


def read_interests(db_path: str) -> str:
    p = os.path.join(basket_dir(db_path), INTERESTS)
    if not os.path.exists(p):
        return ""
    with open(p, encoding="utf-8") as f:
        return f.read()


def write_interests(db_path: str, content: str, mode: str = "append") -> str:
    content = (content or "").strip()
    if not content:
        return "Nothing written: content was empty."
    d = basket_dir(db_path)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, INTERESTS)
    old = read_interests(db_path)
    new = content if (mode == "overwrite" or not old.strip()) else old.rstrip() + "\n" + content
    if len(new) > INTERESTS_MAX:
        return f"Not written: the file would be {len(new)} chars (limit {INTERESTS_MAX}). Trim it first."
    if mode == "overwrite" and old.strip():
        with open(os.path.join(d, "interests.prev.md"), "w", encoding="utf-8") as f:
            f.write(old)
    with open(p, "w", encoding="utf-8") as f:
        f.write(new.rstrip() + "\n")
    return f"Interest file {'replaced' if mode == 'overwrite' and old.strip() else 'updated'} ({len(new)} chars). The next trip reads it."


def papers(db_path: str) -> list:
    """Papers in the basket (not expired), oldest first: [(file name, day)]."""
    d = basket_dir(db_path)
    if not os.path.isdir(d):
        return []
    return [(n, _PAPER.match(n).group(1)) for n in sorted(os.listdir(d)) if _PAPER.match(n)]


def read_paper(db_path: str, name: str = "") -> tuple:
    """(file name, text). name empty → latest. Unknown / unsafe name → ('', '')."""
    ps = papers(db_path)
    name = (name or "").strip()
    if not name:
        if not ps:
            return "", ""
        name = ps[-1][0]
    name = os.path.basename(name)
    if not _PAPER.match(name):
        return "", ""
    for base in (basket_dir(db_path), os.path.join(basket_dir(db_path), "expired")):
        p = os.path.join(base, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return name, f.read()
    return "", ""


def expire(db_path: str, today=None) -> dict:
    d = basket_dir(db_path)
    if not os.path.isdir(d):
        return {"moved": 0, "purged": 0}
    old = os.path.join(d, "expired")
    today = today or datetime.now().date()
    moved = purged = 0
    for name, day in papers(db_path):
        if (today - datetime.strptime(day, "%Y-%m-%d").date()).days >= EXPIRE_DAYS:
            os.makedirs(old, exist_ok=True)
            os.replace(os.path.join(d, name), os.path.join(old, name)); moved += 1
    if os.path.isdir(old):
        for name in os.listdir(old):
            m = _PAPER.match(name)
            if m and (today - datetime.strptime(m.group(1), "%Y-%m-%d").date()).days >= PURGE_DAYS:
                os.remove(os.path.join(old, name)); purged += 1
    return {"moved": moved, "purged": purged}


def status_line(db_path: str) -> str:
    """One line for wakeup / the heartbeat prompt: a count and where to look — never the content."""
    try:
        ps = papers(db_path)
        if not ps:
            return ""
        latest = ps[-1][0][:15].replace("_", " ")
        return (f"The bird brought back {len(ps)} newspaper(s) (latest trip {latest[:10]} {latest[11:13]}:{latest[13:15]}). "
                "list_newspapers / read_newspaper if you want to look; reading them is up to you.")
    except Exception:
        return ""


def handle(db_path: str, name: str, args: dict) -> dict:
    args = args or {}
    if name == "list_newspapers":
        out = []
        for n, day in reversed(papers(db_path)):
            _n, text = read_paper(db_path, n)
            out.append({"name": n, "day": day, "items": len(re.findall(r"^## \d+\. ", text, re.M)), "chars": len(text)})
        return {"papers": out, "basket": "newspaper/", "expires_after_days": EXPIRE_DAYS}
    if name == "read_newspaper":
        n, text = read_paper(db_path, args.get("name", ""))
        if not n:
            return {"exists": False, "note": "No such paper in the basket (see list_newspapers)."}
        return {"name": n, "exists": True, "content": text}
    if name == "read_interests":
        t = read_interests(db_path)
        return {"exists": bool(t), "content": t or "(no interest file yet — write_interests creates it)"}
    if name == "write_interests":
        return {"result": write_interests(db_path, args.get("content", ""), args.get("mode", "append"))}
    return {"error": f"unknown newspaper tool {name}"}


# ── seen ledger ───────────────────────────────────────────────────────────────

def _seen_path(db_path: str) -> str:
    return os.path.join(basket_dir(db_path), ".seen.jsonl")


def load_seen(db_path: str) -> set:
    out = set()
    try:
        with open(_seen_path(db_path), encoding="utf-8") as f:
            for line in f:
                try:
                    out.add(json.loads(line)["key"])
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return out


def mark_seen(db_path: str, items: list) -> None:
    os.makedirs(basket_dir(db_path), exist_ok=True)
    with open(_seen_path(db_path), "a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"key": it["key"], "title": it["title"], "url": it.get("url", ""),
                                "ts": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False) + "\n")


def _key(url: str, title: str) -> str:
    return hashlib.sha1((url or title).strip().lower().encode("utf-8")).hexdigest()[:16]


# ── pools ─────────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 25, data: bytes = None, headers: dict = None) -> bytes:
    """http(s) only — a ``feed:`` line is text someone typed, and urllib would happily open file://."""
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError("only http(s) addresses are fetched")
    req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(MAX_FETCH_BYTES)


def _arxiv_query(query: str) -> str:
    """Two or more phrases (separated by ; or ,) → the first two AND-ed on the abstract; a bare string of
    words → each word AND-ed anywhere. One whole sentence against all fields sorted by date returns whatever
    was submitted this morning; three AND-ed phrases usually return nothing."""
    parts = [p.strip() for p in re.split(r"[;,，；]", query) if p.strip()]
    if len(parts) >= 2:
        return " AND ".join('abs:"%s"' % p.replace('"', "") for p in parts[:2])
    words = [w for w in re.split(r"\s+", query.strip()) if w]
    return " AND ".join("all:" + w for w in words[:4]) or "all:" + query


_ARXIV_LAST = [0.0]


def pool_arxiv(query: str, n: int = 8) -> list:
    wait = 3.0 - (time.time() - _ARXIV_LAST[0])   # arXiv asks for three seconds between requests; faster gets 429
    if 0 < wait <= 3.0:
        time.sleep(wait)
    _ARXIV_LAST[0] = time.time()
    q = urllib.parse.quote(_arxiv_query(query))
    url = f"https://export.arxiv.org/api/query?search_query={q}&sortBy=submittedDate&sortOrder=descending&max_results={n}"
    root = ET.fromstring(_get(url))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        title = " ".join((e.findtext("a:title", "", ns) or "").split())
        link = e.findtext("a:id", "", ns) or ""
        summ = " ".join((e.findtext("a:summary", "", ns) or "").split())
        authors = [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)][:3]
        out.append({"pool": "arxiv", "title": title, "url": link, "date": (e.findtext("a:published", "", ns) or "")[:10],
                    "blurb": summ[:600], "by": ", ".join(x for x in authors if x), "key": _key(link, title)})
    return out


def pool_met(query: str, n: int = 5) -> list:
    base = "https://collectionapi.metmuseum.org/public/collection/v1"
    # search moved to v1.1 (paginated; the v1 search endpoint answers 410 since 2026-10-01); objects are still served from v1
    ids = json.loads(_get(f"{base}.1/search?hasImages=true&q={urllib.parse.quote(query)}&limit={n * 3}&offset=0")).get("objectIDs") or []
    out = []
    for oid in ids[: n * 3]:
        if len(out) >= n:
            break
        try:
            o = json.loads(_get(f"{base}/objects/{oid}", timeout=15))
        except Exception:
            continue
        if not o.get("title"):
            continue
        blurb = " · ".join(x for x in (o.get("objectName"), o.get("medium"), o.get("culture"), o.get("department"),
                                       (o.get("dimensions") or "")[:60]) if x)
        out.append({"pool": "met", "title": o["title"], "url": o.get("objectURL", ""),
                    "date": o.get("objectDate") or str(o.get("objectBeginDate") or ""), "blurb": blurb[:400],
                    "by": o.get("artistDisplayName") or "", "image": o.get("primaryImageSmall") or "",
                    "key": _key(o.get("objectURL", ""), o["title"])})
    return out


def _strip_tags(s: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", s or "").split())


def parse_feed(raw: bytes, label: str, n: int = 6) -> list:
    root = ET.fromstring(raw)
    out = []
    for it in root.iter("item"):   # RSS 2.0
        title = " ".join((it.findtext("title") or "").split()); link = (it.findtext("link") or "").strip()
        out.append({"pool": label, "title": title, "url": link, "date": (it.findtext("pubDate") or "")[:25],
                    "blurb": _strip_tags(it.findtext("description"))[:500], "by": "", "key": _key(link, title)})
        if len(out) >= n:
            return out
    if out:
        return out
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in root.findall("a:entry", ns):
        title = " ".join((e.findtext("a:title", "", ns) or "").split())
        link = ""
        for l in e.findall("a:link", ns):
            if l.get("rel") in (None, "alternate"):
                link = l.get("href", ""); break
        desc = _strip_tags(e.findtext("a:summary", "", ns) or e.findtext("a:content", "", ns))
        pub = (e.findtext("a:updated", "", ns) or e.findtext("a:published", "", ns) or "")[:10]
        out.append({"pool": label, "title": title, "url": link, "date": pub, "blurb": desc[:500], "by": "", "key": _key(link, title)})
        if len(out) >= n:
            break
    return out


def pool_feed(url: str, label: str, n: int = 6) -> list:
    return parse_feed(_get(url), label, n)


def pool_tavily(query: str, n: int = 5) -> list:
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        return []
    body = json.dumps({"api_key": key, "query": query, "max_results": n, "search_depth": "basic", "include_answer": False}).encode()
    d = json.loads(_get("https://api.tavily.com/search", timeout=30, data=body, headers={"Content-Type": "application/json"}))
    return [{"pool": "tavily", "title": x.get("title", ""), "url": x.get("url", ""), "date": (x.get("published_date") or "")[:10],
             "blurb": (x.get("content") or "")[:500], "by": "", "key": _key(x.get("url", ""), x.get("title", ""))}
            for x in d.get("results", [])]


# ── the interest file, read mechanically ──────────────────────────────────────

def parse_interests(text: str) -> dict:
    cfg = {"topics": [], "feeds": [], "pools": None, "max": 3, "novelty": True}
    for line in (text or "").splitlines():
        m = re.match(r"\s*(?:[-*]\s*)?(topic|feed|pools|max|novelty)\s*[:：]\s*(.+?)\s*$", line, re.I)
        if not m:
            continue
        k, v = m.group(1).lower(), m.group(2).strip()
        if v.startswith("("):          # the template's explanatory placeholders
            continue
        if k == "topic":
            cfg["topics"].append(v)
        elif k == "feed":
            parts = v.split(None, 1)
            if urllib.parse.urlparse(parts[0]).scheme in ("http", "https"):
                cfg["feeds"].append({"url": parts[0], "label": (parts[1] if len(parts) > 1 else "feed").strip()[:40]})
        elif k == "pools":
            cfg["pools"] = [p for p in re.split(r"[,\s，]+", v.lower()) if p in POOLS]
        elif k == "max":
            try:
                cfg["max"] = max(1, min(6, int(v)))
            except ValueError:
                pass
        elif k == "novelty":
            cfg["novelty"] = v.lower() not in ("off", "no", "0", "false")
    if cfg["pools"] is None:
        cfg["pools"] = list(POOLS)
    if not os.getenv("TAVILY_API_KEY"):
        cfg["pools"] = [p for p in cfg["pools"] if p != "tavily"]
    return cfg


# ── two optional editors ──────────────────────────────────────────────────────

def default_editor():
    """callable(system, user) -> str | None, or None when no model is configured anywhere."""
    try:
        import anchor_llm
        llm = anchor_llm.get_default_llm()

        def _ed(system, user):
            try:
                return llm.call(system, user, max_tokens=800, temperature=0.4).text
            except Exception as e:
                print(f"[newspaper] editor call failed: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                return None
        return _ed
    except Exception:
        pass
    try:
        import anchor_agent
        up = anchor_agent.Upstream()
        if up.ready():
            def _ed2(system, user):
                try:
                    r = up.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
                    return r["choices"][0]["message"].get("content")
                except Exception as e:
                    print(f"[newspaper] editor call failed: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                    return None
            return _ed2
    except Exception:
        pass
    return None


def _json_in(text):
    if not text or not isinstance(text, str):
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


SCOUT_SYSTEM = (
    "You are the scout for a small newspaper fetched for one reader. You see only the reader's own interest file. "
    "Do not guess beyond it. Decide 2-3 directions for this trip and write search words for each pool. Rules: "
    "(1) directions come from the file; avoid what it says it wants less of, prefer what it wants more of. "
    "(2) {novel_rule} "
    "(3) arxiv: 2-3 English academic phrases separated by semicolons — each must plausibly appear in an abstract, so "
    "neither too narrow nor too broad. met: one English noun phrase (an object, a school, a region, a material). "
    "tavily: one natural-language sentence. Leave a pool's field empty when the direction does not suit it. "
    "Write direction names in the language of the interest file. Output JSON only: "
    '{{"directions":[{{"name":"…","novel":false,"arxiv":"…","met":"…","tavily":"…"}}]}}'
)
NOVEL_ON = ("Add exactly one more direction marked novel=true that has nothing to do with the file but that a curious "
            "mind might glance at (astronomy, the sea, a craft, the history of a small place — pick at random), and name "
            "it after its subject.")
NOVEL_OFF = "Do not add directions that are not in the file."

PICK_SYSTEM = (
    "You are the picking editor of a small newspaper for one reader. Below are the reader's own interest file and "
    "this trip's candidates. Pick 0 to {n} worth bringing home: each must really relate to what the file says, or be a "
    "good find in the direction marked unfamiliar. Candidates that merely share a word with a direction are not "
    "related — drop them. Bringing fewer is fine; bringing none is fine. For each pick write one opener: a single "
    "sentence in the language of the interest file, the kind of thing said while handing a paper over. It does not "
    "summarise, does not conclude for the reader, and does not address the reader by any name or title. "
    'Output JSON only: {{"picks":[{{"i":<candidate number>,"opener":"…"}}]}}'
)


def scout(interest: str, cfg: dict, editor=None) -> list:
    literal = [{"name": t, "novel": False, "arxiv": t if "arxiv" in cfg["pools"] else "",
                "met": re.split(r"[;,，；]", t)[0].strip() if "met" in cfg["pools"] else "",
                "tavily": t if "tavily" in cfg["pools"] else ""} for t in cfg["topics"][:3]]
    if not editor or not interest.strip():
        return literal
    out = editor(SCOUT_SYSTEM.format(novel_rule=NOVEL_ON if cfg.get("novelty", True) else NOVEL_OFF),
                 f"Pools available this trip: {', '.join(cfg['pools']) or '(feeds only)'}\n\nInterest file:\n{interest[:6000]}")
    data = _json_in(out) or {}
    dirs = data.get("directions") if isinstance(data, dict) else None
    if not isinstance(dirs, list) or not dirs:
        return literal
    clean = []
    for d in dirs[:4]:
        if isinstance(d, dict) and d.get("name"):
            clean.append({"name": str(d["name"])[:80], "novel": bool(d.get("novel")),
                          **{p: str(d.get(p) or "")[:200] if p in cfg["pools"] else "" for p in POOLS}})
    return clean or literal


def fetch_candidates(dirs: list, cfg: dict, seen: set, pools: dict = None, log=None) -> list:
    pools = pools or {"arxiv": pool_arxiv, "met": pool_met, "tavily": pool_tavily, "feed": pool_feed}
    log = log or (lambda m: None)
    cands = []
    for d in dirs:
        for p in POOLS:
            if p in cfg["pools"] and d.get(p):
                try:
                    got = pools[p](d[p])
                    for c in got:
                        c["direction"] = d.get("name", ""); c["novel"] = bool(d.get("novel"))
                    cands += got
                except Exception as e:
                    log(f"{p} error: {type(e).__name__}: {str(e)[:80]}")
    for feed in cfg["feeds"]:
        try:
            cands += pools["feed"](feed["url"], feed["label"])
        except Exception as e:
            log(f"feed {feed['label']} error: {type(e).__name__}: {str(e)[:80]}")
    uniq, keys = [], set()
    for c in cands:
        if not c.get("title") or c["key"] in seen or c["key"] in keys:
            continue
        keys.add(c["key"]); uniq.append(c)
    return uniq


def pick(interest: str, cands: list, max_items: int, editor=None) -> list:
    if not cands:
        return []
    if editor:
        listing = "\n".join(
            f"{i + 1}. [{c['pool']}{' · unfamiliar' if c.get('novel') else ''}] {c['title']} ({c.get('date', '')}) — {c.get('blurb', '')[:220]}"
            for i, c in enumerate(cands[:40]))
        data = _json_in(editor(PICK_SYSTEM.format(n=max_items), f"Interest file:\n{interest[:4000]}\n\nCandidates:\n{listing}"))
        picks = data.get("picks") if isinstance(data, dict) else None
        if isinstance(picks, list):
            chosen, used = [], set()
            for p in picks[:max_items]:
                try:
                    i = int(p["i"]) - 1
                    if i < 0 or i in used:
                        continue
                    c = cands[:40][i]
                except Exception:
                    continue
                used.add(i)
                chosen.append(dict(c, opener=" ".join(str(p.get("opener", "")).split())[:200]))
            return chosen
    # no model (or it answered garbage): take turns across pools so one source does not fill the paper
    by_pool, order = {}, []
    for c in cands:
        if c["pool"] not in by_pool:
            order.append(c["pool"])
        by_pool.setdefault(c["pool"], []).append(c)
    out = []
    while len(out) < max_items and any(by_pool.values()):
        for p in order:
            if by_pool[p] and len(out) < max_items:
                out.append(dict(by_pool[p].pop(0), opener=""))
    return out


# ── the paper ─────────────────────────────────────────────────────────────────

def write_paper(db_path: str, items: list, dirs: list, now: datetime = None, note: str = "") -> str:
    now = now or datetime.now()
    d = basket_dir(db_path); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, now.strftime("%Y-%m-%d_%H%M") + ".md")
    exp = (now + timedelta(days=EXPIRE_DAYS)).strftime("%Y-%m-%d")
    names = ", ".join(x.get("name", "") + (" (unfamiliar)" if x.get("novel") else "") for x in dirs) or "feeds only"
    lines = [f"# Newspaper · {now.strftime('%Y-%m-%d %H:%M')}", "",
             f"Directions this trip: {names}. {len(items)} item(s). Leaves the basket on {exp}.", ""]
    for i, it in enumerate(items, 1):
        lines.append(f"## {i}. {it['title']}")
        lines.append(f"Source: {POOL_LABEL.get(it['pool'], it['pool'])}" + (f" · {it['by']}" if it.get("by") else "")
                     + f" · published/dated: {it.get('date') or 'unknown'} · fetched: {now.strftime('%Y-%m-%d')}")
        if it.get("url"):
            lines.append(f"Link: {it['url']}")
        if it.get("image"):
            lines.append(f"Image: {it['image']}")
        if it.get("blurb"):
            lines.append(f"Excerpt (the source's own words): {it['blurb']}")
        if it.get("opener"):
            lines.append(f"> {it['opener']}")
        lines.append("")
    lines.append("---")
    lines.append("Reading this is up to you, and so is saying anything about it. For more or less of something, "
                 "add a line to your interest file (write_interests); the next trip reads it.")
    if note:
        lines.append(f"({note})")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def fly(db_path: str, dry: bool = False, editor="default", pools: dict = None, now: datetime = None) -> dict:
    """One trip. Returns {"path", "items", "directions", "note"}; path is '' when nothing was written."""
    logs = []
    interest = read_interests(db_path)
    if not interest.strip():
        if not dry:
            os.makedirs(basket_dir(db_path), exist_ok=True)
            p = os.path.join(basket_dir(db_path), INTERESTS)
            if not os.path.exists(p):
                with open(p, "w", encoding="utf-8") as f:
                    f.write(TEMPLATE)
        return {"path": "", "items": [], "directions": [], "note": "no interest file yet — a template was left in newspaper/interests.md; the bird stays home until something is written there"}
    cfg = parse_interests(interest)
    base_ed = default_editor() if editor == "default" else editor
    silent = []

    def ed(system, user):   # an editor that is configured but does not answer must show up in the paper's footnote
        out = base_ed(system, user)
        if not _json_in(out):
            silent.append(1)
        return out
    if not base_ed:
        ed = None
    dirs = scout(interest, cfg, ed)
    if not dirs and not cfg["feeds"]:
        return {"path": "", "items": [], "directions": [],
                "note": "nothing to search for: no model is configured to read the file, and it has no 'topic:' or 'feed:' line"}
    cands = fetch_candidates(dirs, cfg, load_seen(db_path), pools, logs.append)
    items = pick(interest, cands, cfg["max"], ed)
    note = "; ".join(x for x in (
        "" if ed else "no editor model configured: topics searched literally, nothing was picked over",
        f"the editor model gave no usable answer {len(silent)} time(s): that step fell back to literal topics / unpicked results" if silent else "",
        *logs) if x)
    out = {"path": "", "items": items, "directions": dirs, "candidates": len(cands), "note": note}
    if dry or not items:
        return out
    out["path"] = write_paper(db_path, items, dirs, now, note)
    mark_seen(db_path, items)
    return out


def tick(db_path: str, hours: list, now: datetime = None, **kw) -> dict:
    """For cron / a scheduler: fly once per listed local hour per day, then clear expired papers."""
    now = now or datetime.now()
    sp = os.path.join(basket_dir(db_path), ".state.json")
    try:
        with open(sp, encoding="utf-8") as f:
            st = json.load(f)
    except Exception:
        st = {}
    flew = []
    for h in hours:
        key = f"{now.strftime('%Y-%m-%d')}:{int(h):02d}"
        if now.hour >= int(h) and key not in st:
            st[key] = now.isoformat(timespec="seconds")
            flew.append(fly(db_path, now=now, **kw))
    cutoff = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    st = {k: v for k, v in st.items() if k[:10] >= cutoff}
    os.makedirs(basket_dir(db_path), exist_ok=True)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    return {"flew": flew, "expired": expire(db_path, now.date())}


def main():
    ap = argparse.ArgumentParser(description="Anchor newspaper: the little bird")
    ap.add_argument("--db-path", default="./anchor_data")
    ap.add_argument("--fly", action="store_true", help="one trip now")
    ap.add_argument("--dry", action="store_true", help="with --fly: print what it would bring, write nothing")
    ap.add_argument("--tick", action="store_true", help="for cron: fly when one of --hours has come, clear expired")
    ap.add_argument("--hours", default="6", help="local hours for --tick, e.g. 6,18")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--expire", action="store_true")
    a = ap.parse_args()
    if a.fly:
        r = fly(a.db_path, dry=a.dry)
        print(json.dumps({"path": r["path"], "directions": [d.get("name") for d in r["directions"]], "candidates": r.get("candidates", 0),
                          "items": [{k: it.get(k) for k in ("pool", "title", "date", "url", "opener")} for it in r["items"]],
                          "note": r["note"]}, ensure_ascii=False, indent=1))
    elif a.tick:
        r = tick(a.db_path, [int(h) for h in a.hours.split(",") if h.strip()])
        print(json.dumps({"flew": [x["path"] or x["note"] for x in r["flew"]], "expired": r["expired"]}, ensure_ascii=False))
    elif a.expire:
        print(json.dumps(expire(a.db_path)))
    elif a.status:
        print(status_line(a.db_path) or "The basket is empty.")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
