"""anchor_hub.py — a local page of the AI's days: heartbeat, room, journal, newspapers (v1.30).

    python3 anchor_hub.py --db-path ./anchor_data                 # http://127.0.0.1:8377
    python3 anchor_hub.py --db-path ./anchor_data --out day.html  # write today's page to a file and exit
    python3 anchor_hub.py --db-path ./anchor_data --out day.html --date 2026-11-16

Zero model, standard library only. It reads the folders anchor_presence writes —
``heartbeat/``, ``room/``, ``journal/``, ``newspaper/`` and ``presence_log.jsonl`` — and shows one day
per page with a strip of the days that have something.

What it does not show, by construction:

* the vault (never opened here);
* alarm notes (the AI was told nobody else reads them);
* anything the AI wrapped in ``<!-- private -->`` … ``<!-- /private -->``. The file on disk is
  untouched; the page puts a visible lock line in its place, so a hidden passage does not look like a
  broken page. The door signs of heartbeat, room and journal tell the AI about this marker.

It binds 127.0.0.1 unless told otherwise. There is no login: putting it on another interface means
anyone who can reach the port can read these files.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PRIVATE = re.compile(r"<!--\s*private\s*-->.*?(?:<!--\s*/private\s*-->|\Z)", re.S | re.I)
LOCK = "\n\n🔒 *A passage here is marked private. It is intact in the file and not shown on this page.*\n\n"
SECTIONS = (("heartbeat", "Heartbeat"), ("room", "Room"), ("journal", "Journal"))


def strip_private(text: str) -> tuple:
    """(text with private passages replaced by a lock line, whether any was hidden). An opening marker
    with no closing one hides everything after it — failing closed."""
    if "<!--" not in (text or ""):
        return text or "", False
    out, n = _PRIVATE.subn(LOCK, text)
    return out, bool(n)


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def md_to_html(text: str) -> str:
    """A few lines of markdown, after escaping everything: headings, bold, italics, quotes, rules, links."""
    blocks = []
    for block in re.split(r"\n{2,}", html.escape(text.strip())):
        b = block.strip()
        if not b:
            continue
        m = re.match(r"^(#{1,4}) (.+)$", b)
        if m and "\n" not in b:
            lvl = min(len(m.group(1)) + 2, 6)
            blocks.append(f"<h{lvl}>{m.group(2)}</h{lvl}>")
            continue
        if re.fullmatch(r"-{3,}", b):
            blocks.append("<hr>")
            continue
        lines = []
        for ln in b.split("\n"):
            hm = re.match(r"^(#{1,4}) (.+)$", ln)
            if hm:
                lvl = min(len(hm.group(1)) + 2, 6)
                lines.append(f"</p><h{lvl}>{hm.group(2)}</h{lvl}><p>")
            elif re.fullmatch(r"-{3,}", ln.strip()):
                lines.append("</p><hr><p>")
            elif ln.startswith("&gt; "):
                lines.append(f"<span class=\"q\">{ln[5:]}</span>")
            else:
                lines.append(ln)
        p = re.sub(r"(?:<br>)?(</p><h\d>.*?</h\d><p>|</p><hr><p>)(?:<br>)?", r"\1", "<br>".join(lines))
        p = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", p)
        p = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<em>\1</em>", p)
        p = re.sub(r"(?<![\"=>])(https?://[^\s<]+[^\s<.,;:!?)\]])", r'<a href="\1" rel="noreferrer noopener">\1</a>', p)
        blocks.append(f"<p>{p}</p>")
    return "\n".join(blocks).replace("<p></p>", "").replace("<p><br>", "<p>")


def available_dates(db_path: str) -> list:
    days = set()
    for sub, _ in SECTIONS:
        d = os.path.join(db_path, sub)
        if os.path.isdir(d):
            days.update(n[:-3] for n in os.listdir(d) if n.endswith(".md") and _DATE.match(n[:-3]))
    nd = os.path.join(db_path, "newspaper")
    for base in (nd, os.path.join(nd, "expired")):
        if os.path.isdir(base):
            days.update(n[:10] for n in os.listdir(base) if re.match(r"^\d{4}-\d{2}-\d{2}_\d{4}\.md$", n))
    return sorted(days)


def day_data(db_path: str, date: str) -> dict:
    out = {"date": date, "sections": [], "papers": [], "log": []}
    for sub, label in SECTIONS:
        raw = _read(os.path.join(db_path, sub, f"{date}.md"))
        if raw.strip():
            text, hidden = strip_private(raw)
            text = re.sub(rf"^# {date}\s*\n", "", text.lstrip(), count=1)   # the page already says the date
            out["sections"].append({"key": sub, "label": label, "text": text, "hidden": hidden})
    nd = os.path.join(db_path, "newspaper")
    for base, expired in ((nd, False), (os.path.join(nd, "expired"), True)):
        if os.path.isdir(base):
            for n in sorted(os.listdir(base)):
                if re.match(rf"^{date}_\d{{4}}\.md$", n):
                    out["papers"].append({"name": n, "expired": expired, "text": _read(os.path.join(base, n))})
    try:
        with open(os.path.join(db_path, "presence_log.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.startswith('{"ts": "' + date):
                    try:
                        out["log"].append(json.loads(line))
                    except Exception:
                        pass
    except FileNotFoundError:
        pass
    return out


CSS = """
:root{--bg:#f6f4ef;--card:#fffdf8;--ink:#2b2a27;--dim:#8a8578;--line:#e4dfd3;--accent:#8a5a44;--chip:#efe9dc}
@media (prefers-color-scheme:dark){:root{--bg:#161513;--card:#1e1d1a;--ink:#dcd8cf;--dim:#8d887c;--line:#2e2c27;--accent:#d9a48a;--chip:#26241f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 Georgia,'Songti SC','Noto Serif CJK SC',serif}
main{max-width:760px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:1.5rem;margin:0 0 2px}h2{font-size:.8rem;letter-spacing:.12em;text-transform:uppercase;color:var(--dim);margin:34px 0 10px;font-family:system-ui,sans-serif}
h3,h4,h5,h6{font-size:1rem;margin:18px 0 6px;color:var(--accent);font-family:system-ui,sans-serif}
.sub{color:var(--dim);font-size:.9rem;font-family:system-ui,sans-serif}
nav{display:flex;gap:10px;align-items:center;margin:14px 0 6px;font-family:system-ui,sans-serif;font-size:.9rem}
nav a,.days a{color:var(--accent);text-decoration:none}nav .sp{flex:1}
.days{display:flex;gap:6px;overflow-x:auto;padding:6px 0 10px;font-family:system-ui,sans-serif;font-size:.78rem}
.days a{flex:none;padding:3px 9px;border-radius:99px;background:var(--chip);white-space:nowrap}
.days a.on{background:var(--accent);color:var(--bg)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:6px 20px 14px;overflow-wrap:anywhere}
.card p{margin:10px 0}.card hr{border:0;border-top:1px solid var(--line);margin:14px 0}.card a{color:var(--accent)}
.q{display:block;border-left:3px solid var(--line);padding-left:12px;color:var(--dim);font-style:italic}
.empty{color:var(--dim);font-style:italic;font-size:.95rem}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 20px;margin-bottom:10px;overflow-wrap:anywhere}
summary{cursor:pointer;font-family:system-ui,sans-serif;font-size:.9rem;color:var(--accent)}
table{width:100%;border-collapse:collapse;font:.82rem/1.5 system-ui,sans-serif}.tw{overflow-x:auto}
td,th{text-align:left;padding:5px 10px 5px 0;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--dim);font-weight:500}
.err{color:#b4442f}footer{margin-top:40px;color:var(--dim);font:.78rem/1.5 system-ui,sans-serif}
"""


def render_day(db_path: str, date: str = None, static: bool = False) -> str:
    """One day as a complete HTML page. static=True → no navigation links (for --out)."""
    today = datetime.now().strftime("%Y-%m-%d")
    date = date if (date and _DATE.match(date)) else today
    data = day_data(db_path, date)
    dates = available_dates(db_path)
    try:
        nice = datetime.strptime(date, "%Y-%m-%d").strftime("%A, %B %d, %Y").replace(" 0", " ")
    except ValueError:
        nice = date
    earlier = [d for d in dates if d < date]
    later = [d for d in dates if d > date]
    nav = ""
    if not static:
        prev = f'<a href="/?date={earlier[-1]}">&larr; {earlier[-1]}</a>' if earlier else ""
        nxt = f'<a href="/?date={later[0]}">{later[0]} &rarr;</a>' if later else ""
        tod = '<a href="/">today</a>' if date != today else ""
        chips = "".join(f'<a href="/?date={d}" class="{"on" if d == date else ""}">{d[5:]}</a>' for d in dates[-45:])
        nav = f'<nav>{prev}<span class="sp"></span>{tod}<span class="sp"></span>{nxt}</nav><div class="days">{chips}</div>'
    body = []
    for sec in data["sections"]:
        body.append(f'<h2>{sec["label"]}</h2><div class="card">{md_to_html(sec["text"])}</div>')
    if data["papers"]:
        body.append("<h2>Newspapers</h2>")
        for p in data["papers"]:
            items = len(re.findall(r"^## \d+\. ", p["text"], re.M))
            tag = " · expired" if p["expired"] else ""
            body.append(f'<details><summary>{html.escape(p["name"][11:13] + ":" + p["name"][13:15])} · {items} item(s){tag}</summary>'
                        f'{md_to_html(p["text"])}</details>')
    if not data["sections"] and not data["papers"]:
        body.append('<p class="empty">Nothing was written on this day.</p>')
    if data["log"]:
        rows = []
        for r in data["log"]:
            tools = ", ".join(r.get("tools") or []) or "—"
            what = f'<span class="err">{html.escape(str(r["error"])[:200])}</span>' if r.get("error") else (
                f'{r.get("items", 0)} item(s)' if r.get("kind") == "newspaper" else f'{r.get("chars", 0)} chars')
            rows.append(f'<tr><td>{html.escape(str(r.get("ts", ""))[11:16])}</td><td>{html.escape(str(r.get("kind", "")))}'
                        f'{" (alarm)" if r.get("alarm") else ""}</td><td>{html.escape(tools)}</td><td>{what}</td></tr>')
        body.append('<h2>What the machine did</h2><div class="card tw"><table><tr><th>time</th><th>turn</th><th>tools called</th>'
                    f'<th>result</th></tr>{"".join(rows)}</table></div>')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(date)} · Anchor</title><style>{CSS}</style></head><body><main>'
            f'<h1>{html.escape(nice)}</h1><div class="sub">{html.escape(date)}</div>{nav}{"".join(body)}'
            f'<footer>Read from {html.escape(os.path.abspath(db_path))}. The vault and alarm notes are never shown; passages marked '
            f'&lt;!-- private --&gt; are hidden here and intact on disk.</footer></main></body></html>')


def make_handler(db_path: str):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlparse(self.path)
            if u.path != "/":
                self.send_error(404)
                return
            date = (parse_qs(u.query).get("date") or [""])[0]
            page = render_day(db_path, date if _DATE.match(date) else None).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, *a):
            pass
    return H


def main():
    ap = argparse.ArgumentParser(description="Anchor hub: a local page of the AI's days")
    ap.add_argument("--db-path", default="./anchor_data")
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 unless you mean to share these files; there is no login")
    ap.add_argument("--port", type=int, default=8377)
    ap.add_argument("--out", default="", help="write one day's page to this file and exit")
    ap.add_argument("--date", default="", help="with --out: YYYY-MM-DD (default today)")
    a = ap.parse_args()
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(render_day(a.db_path, a.date or None, static=True))
        print(f"wrote {a.out}")
        return
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[hub] listening on {a.host}: anyone who can reach this port can read these files (no login).", file=sys.stderr)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(a.db_path))
    print(f"[hub] http://{a.host}:{a.port}  (db: {os.path.abspath(a.db_path)})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
