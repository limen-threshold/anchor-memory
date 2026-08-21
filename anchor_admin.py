#!/usr/bin/env python3
"""anchor_admin — deterministic bulk maintenance for an Anchor memory store.

No LLM involved. Everything here is filter → preview → act, in that order,
and every destructive action makes a backup first.

Memories live in TWO layers: SQLite (memories.db) and the Chroma vector store
(chroma/). Deleting from only one layer leaves ghosts the other layer can
still surface — this tool always operates on both.

Usage:
    python3 anchor_admin.py DATA_DIR COMMAND [filters] [options]

Commands:
    count               how many memories match the filters
    list                print matching memories (id | tier | date | text head)
    delete              delete matching memories from BOTH layers (asks --yes)
    demote --to TIER    change matching memories' tier (e.g. core → long)
    backup              copy DATA_DIR to DATA_DIR_backup_<timestamp>
    ui [--port N]       open a local web page (127.0.0.1 only) to browse,
                        select and delete/retier with your mouse — no
                        commands needed beyond starting it

Filters (combine freely; all optional):
    --tier core|long|short
    --source SOURCE         exact match on the source column
    --since YYYY-MM-DD      timestamp >= this date
    --until YYYY-MM-DD      timestamp < this date (exclusive)
    --ids id1,id2,...       explicit id list (bypasses other filters)
    --keep id1,id2,...      ids to exclude from delete/demote

Options:
    --limit N               list: max rows (default 50)
    --yes                   delete/demote: actually do it (default: dry-run preview)
    --no-backup             delete/demote: skip the automatic backup

Examples:
    # See what a runaway import wrote
    python3 anchor_admin.py ~/anchor_data count --tier core --since 2026-08-01
    python3 anchor_admin.py ~/anchor_data list  --tier core --since 2026-08-01

    # Preview, then delete it (both layers), keeping two entries
    python3 anchor_admin.py ~/anchor_data delete --tier core --since 2026-08-01
    python3 anchor_admin.py ~/anchor_data delete --tier core --since 2026-08-01 \
        --keep mem_aaa,mem_bbb --yes

    # Or keep the text but drop the permanence
    python3 anchor_admin.py ~/anchor_data demote --tier core --to long --yes
"""
import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime


def _connect(data_dir):
    db_path = os.path.join(data_dir, "memories.db")
    if not os.path.exists(db_path):
        sys.exit(f"error: {db_path} not found — is {data_dir} an Anchor data directory?")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _has_column(conn, name):
    return any(r[1] == name for r in conn.execute("PRAGMA table_info(memories)"))


def _where(args, conn):
    if args.ids:
        ids = [i.strip() for i in args.ids.split(",") if i.strip()]
        ph = ",".join("?" for _ in ids)
        return f"memory_id IN ({ph})", ids
    clauses, params = [], []
    if args.tier:
        clauses.append("tier = ?")
        params.append(args.tier)
    if args.source:
        if not _has_column(conn, "source"):
            sys.exit("error: this store has no source column — drop --source")
        clauses.append("source = ?")
        params.append(args.source)
    if args.since:
        clauses.append("timestamp >= ?")
        params.append(args.since)
    if args.until:
        clauses.append("timestamp < ?")
        params.append(args.until)
    if not clauses:
        sys.exit("error: no filters given — refusing to match the whole store. "
                 "Use --tier/--source/--since/--until/--ids explicitly.")
    return " AND ".join(clauses), params


def _match(conn, args):
    where, params = _where(args, conn)
    rows = conn.execute(
        f"SELECT memory_id, tier, timestamp, text FROM memories WHERE {where} "
        "ORDER BY timestamp", params).fetchall()
    if args.keep:
        keep = {i.strip() for i in args.keep.split(",")}
        rows = [r for r in rows if r["memory_id"] not in keep]
    return rows


def _backup(data_dir):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = data_dir.rstrip("/") + f"_backup_{stamp}"
    shutil.copytree(data_dir, dest)
    print(f"backup: {dest}")
    return dest


def cmd_count(conn, args, data_dir):
    print(len(_match(conn, args)))


def cmd_list(conn, args, data_dir):
    rows = _match(conn, args)
    for r in rows[: args.limit]:
        head = " ".join((r["text"] or "").split())[:60]
        print(f"{r['memory_id']} | {r['tier']} | {r['timestamp'][:10]} | {head}")
    if len(rows) > args.limit:
        print(f"... and {len(rows) - args.limit} more (use --limit)")


def cmd_delete(conn, args, data_dir):
    rows = _match(conn, args)
    if not rows:
        print("nothing matches — nothing deleted")
        return
    if not args.yes:
        cmd_list(conn, args, data_dir)
        print(f"\ndry-run: {len(rows)} memories would be deleted from BOTH layers. "
              "Re-run with --yes to proceed.")
        return
    if not args.no_backup:
        _backup(data_dir)
    # Chroma layer first (ids that miss are ignored by chroma), then SQLite
    # (AnchorDB.delete also clears edges). Chroma import is local so that
    # count/list/demote work even on a machine without chromadb installed.
    import chromadb
    collection = chromadb.PersistentClient(
        path=os.path.join(data_dir, "chroma")).get_or_create_collection(name="memories")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from anchor_db import AnchorDB
    db = AnchorDB(os.path.join(data_dir, "memories.db"))
    ids = [r["memory_id"] for r in rows]
    for chunk_start in range(0, len(ids), 100):
        collection.delete(ids=ids[chunk_start:chunk_start + 100])
    for mid in ids:
        db.delete(mid)
    remaining = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    print(f"deleted {len(ids)} memories (both layers, edges cleared); "
          f"{remaining} remain in the store")


def cmd_demote(conn, args, data_dir):
    if not args.to:
        sys.exit("error: demote needs --to core|long|short")
    rows = _match(conn, args)
    if not rows:
        print("nothing matches — nothing changed")
        return
    if not args.yes:
        cmd_list(conn, args, data_dir)
        print(f"\ndry-run: {len(rows)} memories would become tier={args.to}. "
              "Re-run with --yes to proceed.")
        return
    if not args.no_backup:
        _backup(data_dir)
    ids = [r["memory_id"] for r in rows]
    ph = ",".join("?" for _ in ids)
    conn.execute(f"UPDATE memories SET tier = ? WHERE memory_id IN ({ph})",
                 [args.to] + ids)
    conn.commit()
    print(f"retiered {len(ids)} memories to {args.to}")
    # tier lives only in SQLite (Chroma metadata doesn't carry it) — no second layer to sync.


def cmd_backup(conn, args, data_dir):
    _backup(data_dir)


# ── Local web UI ─────────────────────────────────────────────────────────────
# `ui` serves a browse/select/confirm page on 127.0.0.1 only. Same rules as
# the CLI: preview before action, automatic backup before anything
# destructive, deletes hit both layers. Stdlib only.

_UI_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Anchor Admin</title>
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2em auto;
        max-width: 60em; padding: 0 1em; color: #222; }}
 table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
 th, td {{ border-bottom: 1px solid #ddd; padding: .4em .6em; text-align: left;
          font-size: .9em; }}
 .tier-core {{ color: #b00; font-weight: 600; }}
 .bar {{ background: #f5f5f5; padding: .8em 1em; border-radius: .5em;
        margin: 1em 0; }}
 .warn {{ background: #fff3e0; border: 1px solid #f0c080; padding: .8em 1em;
         border-radius: .5em; }}
 button {{ padding: .45em 1.1em; border-radius: .4em; border: 1px solid #999;
          background: #fff; cursor: pointer; }}
 button.danger {{ border-color: #b00; color: #b00; }}
 input, select {{ padding: .3em; }}
 small {{ color: #777; }}
</style></head><body>
<h2>Anchor Admin</h2>
<p><small>store: {data_dir} &nbsp;·&nbsp; {counts}</small></p>
<form method="get" class="bar">
 tier <select name="tier"><option value="">any</option>
   <option {sel_core}>core</option><option {sel_long}>long</option>
   <option {sel_short}>short</option></select>
 &nbsp; from <input name="since" value="{since}" placeholder="YYYY-MM-DD" size="10">
 &nbsp; to <input name="until" value="{until}" placeholder="YYYY-MM-DD" size="10">
 &nbsp; <button>filter</button>
</form>
{body}
</body></html>"""


def _ui_counts(conn):
    rows = conn.execute(
        "SELECT tier, COUNT(*) FROM memories GROUP BY tier ORDER BY tier").fetchall()
    total = sum(r[1] for r in rows)
    return f"{total} memories (" + ", ".join(f"{r[0]}: {r[1]}" for r in rows) + ")"


def _ui_rows(conn, q):
    clauses, params = [], []
    if q.get("tier"):
        clauses.append("tier = ?"); params.append(q["tier"])
    if q.get("since"):
        clauses.append("timestamp >= ?"); params.append(q["since"])
    if q.get("until"):
        clauses.append("timestamp < ?"); params.append(q["until"])
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return conn.execute(
        f"SELECT memory_id, tier, timestamp, text FROM memories{where} "
        "ORDER BY timestamp DESC LIMIT 500", params).fetchall()


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def cmd_ui(conn, args, data_dir):
    import http.server
    import urllib.parse
    import webbrowser

    port = args.port

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, html, code=200):
            body = html.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _page(self, q, body):
            tier = q.get("tier", "")
            return _UI_PAGE.format(
                data_dir=_esc(data_dir), counts=_esc(_ui_counts(conn)),
                sel_core="selected" if tier == "core" else "",
                sel_long="selected" if tier == "long" else "",
                sel_short="selected" if tier == "short" else "",
                since=_esc(q.get("since", "")), until=_esc(q.get("until", "")),
                body=body)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
            rows = _ui_rows(conn, q)
            tr = "".join(
                f'<tr><td><input type="checkbox" name="id" value="{_esc(r["memory_id"])}"></td>'
                f'<td>{_esc(r["memory_id"])}</td>'
                f'<td class="tier-{_esc(r["tier"])}">{_esc(r["tier"])}</td>'
                f'<td>{_esc(r["timestamp"][:10])}</td>'
                f'<td>{_esc(" ".join((r["text"] or "").split())[:70])}</td></tr>'
                for r in rows)
            body = (
                f'<form method="post" action="/act">'
                f'<input type="hidden" name="tier_f" value="{_esc(q.get("tier", ""))}">'
                f'<p>{len(rows)} match{"" if len(rows) == 1 else "es"} '
                f'(showing up to 500). '
                f'<label><input type="checkbox" '
                f'onclick="document.querySelectorAll(\'input[name=id]\')'
                f'.forEach(c=>c.checked=this.checked)"> select all shown</label></p>'
                f'<table><tr><th></th><th>id</th><th>tier</th><th>date</th>'
                f'<th>text</th></tr>{tr}</table>'
                f'<div class="bar">with selected: '
                f'<button class="danger" name="op" value="delete">delete (both layers)</button>'
                f' &nbsp; <button name="op" value="demote">change tier to</button> '
                f'<select name="to"><option>long</option><option>short</option>'
                f'<option>core</option></select>'
                f'<br><small>either action backs up the whole data folder first; '
                f'a confirmation page shows exactly what will change before '
                f'anything happens</small></div></form>')
            self._send(self._page(q, body))

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            form = urllib.parse.parse_qs(self.rfile.read(length).decode())
            ids = form.get("id", [])
            op = form.get("op", [""])[0]
            to = form.get("to", ["long"])[0]
            confirmed = form.get("confirmed", [""])[0] == "1"
            if not ids:
                self._send(self._page({}, '<p class="warn">nothing selected</p>'))
                return
            if self.path == "/act" and not confirmed:
                # Confirmation page — re-post the same ids with confirmed=1.
                hidden = "".join(
                    f'<input type="hidden" name="id" value="{_esc(i)}">' for i in ids)
                verb = ("delete from BOTH layers" if op == "delete"
                        else f"change tier to {to}")
                listing = "".join(f"<li>{_esc(i)}</li>" for i in ids[:50])
                more = (f"<li>... and {len(ids) - 50} more</li>"
                        if len(ids) > 50 else "")
                body = (
                    f'<div class="warn"><p><b>{len(ids)}</b> memories will '
                    f'{verb}. The data folder is backed up first.</p>'
                    f'<ul>{listing}{more}</ul>'
                    f'<form method="post" action="/act">{hidden}'
                    f'<input type="hidden" name="op" value="{_esc(op)}">'
                    f'<input type="hidden" name="to" value="{_esc(to)}">'
                    f'<input type="hidden" name="confirmed" value="1">'
                    f'<button class="danger">yes, do it</button> '
                    f'<a href="/">cancel</a></form></div>')
                self._send(self._page({}, body))
                return
            bak = _backup(data_dir)
            if op == "delete":
                import chromadb
                collection = chromadb.PersistentClient(
                    path=os.path.join(data_dir, "chroma")
                ).get_or_create_collection(name="memories")
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from anchor_db import AnchorDB
                db = AnchorDB(os.path.join(data_dir, "memories.db"))
                for i in range(0, len(ids), 100):
                    collection.delete(ids=ids[i:i + 100])
                for mid in ids:
                    db.delete(mid)
                done = f"deleted {len(ids)} memories (both layers, edges cleared)"
            else:
                ph = ",".join("?" for _ in ids)
                conn.execute(
                    f"UPDATE memories SET tier = ? WHERE memory_id IN ({ph})",
                    [to] + ids)
                conn.commit()
                done = f"changed {len(ids)} memories to tier={to}"
            self._send(self._page({}, (
                f'<div class="bar"><p>{_esc(done)}</p>'
                f'<p><small>backup: {_esc(bak)}</small></p>'
                f'<p><a href="/">back to the list</a></p></div>')))

    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Anchor Admin UI: {url}  (Ctrl-C to stop)")
    print("Note: the page is served on THIS machine only. If Anchor runs on a "
          "remote server, tunnel first from your own computer:\n"
          f"      ssh -L {port}:127.0.0.1:{port} user@server\n"
          f"      then open {url} in your local browser. "
          "(Don't bind this to a public address — it has no login.)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir")
    p.add_argument("command", choices=["count", "list", "delete", "demote", "backup", "ui"])
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--tier", choices=["core", "long", "short"])
    p.add_argument("--source")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--ids")
    p.add_argument("--keep")
    p.add_argument("--to", choices=["core", "long", "short"])
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--yes", action="store_true")
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    conn = _connect(data_dir)
    {"count": cmd_count, "list": cmd_list, "delete": cmd_delete,
     "demote": cmd_demote, "backup": cmd_backup,
     "ui": cmd_ui}[args.command](conn, args, data_dir)


if __name__ == "__main__":
    main()
