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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir")
    p.add_argument("command", choices=["count", "list", "delete", "demote", "backup"])
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
     "demote": cmd_demote, "backup": cmd_backup}[args.command](conn, args, data_dir)


if __name__ == "__main__":
    main()
