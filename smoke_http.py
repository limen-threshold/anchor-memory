#!/usr/bin/env python3
"""v1.16 smoke: the HTTP transport answers the same as stdio.

Starts anchor_mcp.py --http on a free port against a throwaway db, then:
  1. Streamable HTTP: initialize → tools/list → tools/call(store) → tools/call(search)
     (+ notification → 202, batch, bad token → 401, GET → 405)
  2. Legacy SSE: GET /sse gives an endpoint event; POST /messages answers over the stream
  3. stdio parity: same initialize/tools/list over the stdio transport → identical tool list

Run:  python3 smoke_http.py        (needs fastapi + uvicorn, like the proxy)
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TOKEN = "smoke-token"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def post(url, obj, token=TOKEN, raw=False):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Accept": "application/json, text/event-stream",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
            return r.status, dict(r.headers), (body if raw else (json.loads(body) if body else None))
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), None


def rpc(id_, method, params=None):
    m = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        m["params"] = params
    return m


def main():
    db = tempfile.mkdtemp(prefix="anchor_smoke_http_")
    port = free_port()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    srv = subprocess.Popen([PY, os.path.join(HERE, "anchor_mcp.py"), "--http", "--port", str(port),
                            "--db-path", db, "--token", TOKEN],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
    base = f"http://127.0.0.1:{port}"
    ok = 0
    try:
        # wait for the port
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/", timeout=2).read()
                break
            except Exception:
                if srv.poll() is not None:
                    print(srv.stdout.read()); sys.exit("server died")
                time.sleep(0.5)
        else:
            sys.exit("server never came up")

        # ── 1. Streamable HTTP ──
        st, hdr, r = post(base + "/mcp", rpc(1, "initialize", {"protocolVersion": "2025-03-26",
                                                                 "capabilities": {}, "clientInfo": {"name": "smoke", "version": "0"}}))
        assert st == 200 and r["result"]["protocolVersion"] == "2025-03-26", (st, r)
        assert r["result"]["serverInfo"]["name"] == "anchor-memory"
        assert any(k.lower() == "mcp-session-id" for k in hdr), hdr
        ok += 1; print("✓ initialize (echoes protocolVersion, sets Mcp-Session-Id)")

        st, _, r = post(base + "/mcp", {"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert st == 202 and r is None, (st, r)
        ok += 1; print("✓ notification → 202, no body")

        st, _, r = post(base + "/mcp", rpc(2, "tools/list"))
        http_tools = [t["name"] for t in r["result"]["tools"]]
        assert "store_memory" in http_tools and "search_memory" in http_tools
        ok += 1; print(f"✓ tools/list ({len(http_tools)} tools)")

        st, _, r = post(base + "/mcp", rpc(3, "tools/call", {"name": "store_memory",
                                                            "arguments": {"text": "the lighthouse keeper waved at the ferry", "tag": "smoke"}}))
        assert st == 200
        stored = json.loads(r["result"]["content"][0]["text"])
        assert stored.get("memory_id") or stored.get("id") or stored.get("status"), stored
        ok += 1; print(f"✓ tools/call store_memory → {str(stored)[:80]}")

        st, _, r = post(base + "/mcp", rpc(4, "tools/call", {"name": "search_memory",
                                                            "arguments": {"query": "lighthouse ferry", "limit": 3}}))
        found = json.loads(r["result"]["content"][0]["text"])
        assert "lighthouse" in json.dumps(found, ensure_ascii=False), found
        ok += 1; print("✓ tools/call search_memory finds it")

        st, _, r = post(base + "/mcp", [rpc(5, "ping"), rpc(6, "tools/list")])
        assert st == 200 and isinstance(r, list) and len(r) == 2, (st, r)
        ok += 1; print("✓ batch")

        st, _, _ = post(base + "/mcp", rpc(7, "ping"), token="wrong")
        assert st == 401, st
        ok += 1; print("✓ bad token → 401")

        st, _, r = post(base + "/mcp", rpc(8, "nope/method"))
        assert st == 200 and r["error"]["code"] == -32601
        ok += 1; print("✓ unknown method → JSON-RPC -32601")

        try:
            urllib.request.urlopen(urllib.request.Request(base + "/mcp", headers={"Authorization": f"Bearer {TOKEN}"}), timeout=5)
            raise AssertionError("GET should be 405")
        except urllib.error.HTTPError as e:
            assert e.code == 405, e.code
        ok += 1; print("✓ GET /mcp → 405")

        # ── 2. Legacy SSE ──
        req = urllib.request.Request(base + "/sse", headers={"Authorization": f"Bearer {TOKEN}"})
        stream = urllib.request.urlopen(req, timeout=30)
        first = b""
        while b"\n\n" not in first:
            first += stream.readline()
        assert b"event: endpoint" in first, first
        endpoint = first.decode().split("data:")[1].split("\n")[0].strip()
        assert endpoint.startswith("/messages?sessionId="), endpoint
        ok += 1; print(f"✓ SSE endpoint event → {endpoint}")
        st, _, _ = post(base + endpoint, rpc(9, "tools/list"))
        assert st == 202, st
        got = b""
        deadline = time.time() + 30
        while b"event: message" not in got and time.time() < deadline:
            got += stream.readline()
        while b"\n\n" not in got.split(b"event: message", 1)[1] if b"event: message" in got else True:
            line = stream.readline()
            got += line
            if not line or time.time() > deadline:
                break
        payload = got.split(b"event: message", 1)[1].split(b"data:", 1)[1].split(b"\n\n", 1)[0].strip()
        msg = json.loads(payload)
        assert msg["id"] == 9 and [t["name"] for t in msg["result"]["tools"]] == http_tools
        stream.close()
        ok += 1; print("✓ SSE: POST /messages answered over the stream, same tool list")

        # ── 3. stdio parity ──
        p = subprocess.Popen([PY, os.path.join(HERE, "anchor_mcp.py"), "--db-path", db],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
        p.stdin.write(json.dumps(rpc(1, "initialize", {"protocolVersion": "2024-11-05"})) + "\n")
        p.stdin.write(json.dumps(rpc(2, "tools/list")) + "\n")
        p.stdin.flush()
        init = json.loads(p.stdout.readline())
        lst = json.loads(p.stdout.readline())
        p.stdin.close(); p.wait(timeout=30)
        assert init["result"]["protocolVersion"] == "2024-11-05"
        assert [t["name"] for t in lst["result"]["tools"]] == http_tools
        ok += 1; print("✓ stdio parity: identical tool list")

        print(f"\nALL {ok} PASS")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:
            srv.kill()
        shutil.rmtree(db, ignore_errors=True)


if __name__ == "__main__":
    main()
