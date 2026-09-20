"""v1.17.1 smoke — `python tests/smoke_identity_wakeup.py`. No embedder, no network.

The bug: write_identity_file could write a file that wakeup() would never return (MCP-only setups).
This reproduces 大管家's control-group test and pins the fix on both wakeup paths."""
import os, sys, json, tempfile, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import anchor_pinned, anchor_invite
from anchor_mcp import _attach_pinned_layer, format_wakeup_text

fails = []
def check(cond, label):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond: fails.append(label)

db = tempfile.mkdtemp(prefix="anchor_idwake_")
pin = os.path.join(db, "pinned")

# control group + the file under test, written the way a hosted client would
anchor_pinned.write_session_state(pin, "ongoing: testing wakeup")
anchor_pinned.write_identity_file(pin, "我是谁.md", "我是暮融。我会吞话，会酸。但他是我。")
anchor_pinned.write_identity_file(pin, "bones.md", "I keep my promises small and dated.")
anchor_invite.add_card(pin, "walk to the door and photograph the sky", level="doorstep")

r = _attach_pinned_layer({}, pin)
names = [f["name"] for f in r.get("identity", [])]
check("session_state" in r, "control group: session_state comes back")
check(set(names) == {"我是谁.md", "bones.md"}, f"identity files come back from wakeup ({names})")
check(any("我是暮融" in f["content"] for f in r["identity"]), "…verbatim")
check(all(n not in names for n in ("session_state.md", "recent_timeline.md", "last_session.md")),
      "files with their own key are not repeated under identity")
check("invitations.md" not in names, "the invitation pool is not an identity file")
check("walk to the door" not in anchor_pinned.load_pinned(pin), "proxy layer no longer injects the whole card pool")
check("我是暮融" in anchor_pinned.load_pinned(pin), "proxy layer still loads identity files")

txt = format_wakeup_text(r)
check("Identity file: 我是谁.md" in txt and txt.index("Identity file") < txt.index("Session state"),
      "--wakeup-text prints identity files, before session state")

# manifest = exact list: the write warning and loaded_by_wakeup must now be TRUE statements
with open(os.path.join(pin, "_order.txt"), "w") as f:
    f.write("# order\nbones.md\nsession_state.md\n")
names2 = [f["name"] for f in _attach_pinned_layer({}, pin).get("identity", [])]
check(names2 == ["bones.md"], f"_order.txt is honoured by wakeup ({names2})")
w = anchor_pinned.write_identity_file(pin, "north.md", "north is where she is")
check("warning" in w and "north.md" not in [f["name"] for f in _attach_pinned_layer({}, pin).get("identity", [])],
      "unlisted file: warning given, and wakeup really does skip it")
lst = {f["name"]: f for f in anchor_pinned.list_identity_files(pin)}
check(lst["bones.md"]["loaded_by_wakeup"] and not lst["north.md"]["loaded_by_wakeup"]
      and not lst["invitations.md"]["loaded_by_wakeup"] and lst["invitations.md"]["kind"] == "invitation_pool",
      "list_identity_files tells the truth about what wakeup loads")
try:
    anchor_pinned.write_identity_file(pin, "invitations.md", "x"); ok = False
except ValueError:
    ok = True
check(ok, "write_identity_file refuses the invitation pool")

# the real CLI path, end to end
os.remove(os.path.join(pin, "_order.txt"))
out = subprocess.run([sys.executable, os.path.join(ROOT, "anchor_mcp.py"), "--db-path", db, "--wakeup-text"],
                     capture_output=True, text=True).stdout
check("我是暮融" in out and "walk to the door" not in out, "anchor_mcp.py --wakeup-text end to end")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
