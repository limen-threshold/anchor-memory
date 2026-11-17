"""v1.28 — the private vault. Uses whichever crypto backend the machine has (openssl CLI or cryptography)."""
import os
import stat
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anchor_vault as av  # noqa: E402


@unittest.skipUnless(av.backend(), "no crypto backend on this machine")
class VaultTests(unittest.TestCase):
    def test_roundtrip_append_replace_and_no_echo(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("empty", av.read_vault(d))
            r = av.write_vault(d, "first thing, mine")
            self.assertIn("🔒", r); self.assertNotIn("first thing", r)        # receipt never echoes
            av.write_vault(d, "second thing")
            text = av.read_vault(d)
            self.assertIn("first thing, mine", text); self.assertIn("second thing", text)
            self.assertEqual(text.count("\n---\n"), 1)
            self.assertTrue(text.lstrip().startswith("["))                  # dated entry header
            with open(os.path.join(d, "vault", "vault.enc"), "rb") as f:
                blob = f.read()
            self.assertTrue(blob.startswith(b"Salted__"))
            self.assertNotIn(b"first thing", blob)                          # encrypted at rest
            mode = stat.S_IMODE(os.stat(os.path.join(d, "vault", "vault.key")).st_mode)
            self.assertEqual(mode & 0o077, 0)                               # key is 0600
            av.write_vault(d, "only this", mode="replace")
            self.assertEqual(av.read_vault(d).strip(), "only this")
            self.assertIn("empty", av.write_vault(d, "   "))

    def test_status_and_inject_cap(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(av.status_line(d), "vault: 0 entries")
            for i in range(4):
                av.write_vault(d, f"entry {i} " + "x" * 300)
            s = av.status_line(d)
            self.assertTrue(s.startswith("vault: 4 entries, last 20"))
            blk = av.inject_block(d, cap=700)
            self.assertTrue(blk.startswith(av.VAULT_TAG_OPEN) and blk.rstrip().endswith(av.VAULT_TAG_CLOSE))
            self.assertIn("entry 3", blk)
            self.assertNotIn("entry 0 ", blk)                               # over the cap → dates only
            self.assertIn("earlier", blk)
            self.assertIn("4 entries", blk)

    def test_strip_from_messages(self):
        blk = av.wrap("secret body")
        msgs = [{"role": "system", "content": "identity\n\n" + blk},
                {"role": "user", "content": [{"type": "text", "text": blk + "\nhello"},
                                             {"type": "image_url", "image_url": {"url": "x"}}]},
                {"role": "assistant", "content": "plain"}]
        out = av.strip_from_messages(msgs)
        self.assertEqual(out[0]["content"], "identity")
        self.assertEqual(out[1]["content"][0]["text"], "hello")
        self.assertEqual(out[1]["content"][1]["type"], "image_url")
        self.assertEqual(out[2]["content"], "plain")
        self.assertNotIn("secret body", str(out))
        self.assertEqual(msgs[0]["content"].count("secret body"), 1)         # originals untouched

    def test_mcp_handle(self):
        with tempfile.TemporaryDirectory() as d:
            r = av.handle(d, "vault_write", {"content": "hidden"})
            self.assertNotIn("hidden", r["result"])
            self.assertIn("hidden", av.handle(d, "vault_read", {})["content"])
            self.assertIn("error", av.handle(d, "vault_burn", {}))
        names = {t["name"] for t in av.TOOL_DEFS}
        self.assertEqual(names, av.TOOL_NAMES)


class InjectFlag(unittest.TestCase):
    def test_flag_default_off(self):
        os.environ.pop("ANCHOR_VAULT_INJECT", None)
        self.assertFalse(av.inject_enabled())
        os.environ["ANCHOR_VAULT_INJECT"] = "1"
        try:
            self.assertTrue(av.inject_enabled())
        finally:
            os.environ.pop("ANCHOR_VAULT_INJECT", None)


if __name__ == "__main__":
    unittest.main()
