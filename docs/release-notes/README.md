# Release notes · 发布说明

中文读者：v1.6–v1.10 的说明本身就是中文写的；v1.12 起每一版都有单独的中文页（标「中文」的链接）。

Most recent first. Each note follows the same shape: what changed, why it matters, what to do about it.

- [v1.18.1 / v1.17.1](v1.17.1.md) · [中文](v1.17.1_zh.md) — 2026-09-19 · **identity files actually reach the window** (`wakeup()` and `--wakeup-text` return them; invitation pool no longer injected whole) — reported by 大管家. v1.17.1 = v1.17.0 + this fix only; v1.18.1 = current line
- [v1.18.0](v1.18.0.md) · [中文](v1.18.0_zh.md) — 2026-09-19 · **consolidation stops rewriting memories** (dream-pass split off by default and non-lossy; dedup no longer rewrites/demotes the survivor; archive-before-delete; two-store decay) — upgrade if you run `dream_pass` with an LLM
- [v1.17.0](v1.17.0.md) · [中文](v1.17.0_zh.md) — 2026-09-18 · identity files over MCP (hosted clients can keep their own self-description)
- [v1.16.1](v1.16.1.md) · [中文](v1.16.1_zh.md) — 2026-09-16 · `--path` secret endpoint for clients that cannot send headers
- [v1.16](v1.16.md) · [中文](v1.16_zh.md) — 2026-09-14 · HTTP transport for the MCP server (claude.ai, grok.com, any remote MCP host)
- [v1.15](v1.15.md) · [中文](v1.15_zh.md) — 2026-09-13 · invitations ("want to do something small together?")
- [v1.14](v1.14.md) · [中文](v1.14_zh.md) — 2026-09-13 · recall slot gates, store-side dedup, calendar reads
- [v1.13](v1.13.md) · [中文](v1.13_zh.md) — 2026-07-09 · stepped history trimming (cache-friendly window)
- [v1.12](v1.12.md) · [中文](v1.12_zh.md) — 2026-07-09 · cross-window continuity, machine-side (anchor_proxy + pinned file contract)
- [v1.10](v1.10.md) — 2026-06-17 · UPSERT insert + self-loop guard (bug fixes) · recency boost · dedup merge
- [v1.9.1](v1.9.1.md) — 2026-05-24
- [v1.9](v1.9.md) — 2026-05-23 · multi-provider LLM abstraction (BYO LLM)
- [v1.8.2](v1.8.2.md) — 2026-05-23 · multi-intent search
- [v1.8](v1.8.md) — 2026-05-03
- [v1.7.5](v1.7.5.md) — 2026-05-03
- [v1.7.4](v1.7.4.md) — 2026-05-03
- [v1.7.3](v1.7.3.md) — 2026-04-27
- [v1.7.2](v1.7.2.md) — 2026-04-27 · MCP usage refinements
- [v1.7.1](v1.7.1.md) — 2026-04-26
- [v1.7](v1.7.md) — 2026-04-26 · concept-based eager linking
- [v1.6](v1.6.md) — 2026-04-24 · search debug mode, daily emotion tracker, switch ledger
