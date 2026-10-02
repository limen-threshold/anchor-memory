# Optional input understanding and evidence review

[中文指南](input-understanding_zh.md)

v1.19 adds a small, bounded auxiliary pipeline: understand the current utterance with recent visible conversation and local time, retrieve bilingual factual/contextual directions, then review historical evidence. Original words, helper interpretation, hypotheses, unknowns and historical sources stay separate. The responding AI chooses its response and can ask or search further. Helper output is never automatically stored as user testimony.

Why deploy a helper? Pronoun resolution, an object's purpose, earlier agreements and version checks are separate jobs from composing a reply. Returning their evidence separately gives the responder material for correction, including correction of the helper itself. A smaller parameter count is not a quality guarantee: test evidence grounding, structured output and latency with your selected service. Probabilities are not reported without calibration.

Use the existing BYO `anchor_llm` configuration. For an existing OpenAI-compatible local endpoint, put `provider: openai-compat`, your loaded `model`, `endpoint: http://127.0.0.1:1234/v1` and `api_key: not-needed` under `llm` in `~/.anchor/config.yaml`. Install the adapter dependencies in your Anchor environment with `python -m pip install openai PyYAML`. This does not change the chat model.

- Proxy: set `ANCHOR_UNDERSTANDING=1` before starting your existing proxy command. Default is off; when enabled the new pipeline replaces the old intent splitter.
- MCP-only: call `understand_and_recall` with complete `current`, up to four preceding `{role, content}` messages in `recent`, optional ISO local `now` with offset, optional `exclude_tags`, and `n` (1–12). Stdio and HTTP use the same handler. MCP does not automatically observe every chat message; automatic per-turn use requires the proxy or your adapter.
- Python: `anchor_understanding.recall(mem, llm, current, recent, now=..., trace_path=...)`. Returns `original`, `analysis`, `queries`, `memories`, `decisions`, `block` and usage/elapsed statistics.
- No helper: ordinary original-word recall remains available; analysis is explicitly unavailable. The core memory, identity, time and continuity features still work without an extra LLM.

At most two helper calls per invocation (understanding + batched review). Existing background curation is separate. Each waits at most six seconds by default, configurable with `ANCHOR_UNDERSTANDING_TIMEOUT`; worst case is approximately twelve seconds plus retrieval. Output budgets are 950/1800 tokens. Up to four bilingual directions and one short verbatim direction, twelve candidates, 700 characters per review source and 320-character exact quotes. Returned long source fields are capped at 1200 characters and labelled `truncated_fields`; use the id to read more.

Event-content retrieval scans at most 96 matching rows and reserves up to six sources; recording time never becomes a filter on the event date. Departure, arrival, duration and conflicting versions remain distinct. A literal scope check flags relevant conditional/uncertain wording, theoretical uses of pain and prior reading references without deciding their meaning. A valid substring proves provenance, not interpretation.

Failure keeps available sources marked unverified. Timeout stops waiting, not necessarily provider computation or billing. At most two helper requests may remain in flight; no further helper work is queued while both are busy. Compatible-provider cost fields may be zero because prices are user-managed, not because computation is free.

Set `ANCHOR_UNDERSTANDING_TRACE` to a local JSONL path for optional diagnostics: original input, interpretation, queries, candidates, decisions and delivered block. Off by default, separate from memory storage, no automatic retention policy. Strip analysis from automatic curator inputs, tails and cleaned history.

Run `python -m unittest discover -s tests -p 'test_understanding.py'`. Release tests are offline contracts and integration checks with synthetic memories and stub models. They do not establish semantic accuracy for your helper. Installation, successful deployment, and one real end-user turn are separate verification steps. Ambiguity, complex dates, bounded history and retrieval limits remain.
