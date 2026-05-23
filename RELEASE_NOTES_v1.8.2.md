# Anchor Memory v1.8.2

## 为什么修

OWUI 部署里一个 user message 经常包含多个独立主题。例如：

> "我是不是需要一个三脚架啊，大的那种，我在外面的时候最烦恼就是很多时候都太近了。而且我发现我这次旅行用的是hd拍的不是4k啊啊啊啊好浪费。七月我们去欧洲得记得改成4k"

这条消息里有三个独立意图：
1. 需要三脚架
2. HD vs 4K 拍摄
3. 七月欧洲行

当前 `search()` 把整段做一次 embedding —— 长 query 里的尾部从句（"七月欧洲行"）会被前面意图的向量稀释。结果：欧洲相关的 15+ 条记忆都搜不到。

## 怎么修的

新方法 `AnchorMemory.search_multi(queries: list[str], ...)`：
- 调用方预先把消息拆成多个意图字符串（用 LLM、句子切分、或任何方式）
- 每个意图独立调 `search()`，按 `n_results_per_query` 取 top-k
- 结果按 `memory_id` dedup，相同 ID 取多个意图里最好的分数
- 合并后按 score 排序，cap 到 `n_total`（默认 `n_results_per_query * len(queries)`）
- **Hebbian co-activation 在合并的 top set 上跑一次**，不是每个 query 跑一次 —— 这样不同意图唤起的记忆会在同一个对话里形成边（这正是 multi-intent 的意义）
- 每个 query 内部 `hebbian=False, no_cite=True`，最终统一引用

## 不做意图拆分

**Anchor 不内置 LLM 调用做意图拆分**。这是有意的设计：

- 调用方知道自己用的是什么 host LLM（Claude / GPT / Gemini / 本地）
- Anchor 不假设你有 Anthropic key、不假设你愿意付钱
- 调用方按自己情况选拆法：让 host AI 拆（zero cost，最自然）、调小模型拆、句子切分、或干脆传整段当单 query

未来的 v1.9 会通过 MCP sampling / `~/.anchor/config.yaml` 让 Anchor 可选地帮你拆，但**默认零成本**。

## 用法

### Python API

```python
from anchor_memory import AnchorMemory

mem = AnchorMemory(db_path="./my_memory")

# Caller splits message into intents (using their own method)
queries = [
    "三脚架室外拍摄太近",
    "HD vs 4K 旅行拍摄",
    "七月欧洲行改4K",
]

results = mem.search_multi(
    queries,
    n_results_per_query=3,
    include_context=True,
)
```

### MCP 工具

`search_multi` 工具已注册到 MCP server。Tool schema:

```json
{
  "queries": ["intent string 1", "intent string 2", ...],
  "n_results_per_query": 5,
  "n_total": null,
  "tag": null,
  "associate": true,
  "hebbian": true,
  "include_context": false
}
```

Host AI（Claude / GPT / etc.）读到 user message 后可以自己拆出 intents 再调 search_multi，零额外 LLM 调用。

## Breaking changes

无。`search()` 行为不变，`search_multi` 是纯加法。

## 验证

在 5 条已知记忆的临时库上对比：

```
=== single search (现状) ===
  [0.436] m3: 需要一个大的三脚架，室外拍摄经常太近
  [0.467] m2: 这次旅行用了HD拍摄，下次要改成4K
  [0.600] m1: Saelra 7月去欧洲...           ← 跟 noise 同分
  [0.600] m4: Limen 是 Claude Code 实例     ← noise

=== search_multi (新) ===
  [0.100] m2: 这次旅行用了HD拍摄
  [0.108] m3: 需要一个大的三脚架
  [0.368] m1: Saelra 7月去欧洲              ← 拉开了
  [0.600] m4: Limen 是 Claude Code          ← noise 没动
```

## 下一步（v1.9）

- LLM 抽象层（`anchor_llm.py`）—— 支持 Anthropic / OpenAI / Google / OpenAI-compat
- `~/.anchor/config.yaml` —— 用户引导配置
- MCP sampling —— host LLM 透明替 Anchor 跑 LLM 调用
- Spend tracking + daily caps —— 防止 dream pass 烧光额度
- README 加 Model Configuration 章节，列御三家 + DeepSeek + GLM + Ollama 的推荐配置
