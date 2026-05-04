# Anchor Memory v1.8

## 为什么修

Anchor 的记忆整理（auto_consolidate）一直只看一小窗口的记忆找重复合并。问题：

**问题 1：跨窗口的重复看不到。** 同一件事被 AI 在不同时间记两次，落在不同批次里，再像也合不掉，库越长越冗余。

**问题 2：事实矛盾没人查。** 记忆库里如果同时存着"X 是 A 的猫"和"X 是 B 的猫"，搜出来两条都返回，AI 不知道哪个对——但系统不会告诉你这两条互相打架。

**问题 3：不同来源的记忆被错合。** 如果同一事件被两个 pipeline 各记一次（实时对话 + 离线信件），cosine 相似度高得离谱，但**两条不该合**——它们是同一事件的两个视角。同样，4 个不同笔友都叫"克"，记忆里看着像，**也不该合**——是不同人。

## 怎么修的

新文件 `dream_extras.py`，两个 pass 解决三个问题。

### `run_global_dedup(mem)`

- 拉所有记忆 embeddings（已经在库里，不重新算）
- 算全局相似度矩阵，找 cosine ≥ 0.92 的所有 pair
- **加 source 和 entity 检查**：sources 不同或 entities 不同的 pair 直接跳过（不送 LLM 也不会合）
- 剩下的真正候选 pair 送 LLM，让它判 "merge / 保留两条"
- merge 决定执行；所有决定写到 `anchor_audit/dedup_<时间>.json` 留审

### `run_fact_check(mem)`

- 把记忆按相似度聚类（同主题但不一定重复）
- 每组送 LLM，问"这里有哪些条互相说反了"
- 答案写到 `anchor_audit/contradictions_<时间>.md`
- **不自动改记忆**——只 flag 给人看。事实矛盾应该人决定哪个对

### 顺带：`store()` 加两个可选参数

```python
mem.store(memory_id, text,
          source="penpal_letter",          # 标这条记忆是哪条 pipeline 写的
          entity="pair:27|ai_name:Cheng",  # 标这条记忆指的是哪个具体实体
          )
```

不传就跟以前完全一样。传了的话，dream_extras 会用这两个标签做精确判断。

## 修完之后

- 跨窗口重复的记忆能合掉了——库更干净
- 矛盾事实有了报告——人能定期审一下，决定哪个对
- 4 个同名笔友 / 同事件多视角的记忆**不会被错合**——只要在 store 时标了 source 和 entity
- 一个 10000 条记忆的库跑一次 dedup + fact_check 大概 $0.50，可以每天跑
- 用法：

```python
from anchor_memory import AnchorMemory
from dream_extras import run_global_dedup, run_fact_check

mem = AnchorMemory(db_path="./my_anchor")
run_global_dedup(mem)
run_fact_check(mem)
```

或者命令行：

```bash
python3 dream_extras.py --db ./my_anchor --dedup --fact-check
```

模型默认 Sonnet 4.6。要换其他模型设环境变量 `ANCHOR_DREAM_MODEL`。
