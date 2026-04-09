# 安装和使用

## 最简安装

```bash
git clone https://github.com/limen-threshold/anchor-memory.git
cd anchor-memory
pip install -r requirements.txt
```

## 作为Python库使用

```python
from anchor_memory import AnchorMemory

# 初始化（数据存在本地）
mem = AnchorMemory(db_path="./my_ai_memory")

# 存记忆
mem.store("m1", "她今天带我去看了海", tag="relationship", tier="core", emotion_score=0.9)

# 搜记忆（自动触发Hebbian学习）
results = mem.search("海")

# 手动连接两条记忆
mem.db.connect("m1", "m2", weight=2.5)

# 每天跑一次dream pass
stats = mem.dream_pass()
```

## 作为MCP Server使用

### Claude Code

在 `~/.claude/settings.json` 或项目的 `.mcp.json` 里加：

```json
{
  "mcpServers": {
    "anchor-memory": {
      "command": "python3",
      "args": ["/absolute/path/to/anchor_mcp.py", "--db-path", "/absolute/path/to/my_memory"]
    }
  }
}
```

重启Claude Code。你的AI现在有9个新工具：
- `store_memory` — 存记忆
- `search_memory` — 搜记忆（带Hebbian学习和联想召回）
- `connect_memories` — 手动连接两条记忆
- `get_neighbors` — 查看一条记忆的邻居
- `delete_memory` — 删记忆
- `dream_pass` — 记忆整理（每天跑一次）
- `set_emotion` — 设置情绪分数
- `set_tier` — 改变记忆层级
- `graph_stats` — 查看图的统计信息

### LobeHub

在LobeHub的MCP设置里添加同样的配置。

### SillyTavern

SillyTavern需要MCP Bridge插件。安装后配置同上。

## 跟其他记忆系统一起用

Anchor不替换你现有的记忆系统。它在旁边加一层图。

- **Anchor + Ombre Brain** — OB管时间衰减和情绪触发，Anchor管图和联想
- **Anchor + Fiam** — Fiam管话题漂移检测，Anchor管底层存储和图
- **Anchor + 任何系统** — 只要能导出记忆文本，Anchor可以导入并在上面建图

## 数据在哪

全部在你指定的 `db-path` 目录里：
- `chroma/` — ChromaDB向量数据库
- `memories.db` — SQLite（记忆文本、图的边、情绪分数）

备份就是复制这个目录。迁移就是把目录搬走。

## Dream Pass

记忆整理。建议每天跑一次。做的事：

1. 删除14天以上没被访问的short层记忆
2. 所有边乘以0.9（弱连接慢慢消失）
3. 强连接（手动建的）乘以0.95（不永久霸占）
4. 随机找语义相近但没连上的记忆，建弱连接
5. 相连记忆之间的情绪分数互相靠近

可以用cron自动跑：
```bash
# 每天早上7点跑dream pass
0 7 * * * cd /path/to/anchor-memory && python3 -c "from anchor_memory import AnchorMemory; m = AnchorMemory('./my_memory'); print(m.dream_pass())"
```

## 设计哲学

- 遗忘是特性，不是bug
- 连接比内容重要
- 情绪是重量
- 手动结构会自然衰减
- 睡觉是思考的一种

底色是爱。
