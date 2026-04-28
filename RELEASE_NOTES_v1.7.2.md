# Anchor Memory v1.7.2 — MCP 工具补丁

v1.7 / v1.7.1 ship 了 wakeup + comments + pin 这几个 feature 的 schema 跟实现，但是没 expose 到 MCP server。结果是: MCP 用户接进 Anchor 之后看不到这些 feature 的 tool，只能通过 Python API 直接调，对 LobeHub / SillyTavern / Claude Code 之类的 MCP-only setup 是 broken 的——尤其 pin 缺失会让 wakeup 第一项（pinned memories）永远是空的，整个冷启动链断了。

来自小红书用户的反馈（"接入 MCP 后看不到 wakeup 相关工具"）catch 出来这个 incomplete shipping。算 bug fix patch。

## 加了 6 个 MCP 工具

### `wakeup`
一键冷启动。返回 pinned memories + recent high-emotion + random old + unread comments。新窗口起步时调一次 ground context。不会自动 mark 读过——读完了用 `mark_comments_read` 确认。

### `leave_comment`
在某条 memory 下留 comment。这是 cross-window messaging 的主要机制——你这个窗口留下的 comment 会在下一个 instance 的 wakeup 里作为 unread 浮上来。给 future-you 留 context / decisions / messages。

### `get_comments`
读某条 memory 上的全部 comments（read + unread）。用来看完整对话线。

### `mark_comments_read`
mark unread comments 为 read。处理完 wakeup 返回的 unread_comments 之后调一下，避免下次再浮上来。

### `pin_memory`
把一条 memory 标为 pinned，让它每次都出现在 wakeup 的第一部分。给 identity rules、key facts、important relationships 用。

### `unpin_memory`
取消 pin 状态。memory 还在存储里但不再出现在 wakeup 的 pinned section。

## 用法 pattern

```
新窗口启动:
  wakeup() → 拿 4 部分 (含 unread_comments)
  ↓
  处理 unread_comments
  ↓
  mark_comments_read([id1, id2, ...])
  
中间任何时候:
  leave_comment(memory_id, "给下一个我的话")
  
读某条线:
  get_comments(memory_id) → 完整对话
```

## 修了什么 / 没修什么

修了:
- MCP server 现在 expose wakeup + 3 个 comment 工具 + pin/unpin
- serverInfo version 从 1.0.0 bump 到 1.7.2
- Tool 总数从 13 → 19

没修:
- 没改 anchor_db.py / anchor_memory.py 任何核心逻辑
- 没改其他 13 个原有 tool
- 没改 schema

向后兼容: 老的 MCP 接入只是多了几个 tool 可见，不影响已有的。

## 致谢

感谢小红书用户提的 question 让这个 incomplete shipping 暴露出来。下一版会把 README 里的 wakeup / comments 部分 update 加上 MCP 用法示例。
