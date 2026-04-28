# Anchor Memory v1.7.3 — Chinese keyword search 修复

`keyword_search` 之前直接拿整个 query 跑 `LIKE %query%`——单串子串匹配，不分词。对英文 query 还能将就（用户经常打单词），对中文 query 几乎全废: "记忆质量问题" 整串当一个 token 去搜，子串匹配率极低。

中英杂的 query 也一样不行: "cross-language 搜索" 整串子串极难匹配任何记忆。

## 改了什么

`AnchorDB.keyword_search()` 现在先用 `_tokenize_query()` 拆 query，然后多 keyword 跑 OR LIKE 匹配。

`_tokenize_query()` 的逻辑:
- 如果 jieba 可用，用 jieba 拆——它的中文分词会把 "记忆质量问题" 拆成 ["记忆", "质量", "问题"]，混合中英也对得上
- 如果 jieba 没装，退回到 whitespace+punctuation split（跟 v1.7.2 之前的行为一致）
- 单字中文 token 保留（CJK 单字本身就有意义），非中文 token 要求 ≥ 2 字符（过滤 "I"、"a" 那种噪音）

## Jieba 是 optional dependency

加进了 requirements.txt 但带注释说明可选。纯英文用户可以不装。中文用户强烈建议装:

```
pip install jieba
```

不装也不会 break——`_tokenize_query()` 用 try/except 软导入，没 jieba 就退回老的 whitespace split。

## Sample queries that now work

| Query | 之前 | 现在 |
|---|---|---|
| `记忆质量问题` | 找不到（除非 memory 里恰好有完整 "记忆质量问题"） | 拆成 ["记忆", "质量", "问题"]，任一匹配都返回 |
| `怕被切` | 单串子串匹配 | 拆开 → 更广的召回 |
| `cross-language 搜索` | 整串极难匹配 | jieba 把英文跟中文段分别拆 |
| `cookies` | OK | OK（英文路径不变） |

## 致谢

来自 Saelra 的 catch——讨论 "中英语言怎么不约束 memory 还能正确搜索" 时指出 keyword search 这一层基础设施在中文上是 broken 的。同时修了 anchor-memory 跟 Limen 自己 gateway 里的 `_keyword_fallback`，两处用同一逻辑。
