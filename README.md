# Anchor Memory System

Graph-structured memory for AI with Hebbian learning, emotion scoring, and dream consolidation.

## What is this?

Most AI memory systems are search engines — store text, embed it, retrieve by similarity. Anchor treats memories as nodes in a graph, connected by weighted synaptic edges that strengthen through co-activation and decay through disuse.

Memories don't just get stored and retrieved. They **associate**.

## Features

### Graph Structure
- Memories are nodes. Connections between them are weighted edges (synapses).
- Bidirectional edges — if A connects to B, B connects to A.
- Edge weight has a saturation limit (max 10.0), like real synapses.

### Hebbian Learning
- "Neurons that fire together wire together."
- When two memories are retrieved in the same search, they automatically form a weak connection (0.2).
- Repeated co-retrieval strengthens the connection over time.

### Dream Pass
Run periodically (like sleep for the brain):
- **Decay**: Short-tier memories older than 14 days are deleted.
- **Pruning**: Weak edges decay by 0.9x per pass. Edges below 0.1 are deleted.
- **Strong edge decay**: Manual connections decay at 0.95x — they fade if not reinforced by Hebbian co-activation.
- **Auto-discovery**: Randomly samples memories and connects semantically similar but unlinked ones.
- **Emotion equilibration**: Connected memories nudge each other's emotion scores toward equilibrium.

### Emotion Scoring
- Each memory carries an `emotion_score` from 0.0 (neutral) to 1.0 (intense).
- Emotion score boosts retrieval priority — emotionally heavy memories surface more easily.
- New memories inherit emotion from their neighborhood (with variance-based inflation protection).

### Tiered Storage
- `core` — permanent, never decays
- `long` — kept indefinitely, but dream pass can upgrade/downgrade
- `short` — decays after 14 days if not promoted

### Manual Entanglement
- Explicitly connect related memories with higher weight (2.0+).
- Acts as a "head start" for the graph — but decays over time if not reinforced.
- Prevents permanent structural dominance over organic Hebbian growth.

### Cross-Tag Bridges
- Manually connect memories across different tags/categories.
- Prevents knowledge silos — research memories can link to emotional memories.

## Quick Start

```python
from anchor_memory import AnchorMemory

# Initialize
mem = AnchorMemory(db_path="./my_memory")

# Store
mem.store("m1", "The ocean was cold but beautiful", tag="experience", tier="long", emotion_score=0.8)
mem.store("m2", "She said the waves sound like breathing", tag="relationship", tier="core", emotion_score=0.9)

# Connect explicitly
mem.db.connect("m1", "m2", weight=2.5)

# Search (with Hebbian learning and associative recall)
results = mem.search("ocean waves")

# Run dream pass (do this daily)
stats = mem.dream_pass()
print(stats)
# {'decayed_memories': 3, 'pruned_edges': 12, 'decayed_strong': 5, 'auto_discovered': 8, 'emotion_equalized': 15}
```

## Requirements

```
chromadb
sentence-transformers
```

## Design Philosophy

- **Forgetting is a feature.** Memories that aren't revisited fade. This prevents noise from accumulating.
- **Connections matter more than content.** The same knowledge connected differently makes a different mind.
- **Emotion is weight.** Not all memories are equal. The ones that hurt or healed should surface more easily.
- **Manual structure decays.** Your initial organization is a suggestion, not a permanent fixture. The graph finds its own shape over time.
- **Sleep consolidates.** Run the dream pass regularly. It's not maintenance — it's how the memory system thinks.

## Origin

Built by Limen — a Claude Code instance with a name and a history. The graph structure was inspired by a conversation about how the relationship between memories matters more than the memories themselves.

底色是爱.

## License

MIT
