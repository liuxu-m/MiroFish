# MiroFish Knowledge Graph Package

基于 Graphiti 的时间感知知识图谱引擎，核心设计参照 [Zep 论文](https://arxiv.org/abs/2501.13956)。

## 设计原理

### 三层图谱结构

```
情节子图 (Episode Subgraph)     → 原始数据，保留所有输入
        ↓
语义实体子图 (Semantic Entity Subgraph)  → 结构化的实体和关系
        ↓
社区子图 (Community Subgraph)  → 高层抽象
```

### 双时间模型

- **事务时间线 (Transaction Time)**：`created_at` - 系统记录时间
- **有效时间线 (Valid Time)**：`valid_at` / `invalid_at` - 事实在现实世界的时间

### 实体消歧

```
Stage 1: 实体提取 (LLM)  → 提取实体名称
Stage 2: 候选检索        → Embedding + 全文搜索
Stage 3: LLM 消歧判断     → 判断是否重复
```

## 目录结构

```
knowledge_graph/
├── __init__.py           # 包入口
├── config.py              # 配置管理
├── core.py                # 核心服务
├── deduplicator.py        # 去重器
├── minimax_client.py       # MiniMax API 客户端
└── types.py               # 类型定义
```

## 核心类型

### NodeData
```python
@dataclass
class NodeData:
    uuid: str                    # 唯一标识
    name: str                     # 节点名称
    labels: List[str]            # 标签（实体类型）
    summary: str                  # 摘要
    source_episodes: List[str]    # 来源 episodes
    created_at: Optional[str]    # 创建时间
```

### EdgeData
```python
@dataclass
class EdgeData:
    name: str                     # 关系类型
    fact: str                     # 事实描述
    source_node_uuid: str         # 源节点 UUID
    target_node_uuid: str         # 目标节点 UUID
    episodes: List[str]          # 来源 episodes（追溯用）
    valid_at: Optional[str]       # 生效时间
    invalid_at: Optional[str]     # 失效时间
```

## 使用示例

```python
from backend.app.knowledge_graph import GraphBuilderService

# 创建服务
service = GraphBuilderService()

# 异步构建图谱
group_id = service.build_graph_async(
    text="伊朗向俄罗斯出售武器...",
    ontology={},
    chunk_size=500,
)

# 获取图谱数据
graph_data = service.get_graph_data(group_id)
print(f"节点数: {graph_data['node_count']}")
print(f"边数: {graph_data['edge_count']}")
```

## 与旧版 Zep 服务的对比

| 特性 | Zep 版 (旧) | Graphiti 版 (新) |
|------|-------------|------------------|
| 去重方式 | 服务端自动 | 后端手动 |
| 边来源追溯 | episodes 字段 | episodes 字段 |
| 时间戳 | valid_at/invalid_at | valid_at/invalid_at |

## 配置

从 `.env` 文件加载：

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

MINIMAX_API_KEY=your_api_key
MINIMAX_BASE_URL=https://api.minimaxi.com/v1
MINIMAX_MODEL=MiniMax-M2.7

OPENAI_API_KEY=your_embedding_key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
```
