# Zep → Graphiti + Neo4j 迁移设计文档

## 一、概述

### 1.1 项目背景

MiroFish 项目当前使用 Zep Cloud 服务作为知识图谱后端，提供图谱构建、检索、实体读取等功能。为减少对云服务的依赖，迁移到自托管的 Graphiti + Neo4j 方案。

### 1.2 当前架构

```
┌─────────────┐      ┌─────────────┐
│   Frontend  │─────▶│   Backend   │─────▶ Zep Cloud API
└─────────────┘      └─────────────┘
                              │
                         ┌────▼────┐
                         │  Zep    │
                         │ Cloud  │
                         └────────┘
```

### 1.3 新架构

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Frontend  │─────▶│   Backend  │─────▶│ Graphiti  │
└─────────────┘      └─────────────┘      │   Core    │
                                         └────┬────┘
                                              │
                                        ┌────▼────┐
                                        │  Neo4j  │
                                        └────────┘
```

### 1.4 部署信息

- **Neo4j**: Docker 部署在本机
- **容器名**: neo4j
- **端口**: 7474 (HTTP), 7687 (Bolt)
- **认证**: neo4j / password
- **版本**: neo4j:5.26-community

---

## 二、后端设计

### 2.1 配置文件变更

文件: `backend/app/config.py`

```python
# 新增配置
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')

# 保留（可选用于兼容性检测）
ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
```

### 2.2 Graphiti 服务封装

文件: `backend/app/services/graphiti_service.py` (新建)

#### 2.2.1 MiniMax 客户端

```python
from graphiti_core.llm_client import OpenAIGenericClient, LLMConfig
import re

class MiniMaxClient(OpenAIGenericClient):
    """支持思考内容移除的 MiniMax 客户端"""

    def __init__(self, config=None, **kwargs):
        if config is None:
            config = LLMConfig(
                api_key=os.getenv("MINIMAX_API_KEY"),
                base_url="https://api.minimax.io/v1",
                model="MiniMax-M2.5-thinking-128k"
            )
        super().__init__(config, **kwargs)

    async def _generate_response(self, messages, **kwargs):
        response = await super()._generate_response(messages, **kwargs)
        if "content" in response:
            content = re.sub(r'<think>[\s\S]*?</think>', '', response["content"]).strip()
            response["content"] = content
        return response
```

#### 2.2.2 Graphiti 服务类

```python
from graphiti_core import Graphiti
from graphiti_core.embedder import OpenAIEmbedder

class GraphitiService:
    """Graphiti 服务封装"""

    def __init__(self):
        self.client = Graphiti(
            uri=Config.NEO4J_URI,
            user=Config.NEO4J_USER,
            password=Config.NEO4J_PASSWORD,
            llm_client=MiniMaxClient(),
            embedder=OpenAIEmbedder()  # 后续配置 embedding 模型
        )

    def add_episodes(self, group_id: str, content: str):
        """添加文本到图谱"""
        return self.client.add_episodes(
            group_id=group_id,
            episodes=[{"content": content, "type": "text"}]
        )

    def search(self, group_id: str, query: str, limit: int = 10):
        """混合搜索"""
        return self.client.search(
            group_id=group_id,
            query=query,
            limit=limit
        )
```

### 2.3 图谱构建服务

文件: `backend/app/services/graph_builder.py` - 需要重写

| 当前方法 | 新实现 |
|---------|-------|
| `create_graph()` | Graphiti 初始化时自动创建 |
| `set_ontology()` | Pydantic 模型预定义 |
| `add_text_batches()` | `client.add_episodes()` |
| `get_graph_data()` | `client.nodes.get()` + `client.edges.get()` |
| `delete_graph()` | `client.delete()` |

### 2.4 图谱检索服务

文件: `backend/app/services/zep_tools.py` → `backend/app/services/graphiti_tools.py` (新建)

| 当前方法 | 新实现 |
|---------|-------|
| `insight_forge()` | 基于 Graphiti search 封装 |
| `panorama_search()` | 混合搜索 |
| `quick_search()` | 向量搜索 |
| `get_node_detail()` | `client.nodes.get()` |
| `get_entities_by_type()` | 按 group_id + type 查询 |

### 2.5 实体读取服务

文件: `backend/app/services/zep_entity_reader.py` → `backend/app/services/entity_reader.py` (新建)

| 当前方法 | 新实现 |
|---------|-------|
| `get_all_nodes()` | `client.nodes.get_by_group_id()` |
| `get_all_edges()` | `client.edges.get_by_group_id()` |
| `get_node_edges()` | `client.edges.get()` |

### 2.6 动态记忆更新

文件: `backend/app/services/zep_graph_memory_updater.py` → `backend/app/services/memory_updater.py`

类似当前逻辑，调用 Graphiti 的 `add_episodes()` 实现增量更新。

---

## 三、前端设计

### 3.1 API 兼容

- 图谱 ID 格式: `mirofish_{project_id}`
- API 端点尽量保持不变

### 3.2 文件变更

| 文件 | 变更内容 |
|------|---------|
| `src/api/graph.js` | 可能需要调整请求参数 |
| `src/components/GraphPanel.vue` | 适配节点/边数据结构 |
| `src/components/Step2EnvSetup.vue` | 移除 "Zep" 日志提示 |

### 3.3 变量命名

- `graph_id` / `graphId` - 保留，仅数据来源变化

---

## 四、依赖变更

文件: `backend/requirements.txt`

```txt
# 新增
graphiti-core>=0.1.0
neo4j>=5.0

# 可选（如果用 embedding）
openai>=1.0
anthropic>=0.18.0
```

---

## 五、数据流程

### 5.1 图谱构建

```
1. 前端发起构建请求
2. Backend → GraphBuilderService
3. Graphiti 处理分块 → 调用 LLM 提取实体
4. 存储到 Neo4j
5. 返回图谱 ID (group_id)
```

### 5.2 图谱检索

```
1. 前端发起检索请求
2. Backend → GraphitiService.search()
3. Graphiti 执行混合搜索
4. 返回结果
```

---

## 六、实施顺序

1. **Phase 1**: 环境搭建
   - 安装 graphiti-core 依赖
   - 测试 Neo4j 连接

2. **Phase 2**: 核心模块重写
   - graphiti_service.py
   - graph_builder.py

3. **Phase 3**: 检索和读取服务
   - graphiti_tools.py
   - entity_reader.py

4. **Phase 4**: 动态更新
   - memory_updater.py

5. **Phase 5**: 后端适配
   - simulation_manager.py
   - report_agent.py
   - API 端点

6. **Phase 6**: 前端适配
   - API 调用
   - 图谱展示

7. **Phase 7**: 测试
   - 端到端测试

---

## 七、风险和注意事项

### 7.1 LLM 兼容性

- MiniMax 推理模型输出包含 `<think>` 思考内容，需要移除
- 已封装 MiniMaxClient 处理

### 7.2 Embedding 模型

- 当前未配置，需要后续提供
- 可以先用 OpenAI text-embedding-3-small

### 7.3 数据迁移

- 新系统作为全新开始，不迁移历史数据

### 7.4 兼容性

- 保留 ZEP_API_KEY 配置用于检测是否迁移完成

---

## 八、环境信息

- **Python**: 3.12.4
- **Conda 环境**: MiroFish
- **Neo4j**: 本机 Docker (neo4j:5.26-community)
- **端口**: 7474, 7687
- **认证**: neo4j / password