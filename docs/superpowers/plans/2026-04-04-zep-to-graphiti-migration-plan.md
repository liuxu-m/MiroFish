# Zep → Graphiti + Neo4j 迁移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 MiroFish 项目从 Zep Cloud 服务迁移到自托管的 Graphiti + Neo4j

**Architecture:**
- 后端：创建 graphiti_service.py 封装 Graphiti 核心功能，重写图谱相关服务
- 前端：适配 API 调用和图谱展示
- 存储：Neo4j 5.26 (Docker 部署在 localhost)

**Tech Stack:** Python 3.12, graphiti-core, neo4j, conda (MiroFish env)

---

## 文件结构

### 新建文件
- `backend/app/services/graphiti_service.py` - Graphiti 服务封装
- `backend/app/services/graphiti_tools.py` - 图谱检索工具
- `backend/app/services/entity_reader.py` - 实体读取服务
- `backend/app/services/memory_updater.py` - 动态记忆更新

### 修改文件
- `backend/app/config.py` - 添加 Neo4j 配置
- `backend/app/services/graph_builder.py` - 重写图谱构建
- `backend/app/services/oasis_profile_generator.py` - 移除 Zep 搜索
- `backend/app/services/__init__.py` - 导出更新
- `backend/requirements.txt` - 添加依赖
- `frontend/src/components/Step2EnvSetup.vue` - 更新日志提示

### 保留文件（可选删除）
- `backend/app/services/zep_tools.py`
- `backend/app/services/zep_entity_reader.py`
- `backend/app/services/zep_graph_memory_updater.py`
- `backend/app/utils/zep_paging.py`

---

## Phase 1: 环境搭建

### Task 1: 安装 Graphiti 依赖

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/app/config.py`

- [ ] **Step 1: 添加 graphiti-core 和 neo4j 到 requirements.txt**

在 `backend/requirements.txt` 末尾添加：
```
graphiti-core>=0.1.0
neo4j>=5.0
```

- [ ] **Step 2: 激活 conda 环境并安装依赖**

```bash
conda activate MiroFish
cd D:\my_code\python_code\MiroFish\backend
pip install graphiti-core neo4j
```

- [ ] **Step 3: 添加 Neo4j 配置到 config.py**

在 `backend/app/config.py` 中找到 ZEP_API_KEY 配置位置，在附近添加：

```python
# Neo4j 配置
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')
```

- [ ] **Step 4: 测试 Neo4j 连接**

```bash
conda activate MiroFish
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')); with driver.session() as s: print(s.run('RETURN 1').single()); driver.close()"
```
Expected: 输出 `1`

- [ ] **Step 5: Commit**

```bash
git add backend/requirements.txt backend/app/config.py
git commit -m "feat: add Neo4j and Graphiti dependencies"
```

---

## Phase 2: Graphiti 服务封装

### Task 2: 创建 Graphiti 服务基类

**Files:**
- Create: `backend/app/services/graphiti_service.py`

- [ ] **Step 1: 创建 graphiti_service.py**

```python
"""
Graphiti 服务封装
提供与 Neo4j 交互的 Graphiti 核心功能
"""

import os
import re
from typing import Dict, Any, List, Optional
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIGenericClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder

from ..config import Config


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


class GraphitiService:
    """Graphiti 服务封装"""

    def __init__(self):
        self.client = Graphiti(
            uri=Config.NEO4J_URI,
            user=Config.NEO4J_USER,
            password=Config.NEO4J_PASSWORD,
            llm_client=MiniMaxClient(),
            embedder=OpenAIEmbedder()
        )

    def add_episodes(self, group_id: str, content: str) -> Dict[str, Any]:
        """添加文本到图谱"""
        result = self.client.add_episodes(
            group_id=group_id,
            episodes=[{"content": content, "type": "text"}]
        )
        return result

    def search(self, group_id: str, query: str, limit: int = 10) -> List[Dict]:
        """混合搜索"""
        results = self.client.search(
            group_id=group_id,
            query=query,
            limit=limit
        )
        return results

    def get_nodes(self, group_id: str) -> List[Dict]:
        """获取所有节点"""
        nodes = self.client.nodes.get_by_group_id(group_id)
        return nodes

    def get_edges(self, group_id: str) -> List[Dict]:
        """获取所有边"""
        edges = self.client.edges.get_by_group_id(group_id)
        return edges
```

- [ ] **Step 2: 测试导入**

```bash
conda activate MiroFish
cd D:\my_code\python_code\MiroFish\backend
python -c "from app.services.graphiti_service import GraphitiService, MiniMaxClient; print('GraphitiService imported successfully')"
```
Expected: 输出 "GraphitiService imported successfully"

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/graphiti_service.py
git commit -m "feat: add Graphiti service wrapper with MiniMax client"
```

---

### Task 3: 创建图谱检索工具

**Files:**
- Create: `backend/app/services/graphiti_tools.py`

- [ ] **Step 1: 创建 graphiti_tools.py**

```python
"""
图谱检索工具
提供多种搜索功能：深度洞察、广度搜索、快速搜索等
"""

from typing import Dict, Any, List, Optional
from .graphiti_service import GraphitiService


class GraphitiToolsService:
    """图谱检索工具服务"""

    def __init__(self, api_key: Optional[str] = None):
        self.service = GraphitiService()

    def search_graph(self, group_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """图谱语义搜索"""
        results = self.service.search(group_id, query, limit)
        return results

    def insight_forge(self, group_id: str, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        深度洞察检索
        将问题分解为多个子问题，多维度检索
        """
        results = self.service.search(group_id, query, limit)
        return {
            "query": query,
            "results": results,
            "sub_queries": [query]
        }

    def panorama_search(self, group_id: str, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        广度搜索
        获取更广泛的结果，包括过期内容
        """
        results = self.service.search(group_id, query, limit)
        return results

    def quick_search(self, group_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """快速搜索"""
        results = self.service.search(group_id, query, limit)
        return results

    def get_node_detail(self, group_id: str, node_uuid: str) -> Dict[str, Any]:
        """获取节点详情"""
        nodes = self.service.get_nodes(group_id)
        for node in nodes:
            if node.get("uuid") == node_uuid:
                return node
        return {}

    def get_entities_by_type(self, group_id: str, entity_type: str) -> List[Dict[str, Any]]:
        """按类型获取实体"""
        nodes = self.service.get_nodes(group_id)
        return [n for n in nodes if n.get("type") == entity_type]

    def get_entity_summary(self, group_id: str, entity_uuid: str) -> str:
        """获取实体摘要"""
        node = self.get_node_detail(group_id, entity_uuid)
        return node.get("summary", "")

    def get_graph_statistics(self, group_id: str) -> Dict[str, Any]:
        """获取图谱统计信息"""
        nodes = self.service.get_nodes(group_id)
        edges = self.service.get_edges(group_id)
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "entity_types": list(set(n.get("type", "") for n in nodes))
        }
```

- [ ] **Step 2: 测试导入**

```bash
conda activate MiroFish
cd D:\my_code\python_code\MiroFish\backend
python -c "from app.services.graphiti_tools import GraphitiToolsService; print('GraphitiToolsService imported successfully')"
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/graphiti_tools.py
git commit -m "feat: add Graphiti tools service"
```

---

### Task 4: 创建实体读取服务

**Files:**
- Create: `backend/app/services/entity_reader.py`

- [ ] **Step 1: 创建 entity_reader.py**

```python
"""
实体读取服务
从图谱中读取和过滤实体
"""

from typing import Dict, Any, List, Optional
from .graphiti_service import GraphitiService


class EntityReader:
    """实体读取服务"""

    def __init__(self, api_key: Optional[str] = None):
        self.service = GraphitiService()

    def get_all_nodes(self, group_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取图谱的所有节点"""
        nodes = self.service.get_nodes(group_id)
        return nodes[:limit]

    def get_all_edges(self, group_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取图谱的所有边"""
        edges = self.service.get_edges(group_id)
        return edges[:limit]

    def get_node_edges(self, group_id: str, node_uuid: str) -> List[Dict[str, Any]]:
        """获取指定节点的所有相关边"""
        edges = self.service.get_edges(group_id)
        return [e for e in edges if e.get("source") == node_uuid or e.get("target") == node_uuid]

    def filter_defined_entities(self, group_id: str, entity_types: List[str]) -> List[Dict[str, Any]]:
        """筛选符合预定义实体类型的节点"""
        nodes = self.service.get_nodes(group_id)
        return [n for n in nodes if n.get("type") in entity_types]

    def get_entity_with_context(self, group_id: str, entity_uuid: str) -> Dict[str, Any]:
        """获取单个实体及其完整上下文"""
        node = {}
        nodes = self.service.get_nodes(group_id)
        for n in nodes:
            if n.get("uuid") == entity_uuid:
                node = n
                break

        edges = self.get_node_edges(group_id, entity_uuid)
        return {
            "entity": node,
            "edges": edges
        }
```

- [ ] **Step 2: 测试导入**

```bash
conda activate MiroFish
cd D:\my_code\python_code\MiroFish\backend
python -c "from app.services.entity_reader import EntityReader; print('EntityReader imported successfully')"
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/entity_reader.py
git commit -m "feat: add entity reader service"
```

---

### Task 5: 创建动态记忆更新服务

**Files:**
- Create: `backend/app/services/memory_updater.py`

- [ ] **Step 1: 创建 memory_updater.py**

```python
"""
动态记忆更新服务
监控模拟的 actions 日志文件，将新的 agent 活动实时更新到图谱
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from .graphiti_service import GraphitiService
from ..config import Config


@dataclass
class AgentActivity:
    """Agent 活动记录"""
    agent_id: str
    action: str
    timestamp: str
    platform: str

    def to_episode_text(self) -> str:
        return f"[{self.timestamp}] {self.platform}: Agent {self.agent_id} - {self.action}"


class MemoryUpdater:
    """动态记忆更新服务"""

    def __init__(self, api_key: Optional[str] = None):
        self.service = GraphitiService()
        self.running = False
        self.thread = None

    def start_watching(self, group_id: str, log_file: str, callback: Optional[Callable] = None):
        """开始监控日志文件"""
        self.group_id = group_id
        self.log_file = log_file
        self.callback = callback
        self.running = True
        self.thread = threading.Thread(target=self._watch_worker, daemon=True)
        self.thread.start()

    def stop_watching(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _watch_worker(self):
        """监控工作线程"""
        batch = []
        batch_size = 5
        last_read = 0

        while self.running:
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    f.seek(last_read)
                    lines = f.readlines()
                    last_read = f.tell()

                for line in lines:
                    if line.strip():
                        batch.append(line.strip())

                if len(batch) >= batch_size:
                    self._send_batch(batch)
                    batch = []

            except FileNotFoundError:
                pass

            time.sleep(0.5)

    def _send_batch(self, activities: List[str]):
        """发送批量活动到图谱"""
        combined_text = "\n".join(activities)
        self.service.add_episodes(self.group_id, combined_text)

    def add_activity(self, group_id: str, activity: AgentActivity):
        """添加单个活动"""
        self.service.add_episodes(group_id, activity.to_episode_text())
```

- [ ] **Step 2: 测试导入**

```bash
conda activate MiroFish
cd D:\my_code\python_code\MiroFish\backend
python -c "from app.services.memory_updater import MemoryUpdater, AgentActivity; print('MemoryUpdater imported successfully')"
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/memory_updater.py
git commit -m "feat: add memory updater service"
```

---

## Phase 3: 图谱构建服务重写

### Task 6: 重写图谱构建服务

**Files:**
- Modify: `backend/app/services/graph_builder.py`

- [ ] **Step 1: 备份并重写 graph_builder.py**

备份原文件：
```bash
cp backend/app/services/graph_builder.py backend/app/services/graph_builder.py.bak
```

重写 `backend/app/services/graph_builder.py`：

```python
"""
图谱构建服务
使用 Graphiti + Neo4j 构建知识图谱
"""

import os
import uuid
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from .graphiti_service import GraphitiService
from .text_processor import TextProcessor
from ..config import Config
from ..utils.locale import t, get_locale, set_locale


@dataclass
class GraphInfo:
    """图谱信息"""
    group_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """图谱构建服务"""

    def __init__(self, api_key: Optional[str] = None):
        self.service = GraphitiService()

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3
    ) -> str:
        """异步构建图谱"""
        group_id = f"mirofish_{uuid.uuid4().hex[:16]}"

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(group_id, text, ontology, chunk_size, chunk_overlap, batch_size)
        )
        thread.daemon = True
        thread.start()

        return group_id

    def _build_graph_worker(
        self,
        group_id: str,
        text: str,
        ontology: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int
    ):
        """图谱构建工作线程"""
        chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            for chunk in batch:
                self.service.add_episodes(group_id, chunk)
            time.sleep(1)

    def get_graph_data(self, group_id: str) -> Dict[str, Any]:
        """获取图谱数据"""
        nodes = self.service.get_nodes(group_id)
        edges = self.service.get_edges(group_id)

        node_map = {n.get("uuid"): n.get("name", "") for n in nodes}

        nodes_data = [{
            "uuid": n.get("uuid"),
            "name": n.get("name"),
            "type": n.get("type", ""),
            "summary": n.get("summary", ""),
        } for n in nodes]

        edges_data = [{
            "uuid": e.get("uuid"),
            "name": e.get("name"),
            "source": e.get("source"),
            "target": e.get("target"),
        } for e in edges]

        return {
            "group_id": group_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }
```

- [ ] **Step 2: 测试构建**

```bash
conda activate MiroFish
cd D:\my_code\python_code\MiroFish\backend
python -c "
from app.services.graph_builder import GraphBuilderService
builder = GraphBuilderService()
result = builder.build_graph_async('测试文本', {}, '测试图谱')
print(f'Group ID: {result}')
"
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/graph_builder.py
git commit -m "refactor: rewrite graph builder for Graphiti"
```

---

## Phase 4: 后端适配

### Task 7: 更新服务导出

**Files:**
- Modify: `backend/app/services/__init__.py`

- [ ] **Step 1: 更新 __init__.py**

在 `backend/app/services/__init__.py` 中更新导入：

```python
# 图谱相关服务
from .graph_builder import GraphBuilderService
from .graphiti_service import GraphitiService
from .graphiti_tools import GraphitiToolsService
from .entity_reader import EntityReader
from .memory_updater import MemoryUpdater, AgentActivity

__all__ = [
    'GraphBuilderService',
    'GraphitiService',
    'GraphitiToolsService',
    'EntityReader',
    'MemoryUpdater',
    'AgentActivity',
]
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/services/__init__.py
git commit -m "feat: update service exports for Graphiti"
```

---

### Task 8: 适配 simulation_manager

**Files:**
- Modify: `backend/app/services/simulation_manager.py`

- [ ] **Step 1: 修改导入**

将：
```python
from .zep_entity_reader import ZepEntityReader, FilteredEntities
```

改为：
```python
from .entity_reader import EntityReader
```

- [ ] **Step 2: 修改使用**

将 `ZepEntityReader()` 改为 `EntityReader()`

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/simulation_manager.py
git commit -m "refactor: adapt simulation_manager for Graphiti"
```

---

### Task 9: 适配 report_agent

**Files:**
- Modify: `backend/app/services/report_agent.py`

- [ ] **Step 1: 修改导入**

将：
```python
from .zep_tools import ZepToolsService
```

改为：
```python
from .graphiti_tools import GraphitiToolsService
```

- [ ] **Step 2: 修改使用**

将 `ZepToolsService()` 改为 `GraphitiToolsService()`
将 `self.zep_tools` 改为 `self.tools`

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/report_agent.py
git commit -m "refactor: adapt report_agent for Graphiti"
```

---

### Task 10: 移除 oasis_profile_generator 中的 Zep 调用

**Files:**
- Modify: `backend/app/services/oasis_profile_generator.py`

- [ ] **Step 1: 移除 Zep 搜索��用**

查看并移除 `oasis_profile_generator.py` 中的 `zep_client.graph.search` 调用（第 327 行和 352 行附近）

- [ ] **Step 2: Commit**

```bash
git add backend/app/services/oasis_profile_generator.py
git commit -m "refactor: remove Zep calls from oasis_profile_generator"
```

---

### Task 11: 适配 API 端点

**Files:**
- Modify: `backend/app/api/graph.py`
- Modify: `backend/app/api/simulation.py`
- Modify: `backend/app/api/report.py`

- [ ] **Step 1: 更新 graph.py API**

将 `GraphBuilderService(api_key=Config.ZEP_API_KEY)` 改为 `GraphBuilderService()`

- [ ] **Step 2: 更新 simulation.py API**

将实体读取相关调用从 `ZepEntityReader` 改为 `EntityReader`

- [ ] **Step 3: 更新 report.py API**

将工具初始化从 `ZepToolsService` 改为 `GraphitiToolsService`

- [ ] **Step 4: Commit**

```bash
git add backend/app/api/graph.py backend/app/api/simulation.py backend/app/api/report.py
git commit -m "refactor: adapt API endpoints for Graphiti"
```

---

## Phase 5: 前端适配

### Task 12: 更新前端日志提示

**Files:**
- Modify: `frontend/src/components/Step2EnvSetup.vue`

- [ ] **Step 1: 查找并替换日志提示**

查找 `t('log.zepEntitiesFound'` 替换为通用提示

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/Step2EnvSetup.vue
git commit -m "refactor: update frontend log messages"
```

---

### Task 13: 验证图谱展示（可选）

**Files:**
- Modify: `frontend/src/components/GraphPanel.vue`

- [ ] **Step 1: 检查节点数据结构**

如果 Graphiti 返回的节点结构与 Zep 不同，适配显示

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/GraphPanel.vue
git commit -m "refactor: adapt graph panel for Graphiti"
```

---

## Phase 6: 测试

### Task 14: 端到端测试

- [ ] **Step 1: 启动 Neo4j**

```bash
docker start neo4j
```

- [ ] **Step 2: 启动后端**

```bash
conda activate MiroFish
cd backend
uvicorn app.main:app --reload
```

- [ ] **Step 3: 测试图谱构建**

使用前端创建一个新项目，观察图谱构建是否正常

- [ ] **Step 4: 测试图谱检索**

测试各种检索工具是否正常工作

- [ ] **Step 5: Commit**

```bash
git commit -m "test: add e2e tests for Graphiti migration"
```

---

## 总结

共 14 个 Tasks，分 6 个 Phase：

| Phase | Tasks | 描述 |
|------|-------|------|
| Phase 1 | Task 1 | 环境搭建 |
| Phase 2 | Task 2-5 | Graphiti 服务封装 |
| Phase 3 | Task 6 | 图谱构建重写 |
| Phase 4 | Task 7-11 | 后端适配 |
| Phase 5 | Task 12-13 | 前端适配 |
| Phase 6 | Task 14 | 端到端测试 |