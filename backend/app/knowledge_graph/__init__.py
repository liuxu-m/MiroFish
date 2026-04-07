"""
MiroFish Knowledge Graph Package
基于 Graphiti 的时间感知知识图谱引擎

核心设计参照 Zep 论文：
- 三层图谱结构：情节子图 -> 语义实体子图 -> 社区子图
- 双时间模型：事务时间 + 有效时间
- 实体消歧：Embedding + LLM 判断
"""

from .core import GraphBuilderService, GraphitiService
from .types import GraphData, NodeData, EdgeData, EpisodeData, GraphInfo
from .deduplicator import NodeDeduplicator, EdgeDeduplicator
from .config import GraphConfig, load_config

__all__ = [
    # 核心服务
    "GraphBuilderService",
    "GraphitiService",
    # 类型
    "GraphData",
    "NodeData",
    "EdgeData",
    "EpisodeData",
    "GraphInfo",
    # 去重
    "NodeDeduplicator",
    "EdgeDeduplicator",
    # 配置
    "GraphConfig",
    "load_config",
]

__version__ = "0.1.0"
