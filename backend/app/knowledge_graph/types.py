"""
图谱数据类型定义
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NodeData:
    """节点数据"""
    uuid: str                           # 唯一标识
    name: str                           # 节点名称
    labels: List[str] = field(default_factory=list)    # 标签（实体类型）
    summary: str = ""                    # 摘要
    entity_type: str = ""                # 实体类型
    attributes: Dict[str, Any] = field(default_factory=dict)   # 属性
    created_at: Optional[str] = None     # 创建时间
    source_episodes: List[str] = field(default_factory=list)  # 来源 episodes（参照 Zep）
    name_orig: Optional[str] = None      # 原始名称


@dataclass
class EdgeData:
    """边数据"""
    name: str                            # 边名称（关系类型）
    fact: str = ""                       # 事实描述
    summary: str = ""
    source: str = ""                      # 源节点 UUID
    target: str = ""                      # 目标节点 UUID
    source_node_uuid: str = ""            # 源节点 UUID（兼容字段）
    target_node_uuid: str = ""             # 目标节点 UUID（兼容字段）
    fact_type: str = ""                   # 关系类型
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None     # 创建时间（事务时间线）
    valid_at: Optional[str] = None       # 生效时间（有效时间线）
    invalid_at: Optional[str] = None      # 失效时间
    expired_at: Optional[str] = None     # 过期时间
    episodes: List[str] = field(default_factory=list)  # 来源 episodes（关键！追溯来源）
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeData:
    """情节数据（Episode Subgraph）"""
    uuid: str
    content: str                         # 原始内容
    source: str = ""                     # 来源描述
    created_at: Optional[str] = None
    group_id: str = ""


@dataclass
class GraphData:
    """完整图谱数据"""
    group_id: str
    nodes: List[NodeData] = field(default_factory=list)
    edges: List[EdgeData] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    entity_types: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "group_id": self.group_id,
            "nodes": [n.__dict__ if isinstance(n, NodeData) else n for n in self.nodes],
            "edges": [e.__dict__ if isinstance(e, EdgeData) else e for e in self.edges],
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


@dataclass
class GraphInfo:
    """图谱信息"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }
