"""
图谱去重器
参照 Zep 的实体消歧设计：
- 节点去重：按名称去重，保留第一个
- 边去重：按 (source, target, name, fact) 精确去重
- 来源追溯：记录节点和边的来源 episodes
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


class NodeDeduplicator:
    """
    节点去重器
    参照 Zep 论文的实体消歧设计
    """

    def __init__(self):
        self.seen_names: Dict[str, str] = {}  # name -> first_uuid
        self.uuid_to_name: Dict[str, str] = {}  # uuid -> name
        self.name_sources: Dict[str, List[str]] = {}  # name -> [episodes]

    def add_node(self, node: Dict[str, Any]) -> bool:
        """
        添加节点，返回是否应该保留

        Returns:
            True: 保留该节点（新节点或第一个出现的名称）
            False: 跳过该节点（重复名称）
        """
        name = node.get('name', '')
        uuid = node.get('uuid', '')

        if not name:
            return True

        self.uuid_to_name[uuid] = name

        if name in self.seen_names:
            # 重复节点，记录来源
            if 'source_episodes' not in self.name_sources.get(name, []):
                if name not in self.name_sources:
                    self.name_sources[name] = []
                episodes = node.get('episodes', [])
                if isinstance(episodes, list):
                    self.name_sources[name].extend(episodes)
            logger.debug(f"[去重] 跳过重复节点: {name} (uuid={uuid} -> {self.seen_names[name]})")
            return False

        self.seen_names[name] = uuid
        self.name_sources[name] = node.get('episodes', []) or []
        return True

    def get_canonical_uuid(self, name: str) -> Optional[str]:
        """获取名称对应的规范 UUID"""
        return self.seen_names.get(name)

    def get_canonical_uuid_by_uuid(self, uuid: str) -> Optional[str]:
        """根据 UUID 获取规范 UUID（如果当前 UUID 是重复的）"""
        name = self.uuid_to_name.get(uuid)
        if name:
            return self.seen_names.get(name)
        return uuid

    def get_source_episodes(self, name: str) -> List[str]:
        """获取节点名称对应的所有来源"""
        return self.name_sources.get(name, [])

    def get_all_episodes(self) -> List[str]:
        """获取所有去重后的 episodes"""
        all_episodes = set()
        for episodes in self.name_sources.values():
            all_episodes.update(episodes)
        return list(all_episodes)


class EdgeDeduplicator:
    """
    边去重器
    策略：按 (source, target, name, fact) 精确去重
    不同的事实保留，因为可能来自不同的文本块
    """

    def __init__(self, node_deduplicator: Optional[NodeDeduplicator] = None):
        self.node_deduplicator = node_deduplicator
        self.seen_edges: Set[Tuple] = set()
        self.edge_sources: Dict[Tuple, List[str]] = {}  # edge_key -> [episodes]

    def should_keep_edge(self, edge: Dict[str, Any]) -> bool:
        """
        判断是否应该保留边

        策略：
        - 按 (source, target, name, fact[:100]) 去重
        - 只有当所有字段都相同时才认为是重复边
        """
        source = edge.get('source', '')
        target = edge.get('target', '')
        name = edge.get('name', '')
        fact = (edge.get('fact', '') or '')[:100]

        # 规范化 UUID（如果使用了节点去重器）
        if self.node_deduplicator:
            source = self.node_deduplicator.get_canonical_uuid_by_uuid(source) or source
            target = self.node_deduplicator.get_canonical_uuid_by_uuid(target) or target

        edge_key = (source, target, name, fact)

        if edge_key in self.seen_edges:
            # 记录来源
            episodes = edge.get('episodes', [])
            if isinstance(episodes, list) and edge_key not in self.edge_sources:
                self.edge_sources[edge_key] = episodes
            elif isinstance(episodes, list):
                self.edge_sources[edge_key].extend(episodes)
            logger.debug(f"[去重] 跳过完全重复的边: {name} ({source} -> {target})")
            return False

        self.seen_edges.add(edge_key)
        self.edge_sources[edge_key] = edge.get('episodes', []) or []
        return True

    def get_edge_sources(self, source: str, target: str, name: str, fact: str) -> List[str]:
        """获取边的所有来源 episodes"""
        edge_key = (source, target, name, fact[:100])
        return self.edge_sources.get(edge_key, [])


class SemanticDeduplicator:
    """
    语义去重器（高级版）
    使用 embedding 相似度判断是否重复
    目前暂未实现，保留接口
    """

    def __init__(self, embedder=None):
        self.embedder = embedder
        self.similarity_threshold = 0.85

    async def is_duplicate_node(self, node1: Dict, node2: Dict) -> bool:
        """
        判断两个节点是否是重复的（基于语义）

        参照 Zep 论文的两阶段设计：
        1. 向量相似度匹配
        2. LLM 消歧判断

        目前暂未实现
        """
        # TODO: 实现语义去重
        # 1. 计算 embedding 相似度
        # 2. 如果相似度高，再用 LLM 判断
        raise NotImplementedError("语义去重暂未实现")

    async def is_duplicate_edge(self, edge1: Dict, edge2: Dict) -> bool:
        """判断两条边是否是重复的（基于语义）"""
        raise NotImplementedError("语义边去重暂未实现")


def parse_attributes(attributes: Any) -> Dict[str, Any]:
    """解析 attributes 字段"""
    if isinstance(attributes, dict):
        return attributes
    if isinstance(attributes, str):
        try:
            return json.loads(attributes)
        except:
            pass
    return {}
