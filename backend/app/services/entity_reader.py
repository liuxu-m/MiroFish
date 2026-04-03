"""
实体读取服务
从图谱中读取和过滤实体
"""

import asyncio
from typing import Dict, Any, List, Optional
from .graphiti_service import GraphitiService


class EntityReader:
    """实体读取服务"""

    def __init__(self, api_key: Optional[str] = None):
        self.service = GraphitiService()

    def get_all_nodes(self, group_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取图谱的所有节点"""
        nodes = asyncio.run(self.service.get_nodes(group_id))
        return nodes[:limit]

    def get_all_edges(self, group_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取图谱的所有边"""
        edges = asyncio.run(self.service.get_edges(group_id))
        return edges[:limit]

    def get_node_edges(self, group_id: str, node_uuid: str) -> List[Dict[str, Any]]:
        """获取指定节点的所有相关边"""
        edges = asyncio.run(self.service.get_edges(group_id))
        return [e for e in edges if e.get("source") == node_uuid or e.get("target") == node_uuid]

    def filter_defined_entities(self, group_id: str, entity_types: List[str]) -> List[Dict[str, Any]]:
        """筛选符合预定义实体类型的节点"""
        nodes = asyncio.run(self.service.get_nodes(group_id))
        return [n for n in nodes if n.get("type") in entity_types]

    def get_entity_with_context(self, group_id: str, entity_uuid: str) -> Dict[str, Any]:
        """获取单个实体及其完整上下文"""
        node = {}
        nodes = asyncio.run(self.service.get_nodes(group_id))
        for n in nodes:
            if n.get("uuid") == entity_uuid:
                node = n
                break

        edges = self.get_node_edges(group_id, entity_uuid)
        return {
            "entity": node,
            "edges": edges
        }