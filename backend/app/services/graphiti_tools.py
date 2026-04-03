"""
图谱检索工具
提供多种搜索功能：深度洞察、广度搜索、快速搜索等
"""

import asyncio
from typing import Dict, Any, List, Optional

from .graphiti_service import GraphitiService


class GraphitiToolsService:
    """图谱检索工具服务"""

    def __init__(self, api_key: Optional[str] = None):
        self.service = GraphitiService()

    def search_graph(self, group_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """图谱语义搜索"""
        results = asyncio.run(self.service.search(group_id, query, limit))
        return results

    def insight_forge(self, group_id: str, query: str, limit: int = 10) -> Dict[str, Any]:
        """深度洞察检索"""
        results = asyncio.run(self.service.search(group_id, query, limit))
        return {
            "query": query,
            "results": results,
            "sub_queries": [query]
        }

    def panorama_search(self, group_id: str, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """广度搜索"""
        results = asyncio.run(self.service.search(group_id, query, limit))
        return results

    def quick_search(self, group_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """快速搜索"""
        results = asyncio.run(self.service.search(group_id, query, limit))
        return results

    def get_node_detail(self, group_id: str, node_uuid: str) -> Dict[str, Any]:
        """获取节点详情"""
        nodes = asyncio.run(self.service.get_nodes(group_id))
        for node in nodes:
            if node.get("uuid") == node_uuid:
                return node
        return {}

    def get_entities_by_type(self, group_id: str, entity_type: str) -> List[Dict[str, Any]]:
        """按类型获取实体"""
        nodes = asyncio.run(self.service.get_nodes(group_id))
        return [n for n in nodes if n.get("type") == entity_type]

    def get_entity_summary(self, group_id: str, entity_uuid: str) -> str:
        """获取实体摘要"""
        node = self.get_node_detail(group_id, entity_uuid)
        return node.get("summary", "")

    def get_graph_statistics(self, group_id: str) -> Dict[str, Any]:
        """获取图谱统计信息"""
        nodes = asyncio.run(self.service.get_nodes(group_id))
        edges = asyncio.run(self.service.get_edges(group_id))
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "entity_types": list(set(n.get("type", "") for n in nodes))
        }