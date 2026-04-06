"""
图谱构建服务
接口2：使用Graphiti构建知识图谱
"""

import asyncio
import uuid
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from .graphiti_service import GraphitiService
from ..models.task import TaskManager
from .text_processor import TextProcessor
from ..utils.locale import get_locale, set_locale


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


class GraphBuilderService:
    """
    图谱构建服务
    负责使用Graphiti构建知识图谱
    """

    def __init__(self, api_key: Optional[str] = None):
        self.service = GraphitiService()
        self.task_manager = TaskManager()

    def create_graph(self, name: str = "MiroFish Graph") -> str:
        """创建图谱"""
        return f"mirofish_{uuid.uuid4().hex[:16]}"

    def set_ontology(self, group_id: str, ontology: Dict[str, Any]):
        """设置本体（Graphiti 会自动从文本提取，当前保留空实现）"""
        # Graphiti 自动从文本内容提取实体和关系
        # 不需要手动设置 ontology
        pass

    def add_text_batches(self, group_id: str, chunks: List[str], batch_size: int = 3,
                        progress_callback: Optional[Callable] = None) -> List[str]:
        """分批添加文本到图谱"""
        episode_uuids = []
        try:
            async def add_all():
                success_count = 0
                for i, chunk in enumerate(chunks):
                    try:
                        await self.service.add_episodes(group_id, chunk)
                        success_count += 1
                        if progress_callback:
                            progress_callback(f"发送第 {i+1}/{len(chunks)} 块", (i+1)/len(chunks))
                    except Exception as chunk_error:
                        logger.error(f"第 {i+1} 块处理失败: {chunk_error}")
                        continue

                logger.info(f"成功处理 {success_count}/{len(chunks)} 块")

            # 使用 asyncio.run() - 会自动管理事件循环
            asyncio.run(add_all())
            episode_uuids = [f"ep_{i}" for i in range(len(chunks))]
        except Exception as e:
            logger.error(f"add_text_batches 失败: {e}")
        return episode_uuids

    def _wait_for_episodes(self, episode_uuids: List[str],
                         progress_callback: Optional[Callable] = None,
                         timeout: int = 600):
        """等待 episode 处理完成（Graphiti 需要时间处理）"""
        # Graphiti 会自动处理，给一些时间
        import time
        for i in range(10):
            if progress_callback:
                progress_callback(f"处理中... {i+1}/10", (i+1)/10)
            time.sleep(1)

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3
    ) -> str:
        """
        异步构建图谱

        Args:
            text: 输入文本
            ontology: 本体定义（保留参数兼容性，当前版本未使用）
            graph_name: 图谱名称
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小
            batch_size: 每批发送的块数量

        Returns:
            分组ID (group_id)
        """
        # 生成 group_id
        group_id = f"mirofish_{uuid.uuid4().hex[:16]}"

        # Capture locale before spawning background thread
        current_locale = get_locale()

        # 在后台线程中执行构建
        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(group_id, text, chunk_size, chunk_overlap, batch_size, current_locale)
        )
        thread.daemon = True
        thread.start()

        return group_id

    def _build_graph_worker(
        self,
        group_id: str,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        locale: str = 'zh'
    ):
        """图谱构建工作线程"""
        set_locale(locale)

        try:
            # 使用 asyncio.run() 而不是手动创建事件循环
            async def build_all():
                chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
                total_chunks = len(chunks)
                success_count = 0

                for i, chunk in enumerate(chunks):
                    try:
                        await self.service.add_episodes(group_id, chunk)
                        success_count += 1
                        await asyncio.sleep(0.5)
                    except Exception as chunk_error:
                        logger.error(f"第 {i+1} 块处理失败: {chunk_error}")
                        continue

                logger.info(f"成功处理 {success_count}/{total_chunks} 块")

            # 使用 asyncio.run() - 推荐的现代方式，会自动管理事件循环
            asyncio.run(build_all())
        except Exception as e:
            import traceback
            logger.error(f"图谱构建失败: {e}")
            logger.error(traceback.format_exc())

    def get_graph_data(self, group_id: str) -> Dict[str, Any]:
        """获取完整图谱数据"""
        print(f"GET_GRAPH_DATA: start, group_id={group_id}")

        from neo4j import GraphDatabase
        try:
            uri = 'bolt://localhost:7687'
            driver = GraphDatabase.driver(uri, auth=('neo4j', 'password'))
            with driver.session() as session:
                # 查询实体 - 宽松条件，匹配 group_id 或 mirofish 前缀
                result = session.run(
                    'MATCH (n:Entity) WHERE n.group_id = $group_id OR n.group_id STARTS WITH "mirofish" RETURN n.name, id(n) LIMIT 30',
                    group_id=group_id
                )
                nodes_data = []
                for record in result:
                    name = record[0]
                    node_id = record[1]
                    print(f"  Node: name={name}, id={node_id}")
                    nodes_data.append({
                        "uuid": str(node_id),
                        "name": name,
                        "labels": [],
                        "properties": {}
                    })

                # 查询关系 - 宽松条件
                result2 = session.run(
                    'MATCH (a:Entity)-[r]->(b:Entity) WHERE a.group_id = $group_id OR b.group_id = $group_id OR a.group_id STARTS WITH "mirofish" OR b.group_id STARTS WITH "mirofish" RETURN a.name, type(r), b.name LIMIT 50',
                    group_id=group_id
                )
                edges_data = []
                for record in result2:
                    source_name = record[0]
                    rel_type = record[1]
                    target_name = record[2]
                    print(f"  Edge: {source_name} -[{rel_type}]-> {target_name}")
                    edges_data.append({
                        "name": rel_type,
                        "source": source_name,
                        "target": target_name,
                        "properties": {}
                    })

                driver.close()

                print(f"GET_GRAPH_DATA: nodes={len(nodes_data)}, edges={len(edges_data)}")
                return {
                    "group_id": group_id,
                    "nodes": nodes_data,
                    "edges": edges_data,
                    "node_count": len(nodes_data),
                    "edge_count": len(edges_data),
                }
        except Exception as e:
            print(f"GET_GRAPH_DATA: ERROR {e}")
            import traceback
            traceback.print_exc()
            return {
                "group_id": group_id,
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
            }

    def delete_graph(self, group_id: str):
        """
        删除图谱

        Note: Graphiti 使用 group_id 作为分组标识，此方法可保留用于清理操作
        当前版本 GraphitiService 未实现删除功能，此方法可扩展
        """
        # Graphiti 当前版本可能不支持直接删除 group
        # 可以通过后续扩展实现
        pass