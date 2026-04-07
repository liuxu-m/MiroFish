"""
图谱核心服务
包含 GraphBuilderService 和 GraphitiService
"""

import asyncio
import json
import logging
import os
import uuid
import time
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from .types import GraphData, NodeData, EdgeData, GraphInfo
from .deduplicator import NodeDeduplicator, EdgeDeduplicator
from .config import load_config, GraphConfig

logger = logging.getLogger(__name__)


class GraphitiService:
    """
    Graphiti 知识图谱服务
    封装 graphiti_core 的操作

    核心功能：
    1. 添加文本到图谱 (add_episodes)
    2. 获取节点 (get_nodes)
    3. 获取边 (get_edges)
    """

    def __init__(self, config: GraphConfig = None):
        self.config = config or load_config()
        self._load_env()

    def _load_env(self):
        """加载环境变量"""
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / '.env',
            Path.cwd() / '.env',
        ]
        for p in possible_paths:
            if p.exists():
                load_dotenv(str(p), override=True)
                break

    def _create_graphiti(self):
        """创建 Graphiti 实例"""
        from graphiti_core import Graphiti
        from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
        from .minimax_client import MiniMaxCompatibleClient

        llm_client = MiniMaxCompatibleClient(
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
            model=self.config.llm.model,
        )

        embedder_config = OpenAIEmbedderConfig(
            api_key=self.config.embedder.api_key,
            base_url=self.config.embedder.base_url,
            embedding_model=self.config.embedder.model,
        )
        embedder = OpenAIEmbedder(config=embedder_config)

        return Graphiti(
            uri=self.config.neo4j.uri,
            user=self.config.neo4j.user,
            password=self.config.neo4j.password,
            llm_client=llm_client,
            embedder=embedder,
        )

    async def add_episodes(
        self,
        group_id: str,
        content: str,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        添加文本到知识图谱

        Args:
            group_id: 分组ID
            content: 要添加的文本内容
            max_retries: 最大重试次数

        Returns:
            添加结果
        """
        from datetime import datetime

        graphiti = self._create_graphiti()
        last_error = None

        for attempt in range(max_retries):
            try:
                result = await graphiti.add_episode(
                    name=f"Episode_{group_id}",
                    episode_body=content,
                    source_description="MiroFish simulation data",
                    group_id=group_id,
                    reference_time=datetime.now(),
                )
                await graphiti.close()
                return {
                    "success": True,
                    "group_id": group_id,
                    "message": "Episode added successfully",
                    "result": str(result) if result else None,
                }
            except Exception as e:
                last_error = e
                error_str = str(e)

                if "SummarizedEntities" in error_str and "summaries" in error_str:
                    logger.warning(f"实体摘要提取失败 (尝试 {attempt+1}/{max_retries}): {error_str[:100]}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                else:
                    logger.error(f"add_episodes 错误: {e}")
                    break
            finally:
                try:
                    await graphiti.close()
                except:
                    pass

        return {
            "success": False,
            "group_id": group_id,
            "error": f"实体摘要提取失败: {last_error}",
            "partial": True,
        }

    async def get_nodes(self, group_id: str) -> List[Dict[str, Any]]:
        """
        获取指定分组的全部节点

        Args:
            group_id: 分组ID

        Returns:
            节点列表
        """
        from neo4j import GraphDatabase

        try:
            driver = GraphDatabase.driver(
                self.config.neo4j.uri,
                auth=(self.config.neo4j.user, self.config.neo4j.password),
            )
            with driver.session() as session:
                result = session.run(
                    '''MATCH (n:Entity)
                    WHERE n.group_id = $group_id OR n.group_id STARTS WITH $prefix
                    RETURN n.name, elementid(n), labels(n), n.summary, n.created_at, n.uuid''',
                    group_id=group_id,
                    prefix=self.config.group_id_prefix,
                )

                nodes = []
                seen_names = set()
                for record in result:
                    name = record[0]
                    node_id = record[1]
                    labels = record[2] or []
                    summary = record[3] or ""
                    created_at = record[4]
                    node_uuid = record[5] or str(node_id)

                    # 按名称去重
                    if name in seen_names:
                        continue
                    seen_names.add(name)

                    nodes.append({
                        "uuid": node_uuid,
                        "name": name,
                        "labels": labels,
                        "summary": summary,
                        "created_at": str(created_at) if created_at else None,
                    })

            driver.close()
            return nodes
        except Exception as e:
            logger.error(f"get_nodes 错误: {e}")
            return []

    async def get_edges(self, group_id: str) -> List[Dict[str, Any]]:
        """
        获取指定分组的全部边

        Args:
            group_id: 分组ID

        Returns:
            边列表
        """
        from neo4j import GraphDatabase

        try:
            driver = GraphDatabase.driver(
                self.config.neo4j.uri,
                auth=(self.config.neo4j.user, self.config.neo4j.password),
            )
            with driver.session() as session:
                result = session.run(
                    '''MATCH (a:Entity)-[r]->(b:Entity)
                    WHERE a.group_id = $group_id OR b.group_id = $group_id
                       OR a.group_id STARTS WITH $prefix OR b.group_id STARTS WITH $prefix
                    RETURN a.name, type(r), b.name, r.name, r.fact, r.summary,
                           r.created_at, a.uuid, b.uuid, r.episodes, r.valid_at, r.invalid_at''',
                    group_id=group_id,
                    prefix=self.config.group_id_prefix,
                )

                edges = []
                seen_edge_keys = set()
                for record in result:
                    source_name = record[0]
                    rel_type = record[1]
                    target_name = record[2]
                    rel_name = record[3] or ""
                    fact = record[4] or ""
                    summary = record[5] or ""
                    created_at = record[6]
                    source_uuid = record[7] or source_name
                    target_uuid = record[8] or target_name
                    episodes = record[9] or []
                    valid_at = record[10]
                    invalid_at = record[11]

                    # 按 source-target-relation 去重
                    edge_key = (source_uuid, target_uuid, rel_name or rel_type)
                    if edge_key in seen_edge_keys:
                        continue
                    seen_edge_keys.add(edge_key)

                    edges.append({
                        "name": rel_name or rel_type,
                        "fact": fact,
                        "summary": summary,
                        "source": source_uuid,
                        "target": target_uuid,
                        "source_node_uuid": source_uuid,
                        "target_node_uuid": target_uuid,
                        "fact_type": rel_type,
                        "created_at": str(created_at) if created_at else None,
                        "valid_at": str(valid_at) if valid_at else None,
                        "invalid_at": str(invalid_at) if invalid_at else None,
                        "episodes": episodes if isinstance(episodes, list) else [],
                    })

            driver.close()
            return edges
        except Exception as e:
            logger.error(f"get_edges 错误: {e}")
            return []


class GraphBuilderService:
    """
    图谱构建服务
    负责使用 Graphiti 构建知识图谱

    核心设计参照 Zep 论文：
    1. 节点去重 + 来源记录
    2. 边去重 + 来源记录
    3. 支持追溯来源
    """

    def __init__(self, config: GraphConfig = None):
        self.config = config or load_config()
        self.service = GraphitiService(self.config)
        self.node_deduplicator = NodeDeduplicator()
        self.edge_deduplicator = EdgeDeduplicator(self.node_deduplicator)

    def create_graph(self, name: str = "MiroFish Graph") -> str:
        """创建图谱"""
        return f"{self.config.group_id_prefix}_{uuid.uuid4().hex[:16]}"

    def set_ontology(self, group_id: str, ontology: Dict[str, Any]):
        """
        设置本体（Graphiti 会自动从文本提取，当前保留空实现）
        """
        pass

    def add_text_batches(
        self,
        group_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
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

            asyncio.run(add_all())
            episode_uuids = [f"ep_{i}" for i in range(len(chunks))]
        except Exception as e:
            logger.error(f"add_text_batches 失败: {e}")
        return episode_uuids

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3,
    ) -> str:
        """异步构建图谱"""
        group_id = self.create_graph(graph_name)

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(group_id, text, chunk_size, chunk_overlap, batch_size),
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
    ):
        """图谱构建工作线程"""
        try:
            async def build_all():
                chunks = self._split_text(text, chunk_size, chunk_overlap)
                total_chunks = len(chunks)
                success_count = 0

                for i, chunk in enumerate(chunks):
                    try:
                        await self.service.add_episodes(group_id, chunk)
                        success_count += 1
                    except Exception as chunk_error:
                        logger.error(f"第 {i+1} 块处理失败: {chunk_error}")
                        continue
                    await asyncio.sleep(0.5)

                logger.info(f"成功处理 {success_count}/{total_chunks} 块")

            asyncio.run(build_all())
        except Exception as e:
            logger.error(f"图谱构建失败: {e}")

    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """文本分块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - overlap
            if start >= len(text):
                break
        return chunks

    def get_graph_data(self, group_id: str) -> Dict[str, Any]:
        """
        获取完整图谱数据

        核心功能：
        1. 节点去重（按名称）
        2. 边去重（按 source-target-name-fact）
        3. 边的节点 UUID 规范化
        4. 记录来源信息
        """
        from neo4j import GraphDatabase

        try:
            driver = GraphDatabase.driver(
                self.config.neo4j.uri,
                auth=(self.config.neo4j.user, self.config.neo4j.password),
            )

            with driver.session() as session:
                # 查询所有边（先查询，因为边包含节点的 facts）
                result2 = session.run(
                    '''MATCH (a:Entity)-[r]->(b:Entity)
                    WHERE a.group_id = $group_id OR b.group_id = $group_id
                       OR a.group_id STARTS WITH $prefix OR b.group_id STARTS WITH $prefix
                    RETURN a.name, type(r), b.name, r.name, r.fact, r.summary,
                           r.created_at, a.uuid, b.uuid, r.episodes, r.valid_at, r.invalid_at''',
                    group_id=group_id,
                    prefix=self.config.group_id_prefix,
                )

                # 构建节点的 facts 映射
                node_facts = {}
                edges_data = []
                for record in result2:
                    source_name = record[0]
                    rel_type = record[1]
                    target_name = record[2]
                    rel_name = record[3] or ""
                    fact = record[4] or ""
                    summary = record[5] or ""
                    created_at = record[6]
                    source_uuid = record[7] or source_name
                    target_uuid = record[8] or target_name
                    episodes = record[9] or []
                    valid_at = record[10]
                    invalid_at = record[11]

                    edges_data.append({
                        "name": rel_name or rel_type,
                        "fact": fact,
                        "summary": summary,
                        "source": source_uuid,
                        "target": target_uuid,
                        "source_node_uuid": source_uuid,
                        "target_node_uuid": target_uuid,
                        "fact_type": rel_type,
                        "created_at": str(created_at) if created_at else None,
                        "valid_at": str(valid_at) if valid_at else None,
                        "invalid_at": str(invalid_at) if invalid_at else None,
                        "episodes": episodes if isinstance(episodes, list) else [],
                    })

                    # 收集 facts 用于节点摘要
                    if fact:
                        for node_name in [source_name, target_name]:
                            if node_name not in node_facts:
                                node_facts[node_name] = []
                            if fact not in node_facts[node_name]:
                                node_facts[node_name].append(fact)

                # 查询所有节点
                result = session.run(
                    '''MATCH (n:Entity)
                    WHERE n.group_id = $group_id OR n.group_id STARTS WITH $prefix
                    RETURN n.name, elementid(n), labels(n), n.summary, n.created_at, n.uuid''',
                    group_id=group_id,
                    prefix=self.config.group_id_prefix,
                )

                # 收集所有节点，建立映射
                all_nodes = []
                uuid_to_name = {}
                for record in result:
                    name = record[0]
                    node_id = record[1]
                    labels = record[2] or []
                    summary = record[3] or ""
                    created_at = record[4]
                    node_uuid = record[5] or str(node_id)

                    uuid_to_name[node_uuid] = name

                    all_nodes.append({
                        "uuid": node_uuid,
                        "name": name,
                        "node_id": node_id,
                        "labels": labels,
                        "summary": summary,
                        "created_at": created_at,
                    })

                # 节点去重
                nodes_data = []
                seen_names = {}
                name_to_uuid = {}
                for node in all_nodes:
                    name = node['name']
                    if name in seen_names:
                        continue
                    seen_names[name] = node['uuid']
                    name_to_uuid[name] = node['uuid']

                    # 如果没有摘要，从 facts 生成
                    node_summary = node['summary']
                    if not node_summary and name in node_facts:
                        facts = node_facts[name][:3]
                        if facts:
                            node_summary = " | ".join(facts)

                    nodes_data.append({
                        "uuid": node['uuid'],
                        "name": name,
                        "labels": node['labels'],
                        "summary": node_summary,
                        "created_at": str(node['created_at']) if node['created_at'] else None,
                        "name_orig": str(node['node_id']),
                    })

                # 边去重并规范化 UUID
                deduplicated_edges = []
                seen_edges = set()
                for edge in edges_data:
                    fact = (edge.get('fact', '') or '')[:self.config.edge_fact_prefix_length]
                    edge_key = (edge.get('source'), edge.get('target'), edge.get('name', ''), fact)
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)

                    # 规范化 UUID
                    source_uuid = edge.get('source')
                    target_uuid = edge.get('target')
                    source_name = uuid_to_name.get(source_uuid)
                    target_name = uuid_to_name.get(target_uuid)

                    if source_name and source_name in name_to_uuid:
                        canonical_source = name_to_uuid[source_name]
                        edge['source'] = canonical_source
                        edge['source_node_uuid'] = canonical_source
                    if target_name and target_name in name_to_uuid:
                        canonical_target = name_to_uuid[target_name]
                        edge['target'] = canonical_target
                        edge['target_node_uuid'] = canonical_target

                    deduplicated_edges.append(edge)
                edges_data = deduplicated_edges

            driver.close()

            return {
                "group_id": group_id,
                "nodes": nodes_data,
                "edges": edges_data,
                "node_count": len(nodes_data),
                "edge_count": len(edges_data),
            }

        except Exception as e:
            logger.error(f"get_graph_data 错误: {e}")
            import traceback
            traceback.print_exc()
            return {
                "group_id": group_id,
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
                "error": str(e),
            }

    def delete_graph(self, group_id: str):
        """删除图谱（当前未实现）"""
        pass
