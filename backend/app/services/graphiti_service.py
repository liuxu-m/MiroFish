"""
Graphiti 知识图谱服务
提供基于 Graphiti 的知识图谱构建和检索功能
"""

import os
import re
from typing import Dict, Any, List, Optional

from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig

from ..config import Config


class MiniMaxClient:
    """
    MiniMax LLM 客户端封装
    使用 OpenAIClient 配置 MiniMax API，并移除思考内容
    """

    def __init__(self):
        """
        初始化 MiniMaxClient
        从环境变量读取配置
        """
        # 从 .env 读取 MiniMax 配置
        api_key = os.environ.get('MINIMAX_API_KEY', '')
        base_url = os.environ.get('MINIMAX_BASE_URL', 'https://api.minimax.io/v1')
        model = os.environ.get('MINIMAX_MODEL', 'MiniMax-M2.5-thinking-128k')

        # 构建 LLMConfig
        llm_config = LLMConfig(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        # 使用 OpenAIClient 作为底层客户端
        self._client = OpenAIClient(
            config=llm_config,
            reasoning="minimal"
        )

    async def generate_response(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        生成响应，并移除思考内容

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            移除思考内容后的响应
        """
        # 调用客户端生成响应
        response = await self._client.generate_response(messages, **kwargs)

        # 如果是字典，提取 content 字段
        if isinstance(response, dict):
            response = response.get('content', '')

        # 移除思考内容
        response = self._remove_thinking(response)

        return response

    @staticmethod
    def _remove_thinking(text: str) -> str:
        """
        移除思考内容

        移除模式:
        - <think>...</think>
        - <think>...<|im_end|>

        Args:
            text: 原始文本

        Returns:
            移除思考内容后的文本
        """
        if not text:
            return text

        # 移除 <think>...</think> 模式
        text = re.sub(r'<think>[\s\S]*?</think>', '', text)

        # 移除 <think>...<|im_end|> 模式
        text = re.sub(r'<think>[\s\S]*?<\|im_end\|>', '', text)

        # 清理多余空白
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text


class GraphitiService:
    """
    Graphiti 知识图谱服务
    提供知识图谱的构建、搜索和查询功能
    """

    def __init__(self):
        """
        初始化 Graphiti 服务
        """
        # 初始化 MiniMax LLM 客户端
        self.llm_client = MiniMaxClient()

        # 初始化 Embedder（SiliconFlow）
        embedder_api_key = os.environ.get('OPENAI_API_KEY', 'sk-hyidmjeezkubefcmhueqvylalmdbsdwmaqniwvwdfqrxdhmo')
        embedder_base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.siliconflow.cn/v1')
        embedder_model = os.environ.get('OPENAI_EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B')

        embedder_config = OpenAIEmbedderConfig(
            api_key=embedder_api_key,
            base_url=embedder_base_url,
            embedding_model=embedder_model
        )
        self.embedder = OpenAIEmbedder(config=embedder_config)

        # 初始化 Graphiti 客户端
        self.graphiti = Graphiti(
            uri=Config.NEO4J_URI,
            user=Config.NEO4J_USER,
            password=Config.NEO4J_PASSWORD,
            llm_client=self.llm_client._client,
            embedder=self.embedder
        )

    async def add_episodes(
        self,
        group_id: str,
        content: str
    ) -> Dict[str, Any]:
        """
        添加文本到知识图谱

        Args:
            group_id: 分组ID
            content: 要添加的文本内容

        Returns:
            添加结果
        """
        try:
            # 添加到图谱
            result = await self.graphiti.add_episode(
                episode_content=content,
                group_id=group_id
            )

            return {
                "success": True,
                "group_id": group_id,
                "message": "Episode added successfully",
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "group_id": group_id,
                "error": str(e)
            }

    async def search(
        self,
        group_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        混合搜索

        Args:
            group_id: 分组ID
            query: 查询文本
            limit: 返回结果数量限制

        Returns:
            搜索结果列表
        """
        try:
            # 执行混合搜索
            results = await self.graphiti.search(
                query=query,
                group_ids=[group_id],
                num_results=limit
            )

            # 转换为可序列化的格式
            return [
                {
                    "name": edge.name,
                    "properties": edge.properties,
                    "source": edge.source_node_uuid,
                    "target": edge.target_node_uuid
                }
                for edge in results
            ]
        except Exception as e:
            return []

    async def get_nodes(
        self,
        group_id: str
    ) -> List[Dict[str, Any]]:
        """
        获取指定分组的全部节点

        Args:
            group_id: 分组ID

        Returns:
            节点列表
        """
        try:
            # 使用 search 获取节点
            results = await self.graphiti.search(
                query="",
                group_ids=[group_id],
                num_results=100
            )

            # 提取节点信息
            nodes = []
            seen_uuids = set()

            for edge in results:
                # 收集源节点
                if edge.source_node_uuid and edge.source_node_uuid not in seen_uuids:
                    seen_uuids.add(edge.source_node_uuid)
                    nodes.append({
                        "uuid": edge.source_node_uuid,
                        "name": getattr(edge, 'source_name', None),
                        "properties": getattr(edge, 'source_properties', {})
                    })

                # 收集目标节点
                if edge.target_node_uuid and edge.target_node_uuid not in seen_uuids:
                    seen_uuids.add(edge.target_node_uuid)
                    nodes.append({
                        "uuid": edge.target_node_uuid,
                        "name": getattr(edge, 'target_name', None),
                        "properties": getattr(edge, 'target_properties', {})
                    })

            return nodes
        except Exception as e:
            return []

    async def get_edges(
        self,
        group_id: str
    ) -> List[Dict[str, Any]]:
        """
        获取指定分组的全部边

        Args:
            group_id: 分组ID

        Returns:
            边列表
        """
        try:
            # 使用 search 获取边
            results = await self.graphiti.search(
                query="",
                group_ids=[group_id],
                num_results=100
            )

            # 提取边信息
            edges = []
            for edge in results:
                edges.append({
                    "name": edge.name,
                    "source": edge.source_node_uuid,
                    "target": edge.target_node_uuid,
                    "properties": edge.properties
                })

            return edges
        except Exception as e:
            return []