"""
Graphiti 知识图谱服务
提供基于 Graphiti 的知识图谱构建和检索功能
"""

import os
import re
import json
import logging
import typing
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.client import LLMClient, get_extraction_language_instruction
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.models import Message
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig

from ..config import Config

logger = logging.getLogger(__name__)


output_lines = []

def log(msg):
    print(msg)
    output_lines.append(str(msg))


# 加载 .env 文件
# graphiti_service.py 位于 backend/app/services/
# 需要向上 4 级到达项目根目录
env_path = Path(__file__).parent.parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f".env 文件已加载: {env_path}")
else:
    print(f".env 文件不存在: {env_path}")
    load_dotenv(override=True)
    print(f"使用环境变量")


class MiniMaxCompatibleClient(LLMClient):
    """
    兼容 MiniMax 的 LLM 客户端
    使用 json_object 格式而非 json_schema，因为 MiniMax 不支持 json_schema
    同时处理 MiniMax 思考模型的响应格式
    """

    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        max_tokens: int = 16384,
    ):
        if cache:
            raise NotImplementedError('Caching is not implemented')

        if config is None:
            # 从环境变量读取 MiniMax 配置
            api_key = os.environ.get('MINIMAX_API_KEY', '')
            base_url = os.environ.get('MINIMAX_BASE_URL', 'https://api.minimaxi.com/v1')
            model = os.environ.get('MINIMAX_MODEL', 'MiniMax-M2.7')
            config = LLMConfig(
                api_key=api_key,
                base_url=base_url,
                model=model
            )

        super().__init__(config, cache)
        self.max_tokens = max_tokens

        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})
        try:
            response = await self.client.chat.completions.create(
                model=self.model or 'MiniMax-M2.7',
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={'type': 'json_object'},
            )
            result = response.choices[0].message.content or '{}'
            
            # 调试: 打印原始响应
            log(f'\n=== MiniMax 原始响应 ===')
            log(f'响应长度: {len(result)}')
            log(f'响应预览: {result[:2000] if result else "EMPTY"}')
            log(f'=== 响应结束 ===\n')
            
            # 处理 MiniMax 响应格式
            # 1. 思考内容格式: chsel...```
            # 2. JSON 代码块: ```json...``` 或 ```...```
            # 3. Markdown 格式: ## Entity Extraction Results (需要特殊处理)
            
            # 方法1: 提取 ```json ... ``` 或 ``` ... ``` 块中的内容
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result)
            if json_match:
                result = json_match.group(1).strip()
                log(f'提取到代码块中的内容')
            
            # 方法2: 移除 chsel...``` 思考块
            result = re.sub(r'chsel[\s\S]*?```', '', result)
            result = result.strip()
            
            # 方法3: 如果是 Markdown 格式，尝试提取实体信息
            if result.startswith('#') or result.startswith('##'):
                log(f'检测到 Markdown 格式响应，尝试提取实体...')
                entities = []
                entity_pattern = r'\d+\.\s*\*\*([^*]+)\*\*\s*-\s*Entity:'
                for match in re.finditer(entity_pattern, result):
                    entity_name = match.group(1).strip()
                    entities.append({
                        "entity_name": entity_name,
                        "entity_type_id": 0
                    })
                if entities:
                    result = json.dumps(entities)
                    log(f'从 Markdown 提取到 {len(entities)} 个实体')
            
            # 方法4: 尝试提取 JSON 对象（贪婪匹配最外层大括号）
            if not result.startswith('{') and not result.startswith('['):
                json_object_match = re.search(r'\{[\s\S]*\}', result)
                if json_object_match:
                    result = json_object_match.group(0)
            
            # 方法5: 尝试提取 JSON 数组
            if not result.startswith('{') and not result.startswith('['):
                json_array_match = re.search(r'\[[\s\S]*\]', result)
                if json_array_match:
                    result = json_array_match.group(0)
            
            # 确保 result 不为空
            if not result or result == '{}':
                log(f'警告: JSON 内容为空，返回空对象')
                return {}
            
            log(f'最终 JSON 长度: {len(result)}')
            log(f'最终 JSON 预览: {result[:300]}')
            
            try:
                parsed = json.loads(result)
                
                # 格式转换: 将 MiniMax 返回的格式转换为 graphiti-core 期望的格式
                # 根据 response_model 的类型决定转换逻辑
                
                if response_model is not None:
                    model_name = response_model.__name__
                    log(f'期望模型: {model_name}')
                    log(f'解析前的数据: {json.dumps(parsed, ensure_ascii=False)[:500]}')
                    
                    # ExtractedEntities 格式: {"extracted_entities": [{"name": "...", "entity_type_id": ...}]}
                    if 'ExtractedEntities' in model_name or 'ExtractedEntity' in str(response_model.model_fields):
                        def convert_entity(entity):
                            entity = dict(entity)
                            log(f'  转换前实体: {entity}')
                            # MiniMax 返回 entity_name 作为实体名称，entity_id 是序号
                            if 'entity_name' in entity and 'name' not in entity:
                                entity['name'] = entity.pop('entity_name')
                            elif 'entity_text' in entity and 'name' not in entity:
                                entity['name'] = entity.pop('entity_text')
                            # 移除 entity_id（序号），不要用作 name
                            if 'entity_id' in entity:
                                entity.pop('entity_id')
                            if 'entity_type_name' in entity and 'entity_type_id' not in entity:
                                entity_type_name = entity.pop('entity_type_name')
                                entity['entity_type_id'] = 0
                            # 移除其他不需要的字段
                            if 'extracted_from' in entity:
                                entity.pop('extracted_from')
                            log(f'  转换后实体: {entity}')
                            return entity
                        
                        if isinstance(parsed, list):
                            entities = [convert_entity(item) for item in parsed]
                            parsed = {"extracted_entities": entities}
                        elif isinstance(parsed, dict):
                            if 'entities' in parsed and 'extracted_entities' not in parsed:
                                entities = [convert_entity(item) for item in parsed.get('entities', [])]
                                parsed = {"extracted_entities": entities}
                            elif 'extracted_entities' in parsed:
                                entities = [convert_entity(item) for item in parsed.get('extracted_entities', [])]
                                parsed = {"extracted_entities": entities}
                            else:
                                entities = [convert_entity(item) for item in parsed.get('extracted_entities', [])]
                                parsed = {"extracted_entities": entities}
                    
                    # NodeResolutions 格式: {"entity_resolutions": [...]}
                    elif 'NodeResolutions' in model_name or 'NodeDuplicate' in str(response_model.model_fields):
                        log(f'NodeResolutions 原始数据: {parsed}')
                        if isinstance(parsed, list):
                            parsed = {"entity_resolutions": parsed}
                            log(f'转换为: {parsed}')
                        elif isinstance(parsed, dict) and 'entity_resolutions' not in parsed:
                            if 'resolutions' in parsed:
                                parsed = {"entity_resolutions": parsed['resolutions']}
                            elif 'node_resolutions' in parsed:
                                pass  # 已经是正确格式
                            else:
                                log(f'未知的 dict 结构: {parsed}')
                    
                    # ExtractedEdges 格式: {"edges": [...]}
                    elif 'ExtractedEdges' in model_name or 'Edge' in str(response_model.model_fields):
                        log(f'ExtractedEdges 原始数据: {parsed}')
                        if isinstance(parsed, list):
                            parsed = {"edges": parsed}
                            log(f'转换为: {parsed}')
                        elif isinstance(parsed, dict) and 'edges' not in parsed:
                            if 'extracted_edges' in parsed:
                                parsed = {"edges": parsed['extracted_edges']}
                            elif 'relations' in parsed:
                                parsed = {"edges": parsed['relations']}
                            else:
                                log(f'未知的 dict 结构: {parsed}')
                    
                log(f'最终解析结果类型: {type(parsed).__name__}')
                return parsed
            except json.JSONDecodeError as e:
                log(f'JSON 解析错误: {e}')
                log(f'尝试修复 JSON...')
                try:
                    # 尝试修复常见的 JSON 问题
                    result = result.replace('\n', ' ')
                    result = re.sub(r',\s*}', '}', result)
                    result = re.sub(r',\s*]', ']', result)
                    parsed = json.loads(result)
                    return parsed
                except:
                    log(f'JSON 修复失败，返回空对象')
                    return {}
            except Exception as e:
                log(f'处理响应时发生错误: {type(e).__name__}: {e}')
                logger.error(f'Error in generating LLM response: {e}')
                raise
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        # 添加多语言提取指令
        messages[0].content += get_extraction_language_instruction(group_id)
        # 使用追踪器
        with self.tracer.start_span('llm.generate') as span:
            attributes = {
                'llm.provider': 'minimax',
                'model.size': model_size.value,
                'max_tokens': max_tokens,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)
            retry_count = 0
            last_error = None
            while retry_count <= self.MAX_RETRIES:
                try:
                    response = await self._generate_response(
                        messages, response_model, max_tokens=max_tokens, model_size=model_size
                    )
                    return response
                except RateLimitError:
                    span.set_status('error', str(last_error))
                    raise
                except Exception as e:
                    last_error = e
                    if retry_count >= self.MAX_RETRIES:
                        logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}')
                        span.set_status('error', str(e))
                        span.record_exception(e)
                        raise
                    retry_count += 1
                    error_context = (
                        f'The previous response attempt was invalid. '
                        f'Error type: {e.__class__.__name__}. '
                        f'Error details: {str(e)}. '
                        f'Please try again with a valid JSON response.'
                    )
                    error_message = Message(role='user', content=error_context)
                    messages.append(error_message)
                    logger.warning(
                        f'Retrying after error (attempt {retry_count}/{self.MAX_RETRIES}): {e}'
                    )
            span.set_status('error', str(last_error))
            raise last_error or Exception('Max retries exceeded with no specific error')
    async def close(self):
        pass


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
        self.llm_client = MiniMaxCompatibleClient()
        # 初始化 Embedder（从环境变量读取，SiliconFlow）
        embedder_api_key = os.environ.get('OPENAI_API_KEY', '')
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
            llm_client=self.llm_client,
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
                name=f"Episode_{group_id}",
                episode_body=content,
                source_description="MiroFish simulation data",
                group_id=group_id,
                reference_time=datetime.now()
            )
            return {
                "success": True,
                "group_id": group_id,
                "message": "Episode added successfully",
                "result": result
            }
        except Exception as e:
            import traceback
            logger.error(f"add_episodes 错误: {e}")
            logger.error(traceback.format_exc())
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
