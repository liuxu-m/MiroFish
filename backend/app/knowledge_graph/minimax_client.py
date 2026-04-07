"""
MiniMax 兼容客户端
封装 MiniMax API 的调用，处理 Graphiti 的 LLMClient 接口
"""

import json
import os
import re
import typing
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.models import Message


def _load_env():
    """加载 .env 文件"""
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / '.env',
        Path.cwd() / '.env',
    ]
    for p in possible_paths:
        if p.exists():
            load_dotenv(str(p), override=True)
            break


_load_env()


class MiniMaxCompatibleClient(LLMClient):
    """
    兼容 MiniMax 的 LLM 客户端

    处理 Graphiti 的 LLMClient 接口，使用 MiniMax API
    处理 MiniMax 思考模型的响应格式
    """

    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        cache: bool = False,
        max_tokens: int = 16384,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        if cache:
            raise NotImplementedError('Caching is not implemented')

        if config is None:
            config = LLMConfig(
                api_key=api_key or os.environ.get('MINIMAX_API_KEY', ''),
                base_url=base_url or os.environ.get('MINIMAX_BASE_URL', 'https://api.minimaxi.com/v1'),
                model=model or os.environ.get('MINIMAX_MODEL', 'MiniMax-M2.7'),
            )

        super().__init__(config, cache)
        self.max_tokens = max_tokens

        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def _generate_response(
        self,
        messages: List[Message],
        response_model: Optional[Type[BaseModel]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> Dict[str, typing.Any]:
        """生成响应"""
        return await self._generate_response_legacy(messages, response_model, max_tokens, model_size)

    async def _generate_response_legacy(
        self,
        messages: List[Message],
        response_model: Optional[Type[BaseModel]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> Dict[str, typing.Any]:
        """传统生成方式（处理 MiniMax 响应格式）"""
        openai_messages = []
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
                extra_body={'reasoning_split': True},
            )

            result = response.choices[0].message.content or '{}'

            # 处理 MiniMax 响应格式
            result = self._parse_response(result, response_model)

            return result

        except Exception as e:
            raise

    def _parse_response(
        self,
        result: str,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """解析 MiniMax 响应"""
        # 移除思考块
        result = re.sub(r'<think[\s\S]*?</think\s*>', '', result)
        result = result.strip()

        # 提取代码块
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result)
        if json_match:
            result = json_match.group(1).strip()

        # 移除 chsel 思考块
        result = re.sub(r'chsel[\s\S]*?```', '', result)
        result = result.strip()

        # 提取 JSON
        if not result.startswith('{') and not result.startswith('['):
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                result = json_match.group(0)

        if not result or result == '{}':
            return {}

        try:
            parsed = json.loads(result)

            # 格式化转换
            if response_model is not None:
                model_name = response_model.__name__

                # ExtractedEntities 格式
                if 'ExtractedEntities' in model_name or 'ExtractedEntity' in str(response_model.model_fields):
                    if isinstance(parsed, list):
                        entities = []
                        for item in parsed:
                            item = dict(item)
                            if 'entity_name' in item and 'name' not in item:
                                item['name'] = item.pop('entity_name')
                            entities.append(item)
                        parsed = {"extracted_entities": entities}

                # NodeResolutions 格式
                elif 'NodeResolutions' in model_name or 'NodeDuplicate' in str(response_model.model_fields):
                    if isinstance(parsed, list):
                        parsed = {"entity_resolutions": parsed}
                    elif isinstance(parsed, dict) and 'entity_resolutions' not in parsed:
                        if 'id' in parsed and 'name' in parsed:
                            parsed = {"entity_resolutions": [parsed]}

                # ExtractedEdges 格式
                elif 'ExtractedEdges' in model_name or 'Edge' in str(response_model.model_fields):
                    if isinstance(parsed, list):
                        converted_edges = []
                        for edge in parsed:
                            if isinstance(edge, dict):
                                converted_edge = {}
                                if 'source_entity_name' in edge:
                                    converted_edge['source_node_name'] = edge['source_entity_name']
                                if 'target_entity_name' in edge:
                                    converted_edge['target_node_name'] = edge['target_entity_name']
                                if 'relation_type' in edge:
                                    converted_edge['name'] = edge['relation_type']
                                if 'fact' in edge:
                                    converted_edge['fact'] = edge['fact']
                                converted_edges.append({**edge, **converted_edge})
                        parsed = {"edges": converted_edges}

            return parsed

        except json.JSONDecodeError:
            return {}

    def _clean_input(self, text: str) -> str:
        """清理输入文本"""
        return text

    async def generate_response(
        self,
        messages: List[Message],
        response_model: Optional[Type[BaseModel]] = None,
        max_tokens: Optional[int] = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: Optional[str] = None,
        prompt_name: Optional[str] = None,
    ) -> Dict[str, typing.Any]:
        """生成响应（带重试）"""
        from graphiti_core.llm_client.client import get_extraction_language_instruction

        if max_tokens is None:
            max_tokens = self.max_tokens

        # 添加多语言指令
        if group_id:
            messages[0].content += get_extraction_language_instruction(group_id)

        retry_count = 0
        last_error = None
        while retry_count <= self.MAX_RETRIES:
            try:
                response = await self._generate_response(
                    messages, response_model, max_tokens=max_tokens, model_size=model_size
                )
                return response
            except Exception as e:
                last_error = e
                if retry_count >= self.MAX_RETRIES:
                    raise
                retry_count += 1

        raise last_error or Exception('Max retries exceeded')

    async def close(self):
        """关闭客户端"""
        pass
