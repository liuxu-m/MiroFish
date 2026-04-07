"""
Graphiti 知识图谱服务
提供基于 Graphiti 的知识图谱构建和检索功能
"""

import os
import re
import json
import logging
import typing
import asyncio
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional
from datetime import datetime, timezone
from typing import get_origin, get_args

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

# LangChain 导入 - 延迟到函数内部，避免 torch DLL 问题
# 在 Windows 上 torch 的 shm.dll 可能无法加载，导致整个模块导入失败
HAS_LANGCHAIN = False

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.client import LLMClient, get_extraction_language_instruction
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.models import Message
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig

from ..config import Config

import sys

logger = logging.getLogger(__name__)

# 确保 print 输出到 stderr，这样 Flask 不会吞掉
def debug_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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


# ============ LangChain 辅助函数 ============

def get_langchain_llm(temperature: float = 0.3):
    """获取 LangChain MiniMax LLM 实例（推荐方案）"""
    global HAS_LANGCHAIN
    if not HAS_LANGCHAIN:
        # 尝试动态导入 langchain_openai
        try:
            from langchain_openai import ChatOpenAI
            HAS_LANGCHAIN = True
        except Exception as e:
            logging.warning(f"LangChain import failed: {e}")
            raise ImportError("LangChain not installed or failed to load")

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model='MiniMax-M2.7',
        temperature=temperature,
        api_key=os.environ.get('MINIMAX_API_KEY', ''),
        base_url='https://api.minimaxi.com/v1',
        extra_body={
            'reasoning_split': True,
            'response_format': {'type': 'json_object'}
        }
    )


def coerce_type(value, target_type):
    """类型转换：将值转换为目标类型"""
    if value is None:
        return None

    origin = get_origin(target_type)

    # 处理 list[X]
    if origin is list:
        elem_type = get_args(target_type)[0] if get_args(target_type) else str
        if isinstance(value, list):
            return [coerce_type(v, elem_type) for v in value]
        elif isinstance(value, str):
            value = value.replace('/', ',')
            return [coerce_type(v.strip(), elem_type) for v in value.split(',') if v.strip()]
        return [value]

    # 处理 float/int
    if target_type in (float, int):
        if isinstance(value, (int, float)):
            return target_type(value)
        if isinstance(value, str):
            numbers = re.findall(r'[\d.]+', value)
            if numbers:
                try:
                    return target_type(float(numbers[0]))
                except:
                    pass
            return target_type(0)
        try:
            return target_type(value)
        except:
            return target_type(0)

    # 处理 str
    if target_type is str:
        return str(value) if value else ""

    return value


def parse_json_response(text: str) -> dict:
    """解析 JSON 响应"""
    text = text.strip()

    # 去掉 markdown 代码块
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        decoder = json.JSONDecoder()
        data, idx = decoder.raw_decode(text)
        return data if data else {}
    except:
        pass

    # 备用：提取第一个 JSON 对象
    start = text.find('{')
    if start >= 0:
        try:
            decoder = json.JSONDecoder()
            return decoder.raw_decode(text[start:])[0]
        except:
            pass

    raise ValueError(f"Cannot parse JSON: {text[:50]}")


def langchain_generate(schema: type, text: str, group_id: str = None) -> dict:
    """
    使用 LangChain 生成结构化输出（推荐方案）

    Args:
        schema: Pydantic 模型类
        text: 输入文本
        group_id: 分组ID（用于多语言指令）

    Returns:
        dict: 解析后的数据字典
    """
    llm = get_langchain_llm()

    # 构建格式字符串
    fields = schema.model_fields
    # 使用 schema 定义的字段名
    format_str = json.dumps({k: "xxx" for k in fields}, ensure_ascii=False)

    # 添加多语言指令
    instruction = ""
    if group_id:
        instruction = get_extraction_language_instruction(group_id)

    # 构建 prompt - 明确要求使用 schema 的字段名
    prompt = f"""从以下文本中提取信息。
要求返回JSON格式: {format_str}
{instruction}
文本: {text}"""

    # 调用 LLM
    response = llm.invoke(prompt)

    # 解析 JSON
    data = parse_json_response(response.content)

    # 解析 JSON
    data = parse_json_response(response.content)

    # 类型转换
    result = {}
    for field_name, field_info in fields.items():
        target_type = field_info.annotation
        if field_name in data:
            result[field_name] = coerce_type(data[field_name], target_type)
        else:
            result[field_name] = coerce_type(None, target_type)

    return result


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
        # 暂时使用传统方式，等 LangChain 格式完全匹配后再启用
        # if response_model and HAS_LANGCHAIN:
        #     return await self._generate_with_langchain(messages, response_model)
        return await self._generate_response_legacy(messages, response_model, max_tokens, model_size)

    async def _generate_with_langchain(
        self,
        messages: list[Message],
        response_model: type[BaseModel],
    ) -> dict[str, typing.Any]:
        """使用 LangChain 生成结构化输出（推荐方案）"""
        # 合并消息为单个文本
        text = ""
        for m in messages:
            text += f"{m.role}: {m.content}\n"

        log(f'\n=== 使用 LangChain 生成 ===')
        log(f'model: {response_model.__name__}')

        try:
            # 使用 LangChain 生成
            result = langchain_generate(response_model, text)
            log(f'结果: {result}')
            return result
        except Exception as e:
            log(f'LangChain 生成失败: {e}')
            # 回退到传统方式
            return await self._generate_response_legacy(messages, response_model)

    async def _generate_response_legacy(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """传统生成方式（备用）"""
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
                extra_body={'reasoning_split': True},
            )
            result = response.choices[0].message.content or '{}'
            
            # 调试: 打印原始响应
            log(f'\n=== MiniMax 原始响应 ===')
            log(f'响应长度: {len(result)}')
            log(f'响应预览: {result[:2000] if result else "EMPTY"}')
            log(f'=== 响应结束 ===\n')
            
            # 处理 MiniMax 响应格式
            # 1. 思考内容格式: <think...</think 或 chsel...```
            # 2. JSON 代码块: ```json...``` 或 ```...```
            # 3. Markdown 格式: ## Entity Extraction Results (需要特殊处理)
            
            # 方法0: 移除 <think...</think 思考块 (MiniMax 新格式)
            result = re.sub(r'<think[\s\S]*?</think\s*>', '', result)
            result = result.strip()
            
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

            # 方法3.5: 处理 **Name**: "summary" 格式的 Markdown 摘要响应
            if '**' in result and '": "' in result:
                log(f'检测到 **Name**: "summary" 格式的 Markdown 摘要')
                summaries = []
                # 匹配 **实体名**: "摘要内容" 格式
                summary_pattern = r'\*\*([^*]+)\*\*:\s*"([^"]*)"'
                for match in re.finditer(summary_pattern, result):
                    entity_name = match.group(1).strip()
                    entity_summary = match.group(2).strip()
                    summaries.append({
                        "name": entity_name,
                        "summary": entity_summary
                    })
                if summaries:
                    parsed = {"summaries": summaries}
                    log(f'从 Markdown 摘要格式提取到 {len(summaries)} 个摘要')
                    # 直接返回，因为已经是正确格式
                    return parsed
            
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
                            # 处理 MiniMax 可能返回的各种字段名
                            if 'resolutions' in parsed:
                                parsed = {"entity_resolutions": parsed['resolutions']}
                            elif 'node_resolutions' in parsed:
                                parsed = {"entity_resolutions": parsed['node_resolutions']}
                            elif 'id' in parsed and 'name' in parsed:
                                # 单个对象: {"id": 0, "name": ..., "duplicate_name": ""}
                                # 需要包装成数组格式: {"entity_resolutions": [{...}]}
                                parsed = {"entity_resolutions": [parsed]}
                            else:
                                log(f'未知的 dict 结构: {parsed}')
                    
                    # ExtractedEdges 格式: {"edges": [...]} 或直接数组
                    elif 'ExtractedEdges' in model_name or 'Edge' in str(response_model.model_fields):
                        log(f'ExtractedEdges 原始数据: {parsed}')
                        if isinstance(parsed, list):
                            # MiniMax 返回的是数组，需要转换字段名
                            converted_edges = []
                            for edge in parsed:
                                if isinstance(edge, dict):
                                    converted_edge = {}
                                    # 转换字段名
                                    if 'source_entity_name' in edge:
                                        converted_edge['source_node_name'] = edge['source_entity_name']
                                    if 'target_entity_name' in edge:
                                        converted_edge['target_node_name'] = edge['target_entity_name']
                                    if 'relation_type' in edge:
                                        converted_edge['name'] = edge['relation_type']
                                    if 'fact' in edge:
                                        converted_edge['fact'] = edge['fact']
                                    if 'valid_at' in edge:
                                        converted_edge['valid_at'] = edge['valid_at']
                                    # 保留原始字段（以防需要）
                                    for k, v in edge.items():
                                        if k not in converted_edge:
                                            converted_edge[k] = v
                                    converted_edges.append(converted_edge)
                            parsed = {"edges": converted_edges}
                            log(f'ExtractedEdges 数组转换后: {converted_edges[:2]}...')
                        elif isinstance(parsed, dict) and 'edges' not in parsed:
                            if 'head' in parsed:
                                parsed = {"edges": parsed['head']}
                            elif 'extracted_edges' in parsed:
                                parsed = {"edges": parsed['extracted_edges']}
                            elif 'relations' in parsed:
                                parsed = {"edges": parsed['relations']}
                            else:
                                log(f'未知的 dict 结构: {parsed}')

                    # SummarizedEntities 格式: {"summaries": [{"name": "...", "summary": "..."}]}
                    elif 'SummarizedEntities' in model_name or 'Summary' in model_name:
                        log(f'SummarizedEntities 原始数据: {parsed}')
                        if isinstance(parsed, dict):
                            # 检查是否有 summaries 字段
                            if 'summaries' not in parsed:
                                # 尝试从其他格式转换
                                summaries = []
                                # 可能是 {"summary": "..."} 格式（单个摘要）
                                if 'summary' in parsed and 'name' in parsed:
                                    summaries = [parsed]
                                # 可能是 {"entity_summaries": [...]} 格式
                                elif 'entity_summaries' in parsed:
                                    summaries = parsed['entity_summaries']
                                # 可能是 {"summaries_list": [...]} 格式
                                elif 'summaries_list' in parsed:
                                    summaries = parsed['summaries_list']
                                # 遍历所有键查找可能的摘要数组
                                else:
                                    for key, value in parsed.items():
                                        if isinstance(value, list) and len(value) > 0:
                                            first_item = value[0]
                                            if isinstance(first_item, dict) and 'summary' in first_item:
                                                summaries = value
                                                break

                                if summaries:
                                    # 转换字段名
                                    converted = []
                                    for item in summaries:
                                        if isinstance(item, dict):
                                            # 确保有 name 和 summary 字段
                                            converted_item = {}
                                            if 'name' in item:
                                                converted_item['name'] = item['name']
                                            elif 'entity_name' in item:
                                                converted_item['name'] = item['entity_name']
                                            if 'summary' in item:
                                                converted_item['summary'] = item['summary']
                                            elif 'entity_summary' in item:
                                                converted_item['summary'] = item['entity_summary']
                                            if converted_item.get('name') and converted_item.get('summary'):
                                                converted.append(converted_item)
                                    if converted:
                                        parsed = {"summaries": converted}
                                        log(f'SummarizedEntities 转换后: {parsed}')
                                elif 'summary' in parsed:
                                    # 只有摘要内容，没有实体名，无法使用
                                    log(f'警告: SummarizedEntities 没有实体名，返回空')
                                    parsed = {"summaries": []}

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
    
    注意: 由于 Windows 上 Neo4j 异步驱动与 asyncio 事件循环的兼容性问题，
    每次操作都会创建新的 Graphiti 实例，避免跨事件循环使用连接。
    """
    
    def _create_graphiti(self):
        """创建新的 Graphiti 实例"""
        llm_client = MiniMaxCompatibleClient()
        embedder_api_key = os.environ.get('OPENAI_API_KEY', '')
        embedder_base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.siliconflow.cn/v1')
        embedder_model = os.environ.get('OPENAI_EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B')
        embedder_config = OpenAIEmbedderConfig(
            api_key=embedder_api_key,
            base_url=embedder_base_url,
            embedding_model=embedder_model
        )
        embedder = OpenAIEmbedder(config=embedder_config)
        return Graphiti(
            uri=Config.NEO4J_URI,
            user=Config.NEO4J_USER,
            password=Config.NEO4J_PASSWORD,
            llm_client=llm_client,
            embedder=embedder
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
        graphiti = self._create_graphiti()
        last_error = None

        for attempt in range(max_retries):
            try:
                result = await graphiti.add_episode(
                    name=f"Episode_{group_id}",
                    episode_body=content,
                    source_description="MiroFish simulation data",
                    group_id=group_id,
                    reference_time=datetime.now()
                )
                await graphiti.close()
                return {
                    "success": True,
                    "group_id": group_id,
                    "message": "Episode added successfully",
                    "result": str(result) if result else None
                }
            except Exception as e:
                last_error = e
                error_str = str(e)

                # 检查是否是 summarization 验证错误
                if "SummarizedEntities" in error_str and "summaries" in error_str:
                    logger.warning(f"实体摘要提取失败 (尝试 {attempt+1}/{max_retries}): {error_str[:100]}")

                    # 如果还有重试机会，尝试继续（跳过这个chunk）
                    if attempt < max_retries - 1:
                        import asyncio
                        await asyncio.sleep(1)  # 等待一下再重试
                        continue
                else:
                    # 其他错误，直接返回失败
                    logger.error(f"add_episodes 错误: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    break
            finally:
                try:
                    await graphiti.close()
                except:
                    pass

        # 所有重试都失败
        return {
            "success": False,
            "group_id": group_id,
            "error": f"实体摘要提取失败: {last_error}",
            "partial": True  # 标记为部分成功，文本可能已添加但没有摘要
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
        graphiti = self._create_graphiti()
        try:
            results = await graphiti.search(
                query=query,
                group_ids=[group_id],
                num_results=limit
            )
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
        finally:
            await graphiti.close()
    async def get_nodes(
        self,
        group_id: str
    ) -> List[Dict[str, Any]]:
        """
        获取指定分组的全部节点
        Args:
            group_id: 分组ID
        Returns:
            节点列表（包含丰富的属性）
        """
        from neo4j import GraphDatabase
        try:
            print(f"GET_NODES: start, group_id={group_id}")
            uri = Config.NEO4J_URI
            user = Config.NEO4J_USER
            password = Config.NEO4J_PASSWORD

            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                # 查询实体 - 获取完整的节点属性
                result = session.run(
                    '''MATCH (n:Entity) WHERE n.group_id = $group_id OR n.group_id STARTS WITH "mirofish"
                    RETURN n.name, id(n), labels(n), n.summary, n.entity_type, n.created_at, n.attributes, n.uuid''',
                    group_id=group_id
                )

                nodes = []
                for record in result:
                    name = record[0]
                    node_id = record[1]
                    labels = record[2] or []
                    summary = record[3] or ""
                    entity_type = record[4] or ""
                    created_at = record[5]
                    attributes = record[6] or {}
                    uuid = record[7] or str(node_id)

                    # 解析 attributes
                    if isinstance(attributes, str):
                        import json
                        try:
                            attributes = json.loads(attributes)
                        except:
                            attributes = {}

                    # 合并 labels 和 entity_type
                    node_labels = labels.copy() if labels else []
                    if entity_type and entity_type not in node_labels:
                        node_labels.append(entity_type)

                    nodes.append({
                        "uuid": uuid,
                        "name": name,
                        "labels": node_labels,
                        "summary": summary,
                        "entity_type": entity_type,
                        "attributes": attributes,
                        "created_at": str(created_at) if created_at else None,
                    })

            driver.close()
            print(f"GET_NODES: returning {len(nodes)} nodes")
            return nodes
        except Exception as e:
            logger.error(f"get_nodes 错误: {e}")
            import traceback
            traceback.print_exc()
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
            边列表（包含丰富的属性）
        """
        from neo4j import GraphDatabase
        try:
            print(f"GET_EDGES: start, group_id={group_id}")
            uri = Config.NEO4J_URI
            user = Config.NEO4J_USER
            password = Config.NEO4J_PASSWORD

            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                # 查询关系 - 获取完整的边属性
                result = session.run(
                    '''MATCH (a:Entity)-[r]->(b:Entity)
                    WHERE a.group_id = $group_id OR b.group_id = $group_id OR a.group_id STARTS WITH "mirofish" OR b.group_id STARTS WITH "mirofish"
                    RETURN a.name, type(r), b.name, r.name, r.fact, r.summary, r.created_at, r.attributes, a.uuid, b.uuid''',
                    group_id=group_id
                )

                edges = []
                for record in result:
                    source_name = record[0]
                    rel_type = record[1]
                    target_name = record[2]
                    rel_name = record[3] or ""
                    fact = record[4] or ""
                    summary = record[5] or ""
                    created_at = record[6]
                    attributes = record[7] or {}
                    source_uuid = record[8] or source_name
                    target_uuid = record[9] or target_name

                    # 解析 attributes
                    if isinstance(attributes, str):
                        import json
                        try:
                            attributes = json.loads(attributes)
                        except:
                            attributes = {}

                    edges.append({
                        "name": rel_name or rel_type,
                        "fact": fact,
                        "summary": summary,
                        "source": source_uuid,
                        "target": target_uuid,
                        "source_node_uuid": source_uuid,
                        "target_node_uuid": target_uuid,
                        "fact_type": rel_type,
                        "attributes": attributes,
                        "created_at": str(created_at) if created_at else None,
                        "properties": attributes,
                    })

            driver.close()
            print(f"GET_EDGES: returning {len(edges)} edges")
            return edges
        except Exception as e:
            logger.error(f"get_edges 错误: {e}")
            import traceback
            traceback.print_exc()
            return []
