"""
图谱配置
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class Neo4jConfig:
    """Neo4j 配置"""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"


@dataclass
class LLMConfig:
    """LLM 配置"""
    api_key: str = ""
    base_url: str = "https://api.minimaxi.com/v1"
    model: str = "MiniMax-M2.7"


@dataclass
class EmbedderConfig:
    """Embedding 配置"""
    api_key: str = ""
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "Qwen/Qwen3-Embedding-0.6B"


@dataclass
class GraphConfig:
    """图谱全局配置"""
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)

    # 去重配置
    deduplicate_nodes: bool = True      # 是否去重节点
    deduplicate_edges: bool = True       # 是否去重边
    edge_fact_prefix_length: int = 100   # 边去重时取 fact 的前 N 个字符

    # 图谱查询配置
    group_id_prefix: str = "mirofish"   # 图谱 group_id 前缀


def load_config(env_path: str = None) -> GraphConfig:
    """从环境变量加载配置"""
    if env_path is None:
        # 尝试多个可能的路径
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / '.env',  # 项目根目录
            Path.cwd() / '.env',
        ]
        for p in possible_paths:
            if p.exists():
                env_path = str(p)
                break

    if env_path and Path(env_path).exists():
        load_dotenv(env_path, override=True)

    return GraphConfig(
        neo4j=Neo4jConfig(
            uri=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
            user=os.environ.get('NEO4J_USER', 'neo4j'),
            password=os.environ.get('NEO4J_PASSWORD', 'password'),
        ),
        llm=LLMConfig(
            api_key=os.environ.get('MINIMAX_API_KEY', ''),
            base_url=os.environ.get('MINIMAX_BASE_URL', 'https://api.minimaxi.com/v1'),
            model=os.environ.get('MINIMAX_MODEL', 'MiniMax-M2.7'),
        ),
        embedder=EmbedderConfig(
            api_key=os.environ.get('OPENAI_API_KEY', ''),
            base_url=os.environ.get('OPENAI_BASE_URL', 'https://api.siliconflow.cn/v1'),
            model=os.environ.get('OPENAI_EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B'),
        ),
    )


# 全局默认配置
default_config = load_config()
