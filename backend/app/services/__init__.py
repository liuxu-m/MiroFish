"""
业务服务模块
"""

from .ontology_generator import OntologyGenerator
from .text_processor import TextProcessor
from .oasis_profile_generator import OasisProfileGenerator, OasisAgentProfile
from .simulation_manager import SimulationManager, SimulationState, SimulationStatus
from .simulation_config_generator import (
    SimulationConfigGenerator,
    SimulationParameters,
    AgentActivityConfig,
    TimeSimulationConfig,
    EventConfig,
    PlatformConfig
)
from .simulation_runner import (
    SimulationRunner,
    SimulationRunState,
    RunnerStatus,
    AgentAction,
    RoundSummary
)
from .simulation_ipc import (
    SimulationIPCClient,
    SimulationIPCServer,
    IPCCommand,
    IPCResponse,
    CommandType,
    CommandStatus
)
from .graphiti_tools import GraphitiToolsService
from .entity_reader import EntityReader
from .memory_updater import MemoryUpdater

# 从 knowledge_graph 包导入图谱服务（核心服务）
from ..knowledge_graph import (
    GraphBuilderService,
    GraphitiService,
    GraphData,
    NodeData,
    EdgeData,
)

# MiniMax 客户端也从 knowledge_graph 导出
from ..knowledge_graph.minimax_client import MiniMaxCompatibleClient

# Zep 服务 - 可选导入（仅当 zep_cloud 已安装时）
try:
    from .zep_entity_reader import ZepEntityReader, EntityNode, FilteredEntities
    from .zep_graph_memory_updater import (
        ZepGraphMemoryUpdater,
        ZepGraphMemoryManager,
        AgentActivity
    )
    HAS_ZEP = True
except ImportError:
    HAS_ZEP = False
    ZepEntityReader = None
    EntityNode = None
    FilteredEntities = None
    ZepGraphMemoryUpdater = None
    ZepGraphMemoryManager = None
    AgentActivity = None

__all__ = [
    'OntologyGenerator',
    'GraphBuilderService',      # 来自 knowledge_graph
    'TextProcessor',
    'OasisProfileGenerator',
    'OasisAgentProfile',
    'SimulationManager',
    'SimulationState',
    'SimulationStatus',
    'SimulationConfigGenerator',
    'SimulationParameters',
    'AgentActivityConfig',
    'TimeSimulationConfig',
    'EventConfig',
    'PlatformConfig',
    'SimulationRunner',
    'SimulationRunState',
    'RunnerStatus',
    'AgentAction',
    'RoundSummary',
    'SimulationIPCClient',
    'SimulationIPCServer',
    'IPCCommand',
    'IPCResponse',
    'CommandType',
    'CommandStatus',
    # Graphiti services（来自 knowledge_graph）
    'GraphitiService',
    'MiniMaxCompatibleClient',
    'GraphitiToolsService',
    'EntityReader',
    'MemoryUpdater',
    # 类型（来自 knowledge_graph）
    'GraphData',
    'NodeData',
    'EdgeData',
    # Zep services (optional)
    'ZepEntityReader',
    'EntityNode',
    'FilteredEntities',
    'ZepGraphMemoryUpdater',
    'ZepGraphMemoryManager',
    'AgentActivity',
    'HAS_ZEP',
]

