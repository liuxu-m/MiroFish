"""
业务服务模块
"""

from .ontology_generator import OntologyGenerator
from .graph_builder import GraphBuilderService
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
from .graphiti_service import GraphitiService, MiniMaxCompatibleClient
from .graphiti_tools import GraphitiToolsService
from .entity_reader import EntityReader
from .memory_updater import MemoryUpdater

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
    'GraphBuilderService',
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
    # Graphiti services
    'GraphitiService',
    'MiniMaxCompatibleClient',
    'GraphitiToolsService',
    'EntityReader',
    'MemoryUpdater',
    # Zep services (optional)
    'ZepEntityReader',
    'EntityNode',
    'FilteredEntities',
    'ZepGraphMemoryUpdater',
    'ZepGraphMemoryManager',
    'AgentActivity',
    'HAS_ZEP',
]

