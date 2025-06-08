"""
Core package para el sistema RAG triple integrado
Contiene el orchestrator principal y configuraciones
"""

from .triple_processor import TripleProcessor, TripleProcessorResult
from .embedding_types import EmbeddingResult
from .pipeline_config import (
    TripleProcessorConfig, VectorConfig, GraphConfig, MetadataConfig,
    ProcessingMode, ErrorStrategy
)
from .unified_metrics import (
    TripleProcessorMetrics, MetricsCollector, ProcessingStage, ProcessingStatus,
    StageMetrics, ComponentMetrics
)

__all__ = [
    # Main processor
    "TripleProcessor",
    "TripleProcessorResult",
    
    # Configuration
    "TripleProcessorConfig",
    "VectorConfig", 
    "GraphConfig",
    "MetadataConfig",
    "ProcessingMode",
    "ErrorStrategy",
    
    # Metrics
    "TripleProcessorMetrics",
    "MetricsCollector",
    "ProcessingStage",
    "ProcessingStatus",
    "StageMetrics",
    "ComponentMetrics"
] 