"""
Pipeline Configuration para sistema RAG triple integrado
Configuración centralizada para Vector + Graph + Metadata processing
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

class ProcessingMode(str, Enum):
    """Modos de procesamiento disponibles"""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only" 
    METADATA_ONLY = "metadata_only"
    VECTOR_GRAPH = "vector_graph"
    VECTOR_METADATA = "vector_metadata"
    GRAPH_METADATA = "graph_metadata"
    TRIPLE_FULL = "triple_full"

class ErrorStrategy(str, Enum):
    """Estrategias de manejo de errores"""
    FAIL_FAST = "fail_fast"           # Falla inmediatamente al primer error
    PARTIAL_SUCCESS = "partial_success"  # Continúa con componentes exitosos
    ROLLBACK_ALL = "rollback_all"     # Rollback completo si hay error

@dataclass
class VectorConfig:
    """Configuración para procesamiento vectorial"""
    enabled: bool = True
    model_name: str = "all-mpnet-base-v2"
    use_static_embeddings: bool = True
    chunk_size: int = 500
    chunk_overlap: int = 50
    batch_size: int = 32
    collection_name: str = "rag_documents"

@dataclass
class GraphConfig:
    """Configuración para procesamiento de grafos"""
    enabled: bool = True
    spacy_model: str = "es_core_news_sm"
    community_algorithm: str = "leiden"  # leiden, louvain
    enable_llm_summarization: bool = False
    min_entities: int = 2
    min_relationships: int = 1
    enable_visualization: bool = True
    output_formats: list = None
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ["knowledge_graph", "networkx_graph", "communities_graph"]

@dataclass
class MetadataConfig:
    """Configuración para sistema de metadata"""
    enabled: bool = True
    database_url: str = "sqlite:///data/metadata/rag_metadata.db"
    enable_api_logging: bool = True
    enable_metrics_collection: bool = True
    auto_cleanup_days: int = 30

@dataclass
class TripleProcessorConfig:
    """Configuración completa del pipeline triple"""
    # Configuraciones por componente
    vector: VectorConfig
    graph: GraphConfig
    metadata: MetadataConfig
    
    # Configuración global
    processing_mode: ProcessingMode = ProcessingMode.TRIPLE_FULL
    error_strategy: ErrorStrategy = ErrorStrategy.PARTIAL_SUCCESS
    enable_parallel_processing: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300
    
    # Logging y monitoring
    enable_detailed_logging: bool = True
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    enable_performance_metrics: bool = True
    
    def __post_init__(self):
        """Validación y ajustes post-inicialización"""
        # Validar consistencia
        if self.processing_mode == ProcessingMode.VECTOR_ONLY:
            self.graph.enabled = False
            self.metadata.enabled = False
        elif self.processing_mode == ProcessingMode.GRAPH_ONLY:
            self.vector.enabled = False
            self.metadata.enabled = False
        elif self.processing_mode == ProcessingMode.METADATA_ONLY:
            self.vector.enabled = False
            self.graph.enabled = False
        elif self.processing_mode == ProcessingMode.VECTOR_GRAPH:
            self.metadata.enabled = False
        elif self.processing_mode == ProcessingMode.VECTOR_METADATA:
            self.graph.enabled = False
        elif self.processing_mode == ProcessingMode.GRAPH_METADATA:
            self.vector.enabled = False
        # TRIPLE_FULL mantiene todo habilitado

    @classmethod
    def create_default(cls) -> 'TripleProcessorConfig':
        """Crear configuración por defecto"""
        return cls(
            vector=VectorConfig(),
            graph=GraphConfig(),
            metadata=MetadataConfig()
        )
    
    @classmethod
    def create_development(cls) -> 'TripleProcessorConfig':
        """Configuración optimizada para desarrollo"""
        return cls(
            vector=VectorConfig(
                use_static_embeddings=True,
                batch_size=16
            ),
            graph=GraphConfig(
                enable_llm_summarization=False,
                enable_visualization=True
            ),
            metadata=MetadataConfig(
                enable_api_logging=True,
                auto_cleanup_days=7
            ),
            enable_parallel_processing=True,
            enable_detailed_logging=True,
            log_level="DEBUG"
        )
    
    @classmethod
    def create_production(cls) -> 'TripleProcessorConfig':
        """Configuración optimizada para producción"""
        return cls(
            vector=VectorConfig(
                use_static_embeddings=True,
                batch_size=64,
                collection_name="production_rag_documents"
            ),
            graph=GraphConfig(
                spacy_model="es_core_news_lg",
                enable_llm_summarization=True,
                enable_visualization=False  # Reducir overhead en producción
            ),
            metadata=MetadataConfig(
                database_url="sqlite:///data/metadata/production_rag_metadata.db",
                enable_api_logging=True,
                auto_cleanup_days=90
            ),
            enable_parallel_processing=True,
            enable_detailed_logging=False,
            log_level="WARNING",
            timeout_seconds=600
        )
    
    @classmethod 
    def create_testing(cls) -> 'TripleProcessorConfig':
        """Configuración para testing automatizado"""
        return cls(
            vector=VectorConfig(
                batch_size=8,
                collection_name="test_rag_documents"
            ),
            graph=GraphConfig(
                enable_llm_summarization=False,
                enable_visualization=False
            ),
            metadata=MetadataConfig(
                database_url="sqlite:///test_rag_metadata.db",
                enable_api_logging=False,
                auto_cleanup_days=1
            ),
            processing_mode=ProcessingMode.TRIPLE_FULL,
            error_strategy=ErrorStrategy.FAIL_FAST,
            enable_parallel_processing=False,  # Secuencial para testing
            max_retries=1,
            timeout_seconds=60,
            enable_detailed_logging=True,
            log_level="DEBUG"
        )

    def set_processing_mode(self, mode: ProcessingMode):
        """Cambiar modo de procesamiento y actualizar configuraciones"""
        self.processing_mode = mode
        self.__post_init__()  # Reaplica la lógica de configuración
    
    def get_enabled_systems(self) -> Dict[str, bool]:
        """Obtener qué sistemas están habilitados"""
        return {
            "vector": self.vector.enabled,
            "graph": self.graph.enabled,
            "metadata": self.metadata.enabled
        }
    
    def validate(self) -> bool:
        """Validar configuración"""
        enabled_systems = self.get_enabled_systems()
        
        # Al menos un sistema debe estar habilitado
        if not any(enabled_systems.values()):
            raise ValueError("Al menos un sistema debe estar habilitado")
        
        # Validar timeouts
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds debe ser positivo")
        
        # Validar retries
        if self.max_retries < 0:
            raise ValueError("max_retries no puede ser negativo")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario"""
        return {
            "vector": {
                "enabled": self.vector.enabled,
                "model_name": self.vector.model_name,
                "use_static_embeddings": self.vector.use_static_embeddings,
                "chunk_size": self.vector.chunk_size,
                "chunk_overlap": self.vector.chunk_overlap,
                "batch_size": self.vector.batch_size,
                "collection_name": self.vector.collection_name
            },
            "graph": {
                "enabled": self.graph.enabled,
                "spacy_model": self.graph.spacy_model,
                "community_algorithm": self.graph.community_algorithm,
                "enable_llm_summarization": self.graph.enable_llm_summarization,
                "min_entities": self.graph.min_entities,
                "min_relationships": self.graph.min_relationships,
                "enable_visualization": self.graph.enable_visualization,
                "output_formats": self.graph.output_formats
            },
            "metadata": {
                "enabled": self.metadata.enabled,
                "database_url": self.metadata.database_url,
                "enable_api_logging": self.metadata.enable_api_logging,
                "enable_metrics_collection": self.metadata.enable_metrics_collection,
                "auto_cleanup_days": self.metadata.auto_cleanup_days
            },
            "global": {
                "processing_mode": self.processing_mode.value,
                "error_strategy": self.error_strategy.value,
                "enable_parallel_processing": self.enable_parallel_processing,
                "max_retries": self.max_retries,
                "timeout_seconds": self.timeout_seconds,
                "enable_detailed_logging": self.enable_detailed_logging,
                "log_level": self.log_level,
                "enable_performance_metrics": self.enable_performance_metrics
            }
        } 