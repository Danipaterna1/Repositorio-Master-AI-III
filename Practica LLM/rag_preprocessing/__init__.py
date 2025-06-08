"""
RAG Preprocessing Module 2025
Sistema moderno de preprocesamiento para RAG con stack dual desarrollo/producción.

Componentes principales:
- ModernEmbeddingManager: Static embeddings + fallback tradicional
- HybridVectorStore: ChromaDB (dev) + Qdrant (prod) 
- DocumentProcessor: Pipeline completo de procesamiento
- RAGConfig: Configuración flexible con flags

Uso básico:
    from rag_preprocessing import get_document_processor
    
    processor = get_document_processor()
    result = processor.process_text("Mi documento aquí")
"""

from .config.settings import (
    RAGConfig, 
    get_config, 
    switch_environment,
    EnvironmentType,
    EmbeddingType,
    VectorStoreType,
    GraphStoreType
)

# Legacy embedding_manager removed - functionality moved to triple_processor
# from .core.embedding_manager import (
#     ModernEmbeddingManager,
#     get_embedding_manager,
#     create_embedding_manager,
#     EmbeddingResult
# )

from .storage.hybrid_vector_store import (
    HybridVectorStore,
    get_vector_store,
    create_vector_store,
    VectorDocument,
    VectorSearchResult
)

# Legacy document_processor removed - functionality moved to triple_processor
# from .core.document_processor import (
#     DocumentProcessor,
#     get_document_processor,
#     create_document_processor,
#     ProcessingResult,
#     DocumentChunk
# )

__version__ = "0.1.0"
__author__ = "RAG Team 2025"

# API principal simplificada
__all__ = [
    # Configuración
    "RAGConfig",
    "get_config", 
    "switch_environment",
    "EnvironmentType",
    "EmbeddingType", 
    "VectorStoreType",
    "GraphStoreType",
    
    # Embedding Manager (legacy - moved to triple_processor)
    # "ModernEmbeddingManager",
    # "get_embedding_manager",
    # "create_embedding_manager", 
    # "EmbeddingResult",
    
    # Vector Store
    "HybridVectorStore",
    "get_vector_store",
    "create_vector_store",
    "VectorDocument",
    "VectorSearchResult",
    
    # Document Processor (legacy - moved to triple_processor)
    # "DocumentProcessor",
    # "get_document_processor",
    # "create_document_processor",
    # "ProcessingResult",
    # "DocumentChunk",
]

def get_version_info():
    """Información de versión y configuración actual"""
    config = get_config()
    return {
        "version": __version__,
        "environment": config.environment.value,
        "vector_store": config.vector_store.store_type.value,
        "graph_store": config.graph_store.store_type.value,
        "embedding_type": config.embedding.embedding_type.value,
        "stack_info": "TIER 2 Development Stack" if config.environment == EnvironmentType.DEVELOPMENT else "TIER 1 Production Stack"
    }

def quick_start_guide():
    """Guía rápida de inicio"""
    return """
    RAG Preprocessing Quick Start:
    
    1. Configuración básica:
       from rag_preprocessing import get_config, get_document_processor
       
    2. Procesar un documento:
       processor = get_document_processor()
       result = processor.process_text("Tu texto aquí")
       
    3. Cambiar entre development/production:
       from rag_preprocessing import switch_environment, EnvironmentType
       switch_environment(EnvironmentType.PRODUCTION)
       
    4. Ver métricas:
       metrics = processor.get_processing_metrics()
       print(metrics)
       
    5. Configuración personalizada:
       from rag_preprocessing import RAGConfig, EnvironmentType
       config = RAGConfig(environment=EnvironmentType.PRODUCTION)
    """ 