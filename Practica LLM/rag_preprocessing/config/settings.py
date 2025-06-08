"""
Sistema de configuración para RAG con stack dual desarrollo/producción.
Permite alternar entre TIER 1 (producción) y TIER 2 (desarrollo) mediante flags.
"""

import os
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class EnvironmentType(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"

class VectorStoreType(Enum):
    CHROMADB = "chromadb"      # TIER 2 - Desarrollo rápido
    QDRANT = "qdrant"          # TIER 1 - Producción enterprise

class GraphStoreType(Enum):
    NETWORKX = "networkx"      # TIER 2 - Desarrollo rápido  
    NEO4J = "neo4j"           # TIER 1 - Producción enterprise

class EmbeddingType(Enum):
    STATIC_FAST = "static"     # Static embeddings para velocidad
    TRADITIONAL = "traditional" # Modelo tradicional para calidad
    HYBRID = "hybrid"          # Ambos con fallback automático

@dataclass
class EmbeddingConfig:
    """Configuración para embeddings modernos 2025"""
    primary_model: str = "all-mpnet-base-v2"  # Modelo unificado para compatibilidad 768D
    fallback_model: str = "all-mpnet-base-v2"  # Traditional embeddings  
    embedding_type: EmbeddingType = EmbeddingType.HYBRID
    truncate_dim: int = 512  # Matryoshka support
    batch_size: int = 32
    use_cpu: bool = True  # Optimizado para CPU

@dataclass
class VectorStoreConfig:
    """Configuración para vector database"""
    store_type: VectorStoreType = VectorStoreType.CHROMADB
    
    # ChromaDB (Development)
    chromadb_path: str = "./data/chromadb"
    chromadb_collection: str = "rag_chunks"
    
    # Qdrant (Production)
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = "rag_vectors"
    
    # Configuración común
    similarity_threshold: float = 0.8
    max_results: int = 10

@dataclass
class GraphStoreConfig:
    """Configuración para graph database"""
    store_type: GraphStoreType = GraphStoreType.NETWORKX
    
    # NetworkX (Development)
    networkx_cache_path: str = "./data/graphs"
    
    # Neo4j (Production)
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password123")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Configuración común
    max_graph_depth: int = 2
    min_entity_confidence: float = 0.7

@dataclass
class ProcessingConfig:
    """Configuración para procesamiento de documentos"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    
    # NER Configuration
    spacy_model: str = "es_core_news_lg"  # Modelo grande para mejor precisión
    use_llm_fallback: bool = True
    llm_model: str = "gemini-1.5-flash"

class RAGConfig:
    """Configuración principal del sistema RAG"""
    
    def __init__(self, environment: EnvironmentType = None):
        # Auto-detectar environment si no se especifica
        if environment is None:
            env_var = os.getenv("RAG_ENVIRONMENT", "development").lower()
            environment = EnvironmentType(env_var)
        
        self.environment = environment
        self.embedding = EmbeddingConfig()
        self.processing = ProcessingConfig()
        
        # Configurar stack según el environment
        if environment == EnvironmentType.DEVELOPMENT:
            self._setup_development_stack()
        else:
            self._setup_production_stack()
    
    def _setup_development_stack(self):
        """Configura TIER 2: Stack de desarrollo rápido"""
        self.vector_store = VectorStoreConfig(
            store_type=VectorStoreType.CHROMADB
        )
        self.graph_store = GraphStoreConfig(
            store_type=GraphStoreType.NETWORKX
        )
        # Embeddings híbridos por defecto
        self.embedding.embedding_type = EmbeddingType.HYBRID
        
    def _setup_production_stack(self):
        """Configura TIER 1: Stack de producción enterprise"""
        self.vector_store = VectorStoreConfig(
            store_type=VectorStoreType.QDRANT
        )
        self.graph_store = GraphStoreConfig(
            store_type=GraphStoreType.NEO4J
        )
        # Solo static embeddings para máxima velocidad en producción
        self.embedding.embedding_type = EmbeddingType.STATIC_FAST
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Retorna resumen de la configuración actual"""
        return {
            "environment": self.environment.value,
            "vector_store": self.vector_store.store_type.value,
            "graph_store": self.graph_store.store_type.value,
            "embedding_type": self.embedding.embedding_type.value,
            "primary_model": self.embedding.primary_model,
            "fallback_model": self.embedding.fallback_model,
        }
    
    def switch_to_production(self):
        """Cambia dinámicamente a stack de producción"""
        self.environment = EnvironmentType.PRODUCTION
        self._setup_production_stack()
    
    def switch_to_development(self):
        """Cambia dinámicamente a stack de desarrollo"""
        self.environment = EnvironmentType.DEVELOPMENT
        self._setup_development_stack()

# Instancia global de configuración
config = RAGConfig()

def get_config() -> RAGConfig:
    """Obtiene la instancia global de configuración"""
    return config

def switch_environment(env_type: EnvironmentType):
    """Cambia el environment globalmente"""
    global config
    config.environment = env_type
    if env_type == EnvironmentType.DEVELOPMENT:
        config._setup_development_stack()
    else:
        config._setup_production_stack()

def create_data_directories():
    """Crea los directorios necesarios para almacenamiento"""
    import os
    
    directories = [
        "./data",
        "./data/chromadb", 
        "./data/graphs",
        "./data/embeddings",
        "./data/metadata"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True) 