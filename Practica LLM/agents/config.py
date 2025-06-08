"""
Kingfisher Configuration System

Sistema de configuración centralizado para el agente A2A Kingfisher.
Maneja configuración de producción, desarrollo y testing.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Configuración de bases de datos"""
    # ChromaDB
    chroma_persist_directory: str = "./data/chromadb"
    chroma_collection_name: str = "kingfisher_documents"
    
    # SQLite
    sqlite_db_path: str = "./data/metadata/kingfisher.db"
    sqlite_pool_size: int = 10
    
    # Graph Database
    graph_db_path: str = "./data/graphs"
    
@dataclass
class ServerConfig:
    """Configuración del servidor A2A"""
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    log_level: str = "info"
    
    # CORS
    cors_origins: list = field(default_factory=lambda: ["*"])
    cors_methods: list = field(default_factory=lambda: ["*"])
    cors_headers: list = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 60
    max_request_size_mb: int = 10

@dataclass
class ProcessingConfig:
    """Configuración de procesamiento"""
    # LLM
    llm_provider: str = "google"  # google, openai, local
    llm_model: str = "gemini-1.5-flash"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 8192
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Processing
    max_concurrent_tasks: int = 5
    task_timeout_seconds: int = 300
    cleanup_interval_hours: int = 24

@dataclass
class LoggingConfig:
    """Configuración de logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "./logs/kingfisher.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    
    # Structured logging
    json_logs: bool = False
    include_trace_id: bool = True

@dataclass
class SecurityConfig:
    """Configuración de seguridad"""
    api_key_required: bool = False
    api_key_header: str = "X-API-Key"
    allowed_api_keys: list = field(default_factory=list)
    
    # Task isolation
    sandbox_mode: bool = False
    max_memory_mb: int = 1024
    max_cpu_percent: int = 80

@dataclass
class KingfisherConfig:
    """Configuración principal de Kingfisher"""
    environment: str = "development"  # development, staging, production
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Agent metadata
    agent_name: str = "Kingfisher RAG Agent"
    agent_version: str = "1.0.0"
    agent_description: str = "Advanced RAG system with triple storage (vector, graph, relational)"
    
    def __post_init__(self):
        """Validación y ajustes post-inicialización"""
        self._validate_config()
        self._setup_directories()
        
    def _validate_config(self):
        """Valida la configuración"""
        if self.environment not in ["development", "staging", "production"]:
            raise ValueError(f"Invalid environment: {self.environment}")
            
        if self.server.port < 1024 or self.server.port > 65535:
            raise ValueError(f"Invalid port: {self.server.port}")
            
    def _setup_directories(self):
        """Crea directorios necesarios"""
        directories = [
            Path(self.database.chroma_persist_directory).parent,
            Path(self.database.sqlite_db_path).parent,
            Path(self.database.graph_db_path),
        ]
        
        if self.logging.file_path:
            directories.append(Path(self.logging.file_path).parent)
            
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'KingfisherConfig':
        """Carga configuración desde variables de entorno"""
        config = cls()
        
        # Environment
        config.environment = os.getenv("KINGFISHER_ENV", config.environment)
        
        # Server
        config.server.host = os.getenv("KINGFISHER_HOST", config.server.host)
        config.server.port = int(os.getenv("KINGFISHER_PORT", config.server.port))
        config.server.log_level = os.getenv("KINGFISHER_LOG_LEVEL", config.server.log_level)
        
        # Database
        config.database.chroma_persist_directory = os.getenv(
            "KINGFISHER_CHROMA_DIR", config.database.chroma_persist_directory
        )
        config.database.sqlite_db_path = os.getenv(
            "KINGFISHER_SQLITE_PATH", config.database.sqlite_db_path
        )
        
        # LLM
        config.processing.llm_provider = os.getenv("KINGFISHER_LLM_PROVIDER", config.processing.llm_provider)
        config.processing.llm_model = os.getenv("KINGFISHER_LLM_MODEL", config.processing.llm_model)
        
        # Security
        config.security.api_key_required = os.getenv("KINGFISHER_API_KEY_REQUIRED", "false").lower() == "true"
        api_keys = os.getenv("KINGFISHER_API_KEYS", "")
        if api_keys:
            config.security.allowed_api_keys = [key.strip() for key in api_keys.split(",")]
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a diccionario"""
        return {
            "environment": self.environment,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "database": {
                "chroma_persist_directory": self.database.chroma_persist_directory,
                "sqlite_db_path": self.database.sqlite_db_path,
                "graph_db_path": self.database.graph_db_path,
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "log_level": self.server.log_level,
            },
            "processing": {
                "llm_provider": self.processing.llm_provider,
                "llm_model": self.processing.llm_model,
                "max_concurrent_tasks": self.processing.max_concurrent_tasks,
            }
        }

# Global configuration instance
_config: Optional[KingfisherConfig] = None

def get_config() -> KingfisherConfig:
    """Obtiene la configuración global"""
    global _config
    if _config is None:
        _config = KingfisherConfig.from_env()
    return _config

def set_config(config: KingfisherConfig):
    """Establece la configuración global"""
    global _config
    _config = config

def reset_config():
    """Resetea la configuración global"""
    global _config
    _config = None

# Alias for backwards compatibility
Config = KingfisherConfig 