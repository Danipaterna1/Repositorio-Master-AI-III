"""
SQLAlchemy models para sistema de metadata RAG
Tracking completo de procesamiento: documentos, entidades, relaciones, comunidades, métricas, API requests
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Document(Base):
    """Tracking de documentos procesados"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    content_hash = Column(String(64), unique=True, index=True)  # SHA256 del contenido
    original_text = Column(Text, nullable=False)
    content_length = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    processing_version = Column(String(20), default="1.1.0")
    
    # Status tracking
    vector_processed = Column(Boolean, default=False)
    graph_processed = Column(Boolean, default=False)
    metadata_processed = Column(Boolean, default=False)
    
    # Relaciones
    entities = relationship("Entity", back_populates="document")
    relationships = relationship("Relationship", back_populates="document")
    communities = relationship("Community", back_populates="document")
    metrics = relationship("ProcessingMetric", back_populates="document")

class Entity(Base):
    """Catálogo de entidades extraídas"""
    __tablename__ = "entities"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Datos de la entidad
    text = Column(String(500), nullable=False, index=True)
    label = Column(String(50), nullable=False, index=True)  # PERSON, ORG, LOC, etc.
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    confidence = Column(Float, default=1.0)
    
    # Metadata
    extraction_method = Column(String(20), default="spacy")  # spacy, llm, hybrid
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    document = relationship("Document", back_populates="entities")

class Relationship(Base):
    """Mapeo de relaciones detectadas"""
    __tablename__ = "relationships"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Entidades relacionadas
    source_entity_id = Column(Integer, ForeignKey("entities.id"), nullable=False)
    target_entity_id = Column(Integer, ForeignKey("entities.id"), nullable=False)
    
    # Datos de la relación
    relation_type = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, default=1.0)
    description = Column(Text)
    
    # Metadata
    extraction_method = Column(String(20), default="spacy")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    document = relationship("Document", back_populates="relationships")
    source_entity = relationship("Entity", foreign_keys=[source_entity_id])
    target_entity = relationship("Entity", foreign_keys=[target_entity_id])

class Community(Base):
    """Estructura de comunidades detectadas"""
    __tablename__ = "communities"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Datos de la comunidad
    community_id = Column(Integer, nullable=False)  # ID interno del algoritmo
    level = Column(Integer, default=0)  # Nivel jerárquico
    algorithm = Column(String(20), default="leiden")  # leiden, louvain
    
    # Entidades en la comunidad
    entity_ids = Column(JSON)  # Lista de IDs de entidades
    entity_count = Column(Integer, default=0)
    
    # Características
    title = Column(String(200))
    summary = Column(Text)
    summary_method = Column(String(20), default="statistical")  # statistical, llm
    importance_score = Column(Float, default=0.0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    document = relationship("Document", back_populates="communities")

class ProcessingMetric(Base):
    """Stats por documento procesado"""
    __tablename__ = "processing_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Tiempos de procesamiento
    vector_processing_time = Column(Float)  # segundos
    graph_processing_time = Column(Float)   # segundos
    total_processing_time = Column(Float)   # segundos
    
    # Conteos
    entities_count = Column(Integer, default=0)
    relationships_count = Column(Integer, default=0)
    communities_count = Column(Integer, default=0)
    
    # Calidad del grafo
    graph_density = Column(Float)
    connected_components = Column(Integer)
    avg_clustering_coefficient = Column(Float)
    
    # Configuración usada
    spacy_model = Column(String(50))
    community_algorithm = Column(String(20))
    llm_enhanced = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    document = relationship("Document", back_populates="metrics")

class APIRequest(Base):
    """Log de requests A2A"""
    __tablename__ = "api_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Request data
    endpoint = Column(String(100), nullable=False, index=True)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    
    # Timing
    request_time = Column(DateTime, default=datetime.utcnow, index=True)
    processing_time = Column(Float)  # segundos
    
    # Client info
    client_ip = Column(String(45))
    user_agent = Column(Text)
    api_key_hash = Column(String(64))  # Hash de la API key
    
    # Request/Response details
    request_size = Column(Integer)   # bytes
    response_size = Column(Integer)  # bytes
    error_message = Column(Text)
    
    # Metadata
    document_id = Column(Integer, ForeignKey("documents.id"))  # Si aplica 