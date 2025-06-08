"""
Pydantic schemas para sistema de metadata RAG
Request/Response models para API y validación de datos
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# ========== SCHEMAS BASE ==========

class DocumentBase(BaseModel):
    """Schema base para documento"""
    original_text: str = Field(..., min_length=1, max_length=1000000)
    processing_version: str = Field(default="1.1.0")

class DocumentCreate(DocumentBase):
    """Schema para crear documento"""
    pass

class DocumentResponse(DocumentBase):
    """Schema para response de documento"""
    id: int
    content_hash: str
    content_length: int
    created_at: datetime
    processed_at: Optional[datetime] = None
    
    # Status flags
    vector_processed: bool = False
    graph_processed: bool = False
    metadata_processed: bool = False
    
    # Conteos relacionados
    entities_count: Optional[int] = 0
    relationships_count: Optional[int] = 0
    communities_count: Optional[int] = 0
    
    class Config:
        from_attributes = True

class EntityBase(BaseModel):
    """Schema base para entidad"""
    text: str = Field(..., min_length=1, max_length=500)
    label: str = Field(..., min_length=1, max_length=50)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extraction_method: str = Field(default="spacy")

class EntityCreate(EntityBase):
    """Schema para crear entidad"""
    document_id: int

class EntityResponse(EntityBase):
    """Schema para response de entidad"""
    id: int
    document_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class RelationshipBase(BaseModel):
    """Schema base para relación"""
    relation_type: str = Field(..., min_length=1, max_length=100)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    description: Optional[str] = None
    extraction_method: str = Field(default="spacy")

class RelationshipCreate(RelationshipBase):
    """Schema para crear relación"""
    document_id: int
    source_entity_id: int
    target_entity_id: int

class RelationshipResponse(RelationshipBase):
    """Schema para response de relación"""
    id: int
    document_id: int
    source_entity_id: int
    target_entity_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class CommunityBase(BaseModel):
    """Schema base para comunidad"""
    community_id: int
    level: int = Field(default=0, ge=0)
    algorithm: str = Field(default="leiden")
    entity_ids: List[int] = Field(default=[])
    entity_count: int = Field(default=0, ge=0)
    title: Optional[str] = Field(None, max_length=200)
    summary: Optional[str] = None
    summary_method: str = Field(default="statistical")
    importance_score: float = Field(default=0.0, ge=0.0, le=1.0)

class CommunityCreate(CommunityBase):
    """Schema para crear comunidad"""
    document_id: int

class CommunityResponse(CommunityBase):
    """Schema para response de comunidad"""
    id: int
    document_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class ProcessingMetricBase(BaseModel):
    """Schema base para métricas"""
    vector_processing_time: Optional[float] = Field(None, ge=0.0)
    graph_processing_time: Optional[float] = Field(None, ge=0.0)
    total_processing_time: Optional[float] = Field(None, ge=0.0)
    
    entities_count: int = Field(default=0, ge=0)
    relationships_count: int = Field(default=0, ge=0)
    communities_count: int = Field(default=0, ge=0)
    
    graph_density: Optional[float] = Field(None, ge=0.0, le=1.0)
    connected_components: Optional[int] = Field(None, ge=0)
    avg_clustering_coefficient: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    spacy_model: Optional[str] = Field(None, max_length=50)
    community_algorithm: Optional[str] = Field(None, max_length=20)
    llm_enhanced: bool = Field(default=False)

class ProcessingMetricCreate(ProcessingMetricBase):
    """Schema para crear métrica"""
    document_id: int

class ProcessingMetricResponse(ProcessingMetricBase):
    """Schema para response de métrica"""
    id: int
    document_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class APIRequestBase(BaseModel):
    """Schema base para request API"""
    endpoint: str = Field(..., min_length=1, max_length=100)
    method: str = Field(..., min_length=1, max_length=10)
    status_code: int = Field(..., ge=100, le=599)
    processing_time: Optional[float] = Field(None, ge=0.0)
    client_ip: Optional[str] = Field(None, max_length=45)
    user_agent: Optional[str] = None
    api_key_hash: Optional[str] = Field(None, max_length=64)
    request_size: Optional[int] = Field(None, ge=0)
    response_size: Optional[int] = Field(None, ge=0)
    error_message: Optional[str] = None
    document_id: Optional[int] = None

class APIRequestCreate(APIRequestBase):
    """Schema para crear request API"""
    pass

class APIRequestResponse(APIRequestBase):
    """Schema para response de request API"""
    id: int
    request_time: datetime
    
    class Config:
        from_attributes = True

# ========== SCHEMAS COMPUESTOS ==========

class DocumentWithDetails(DocumentResponse):
    """Documento con entidades, relaciones y comunidades"""
    entities: List[EntityResponse] = []
    relationships: List[RelationshipResponse] = []
    communities: List[CommunityResponse] = []
    metrics: Optional[ProcessingMetricResponse] = None

class ProcessingStats(BaseModel):
    """Estadísticas generales del sistema"""
    total_documents: int
    total_entities: int
    total_relationships: int
    total_communities: int
    avg_processing_time: float
    avg_entities_per_doc: float
    avg_relationships_per_doc: float
    avg_communities_per_doc: float
    
class SystemHealth(BaseModel):
    """Health check del sistema"""
    status: str = Field(..., description="ok, warning, error")
    database_connected: bool
    vector_db_connected: bool
    graph_db_connected: bool
    total_documents: int
    last_processed: Optional[datetime] = None
    uptime_seconds: float
    memory_usage_mb: float

# ========== SCHEMAS API ==========

class ProcessDocumentRequest(BaseModel):
    """Request para procesar documento"""
    text: str = Field(..., min_length=1, max_length=1000000)
    enable_vector: bool = Field(default=True)
    enable_graph: bool = Field(default=True)
    enable_metadata: bool = Field(default=True)
    enable_llm: bool = Field(default=False)

class ProcessDocumentResponse(BaseModel):
    """Response para procesar documento"""
    document_id: int
    processing_time: float
    vector_processed: bool
    graph_processed: bool
    metadata_processed: bool
    
    # Resultados
    entities_count: int
    relationships_count: int
    communities_count: int
    
    # IDs para retrieval
    vector_ids: List[str] = []  # IDs en ChromaDB
    graph_file: Optional[str] = None  # Archivo HTML generado
    
class QueryRequest(BaseModel):
    """Request para query híbrido"""
    query: str = Field(..., min_length=1, max_length=1000)
    search_type: str = Field(default="hybrid", pattern="^(vector|graph|hybrid)$")
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class QueryResponse(BaseModel):
    """Response para query híbrido"""
    query: str
    search_type: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time: float 