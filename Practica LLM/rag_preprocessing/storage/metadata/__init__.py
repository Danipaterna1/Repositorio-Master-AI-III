"""
Metadata storage module para sistema RAG
Sistema completo de tracking: documentos, entidades, relaciones, comunidades, métricas, API requests
"""

from .models import (
    Base,
    Document,
    Entity, 
    Relationship,
    Community,
    ProcessingMetric,
    APIRequest
)

from .schemas import (
    # Base schemas
    DocumentCreate, DocumentResponse, DocumentWithDetails,
    EntityCreate, EntityResponse,
    RelationshipCreate, RelationshipResponse,
    CommunityCreate, CommunityResponse,
    ProcessingMetricCreate, ProcessingMetricResponse,
    APIRequestCreate, APIRequestResponse,
    
    # Composed schemas
    ProcessingStats,
    SystemHealth,
    
    # API schemas
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    QueryRequest,
    QueryResponse
)

from .sqlite_manager import SQLiteManager

__all__ = [
    # Models
    "Base", "Document", "Entity", "Relationship", "Community", 
    "ProcessingMetric", "APIRequest",
    
    # Schemas
    "DocumentCreate", "DocumentResponse", "DocumentWithDetails",
    "EntityCreate", "EntityResponse",
    "RelationshipCreate", "RelationshipResponse", 
    "CommunityCreate", "CommunityResponse",
    "ProcessingMetricCreate", "ProcessingMetricResponse",
    "APIRequestCreate", "APIRequestResponse",
    "ProcessingStats", "SystemHealth",
    "ProcessDocumentRequest", "ProcessDocumentResponse",
    "QueryRequest", "QueryResponse",
    
    # Manager
    "SQLiteManager"
] 