"""
SQLiteManager para sistema de metadata RAG
CRUD operations completas para tracking de documentos, entidades, relaciones, comunidades, métricas y API requests
"""

import hashlib
import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import create_engine, and_, func, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

from .models import Base, Document, Entity, Relationship, Community, ProcessingMetric, APIRequest
from .schemas import (
    DocumentCreate, EntityCreate, RelationshipCreate, CommunityCreate,
    ProcessingMetricCreate, APIRequestCreate, ProcessingStats, SystemHealth
)

class SQLiteManager:
    """Manager para base de datos SQLite de metadata RAG"""
    
    def __init__(self, database_url: str = "sqlite:///data/metadata/rag_metadata.db"):
        """
        Inicializar manager SQLite
        
        Args:
            database_url: URL de conexión SQLite
        """
        # Crear directorio si no existe
        db_path = database_url.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Crear tablas
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Obtener sesión de base de datos"""
        return self.SessionLocal()
    
    def _generate_content_hash(self, text: str) -> str:
        """Generar hash SHA256 del contenido"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    # ========== CRUD DOCUMENTS ==========
    
    def create_document(self, document: DocumentCreate) -> Document:
        """Crear nuevo documento"""
        with self.get_session() as db:
            content_hash = self._generate_content_hash(document.original_text)
            
            # Verificar si ya existe
            existing = db.query(Document).filter(Document.content_hash == content_hash).first()
            if existing:
                return existing
            
            db_document = Document(
                content_hash=content_hash,
                original_text=document.original_text,
                content_length=len(document.original_text),
                processing_version=document.processing_version
            )
            
            try:
                db.add(db_document)
                db.commit()
                db.refresh(db_document)
                return db_document
            except IntegrityError:
                db.rollback()
                # Si hay conflict, devolver el existente
                return db.query(Document).filter(Document.content_hash == content_hash).first()
    
    def get_document(self, document_id: int) -> Optional[Document]:
        """Obtener documento por ID"""
        with self.get_session() as db:
            return db.query(Document).filter(Document.id == document_id).first()
    
    def get_document_by_hash(self, content_hash: str) -> Optional[Document]:
        """Obtener documento por hash"""
        with self.get_session() as db:
            return db.query(Document).filter(Document.content_hash == content_hash).first()
    
    def update_document_status(self, document_id: int, **status_updates) -> bool:
        """Actualizar status de procesamiento del documento"""
        with self.get_session() as db:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return False
            
            # Actualizar campos permitidos
            allowed_fields = ['vector_processed', 'graph_processed', 'metadata_processed', 'processed_at']
            for field, value in status_updates.items():
                if field in allowed_fields:
                    setattr(document, field, value)
            
            if status_updates:
                document.processed_at = datetime.utcnow()
            
            db.commit()
            return True
    
    def list_documents(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """Listar documentos con paginación"""
        with self.get_session() as db:
            return db.query(Document).offset(skip).limit(limit).all()
    
    # ========== CRUD ENTITIES ==========
    
    def create_entities(self, entities: List[EntityCreate]) -> List[Entity]:
        """Crear múltiples entidades en batch"""
        if not entities:
            return []
        
        with self.get_session() as db:
            db_entities = []
            for entity in entities:
                db_entity = Entity(**entity.dict())
                db_entities.append(db_entity)
            
            db.add_all(db_entities)
            db.commit()
            
            # Refresh para obtener IDs
            for entity in db_entities:
                db.refresh(entity)
            
            return db_entities
    
    def get_entities_by_document(self, document_id: int) -> List[Entity]:
        """Obtener todas las entidades de un documento"""
        with self.get_session() as db:
            return db.query(Entity).filter(Entity.document_id == document_id).all()
    
    def get_entities_by_label(self, label: str, limit: int = 50) -> List[Entity]:
        """Obtener entidades por tipo/label"""
        with self.get_session() as db:
            return db.query(Entity).filter(Entity.label == label).limit(limit).all()
    
    # ========== CRUD RELATIONSHIPS ==========
    
    def create_relationships(self, relationships: List[RelationshipCreate]) -> List[Relationship]:
        """Crear múltiples relaciones en batch"""
        if not relationships:
            return []
        
        with self.get_session() as db:
            db_relationships = []
            for rel in relationships:
                db_rel = Relationship(**rel.dict())
                db_relationships.append(db_rel)
            
            db.add_all(db_relationships)
            db.commit()
            
            for rel in db_relationships:
                db.refresh(rel)
            
            return db_relationships
    
    def get_relationships_by_document(self, document_id: int) -> List[Relationship]:
        """Obtener todas las relaciones de un documento"""
        with self.get_session() as db:
            return db.query(Relationship).filter(Relationship.document_id == document_id).all()
    
    def get_entity_relationships(self, entity_id: int) -> List[Relationship]:
        """Obtener relaciones donde participa una entidad"""
        with self.get_session() as db:
            return db.query(Relationship).filter(
                and_(
                    Relationship.source_entity_id == entity_id,
                    Relationship.target_entity_id == entity_id
                )
            ).all()
    
    # ========== CRUD COMMUNITIES ==========
    
    def create_communities(self, communities: List[CommunityCreate]) -> List[Community]:
        """Crear múltiples comunidades en batch"""
        if not communities:
            return []
        
        with self.get_session() as db:
            db_communities = []
            for community in communities:
                db_community = Community(**community.dict())
                db_communities.append(db_community)
            
            db.add_all(db_communities)
            db.commit()
            
            for community in db_communities:
                db.refresh(community)
            
            return db_communities
    
    def get_communities_by_document(self, document_id: int) -> List[Community]:
        """Obtener todas las comunidades de un documento"""
        with self.get_session() as db:
            return db.query(Community).filter(Community.document_id == document_id).all()
    
    def get_top_communities(self, limit: int = 10) -> List[Community]:
        """Obtener top comunidades por importance score"""
        with self.get_session() as db:
            return db.query(Community).order_by(desc(Community.importance_score)).limit(limit).all()
    
    # ========== CRUD PROCESSING METRICS ==========
    
    def create_processing_metric(self, metric: ProcessingMetricCreate) -> ProcessingMetric:
        """Crear métrica de procesamiento"""
        with self.get_session() as db:
            db_metric = ProcessingMetric(**metric.dict())
            db.add(db_metric)
            db.commit()
            db.refresh(db_metric)
            return db_metric
    
    def get_metric_by_document(self, document_id: int) -> Optional[ProcessingMetric]:
        """Obtener métrica de un documento"""
        with self.get_session() as db:
            return db.query(ProcessingMetric).filter(ProcessingMetric.document_id == document_id).first()
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Calcular métricas promedio del sistema"""
        with self.get_session() as db:
            result = db.query(
                func.avg(ProcessingMetric.total_processing_time).label('avg_processing_time'),
                func.avg(ProcessingMetric.entities_count).label('avg_entities'),
                func.avg(ProcessingMetric.relationships_count).label('avg_relationships'),
                func.avg(ProcessingMetric.communities_count).label('avg_communities'),
                func.avg(ProcessingMetric.graph_density).label('avg_density')
            ).first()
            
            return {
                'avg_processing_time': result.avg_processing_time or 0.0,
                'avg_entities_per_doc': result.avg_entities or 0.0,
                'avg_relationships_per_doc': result.avg_relationships or 0.0,
                'avg_communities_per_doc': result.avg_communities or 0.0,
                'avg_graph_density': result.avg_density or 0.0
            }
    
    # ========== CRUD API REQUESTS ==========
    
    def create_api_request(self, request: APIRequestCreate) -> APIRequest:
        """Registrar request API"""
        with self.get_session() as db:
            db_request = APIRequest(**request.dict())
            db.add(db_request)
            db.commit()
            db.refresh(db_request)
            return db_request
    
    def get_api_requests(self, limit: int = 100, endpoint: Optional[str] = None) -> List[APIRequest]:
        """Obtener requests API con filtros"""
        with self.get_session() as db:
            query = db.query(APIRequest)
            
            if endpoint:
                query = query.filter(APIRequest.endpoint == endpoint)
            
            return query.order_by(desc(APIRequest.request_time)).limit(limit).all()
    
    def get_api_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Obtener estadísticas de API de las últimas N horas"""
        from datetime import timedelta
        
        with self.get_session() as db:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            stats = db.query(
                func.count(APIRequest.id).label('total_requests'),
                func.avg(APIRequest.processing_time).label('avg_processing_time'),
                func.count(func.nullif(APIRequest.status_code >= 400, False)).label('error_count')
            ).filter(APIRequest.request_time >= since).first()
            
            return {
                'total_requests': stats.total_requests or 0,
                'avg_processing_time': stats.avg_processing_time or 0.0,
                'error_count': stats.error_count or 0,
                'error_rate': (stats.error_count or 0) / max(stats.total_requests or 1, 1)
            }
    
    # ========== OPERATIONS COMPLEJAS ==========
    
    def get_document_with_details(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Obtener documento completo con entidades, relaciones y comunidades"""
        with self.get_session() as db:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return None
            
            entities = self.get_entities_by_document(document_id)
            relationships = self.get_relationships_by_document(document_id)
            communities = self.get_communities_by_document(document_id)
            metrics = self.get_metric_by_document(document_id)
            
            return {
                'document': document,
                'entities': entities,
                'relationships': relationships,
                'communities': communities,
                'metrics': metrics
            }
    
    def get_processing_stats(self) -> ProcessingStats:
        """Obtener estadísticas generales del sistema"""
        with self.get_session() as db:
            # Conteos totales
            total_docs = db.query(func.count(Document.id)).scalar() or 0
            total_entities = db.query(func.count(Entity.id)).scalar() or 0
            total_relationships = db.query(func.count(Relationship.id)).scalar() or 0
            total_communities = db.query(func.count(Community.id)).scalar() or 0
            
            # Promedios
            avg_metrics = self.get_average_metrics()
            
            return ProcessingStats(
                total_documents=total_docs,
                total_entities=total_entities,
                total_relationships=total_relationships,
                total_communities=total_communities,
                avg_processing_time=avg_metrics['avg_processing_time'],
                avg_entities_per_doc=avg_metrics['avg_entities_per_doc'],
                avg_relationships_per_doc=avg_metrics['avg_relationships_per_doc'],
                avg_communities_per_doc=avg_metrics['avg_communities_per_doc']
            )
    
    def get_system_health(self) -> SystemHealth:
        """Health check del sistema"""
        try:
            with self.get_session() as db:
                # Test database connection
                total_docs = db.query(func.count(Document.id)).scalar() or 0
                
                # Último documento procesado
                last_doc = db.query(Document).filter(
                    Document.processed_at.isnot(None)
                ).order_by(desc(Document.processed_at)).first()
                
                return SystemHealth(
                    status="ok",
                    database_connected=True,
                    vector_db_connected=True,  # TODO: Check real ChromaDB connection
                    graph_db_connected=True,   # TODO: Check real NetworkX status
                    total_documents=total_docs,
                    last_processed=last_doc.processed_at if last_doc else None,
                    uptime_seconds=0.0,  # TODO: Calculate real uptime
                    memory_usage_mb=0.0  # TODO: Calculate real memory usage
                )
        
        except Exception as e:
            return SystemHealth(
                status="error",
                database_connected=False,
                vector_db_connected=False,
                graph_db_connected=False,
                total_documents=0,
                uptime_seconds=0.0,
                memory_usage_mb=0.0
            )
    
    def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """Limpiar datos antiguos del sistema"""
        from datetime import timedelta
        
        with self.get_session() as db:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Limpiar API requests antiguos
            deleted_requests = db.query(APIRequest).filter(
                APIRequest.request_time < cutoff_date
            ).delete()
            
            db.commit()
            
            return {
                'deleted_api_requests': deleted_requests
            } 