"""
Test completo para sistema de metadata SQLite
Verificación de CRUD operations, schemas, y functionality completa
"""

import os
import sys
sys.path.append('.')

from datetime import datetime, timedelta
from typing import List

from rag_preprocessing.storage.metadata import (
    SQLiteManager, 
    DocumentCreate, EntityCreate, RelationshipCreate, CommunityCreate,
    ProcessingMetricCreate, APIRequestCreate
)

def test_sqlite_manager_initialization():
    """Test inicialización del SQLiteManager"""
    print("🔧 Testeando inicialización SQLiteManager...")
    
    # Usar database temporal para testing
    test_db_url = "sqlite:///test_metadata.db"
    manager = SQLiteManager(test_db_url)
    
    assert manager is not None
    assert manager.engine is not None
    assert manager.SessionLocal is not None
    
    print("✅ SQLiteManager inicializado correctamente")
    return manager

def test_document_crud(manager: SQLiteManager):
    """Test CRUD operations para documentos"""
    print("📄 Testeando CRUD documentos...")
    
    # Crear documento
    doc_data = DocumentCreate(
        original_text="Este es un documento de prueba para el sistema RAG.",
        processing_version="1.1.0"
    )
    
    document = manager.create_document(doc_data)
    assert document is not None
    assert document.id is not None
    assert document.content_hash is not None
    assert document.content_length == len(doc_data.original_text)
    assert not document.vector_processed
    assert not document.graph_processed
    assert not document.metadata_processed
    
    doc_id = document.id
    print(f"   ✅ Documento creado ID: {doc_id}")
    
    # Obtener documento
    retrieved_doc = manager.get_document(doc_id)
    assert retrieved_doc is not None
    assert retrieved_doc.id == doc_id
    print(f"   ✅ Documento recuperado ID: {retrieved_doc.id}")
    
    # Actualizar status
    success = manager.update_document_status(
        doc_id,
        vector_processed=True,
        graph_processed=True,
        metadata_processed=True
    )
    assert success
    
    updated_doc = manager.get_document(doc_id)
    assert updated_doc.vector_processed
    assert updated_doc.graph_processed  
    assert updated_doc.metadata_processed
    assert updated_doc.processed_at is not None
    print(f"   ✅ Status actualizado correctamente")
    
    # Listar documentos
    docs = manager.list_documents(limit=10)
    assert len(docs) >= 1
    print(f"   ✅ Listado documentos: {len(docs)} encontrados")
    
    return doc_id

def test_entity_crud(manager: SQLiteManager, document_id: int):
    """Test CRUD operations para entidades"""
    print("🏷️  Testeando CRUD entidades...")
    
    # Crear entidades
    entities_data = [
        EntityCreate(
            document_id=document_id,
            text="Microsoft",
            label="ORG",
            start_char=0,
            end_char=9,
            confidence=0.95,
            extraction_method="spacy"
        ),
        EntityCreate(
            document_id=document_id,
            text="Dr. García",
            label="PERSON",
            start_char=20,
            end_char=30,
            confidence=0.90,
            extraction_method="spacy"
        )
    ]
    
    entities = manager.create_entities(entities_data)
    assert len(entities) == 2
    assert all(entity.id is not None for entity in entities)
    print(f"   ✅ {len(entities)} entidades creadas")
    
    # Obtener entidades por documento
    doc_entities = manager.get_entities_by_document(document_id)
    assert len(doc_entities) >= 2
    print(f"   ✅ Entidades por documento: {len(doc_entities)}")
    
    # Obtener entidades por label
    org_entities = manager.get_entities_by_label("ORG")
    assert len(org_entities) >= 1
    print(f"   ✅ Entidades ORG: {len(org_entities)}")
    
    return [entity.id for entity in entities]

def test_relationship_crud(manager: SQLiteManager, document_id: int, entity_ids: List[int]):
    """Test CRUD operations para relaciones"""
    print("🔗 Testeando CRUD relaciones...")
    
    if len(entity_ids) < 2:
        print("   ⚠️  Necesitamos al menos 2 entidades para crear relación")
        return []
    
    # Crear relaciones
    relationships_data = [
        RelationshipCreate(
            document_id=document_id,
            source_entity_id=entity_ids[0],
            target_entity_id=entity_ids[1],
            relation_type="works_for",
            confidence=0.85,
            description="Dr. García trabaja para Microsoft",
            extraction_method="spacy"
        )
    ]
    
    relationships = manager.create_relationships(relationships_data)
    assert len(relationships) == 1
    assert relationships[0].id is not None
    print(f"   ✅ {len(relationships)} relaciones creadas")
    
    # Obtener relaciones por documento
    doc_rels = manager.get_relationships_by_document(document_id)
    assert len(doc_rels) >= 1
    print(f"   ✅ Relaciones por documento: {len(doc_rels)}")
    
    return [rel.id for rel in relationships]

def test_community_crud(manager: SQLiteManager, document_id: int, entity_ids: List[int]):
    """Test CRUD operations para comunidades"""
    print("🕸️  Testeando CRUD comunidades...")
    
    # Crear comunidades
    communities_data = [
        CommunityCreate(
            document_id=document_id,
            community_id=0,
            level=0,
            algorithm="leiden",
            entity_ids=entity_ids[:2] if len(entity_ids) >= 2 else entity_ids,
            entity_count=min(2, len(entity_ids)),
            title="Comunidad Tecnológica",
            summary="Comunidad de entidades relacionadas con tecnología",
            summary_method="statistical",
            importance_score=0.75
        )
    ]
    
    communities = manager.create_communities(communities_data)
    assert len(communities) == 1
    assert communities[0].id is not None
    print(f"   ✅ {len(communities)} comunidades creadas")
    
    # Obtener comunidades por documento
    doc_communities = manager.get_communities_by_document(document_id)
    assert len(doc_communities) >= 1
    print(f"   ✅ Comunidades por documento: {len(doc_communities)}")
    
    # Obtener top comunidades
    top_communities = manager.get_top_communities(limit=5)
    assert len(top_communities) >= 1
    print(f"   ✅ Top comunidades: {len(top_communities)}")
    
    return [comm.id for comm in communities]

def test_processing_metrics_crud(manager: SQLiteManager, document_id: int):
    """Test CRUD operations para métricas de procesamiento"""
    print("📊 Testeando CRUD métricas...")
    
    # Crear métrica
    metric_data = ProcessingMetricCreate(
        document_id=document_id,
        vector_processing_time=2.65,
        graph_processing_time=1.94,
        total_processing_time=4.59,
        entities_count=2,
        relationships_count=1,
        communities_count=1,
        graph_density=0.041,
        connected_components=1,
        avg_clustering_coefficient=0.5,
        spacy_model="es_core_news_sm",
        community_algorithm="leiden",
        llm_enhanced=False
    )
    
    metric = manager.create_processing_metric(metric_data)
    assert metric is not None
    assert metric.id is not None
    print(f"   ✅ Métrica creada ID: {metric.id}")
    
    # Obtener métrica por documento
    doc_metric = manager.get_metric_by_document(document_id)
    assert doc_metric is not None
    assert doc_metric.document_id == document_id
    print(f"   ✅ Métrica recuperada para documento {document_id}")
    
    # Obtener métricas promedio
    avg_metrics = manager.get_average_metrics()
    assert 'avg_processing_time' in avg_metrics
    assert avg_metrics['avg_processing_time'] > 0
    print(f"   ✅ Métricas promedio: {avg_metrics['avg_processing_time']:.2f}s")
    
    return metric.id

def test_api_requests_crud(manager: SQLiteManager, document_id: int):
    """Test CRUD operations para requests API"""
    print("🔌 Testeando CRUD API requests...")
    
    # Crear request API
    request_data = APIRequestCreate(
        endpoint="/api/v1/process",
        method="POST",
        status_code=200,
        processing_time=4.59,
        client_ip="127.0.0.1",
        user_agent="test-client/1.0",
        api_key_hash="test_hash_123",
        request_size=1024,
        response_size=2048,
        document_id=document_id
    )
    
    api_request = manager.create_api_request(request_data)
    assert api_request is not None
    assert api_request.id is not None
    print(f"   ✅ API request creado ID: {api_request.id}")
    
    # Obtener requests API
    requests = manager.get_api_requests(limit=10)
    assert len(requests) >= 1
    print(f"   ✅ API requests: {len(requests)} encontrados")
    
    # Obtener stats API
    stats = manager.get_api_stats(hours=24)
    assert 'total_requests' in stats
    assert stats['total_requests'] >= 1
    print(f"   ✅ API stats: {stats['total_requests']} requests en 24h")
    
    return api_request.id

def test_complex_operations(manager: SQLiteManager, document_id: int):
    """Test operaciones complejas del sistema"""
    print("🔬 Testeando operaciones complejas...")
    
    # Obtener documento con detalles completos
    doc_details = manager.get_document_with_details(document_id)
    assert doc_details is not None
    assert 'document' in doc_details
    assert 'entities' in doc_details
    assert 'relationships' in doc_details
    assert 'communities' in doc_details
    assert 'metrics' in doc_details
    print(f"   ✅ Documento completo: {len(doc_details['entities'])} entidades, {len(doc_details['relationships'])} relaciones")
    
    # Obtener estadísticas del sistema
    stats = manager.get_processing_stats()
    assert stats.total_documents >= 1
    assert stats.total_entities >= 1
    print(f"   ✅ Stats sistema: {stats.total_documents} docs, {stats.total_entities} entidades")
    
    # Health check
    health = manager.get_system_health()
    assert health.status == "ok"
    assert health.database_connected
    print(f"   ✅ Health check: {health.status}, {health.total_documents} docs")
    
    return doc_details

def run_complete_test():
    """Ejecutar test completo del sistema metadata"""
    print("🚀 INICIANDO TEST COMPLETO SISTEMA METADATA")
    print("=" * 60)
    
    try:
        # 1. Inicialización
        manager = test_sqlite_manager_initialization()
        
        # 2. Test CRUD completo
        document_id = test_document_crud(manager)
        entity_ids = test_entity_crud(manager, document_id)
        relationship_ids = test_relationship_crud(manager, document_id, entity_ids)
        community_ids = test_community_crud(manager, document_id, entity_ids)
        metric_id = test_processing_metrics_crud(manager, document_id)
        api_request_id = test_api_requests_crud(manager, document_id)
        
        # 3. Operaciones complejas
        doc_details = test_complex_operations(manager, document_id)
        
        print("=" * 60)
        print("🎉 TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("✅ Sistema de metadata SQLite completamente funcional")
        print(f"   📄 Documento ID: {document_id}")
        print(f"   🏷️  Entidades: {len(entity_ids)}")
        print(f"   🔗 Relaciones: {len(relationship_ids)}")
        print(f"   🕸️  Comunidades: {len(community_ids)}")
        print(f"   📊 Métricas: {metric_id}")
        print(f"   🔌 API Requests: {api_request_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR EN TEST: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup test database
        if os.path.exists("test_metadata.db"):
            os.remove("test_metadata.db")
            print("🧹 Test database limpiada")

if __name__ == "__main__":
    success = run_complete_test()
    sys.exit(0 if success else 1) 