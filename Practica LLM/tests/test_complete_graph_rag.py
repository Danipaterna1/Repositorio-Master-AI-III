#!/usr/bin/env python3
"""
Test Completo Microsoft Graph RAG Pipeline
==========================================

Demuestra el pipeline completo de Graph RAG integrado con el sistema existente.
"""

import sys
import os
import time
import json
from pathlib import Path

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_complete_graph_rag_pipeline():
    """Test del pipeline completo de Graph RAG"""
    
    print("🚀 MICROSOFT GRAPH RAG - PIPELINE COMPLETO")
    print("=" * 60)
    
    try:
        from rag_preprocessing.graph_rag.pipeline import GraphRAGPipeline, GraphRAGConfig
        
        # Configurar pipeline
        config = GraphRAGConfig(
            max_community_levels=3,
            min_community_size=2,
            use_llm_extraction=False,  # Solo spaCy por ahora
            community_algorithm="leiden"
        )
        
        # Crear pipeline
        pipeline = GraphRAGPipeline(config)
        
        print(f"✅ Pipeline inicializado con configuración:")
        print(f"   - Algoritmo: {config.community_algorithm}")
        print(f"   - Niveles máximos: {config.max_community_levels}")
        print(f"   - Tamaño mínimo comunidad: {config.min_community_size}")
        print(f"   - LLM Extraction: {config.use_llm_extraction}")
        
        # Documentos de prueba más complejos
        test_documents = [
            """
            Microsoft Research desarrolló Graph RAG en colaboración con OpenAI y Stanford University. 
            El proyecto utiliza algoritmos de detección de comunidades como Leiden y Louvain para 
            identificar clusters de entidades relacionadas. NetworkX proporciona la infraestructura 
            base para manipulación de grafos, mientras que igraph ofrece implementaciones más eficientes.
            
            El Dr. García de la Universidad de Madrid lidera la investigación en sistemas RAG híbridos.
            Su equipo colabora con investigadores de MIT, Stanford y Google DeepMind en el desarrollo
            de arquitecturas de embedding vectorial. Los algoritmos de quantización vectorial mejoran 
            la eficiencia del almacenamiento en sistemas como ChromaDB y Qdrant.
            
            El enfoque de Microsoft combina búsqueda local en comunidades específicas con búsqueda 
            global en resúmenes jerárquicos. Esto permite responder tanto preguntas específicas 
            sobre entidades individuales como preguntas amplias sobre tendencias y patrones generales.
            """
        ]
        
        # 1. Procesar documentos
        print(f"\n📄 PROCESANDO DOCUMENTOS...")
        print(f"   Documentos: {len(test_documents)}")
        print(f"   Longitud total: {sum(len(doc) for doc in test_documents):,} caracteres")
        
        start_time = time.time()
        result = pipeline.process_documents(test_documents)
        processing_time = time.time() - start_time
        
        print(f"✅ Procesamiento completado en {processing_time:.2f}s")
        print(f"\n📊 RESULTADOS DEL PROCESAMIENTO:")
        print(f"   🏷️  Entidades extraídas: {result['entities']}")
        print(f"   🔗 Relaciones detectadas: {result['relationships']}")
        print(f"   🕸️  Comunidades detectadas: {result['communities']}")
        print(f"   📊 Niveles jerárquicos: {result['levels']}")
        print(f"   📝 Reportes generados: {result['reports']}")
        print(f"   ⏱️  Tiempo de procesamiento: {result['processing_time']:.3f}s")
        print(f"   ✅ Estado: {result['status']}")
        
        # 2. Test de consultas
        print(f"\n🔍 TESTING CONSULTAS GRAPH RAG...")
        
        test_queries = [
            "¿Qué algoritmos usa Microsoft Graph RAG?",
            "¿Quién colabora con la Universidad de Madrid?",
            "¿Cómo funciona la búsqueda híbrida en Graph RAG?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 CONSULTA {i}: {query}")
            
            # Test búsqueda local
            local_result = pipeline.query(query, search_type="local")
            print(f"   🏠 Local: {local_result['answer'][:100]}...")
            print(f"      Confidence: {local_result['confidence']:.2f}")
            
            # Test búsqueda global
            global_result = pipeline.query(query, search_type="global")
            print(f"   🌍 Global: {global_result['answer'][:100]}...")
            print(f"      Confidence: {global_result['confidence']:.2f}")
            
            # Test búsqueda híbrida
            hybrid_result = pipeline.query(query, search_type="hybrid")
            print(f"   🔀 Hybrid: {hybrid_result['answer'][:100]}...")
            print(f"      Confidence: {hybrid_result['confidence']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN PIPELINE: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_graph_rag_integration():
    """Test de integración con el sistema RAG principal"""
    
    print(f"\n\n🔗 TESTING INTEGRACIÓN CON SISTEMA RAG PRINCIPAL")
    print("=" * 60)
    
    try:
        # Importar sistema RAG principal
        from rag_preprocessing import get_document_processor, get_vector_store
        from rag_preprocessing.graph_rag.pipeline import GraphRAGPipeline
        
        # Inicializar componentes
        vector_processor = get_document_processor()
        vector_store = get_vector_store()
        graph_pipeline = GraphRAGPipeline()
        
        # Documento de prueba
        test_text = """
        El Graph RAG de Microsoft revoluciona la búsqueda en documentos al combinar 
        embeddings vectoriales con análisis de grafos de conocimiento. El sistema 
        detecta automáticamente entidades como organizaciones, personas y conceptos,
        luego identifica comunidades de entidades relacionadas usando el algoritmo Leiden.
        """
        
        print(f"📄 Procesando documento híbrido...")
        print(f"   Texto: {test_text[:100]}...")
        
        # 1. Procesamiento vectorial tradicional
        print(f"\n🔢 PROCESAMIENTO VECTORIAL:")
        vector_start = time.time()
        vector_result = vector_processor.process_text(
            text=test_text,
            document_id="hybrid_test_doc",
            metadata={"type": "hybrid_test"}
        )
        vector_time = time.time() - vector_start
        
        print(f"   ✅ Chunks: {vector_result.chunks_processed}")
        print(f"   ✅ Embeddings: {vector_result.embeddings_created}")
        print(f"   ⏱️  Tiempo: {vector_time:.3f}s")
        
        # 2. Procesamiento Graph RAG
        print(f"\n🕸️  PROCESAMIENTO GRAPH RAG:")
        graph_start = time.time()
        graph_result = graph_pipeline.process_documents([test_text])
        graph_time = time.time() - graph_start
        
        print(f"   ✅ Entidades: {graph_result['entities']}")
        print(f"   ✅ Comunidades: {graph_result['communities']}")
        print(f"   ⏱️  Tiempo: {graph_time:.3f}s")
        
        # 3. Búsqueda híbrida simulada
        print(f"\n🔍 BÚSQUEDA HÍBRIDA SIMULADA:")
        query = "¿Cómo funciona Graph RAG?"
        
        # Búsqueda vectorial
        vector_search_start = time.time()
        # Generar embedding de consulta
        query_embedding = vector_processor.embedding_manager.encode(query)
        vector_search_results = vector_store.search(
            query_embedding=query_embedding.embeddings[0],
            k=3
        )
        vector_search_time = time.time() - vector_search_start
        
        print(f"   🔢 Vector Search: {len(vector_search_results)} resultados en {vector_search_time*1000:.1f}ms")
        if vector_search_results:
            print(f"      Mejor score: {vector_search_results[0].score:.3f}")
        
        # Búsqueda Graph RAG
        graph_search_start = time.time()
        graph_search_result = graph_pipeline.query(query, search_type="hybrid")
        graph_search_time = time.time() - graph_search_start
        
        print(f"   🕸️  Graph Search: respuesta en {graph_search_time*1000:.1f}ms")
        print(f"      Confidence: {graph_search_result['confidence']:.2f}")
        
        # Métricas comparativas
        print(f"\n📊 MÉTRICAS COMPARATIVAS:")
        print(f"   Procesamiento Vector: {vector_time:.3f}s")
        print(f"   Procesamiento Graph: {graph_time:.3f}s")
        print(f"   Búsqueda Vector: {vector_search_time*1000:.1f}ms")
        print(f"   Búsqueda Graph: {graph_search_time*1000:.1f}ms")
        
        total_processing = vector_time + graph_time
        print(f"   📈 Procesamiento Híbrido Total: {total_processing:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN INTEGRACIÓN: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    
    print("🧪 MICROSOFT GRAPH RAG - TEST COMPLETO")
    print("=" * 70)
    
    # Test 1: Pipeline completo
    success1 = test_complete_graph_rag_pipeline()
    
    # Test 2: Integración con sistema principal
    success2 = test_graph_rag_integration()
    
    # Resultado final
    print("\n" + "=" * 70)
    if success1 and success2:
        print("🎉 TODOS LOS TESTS COMPLETADOS EXITOSAMENTE")
        print("✅ Graph RAG Pipeline: PASS")
        print("✅ Integración Híbrida: PASS")
        print("\n🚀 MICROSOFT GRAPH RAG SPRINT 2: 70% COMPLETADO")
        print("📋 Próximos pasos: Community Summarization + LLM Integration")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        print(f"❌ Graph RAG Pipeline: {'PASS' if success1 else 'FAIL'}")
        print(f"❌ Integración Híbrida: {'PASS' if success2 else 'FAIL'}")

if __name__ == "__main__":
    main() 