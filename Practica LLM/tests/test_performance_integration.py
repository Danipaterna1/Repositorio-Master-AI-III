"""
Test de Integración Performance - Kingfisher A2A

Test completo de los motores de performance integrados con Google A2A Framework.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_performance_engines_import():
    """Test 1: Verificar que los motores de performance se pueden importar"""
    print("\n🔧 TEST 1: Import Performance Engines")
    
    try:
        from rag_preprocessing.core.smart_chunker import SmartSemanticChunker, create_smart_chunker
        print("✅ Smart Chunker importado correctamente")
        
        from rag_preprocessing.core.batch_embedder import BatchEmbedder, create_batch_embedder  
        print("✅ Batch Embedder importado correctamente")
        
        from rag_preprocessing.retrieval.hybrid_retriever import HybridRetriever, create_hybrid_retriever
        print("✅ Hybrid Retriever importado correctamente")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importando performance engines: {e}")
        return False

def test_enhanced_triple_processor():
    """Test 2: Verificar Enhanced Triple Processor"""
    print("\n🔧 TEST 2: Enhanced Triple Processor")
    
    try:
        from rag_preprocessing.core.enhanced_triple_processor import EnhancedTripleProcessor, create_enhanced_triple_processor
        
        # Crear processor
        processor = create_enhanced_triple_processor()
        print("✅ Enhanced Triple Processor creado")
        
        # Test con context manager
        try:
            with processor as p:
                print("✅ Context manager funcional")
                stats = p.get_performance_stats()
                print(f"✅ Performance stats: {stats['engines_available']}")
        except Exception as e:
            print(f"⚠️ Error en context manager: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error con Enhanced Triple Processor: {e}")
        return False

def test_performance_agent():
    """Test 3: Verificar Performance Agent"""
    print("\n🔧 TEST 3: Performance Agent")
    
    try:
        from agents.kingfisher_agent_performance import KingfisherPerformanceAgent
        
        # Crear agent
        agent = KingfisherPerformanceAgent()
        print("✅ Performance Agent creado")
        
        # Verificar capabilities
        capabilities = agent.get_capabilities()
        print(f"✅ Capabilities: {capabilities}")
        
        # Verificar agent card
        agent_card = agent.get_agent_card()
        print(f"✅ Agent Card: {agent_card['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error con Performance Agent: {e}")
        return False

async def test_performance_agent_processing():
    """Test 4: Verificar procesamiento del Performance Agent"""
    print("\n🔧 TEST 4: Performance Agent Processing")
    
    try:
        from agents.kingfisher_agent_performance import KingfisherPerformanceAgent
        
        agent = KingfisherPerformanceAgent()
        
        # Inicializar agent
        await agent.initialize()
        print("✅ Performance Agent inicializado")
        
        # Test document batch processing
        test_documents = [
            {
                "content": "La inteligencia artificial es una rama de la informática que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana. Estos sistemas pueden aprender, razonar y tomar decisiones basadas en datos.",
                "title": "Introducción a la IA"
            },
            {
                "content": "El machine learning es un subconjunto de la inteligencia artificial que permite a las máquinas aprender automáticamente sin ser programadas explícitamente. Utiliza algoritmos que pueden identificar patrones en datos.",
                "title": "Machine Learning Básico"
            }
        ]
        
        batch_task = {
            "id": "test_batch_001",
            "capability": "process_documents_batch",
            "params": {
                "documents": test_documents,
                "processing_mode": "TRIPLE_FULL",
                "batch_size": 32
            }
        }
        
        print("🔄 Procesando batch de documentos...")
        start_time = time.time()
        
        result = await agent.process_task(batch_task)
        
        processing_time = time.time() - start_time
        print(f"✅ Batch procesado en {processing_time:.2f}s")
        
        if result["status"] == "completed":
            batch_result = result["result"]
            print(f"✅ Documentos procesados: {batch_result['documents_processed']}")
            print(f"✅ Chunks generados: {batch_result['total_chunks_generated']}")
            print(f"✅ Embeddings creados: {batch_result['total_embeddings_created']}")
            print(f"✅ Chunks/segundo: {batch_result['avg_chunks_per_second']:.2f}")
            
            # Verificar targets de performance
            if batch_result['avg_chunks_per_second'] > 10:
                print("✅ Target de performance ALCANZADO (>10 chunks/segundo)")
            else:
                print("⚠️ Target de performance no alcanzado")
        else:
            print(f"❌ Batch processing falló: {result.get('error', 'Unknown error')}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error en performance agent processing: {e}")
        return False

async def test_hybrid_retrieval():
    """Test 5: Verificar Hybrid Retrieval"""
    print("\n🔧 TEST 5: Hybrid Retrieval")
    
    try:
        from agents.kingfisher_agent_performance import KingfisherPerformanceAgent
        
        agent = KingfisherPerformanceAgent()
        await agent.initialize()
        
        # Primero procesar algunos documentos para tener data
        test_documents = [
            {
                "content": "Python es un lenguaje de programación de alto nivel, interpretado y de propósito general. Es conocido por su sintaxis clara y legible, lo que lo hace ideal para principiantes.",
                "title": "Python Programming"
            }
        ]
        
        batch_task = {
            "id": "test_setup",
            "capability": "process_documents_batch", 
            "params": {"documents": test_documents}
        }
        
        await agent.process_task(batch_task)
        print("✅ Documentos de prueba indexados")
        
        # Test hybrid query
        query_task = {
            "id": "test_query_001",
            "capability": "retrieve_knowledge_hybrid",
            "params": {
                "query": "¿Qué es Python?",
                "top_k": 5,
                "mode": "hybrid",
                "include_context": True
            }
        }
        
        print("🔄 Ejecutando query híbrida...")
        start_time = time.time()
        
        result = await agent.process_task(query_task)
        
        query_time = time.time() - start_time
        print(f"✅ Query ejecutada en {query_time:.3f}s")
        
        if result["status"] == "completed":
            query_result = result["result"]
            print(f"✅ Resultados encontrados: {query_result['results_count']}")
            print(f"✅ Query time: {query_result['query_time']:.3f}s")
            
            # Verificar target de query time
            if query_result['query_time'] < 0.5:
                print("✅ Target de query time ALCANZADO (<500ms)")
            else:
                print("⚠️ Target de query time no alcanzado")
                
            return True
        else:
            print(f"❌ Query falló: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error en hybrid retrieval: {e}")
        return False

async def test_performance_analysis():
    """Test 6: Verificar Performance Analysis"""
    print("\n🔧 TEST 6: Performance Analysis")
    
    try:
        from agents.kingfisher_agent_performance import KingfisherPerformanceAgent
        
        agent = KingfisherPerformanceAgent()
        await agent.initialize()
        
        analysis_task = {
            "id": "test_analysis_001",
            "capability": "analyze_performance",
            "params": {
                "include_detailed_stats": True,
                "time_range": "all_time"
            }
        }
        
        print("🔄 Analizando performance...")
        result = await agent.process_task(analysis_task)
        
        if result["status"] == "completed":
            analysis = result["result"]
            print("✅ Análisis de performance completado")
            
            # Mostrar métricas clave
            overall = analysis["overall_metrics"]
            print(f"✅ Documentos procesados: {overall['documents_processed']}")
            print(f"✅ Queries procesadas: {overall['queries_processed']}")
            print(f"✅ Avg chunks/segundo: {overall['avg_chunks_per_second']:.2f}")
            print(f"✅ Avg query time: {overall['avg_query_time']:.3f}s")
            
            # Verificar targets
            targets = analysis["performance_targets"]
            print(f"✅ Throughput: {targets['throughput_current']} (target: {targets['throughput_target']})")
            print(f"✅ Query time: {targets['query_time_current']} (target: {targets['query_time_target']})")
            
            return True
        else:
            print(f"❌ Análisis falló: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error en performance analysis: {e}")
        return False

def test_performance_server():
    """Test 7: Verificar Performance Server"""
    print("\n🔧 TEST 7: Performance Server")
    
    try:
        from agents.server.a2a_performance_server import KingfisherA2APerformanceServer
        
        # Crear servidor
        server = KingfisherA2APerformanceServer()
        print("✅ Performance Server creado")
        
        # Verificar FastAPI app
        app = server.get_app()
        if app:
            print("✅ FastAPI app creada")
            print(f"✅ App title: {app.title}")
            print(f"✅ App version: {app.version}")
        else:
            print("❌ FastAPI app no disponible")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error con Performance Server: {e}")
        return False

async def run_all_tests():
    """Ejecutar todos los tests de integración performance"""
    print("🚀 INICIANDO TESTS DE INTEGRACIÓN PERFORMANCE")
    print("=" * 60)
    
    tests = [
        ("Import Performance Engines", test_performance_engines_import),
        ("Enhanced Triple Processor", test_enhanced_triple_processor),
        ("Performance Agent", test_performance_agent),
        ("Performance Agent Processing", test_performance_agent_processing),
        ("Hybrid Retrieval", test_hybrid_retrieval),
        ("Performance Analysis", test_performance_analysis),
        ("Performance Server", test_performance_server)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error ejecutando {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen de resultados
    print("\n" + "="*60)
    print("📊 RESUMEN DE RESULTADOS")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 RESULTADO FINAL: {passed}/{total} tests pasaron")
    
    if passed == total:
        print("🏆 ¡TODOS LOS TESTS DE PERFORMANCE PASARON!")
        print("🚀 Sistema listo para procesamiento masivo con contexto total")
    else:
        print("⚠️ Algunos tests fallaron. Revisar configuración.")
        
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_all_tests()) 