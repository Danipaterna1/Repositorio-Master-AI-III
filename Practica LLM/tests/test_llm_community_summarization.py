#!/usr/bin/env python3
"""
Test LLM-Enhanced Community Summarization
==========================================

Tests para verificar la nueva funcionalidad de Sprint 3:
LLM-enhanced community summarization con Google Gemini.
"""

import sys
import os
import time
from pathlib import Path

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_llm_enhanced_summarization():
    """Test de LLM-enhanced community summarization"""
    
    print("🧠 LLM-ENHANCED COMMUNITY SUMMARIZATION TEST")
    print("=" * 60)
    
    try:
        from rag_preprocessing.graph_rag.pipeline import GraphRAGPipeline, GraphRAGConfig
        
        # Test documento complejo para mejor evaluación LLM
        complex_document = """
        La colaboración entre Microsoft Research, OpenAI y Stanford University ha revolucionado 
        el campo del Graph RAG. El Dr. García de la Universidad Complutense de Madrid lidera 
        un equipo de investigación que incluye a la Dra. Smith del MIT y al Prof. Johnson de 
        Google DeepMind. 
        
        Su trabajo se centra en algoritmos de detección de comunidades como Leiden y Louvain, 
        implementados usando NetworkX e igraph. Los sistemas de embeddings vectoriales utilizan 
        ChromaDB para desarrollo y Qdrant para producción enterprise.
        
        El enfoque híbrido combina búsqueda local en comunidades específicas con búsqueda 
        global usando resúmenes jerárquicos generados por LLMs como Gemini y GPT-4. Los 
        algoritmos de quantización vectorial optimizan el almacenamiento y la velocidad 
        de búsqueda en sistemas de gran escala.
        """
        
        print(f"📄 DOCUMENTO DE PRUEBA:")
        print(f"   Longitud: {len(complex_document):,} caracteres")
        print(f"   Complejidad: Alta (entidades, relaciones, conceptos técnicos)")
        
        # 1. Test con LLM activado
        print(f"\n🧠 TEST 1: LLM-ENHANCED SUMMARIZATION")
        config_llm = GraphRAGConfig(
            max_community_levels=3,
            min_community_size=2,
            use_llm_extraction=False,
            use_llm_summarization=True,  # Activar LLM
            community_algorithm="leiden"
        )
        
        pipeline_llm = GraphRAGPipeline(config_llm)
        
        start_time = time.time()
        result_llm = pipeline_llm.process_documents([complex_document])
        llm_time = time.time() - start_time
        
        print(f"   ✅ LLM Processing: {llm_time:.2f}s")
        print(f"   📊 Entidades: {result_llm['entities']}")
        print(f"   🕸️  Comunidades: {result_llm['communities']}")
        print(f"   📝 Reportes: {result_llm['reports']}")
        print(f"   🧠 LLM Enhanced: {result_llm['llm_enhanced_reports']}")
        print(f"   🔧 Modo: {result_llm['summarization_mode']}")
        
        # 2. Test con LLM desactivado (baseline)
        print(f"\n📊 TEST 2: STATISTICAL BASELINE")
        config_stat = GraphRAGConfig(
            max_community_levels=3,
            min_community_size=2,
            use_llm_extraction=False,
            use_llm_summarization=False,  # Desactivar LLM
            community_algorithm="leiden"
        )
        
        pipeline_stat = GraphRAGPipeline(config_stat)
        
        start_time = time.time()
        result_stat = pipeline_stat.process_documents([complex_document])
        stat_time = time.time() - start_time
        
        print(f"   ✅ Statistical Processing: {stat_time:.2f}s")
        print(f"   📊 Entidades: {result_stat['entities']}")
        print(f"   🕸️  Comunidades: {result_stat['communities']}")
        print(f"   📝 Reportes: {result_stat['reports']}")
        print(f"   🧠 LLM Enhanced: {result_stat['llm_enhanced_reports']}")
        print(f"   🔧 Modo: {result_stat['summarization_mode']}")
        
        # 3. Comparación de calidad
        print(f"\n📈 ANÁLISIS COMPARATIVO:")
        print(f"   ⏱️  Tiempo LLM: {llm_time:.2f}s")
        print(f"   ⏱️  Tiempo Statistical: {stat_time:.2f}s")
        print(f"   🚀 Overhead LLM: {((llm_time - stat_time) / stat_time * 100):.1f}%")
        
        # 4. Verificar títulos de comunidades
        print(f"\n🏷️  COMPARACIÓN DE TÍTULOS DE COMUNIDADES:")
        
        # Test queries para verificar calidad
        test_queries = [
            "¿Qué organizaciones colaboran en Graph RAG?",
            "¿Quién lidera la investigación en Madrid?",
            "¿Qué algoritmos se utilizan?"
        ]
        
        print(f"\n🔍 TEST DE CONSULTAS:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n   📝 CONSULTA {i}: {query}")
            
            # LLM-enhanced query
            llm_result = pipeline_llm.query(query, search_type="hybrid")
            print(f"      🧠 LLM: {llm_result['answer'][:80]}...")
            print(f"         Confidence: {llm_result['confidence']:.2f}")
            
            # Statistical query
            stat_result = pipeline_stat.query(query, search_type="hybrid")
            print(f"      📊 Stat: {stat_result['answer'][:80]}...")
            print(f"         Confidence: {stat_result['confidence']:.2f}")
        
        # 5. Métricas de éxito
        success_metrics = {
            "llm_available": result_llm['llm_enhanced_reports'] > 0,
            "statistical_fallback": result_stat['llm_enhanced_reports'] == 0,
            "same_entities": result_llm['entities'] == result_stat['entities'],
            "same_communities": result_llm['communities'] == result_stat['communities'],
            "performance_acceptable": llm_time < stat_time * 10  # LLM puede ser hasta 10x más lento (realista)
        }
        
        print(f"\n✅ MÉTRICAS DE ÉXITO:")
        for metric, success in success_metrics.items():
            status = "✅" if success else "❌"
            print(f"   {status} {metric}: {success}")
        
        all_success = all(success_metrics.values())
        
        if all_success:
            print(f"\n🎉 TODOS LOS TESTS EXITOSOS")
            print(f"✅ LLM-Enhanced Community Summarization: FUNCIONANDO")
            return True
        else:
            print(f"\n⚠️  ALGUNOS TESTS FALLARON")
            return False
        
    except Exception as e:
        print(f"\n❌ ERROR EN TEST LLM: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_configuration():
    """Test de configuración de API"""
    
    print(f"\n🔧 TEST DE CONFIGURACIÓN API")
    print("=" * 40)
    
    try:
        import os
        from dotenv import load_dotenv
        
        # Cargar variables de entorno
        load_dotenv()
        
        google_api_key = os.getenv('GOOGLE_API_KEY')
        
        print(f"   📁 .env file: {'Found' if os.path.exists('.env') else 'Not found'}")
        print(f"   🗝️  Google API Key: {'Configured' if google_api_key and google_api_key != 'your_google_api_key_here' else 'Not configured'}")
        
        if google_api_key and google_api_key != 'your_google_api_key_here':
            print(f"   ✅ API configurada correctamente")
            print(f"   💡 LLM enhancement estará disponible")
            return True
        else:
            print(f"   ⚠️  API no configurada")
            print(f"   💡 Sistema funcionará en modo estadístico")
            print(f"   📋 Para configurar: edita .env y añade tu Google API key")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking API config: {e}")
        return False

def main():
    """Función principal"""
    
    print("🧪 LLM-ENHANCED GRAPH RAG - SPRINT 3 TEST")
    print("=" * 70)
    
    # Test configuración API
    api_configured = test_api_configuration()
    
    # Test principal
    llm_test_success = test_llm_enhanced_summarization()
    
    print(f"\n" + "=" * 70)
    print(f"📊 RESUMEN DE TESTS:")
    print(f"   🔧 API Configuration: {'✅ PASS' if api_configured else '⚠️  FALLBACK'}")
    print(f"   🧠 LLM Enhancement: {'✅ PASS' if llm_test_success else '❌ FAIL'}")
    
    if llm_test_success:
        print(f"\n🎉 SPRINT 3.1 LLM INTEGRATION: COMPLETADO")
        if api_configured:
            print(f"🧠 LLM-enhanced community summarization funcionando")
        else:
            print(f"📊 Statistical fallback funcionando correctamente")
    else:
        print(f"\n❌ SPRINT 3.1: REQUIERE DEBUGGING")

if __name__ == "__main__":
    main() 