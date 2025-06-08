#!/usr/bin/env python3
"""
RAG System Testing Suite 2025
=============================

Script completo para probar el sistema RAG con diferentes tipos de texto
y generar reportes detallados de performance y comportamiento.

Uso: python test_rag_system.py
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Añadir path para importar rag_preprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_test_documents() -> List[Dict[str, Any]]:
    """Carga todos los documentos de prueba desde test_documents/"""
    
    test_docs = []
    test_dir = Path("test_documents")
    
    if not test_dir.exists():
        logger.error("Directorio test_documents no encontrado")
        return []
    
    for txt_file in test_dir.glob("*.txt"):
        try:
            content = txt_file.read_text(encoding='utf-8')
            
            # Determinar tipo de documento por nombre de archivo
            doc_type = "unknown"
            if "corto" in txt_file.name:
                doc_type = "short_simple"
            elif "largo" in txt_file.name:
                doc_type = "long_complex"
            elif "tecnico" in txt_file.name:
                doc_type = "technical_specialized"
            elif "conversacion" in txt_file.name:
                doc_type = "conversational_dialog"
            
            test_docs.append({
                "id": txt_file.stem,
                "content": content,
                "metadata": {
                    "filename": txt_file.name,
                    "doc_type": doc_type,
                    "file_size": len(content),
                    "word_count": len(content.split()),
                    "line_count": len(content.splitlines())
                }
            })
            
            logger.info(f"Cargado: {txt_file.name} ({len(content):,} chars)")
            
        except Exception as e:
            logger.error(f"Error cargando {txt_file}: {e}")
    
    return test_docs

def analyze_text_complexity(text: str) -> Dict[str, Any]:
    """Analiza la complejidad del texto para predecir qué tipo de embedding usar"""
    
    # Métricas básicas
    char_count = len(text)
    word_count = len(text.split())
    line_count = len(text.splitlines())
    
    # Palabras especializadas (>12 caracteres)
    specialized_words = sum(1 for word in text.split() if len(word.strip('.,;:!?')) > 12)
    
    # Puntuación compleja
    complex_punctuation = text.count(',') + text.count(';') + text.count(':')
    
    # Términos técnicos comunes
    technical_terms = [
        'algorithm', 'implementation', 'architecture', 'optimization', 
        'quantization', 'embeddings', 'vectorial', 'distributed',
        'scalability', 'throughput', 'latency', 'methodology'
    ]
    
    technical_score = sum(1 for term in technical_terms if term.lower() in text.lower())
    
    # Predicción de tipo de embedding
    predicted_embedding = "static"  # Por defecto
    
    # Heurística para embeddings tradicionales
    if (char_count > 2000 or 
        specialized_words > 5 or 
        complex_punctuation > 10 or
        technical_score > 3):
        predicted_embedding = "traditional"
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
        "specialized_words": specialized_words,
        "complex_punctuation": complex_punctuation,
        "technical_score": technical_score,
        "predicted_embedding": predicted_embedding,
        "complexity_score": (specialized_words * 2 + complex_punctuation + technical_score) / max(word_count, 1) * 100
    }

def test_single_document(processor, doc: Dict[str, Any]) -> Dict[str, Any]:
    """Prueba un documento individual y retorna métricas detalladas"""
    
    logger.info(f"\n🧪 Probando documento: {doc['id']}")
    logger.info(f"   Tipo: {doc['metadata']['doc_type']}")
    logger.info(f"   Tamaño: {doc['metadata']['file_size']:,} chars, {doc['metadata']['word_count']:,} words")
    
    # Analizar complejidad
    complexity = analyze_text_complexity(doc['content'])
    logger.info(f"   Complejidad: {complexity['complexity_score']:.1f}% - Predicción: {complexity['predicted_embedding']}")
    
    # Procesar documento
    start_time = time.time()
    result = processor.process_text(
        text=doc['content'],
        document_id=doc['id'],
        metadata=doc['metadata']
    )
    total_time = time.time() - start_time
    
    # Compilar resultados
    test_result = {
        "document_id": doc['id'],
        "doc_type": doc['metadata']['doc_type'],
        "complexity_analysis": complexity,
        "processing_result": {
            "chunks_processed": result.chunks_processed,
            "embeddings_created": result.embeddings_created,
            "storage_ids_count": len(result.storage_ids),
            "processing_time": result.processing_time,
            "errors": result.errors,
            "success": len(result.errors) == 0
        },
        "performance_metrics": {
            "chars_per_second": doc['metadata']['file_size'] / max(result.processing_time, 0.001),
            "chunks_per_second": result.chunks_processed / max(result.processing_time, 0.001),
            "embeddings_per_second": result.embeddings_created / max(result.processing_time, 0.001)
        }
    }
    
    # Mostrar resultados
    print(f"   ✅ Procesado: {result.chunks_processed} chunks, {result.embeddings_created} embeddings")
    print(f"   ⏱️  Tiempo: {result.processing_time:.2f}s ({test_result['performance_metrics']['chars_per_second']:.0f} chars/s)")
    if result.errors:
        print(f"   ⚠️  Errores: {len(result.errors)}")
        for error in result.errors:
            print(f"      - {error}")
    
    return test_result

def test_search_functionality(processor, vector_store) -> Dict[str, Any]:
    """Prueba la funcionalidad de búsqueda vectorial"""
    
    logger.info("\n🔍 Probando funcionalidad de búsqueda...")
    
    # Obtener un embedding para búsqueda
    test_query = "¿Cómo funcionan los sistemas RAG?"
    
    try:
        # Generar embedding de la query
        embedding_manager = processor.embedding_manager
        query_result = embedding_manager.encode(test_query)
        
        # Realizar búsqueda
        search_start = time.time()
        search_results = vector_store.search(
            query_embedding=query_result.embeddings[0],
            k=5
        )
        search_time = time.time() - search_start
        
        # Probar búsqueda con filtros
        filter_start = time.time()
        filtered_results = vector_store.search_with_filters(
            query_embedding=query_result.embeddings[0],
            filters={"doc_type": "conversational_dialog"},
            k=3
        )
        filter_time = time.time() - filter_start
        
        search_metrics = {
            "query_embedding_time": query_result.processing_time,
            "search_time": search_time,
            "filter_search_time": filter_time,
            "results_found": len(search_results),
            "filtered_results_found": len(filtered_results),
            "top_result_score": search_results[0].score if search_results else 0,
            "success": True
        }
        
        print(f"   📊 Query embedding: {query_result.processing_time*1000:.1f}ms")
        print(f"   🔍 Búsqueda: {search_time*1000:.1f}ms, {len(search_results)} resultados")
        print(f"   🔍 Búsqueda filtrada: {filter_time*1000:.1f}ms, {len(filtered_results)} resultados")
        if search_results:
            print(f"   🎯 Mejor score: {search_results[0].score:.3f}")
            print(f"   📄 Mejor resultado: {search_results[0].content[:100]}...")
        
        return search_metrics
        
    except Exception as e:
        logger.error(f"Error en búsqueda: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def generate_report(test_results: List[Dict], search_metrics: Dict, system_metrics: Dict) -> Dict[str, Any]:
    """Genera un reporte completo de los tests"""
    
    # Estadísticas por tipo de documento
    doc_type_stats = {}
    for result in test_results:
        doc_type = result['doc_type']
        if doc_type not in doc_type_stats:
            doc_type_stats[doc_type] = {
                "count": 0,
                "total_processing_time": 0,
                "total_chunks": 0,
                "total_chars": 0,
                "avg_complexity": 0
            }
        
        stats = doc_type_stats[doc_type]
        stats["count"] += 1
        stats["total_processing_time"] += result['processing_result']['processing_time']
        stats["total_chunks"] += result['processing_result']['chunks_processed']
        stats["total_chars"] += result['complexity_analysis']['char_count']
        stats["avg_complexity"] += result['complexity_analysis']['complexity_score']
    
    # Calcular promedios
    for doc_type, stats in doc_type_stats.items():
        if stats["count"] > 0:
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["count"]
            stats["avg_chunks_per_doc"] = stats["total_chunks"] / stats["count"]
            stats["avg_chars_per_doc"] = stats["total_chars"] / stats["count"]
            stats["avg_complexity"] = stats["avg_complexity"] / stats["count"]
            stats["chars_per_second"] = stats["total_chars"] / stats["total_processing_time"] if stats["total_processing_time"] > 0 else 0
    
    # Estadísticas generales
    successful_tests = sum(1 for r in test_results if r['processing_result']['success'])
    total_processing_time = sum(r['processing_result']['processing_time'] for r in test_results)
    total_chunks = sum(r['processing_result']['chunks_processed'] for r in test_results)
    total_embeddings = sum(r['processing_result']['embeddings_created'] for r in test_results)
    
    report = {
        "test_summary": {
            "total_documents": len(test_results),
            "successful_tests": successful_tests,
            "success_rate": successful_tests / len(test_results) * 100 if test_results else 0,
            "total_processing_time": total_processing_time,
            "total_chunks_created": total_chunks,
            "total_embeddings_created": total_embeddings,
            "avg_processing_time": total_processing_time / len(test_results) if test_results else 0
        },
        "doc_type_analysis": doc_type_stats,
        "search_performance": search_metrics,
        "system_metrics": system_metrics,
        "embedding_distribution": {
            "static_calls": system_metrics.get('embedding_metrics', {}).get('static_calls', 0),
            "traditional_calls": system_metrics.get('embedding_metrics', {}).get('traditional_calls', 0),
            "static_percentage": system_metrics.get('embedding_metrics', {}).get('static_percentage', 0)
        }
    }
    
    return report

def print_detailed_report(report: Dict[str, Any]):
    """Imprime un reporte detallado y bonito"""
    
    print("\n" + "="*80)
    print("📊 REPORTE COMPLETO DE TESTING RAG SYSTEM")
    print("="*80)
    
    # Resumen general
    summary = report['test_summary']
    print(f"\n🎯 RESUMEN GENERAL:")
    print(f"   Documentos probados: {summary['total_documents']}")
    print(f"   Tests exitosos: {summary['successful_tests']}/{summary['total_documents']} ({summary['success_rate']:.1f}%)")
    print(f"   Tiempo total: {summary['total_processing_time']:.2f}s")
    print(f"   Chunks creados: {summary['total_chunks_created']:,}")
    print(f"   Embeddings generados: {summary['total_embeddings_created']:,}")
    print(f"   Tiempo promedio/doc: {summary['avg_processing_time']:.2f}s")
    
    # Análisis por tipo de documento
    print(f"\n📋 ANÁLISIS POR TIPO DE DOCUMENTO:")
    for doc_type, stats in report['doc_type_analysis'].items():
        print(f"\n   📄 {doc_type.upper()}:")
        print(f"      Documentos: {stats['count']}")
        print(f"      Tiempo promedio: {stats['avg_processing_time']:.2f}s")
        print(f"      Chunks promedio: {stats['avg_chunks_per_doc']:.1f}")
        print(f"      Velocidad: {stats['chars_per_second']:.0f} chars/s")
        print(f"      Complejidad promedio: {stats['avg_complexity']:.1f}%")
    
    # Performance de búsqueda
    if 'search_performance' in report and report['search_performance'].get('success'):
        search = report['search_performance']
        print(f"\n🔍 PERFORMANCE DE BÚSQUEDA:")
        print(f"   Embedding de query: {search['query_embedding_time']*1000:.1f}ms")
        print(f"   Búsqueda vectorial: {search['search_time']*1000:.1f}ms")
        print(f"   Búsqueda con filtros: {search['filter_search_time']*1000:.1f}ms")
        print(f"   Resultados encontrados: {search['results_found']}")
        print(f"   Mejor score: {search['top_result_score']:.3f}")
    
    # Distribución de embeddings
    if 'embedding_distribution' in report:
        emb = report['embedding_distribution']
        print(f"\n🧠 DISTRIBUCIÓN DE EMBEDDINGS:")
        print(f"   Static embeddings: {emb['static_calls']} ({emb['static_percentage']:.1f}%)")
        print(f"   Traditional embeddings: {emb['traditional_calls']}")
        
        if emb['static_calls'] > 0 and emb['traditional_calls'] > 0:
            print(f"   ✅ Sistema híbrido funcionando correctamente")
        elif emb['static_calls'] > 0:
            print(f"   ⚡ Solo embeddings estáticos utilizados (textos simples)")
        else:
            print(f"   🎯 Solo embeddings tradicionales utilizados (textos complejos)")
    
    # Vector store info
    vs_info = report['system_metrics'].get('vector_store_info', {})
    if vs_info:
        print(f"\n💾 ESTADO DEL VECTOR STORE:")
        print(f"   Tipo: {vs_info.get('type', 'N/A')}")
        print(f"   Documentos almacenados: {vs_info.get('count', 0):,}")
        if 'path' in vs_info:
            print(f"   Path: {vs_info['path']}")

def main():
    """Función principal del testing"""
    
    print("🧪 RAG System Testing Suite 2025")
    print("==================================")
    
    try:
        # Importar sistema RAG
        from rag_preprocessing import get_document_processor, get_vector_store
        
        # Inicializar componentes
        print("\n🏗️  Inicializando sistema RAG...")
        processor = get_document_processor()
        vector_store = get_vector_store()
        
        # Cargar documentos de prueba
        print("\n📂 Cargando documentos de prueba...")
        test_documents = load_test_documents()
        
        if not test_documents:
            print("❌ No se encontraron documentos de prueba en test_documents/")
            return
        
        print(f"✅ Cargados {len(test_documents)} documentos")
        
        # Probar cada documento
        print("\n🧪 Ejecutando tests individuales...")
        test_results = []
        
        for doc in test_documents:
            result = test_single_document(processor, doc)
            test_results.append(result)
        
        # Probar funcionalidad de búsqueda
        search_metrics = test_search_functionality(processor, vector_store)
        
        # Obtener métricas del sistema
        system_metrics = processor.get_processing_metrics()
        
        # Generar reporte
        report = generate_report(test_results, search_metrics, system_metrics)
        
        # Mostrar reporte
        print_detailed_report(report)
        
        # Guardar reporte JSON
        report_file = f"test_report_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Reporte guardado en: {report_file}")
        print(f"\n🎉 Testing completado exitosamente!")
        
    except ImportError as e:
        print(f"❌ Error importando sistema RAG: {e}")
        print("Asegúrate de tener instaladas las dependencias:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error durante testing: {e}")
        logger.error("Error en main", exc_info=True)

if __name__ == "__main__":
    main() 