#!/usr/bin/env python3
"""
Test Microsoft Graph RAG Entity Extraction
==========================================

Script de prueba para verificar que el Entity Extractor funciona correctamente.
"""

import sys
import os
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Añadir path para importar rag_preprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_entity_extraction():
    """Prueba la extracción de entidades con diferentes tipos de texto"""
    
    print("🔍 TESTING MICROSOFT GRAPH RAG - ENTITY EXTRACTION")
    print("=" * 60)
    
    try:
        from rag_preprocessing.graph_rag.entity_extractor import EntityExtractor
        
        # Crear extractor
        extractor = EntityExtractor()
        
        # Textos de prueba
        test_texts = [
            {
                "name": "Texto Simple",
                "content": "Microsoft desarrolló Graph RAG en colaboración con OpenAI. El sistema usa algoritmos Leiden para detectar comunidades en grafos de conocimiento."
            },
            {
                "name": "Texto Técnico", 
                "content": "El algoritmo Leiden mejora el Louvain implementando optimización local y refinamiento. NetworkX proporciona la base para manipulación de grafos mientras que igraph ofrece implementaciones más eficientes."
            },
            {
                "name": "Texto Conversacional",
                "content": "Dr. García trabaja en la Universidad de Madrid desarrollando sistemas RAG. Su equipo colabora con investigadores de Stanford y MIT en proyectos de IA generativa."
            }
        ]
        
        # Procesar cada texto
        for i, test_case in enumerate(test_texts, 1):
            print(f"\n📝 TEST {i}: {test_case['name']}")
            print("-" * 40)
            print(f"Texto: {test_case['content'][:100]}...")
            
            # Extraer entidades
            result = extractor.extract_entities_and_relationships(
                text=test_case['content'],
                use_llm_fallback=False  # Solo spaCy por ahora
            )
            
            print(f"⏱️  Tiempo: {result.processing_time:.3f}s")
            print(f"🔧 Método: {result.method_used}")
            print(f"⭐ Confidence: {result.confidence_score:.2f}")
            print(f"👥 Entidades: {len(result.entities)}")
            print(f"🔗 Relaciones: {len(result.relationships)}")
            
            # Mostrar entidades detectadas
            if result.entities:
                print("\n🏷️  ENTIDADES DETECTADAS:")
                for entity in result.entities[:5]:  # Mostrar solo las primeras 5
                    print(f"  • {entity.name} ({entity.type.value}) - Conf: {entity.confidence:.2f}")
            
            # Mostrar relaciones detectadas
            if result.relationships:
                print("\n🔗 RELACIONES DETECTADAS:")
                for rel in result.relationships[:3]:  # Mostrar solo las primeras 3
                    print(f"  • {rel.source_entity_id} → {rel.relation_type.value} → {rel.target_entity_id}")
        
        # Mostrar métricas finales
        print("\n📊 MÉTRICAS FINALES")
        print("=" * 30)
        metrics = extractor.get_extraction_metrics()
        
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        
        print("\n✅ ENTITY EXTRACTION TEST COMPLETADO")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN ENTITY EXTRACTION: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_community_detection():
    """Prueba rápida de detección de comunidades"""
    
    print("\n\n🕸️  TESTING COMMUNITY DETECTION")
    print("=" * 40)
    
    try:
        from rag_preprocessing.graph_rag.entity_extractor import EntityExtractor
        from rag_preprocessing.graph_rag.community_detector import CommunityDetector
        
        # Crear extractores
        entity_extractor = EntityExtractor()
        community_detector = CommunityDetector()
        
        # Texto más largo para generar más entidades
        long_text = """
        Microsoft Research desarrolló Graph RAG en colaboración con OpenAI y Stanford University. 
        El proyecto utiliza algoritmos de detección de comunidades como Leiden y Louvain.
        NetworkX proporciona la infraestructura base mientras que igraph ofrece implementaciones optimizadas.
        El Dr. García de la Universidad de Madrid lidera la investigación en sistemas RAG híbridos.
        Su equipo colabora con investigadores de MIT, Stanford y Google DeepMind.
        Los algoritmos de quantización vectorial mejoran la eficiencia del almacenamiento.
        ChromaDB se usa para desarrollo mientras que Qdrant escala a nivel de producción.
        """
        
        # 1. Extraer entidades y relaciones
        print("📝 Extrayendo entidades...")
        extraction_result = entity_extractor.extract_entities_and_relationships(long_text, use_llm_fallback=False)
        
        print(f"✅ Extraídas {len(extraction_result.entities)} entidades y {len(extraction_result.relationships)} relaciones")
        
        # 2. Detectar comunidades
        print("🕸️  Detectando comunidades...")
        hierarchy = community_detector.detect_communities(
            entities=extraction_result.entities,
            relationships=extraction_result.relationships,
            max_levels=2,
            min_community_size=2
        )
        
        print(f"✅ Detectadas comunidades en {hierarchy.total_levels} niveles")
        print(f"📊 Total de comunidades: {hierarchy.total_communities}")
        print(f"🔧 Algoritmo usado: {hierarchy.detection_algorithm}")
        print(f"⏱️  Tiempo: {hierarchy.processing_time:.3f}s")
        
        # Mostrar comunidades por nivel
        for level in range(hierarchy.total_levels):
            communities = hierarchy.get_level_communities(level)
            print(f"\n📑 NIVEL {level}: {len(communities)} comunidades")
            
            for i, community in enumerate(communities):
                entity_names = []
                entity_map = {e.id: e.name for e in extraction_result.entities}
                
                for entity_id in list(community.entities)[:3]:  # Primeras 3 entidades
                    if entity_id in entity_map:
                        entity_names.append(entity_map[entity_id])
                
                print(f"  Comunidad {i}: {', '.join(entity_names)}... ({community.size} entidades)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN COMMUNITY DETECTION: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 INICIANDO TESTS DE MICROSOFT GRAPH RAG")
    print("=" * 70)
    
    # Test 1: Entity Extraction
    success1 = test_entity_extraction()
    
    # Test 2: Community Detection
    success2 = test_community_detection()
    
    # Resultado final
    print("\n" + "=" * 70)
    if success1 and success2:
        print("🎉 TODOS LOS TESTS COMPLETADOS EXITOSAMENTE")
        print("✅ Entity Extraction: PASS")
        print("✅ Community Detection: PASS")
        print("\n🚀 Microsoft Graph RAG está listo para Sprint 2!")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        print(f"❌ Entity Extraction: {'PASS' if success1 else 'FAIL'}")
        print(f"❌ Community Detection: {'PASS' if success2 else 'FAIL'}") 