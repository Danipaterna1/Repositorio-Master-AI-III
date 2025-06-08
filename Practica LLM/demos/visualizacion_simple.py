#!/usr/bin/env python3
"""
🎨 VISUALIZACIÓN SIMPLE - MICROSOFT GRAPH RAG
===========================================

Demo rápida de visualización sin abrir navegador automáticamente.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def crear_grafo_simple():
    """Crea una visualización simple del grafo actual"""
    
    from rag_preprocessing.graph_rag.entity_extractor import EntityExtractor
    from rag_preprocessing.graph_rag.graph_visualizer import GraphVisualizer
    
    # Texto de ejemplo
    texto = """
    Microsoft desarrolló Graph RAG junto con OpenAI. El Dr. García lidera la investigación
    en Stanford University. El equipo utiliza spaCy para extracción de entidades y NetworkX
    para análisis de grafos. ChromaDB almacena embeddings vectoriales.
    """
    
    # Extraer entidades y relaciones
    extractor = EntityExtractor()
    result = extractor.extract_entities_and_relationships(texto)
    
    # Crear visualización
    visualizer = GraphVisualizer()
    archivo = visualizer.visualize_knowledge_graph(
        result.entities, 
        result.relationships,
        title="Grafo Simple",
        filename="grafo_simple.html"
    )
    
    print(f"📊 Grafo generado: {archivo}")
    print(f"🏷️  Entidades detectadas: {len(result.entities)}")
    print(f"🔗 Relaciones detectadas: {len(result.relationships)}")
    
    # Mostrar entidades
    print(f"\n🏷️  ENTIDADES:")
    for entity in result.entities:
        print(f"   • {entity.name} ({entity.type.value})")
    
    # Mostrar relaciones
    print(f"\n🔗 RELACIONES:")
    for rel in result.relationships:
        source_name = next((e.name for e in result.entities if e.id == rel.source_entity_id), "Unknown")
        target_name = next((e.name for e in result.entities if e.id == rel.target_entity_id), "Unknown")
        print(f"   • {source_name} --[{rel.relation_type.value}]--> {target_name}")
    
    return archivo

if __name__ == "__main__":
    archivo = crear_grafo_simple()
    print(f"\n🌐 Para ver el grafo, abrir: {archivo}") 