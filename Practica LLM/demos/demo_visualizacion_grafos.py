#!/usr/bin/env python3
"""
🎨 DEMO VISUALIZACIÓN DE GRAFOS - MICROSOFT GRAPH RAG
===================================================

Demostración interactiva de las capacidades de visualización de grafos.
Genera múltiples tipos de visualizaciones usando pyvis y networkx.
"""

import sys
import os
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_graph_visualization():
    """Demostración completa de visualización de grafos"""
    
    print("🎨 DEMO VISUALIZACIÓN DE GRAFOS - GRAPH RAG")
    print("=" * 60)
    
    try:
        from rag_preprocessing.graph_rag.pipeline import GraphRAGPipeline, GraphRAGConfig
        from rag_preprocessing.graph_rag.graph_visualizer import GraphVisualizer
        from rag_preprocessing.graph_rag.entity_extractor import EntityExtractor
        
        print("✅ SISTEMA DE VISUALIZACIÓN INICIALIZADO")
        print("   🎨 Pyvis: Visualizaciones interactivas HTML")
        print("   📊 NetworkX: Análisis de grafos y métricas") 
        print("   🌐 Exportación: HTML navegables localmente")
        
        # Documento complejo para demostración
        documento_rico = """
        El equipo de Microsoft Research, liderado por el Dr. Smith, desarrolló Graph RAG 
        en colaboración con Stanford University y OpenAI. El proyecto utiliza tecnologías
        avanzadas como Qdrant para almacenamiento vectorial y Neo4j para grafos de conocimiento.
        
        La Dra. García de MIT contribuyó con algoritmos de community detection usando Leiden
        y Louvain. Su equipo trabajó junto con investigadores de Google DeepMind y Universidad
        de Barcelona en la optimización de embeddings vectoriales.
        
        El sistema integra ChromaDB para desarrollo rápido, spaCy para extracción de entidades,
        y NetworkX para análisis de grafos. Amazon AWS proporciona infraestructura cloud,
        mientras que Docker facilita el deployment en múltiples entornos.
        
        Entre las tecnologías clave están: sentence-transformers para embeddings, 
        langchain para orquestación, FastAPI para APIs, y Streamlit para interfaces.
        Los algoritmos implementados incluyen HNSW para búsqueda vectorial y algoritmos
        de clustering jerárquico para detección de comunidades.
        """
        
        # Configuración optimizada para visualización
        config = GraphRAGConfig(
            max_community_levels=2,
            min_community_size=2,
            use_llm_extraction=False,
            community_algorithm="leiden"
        )
        
        # Procesar documento
        print(f"\n📄 PROCESANDO DOCUMENTO RICO EN ENTIDADES:")
        print(f"   📊 Caracteres: {len(documento_rico):,}")
        print(f"   📝 Tokens estimados: {len(documento_rico.split()):,}")
        
        pipeline = GraphRAGPipeline(config)
        visualizer = GraphVisualizer(output_dir="data/graphs")
        
        # Procesamiento principal
        start_time = time.time()
        resultado = pipeline.process_documents([documento_rico])
        processing_time = time.time() - start_time
        
        print(f"\n🎯 EXTRACCIÓN COMPLETADA:")
        print(f"   ⏱️  Tiempo: {processing_time:.2f}s")
        print(f"   🏷️  Entidades: {resultado['entities']}")
        print(f"   🔗 Relaciones: {resultado['relationships']}")
        print(f"   🕸️  Comunidades: {resultado['communities']}")
        print(f"   📊 Niveles: {resultado['levels']}")
        
        # Obtener datos para visualización
        extractor = pipeline.entity_extractor
        entity_result = extractor.extract_entities_and_relationships(documento_rico)
        
        # Obtener jerarquía de comunidades
        community_detector = pipeline.community_detector
        hierarchy = community_detector.detect_communities(
            entities=entity_result.entities,
            relationships=entity_result.relationships,
            max_levels=2,
            min_community_size=2
        )
        
        # Obtener reportes de comunidades
        community_summarizer = pipeline.community_summarizer
        reports = community_summarizer.generate_community_reports(
            hierarchy,
            entities=entity_result.entities,
            relationships=entity_result.relationships
        )
        
        # GENERAR VISUALIZACIONES
        print(f"\n🎨 GENERANDO VISUALIZACIONES INTERACTIVAS...")
        
        vis_start = time.time()
        visualization_results = visualizer.generate_visualization_report(
            entities=entity_result.entities,
            relationships=entity_result.relationships,
            communities=hierarchy,
            reports=reports
        )
        vis_time = time.time() - vis_start
        
        print(f"   ⏱️  Tiempo generación: {vis_time:.2f}s")
        print(f"   📁 Archivos generados: {len(visualization_results)}")
        
        # Mostrar archivos generados
        print(f"\n📊 VISUALIZACIONES DISPONIBLES:")
        for i, (tipo, ruta) in enumerate(visualization_results.items(), 1):
            file_size = Path(ruta).stat().st_size / 1024  # KB
            print(f"   {i}. {tipo.replace('_', ' ').title()}")
            print(f"      📁 {ruta}")
            print(f"      💾 Tamaño: {file_size:.1f} KB")
            
            # Abrir automáticamente el primer grafo
            if i == 1:
                try:
                    absolute_path = Path(ruta).resolve()
                    webbrowser.open(f"file://{absolute_path}")
                    print(f"      🌐 Abriendo en navegador...")
                except Exception as e:
                    print(f"      ⚠️  No se pudo abrir automáticamente: {e}")
        
        # Generar estadísticas del grafo
        summary = visualizer.create_graph_summary(
            entity_result.entities, 
            entity_result.relationships, 
            hierarchy, 
            reports
        )
        
        print(f"\n📈 ESTADÍSTICAS DEL GRAFO:")
        print(f"   🔢 Nodos totales: {summary['total_nodes']}")
        print(f"   🔗 Conexiones: {summary['total_edges']}")
        print(f"   📊 Densidad: {summary['density']:.3f}")
        print(f"   📐 Grado promedio: {summary['average_degree']:.1f}")
        print(f"   🧩 Componentes conectados: {summary['connected_components']}")
        
        # Distribución de tipos
        print(f"\n🏷️  DISTRIBUCIÓN POR TIPOS:")
        for tipo, cantidad in summary['entity_type_distribution'].items():
            print(f"   📍 {tipo}: {cantidad} entidades")
        
        # Información de comunidades
        if 'communities' in summary:
            comm_info = summary['communities']
            print(f"\n🕸️  COMUNIDADES DETECTADAS:")
            print(f"   📊 Total: {comm_info['total_communities']}")
            print(f"   📶 Niveles: {comm_info['levels']}")
            print(f"   🎯 Mayor tamaño: {comm_info['largest_community_size']} entidades")
        
        # Top comunidades
        if 'top_communities' in resultado and resultado['top_communities']:
            print(f"\n🏆 COMUNIDADES PRINCIPALES:")
            for i, comunidad in enumerate(resultado['top_communities'], 1):
                print(f"   {i}. {comunidad}")
        
        # Instrucciones de uso
        print(f"\n💡 CÓMO USAR LAS VISUALIZACIONES:")
        print("   🖱️  Clic y arrastre: Mover nodos")
        print("   🔍 Scroll: Zoom in/out")
        print("   ℹ️  Hover: Ver información detallada")
        print("   ⚙️  Panel derecho: Controles de física")
        print("   🎨 Colores: Tipos de entidades y relaciones")
        
        # URLs de acceso
        print(f"\n🌐 ACCESO A VISUALIZACIONES:")
        for tipo, ruta in visualization_results.items():
            if tipo.endswith('graph'):
                absolute_path = Path(ruta).resolve()
                print(f"   📊 {tipo.replace('_', ' ').title()}: file://{absolute_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN VISUALIZACIÓN: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_graph_types():
    """Explicación de los tipos de grafos disponibles"""
    
    print(f"\n\n🎭 TIPOS DE VISUALIZACIONES DISPONIBLES")
    print("=" * 55)
    
    print("1. 📊 KNOWLEDGE GRAPH BÁSICO:")
    print("   • Entidades como nodos coloreados por tipo")
    print("   • Relaciones como conexiones dirigidas") 
    print("   • Tooltips con información detallada")
    print("   • Física interactiva para exploración")
    
    print(f"\n2. 🧠 NETWORKX GRAPH CON MÉTRICAS:")
    print("   • Nodos dimensionados por centralidad")
    print("   • Métricas de importancia calculadas")
    print("   • Análisis de betweenness centrality")
    print("   • Identificación de nodos clave")
    
    print(f"\n3. 🕸️  COMMUNITIES GRAPH:")
    print("   • Comunidades coloreadas distintivamente")
    print("   • Agrupación visual por clusters")
    print("   • Información de importancia por comunidad")
    print("   • Estructura jerárquica visible")
    
    print(f"\n4. 📋 RESUMEN ESTADÍSTICO (JSON):")
    print("   • Métricas cuantitativas del grafo")
    print("   • Distribución de tipos de entidades")
    print("   • Estadísticas de conectividad")
    print("   • Información de comunidades")

def main():
    """Función principal de demostración"""
    
    print("🎨 MICROSOFT GRAPH RAG - VISUALIZACIÓN INTERACTIVA")
    print("=" * 70)
    
    # Demo principal
    success = demo_graph_visualization()
    
    # Explicación de tipos
    if success:
        demo_graph_types()
    
    # Conclusión
    print("\n" + "=" * 70)
    if success:
        print("🎉 ¡VISUALIZACIONES GENERADAS EXITOSAMENTE!")
        print("✅ Grafos interactivos HTML disponibles")
        print("✅ Múltiples perspectivas del knowledge graph")
        print("✅ Análisis de comunidades visualizado")
        print("✅ Métricas de centralidad calculadas")
        print("")
        print("📂 Archivos en: data/graphs/")
        print("🌐 Abrir archivos .html en navegador")
        print("🎯 Explorar interactivamente con mouse/teclado")
    else:
        print("❌ Error en generación de visualizaciones")

if __name__ == "__main__":
    main() 