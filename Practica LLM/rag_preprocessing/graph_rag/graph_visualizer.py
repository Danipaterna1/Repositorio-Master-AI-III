"""
Graph Visualizer for Microsoft Graph RAG
========================================

Visualizador interactivo de knowledge graphs usando pyvis y networkx.
Permite visualizar entidades, relaciones y comunidades detectadas.
"""

import logging
from typing import List, Dict, Any, Optional
import networkx as nx
from pyvis.network import Network
import os
from pathlib import Path

from .entity_extractor import Entity, Relationship, EntityType, RelationType
from .community_detector import Community, CommunityHierarchy
from .community_summarizer import CommunityReport

logger = logging.getLogger(__name__)

class GraphVisualizer:
    """
    Visualizador de knowledge graphs para Graph RAG
    """
    
    def __init__(self, output_dir: str = "data/graphs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración de colores por tipo de entidad
        self.entity_colors = {
            EntityType.PERSON: "#FF6B6B",          # Rojo suave
            EntityType.ORGANIZATION: "#4ECDC4",    # Turquesa
            EntityType.LOCATION: "#45B7D1",        # Azul
            EntityType.TECHNOLOGY: "#96CEB4",      # Verde menta
            EntityType.CONCEPT: "#FFEAA7",         # Amarillo
            EntityType.EVENT: "#DDA0DD",           # Lavanda
            EntityType.PRODUCT: "#F4A460",         # Naranja arena
            EntityType.OTHER: "#D3D3D3"            # Gris claro
        }
        
        # Configuración de colores por tipo de relación
        self.relation_colors = {
            RelationType.WORKS_FOR: "#FF4757",
            RelationType.CREATES: "#2ED573",
            RelationType.USES: "#3742FA",
            RelationType.COLLABORATES: "#FF6348",
            RelationType.LEADS: "#A4B0BE",
            RelationType.RELATED_TO: "#57606F",
            RelationType.OTHER: "#747D8C"
        }
        
        logger.info(f"GraphVisualizer initialized, output dir: {self.output_dir}")
    
    def visualize_knowledge_graph(self, entities: List[Entity], relationships: List[Relationship], 
                                title: str = "Knowledge Graph", filename: str = "knowledge_graph.html") -> str:
        """
        Visualiza el knowledge graph completo
        
        Args:
            entities: Lista de entidades
            relationships: Lista de relaciones
            title: Título del grafo
            filename: Nombre del archivo HTML
            
        Returns:
            Ruta del archivo HTML generado
        """
        # Crear network de pyvis
        net = Network(
            height="800px", 
            width="100%", 
            bgcolor="#ffffff", 
            font_color="black",
            directed=True
        )
        
        # Configuración de física para mejor visualización
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09
            }
          }
        }
        """)
        
        # Agregar entidades como nodos
        entity_dict = {}
        for entity in entities:
            color = self.entity_colors.get(entity.type, self.entity_colors[EntityType.OTHER])
            size = max(20, min(50, len(entity.name) * 2))  # Tamaño basado en longitud del nombre
            
            # Tooltip con información detallada
            tooltip = f"""
            <b>{entity.name}</b><br>
            Tipo: {entity.type.value}<br>
            Confianza: {entity.confidence:.2f}<br>
            Descripción: {entity.description[:100]}...
            """
            
            net.add_node(
                entity.id,
                label=entity.name,
                color=color,
                size=size,
                title=tooltip,
                shape="dot"
            )
            entity_dict[entity.id] = entity
        
        # Agregar relaciones como edges
        for relationship in relationships:
            if (relationship.source_entity_id in entity_dict and 
                relationship.target_entity_id in entity_dict):
                
                color = self.relation_colors.get(relationship.relation_type, 
                                                self.relation_colors[RelationType.OTHER])
                
                # Tooltip para la relación
                tooltip = f"""
                <b>{relationship.relation_type.value}</b><br>
                Confianza: {relationship.confidence:.2f}<br>
                Descripción: {relationship.description}
                """
                
                net.add_edge(
                    relationship.source_entity_id,
                    relationship.target_entity_id,
                    label=relationship.relation_type.value,
                    color=color,
                    title=tooltip,
                    width=2
                )
        
        # Guardar archivo
        output_path = self.output_dir / filename
        net.save_graph(str(output_path))
        
        logger.info(f"Knowledge graph saved to: {output_path}")
        return str(output_path)
    
    def visualize_communities(self, hierarchy: CommunityHierarchy, entities: List[Entity], 
                            reports: List[CommunityReport] = None,
                            filename: str = "communities_graph.html") -> str:
        """
        Visualiza comunidades detectadas con colores distintivos
        
        Args:
            hierarchy: Jerarquía de comunidades
            entities: Lista de entidades
            reports: Reportes de comunidades (opcional)
            filename: Nombre del archivo HTML
            
        Returns:
            Ruta del archivo HTML generado
        """
        net = Network(
            height="800px", 
            width="100%", 
            bgcolor="#f0f0f0", 
            font_color="black",
            directed=False  # Para comunidades usamos grafo no dirigido
        )
        
        # Configuración de física optimizada para comunidades
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 150},
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08,
              "damping": 0.4
            },
            "solver": "forceAtlas2Based"
          }
        }
        """)
        
        # Crear diccionario de entidades
        entity_dict = {entity.id: entity for entity in entities}
        
        # Generar colores únicos para cada comunidad
        community_colors = self._generate_community_colors(hierarchy)
        
        # Procesar cada nivel de la jerarquía
        for level, communities in hierarchy.communities_by_level.items():
            for comm_idx, community in enumerate(communities):
                color = community_colors.get(community.id, "#999999")
                
                # Agregar nodos de la comunidad
                for entity_id in community.entities:
                    if entity_id in entity_dict:
                        entity = entity_dict[entity_id]
                        
                        # Encontrar reporte de comunidad si existe
                        report_info = ""
                        if reports:
                            for report in reports:
                                if report.community_id == community.id:
                                    report_info = f"<br>Comunidad: {report.title}<br>Importancia: {report.importance_score:.2f}"
                                    break
                        
                        tooltip = f"""
                        <b>{entity.name}</b><br>
                        Tipo: {entity.type.value}<br>
                        Nivel: {level}<br>
                        Comunidad: {comm_idx + 1}{report_info}
                        """
                        
                        # Tamaño basado en importancia de la comunidad
                        size = max(15, min(40, community.size * 3))
                        
                        net.add_node(
                            entity_id,
                            label=entity.name,
                            color=color,
                            size=size,
                            title=tooltip,
                            shape="dot"
                        )
                
                # Agregar edges dentro de la comunidad
                for rel_id in community.relationships:
                    # Buscar la relación correspondiente
                    # Nota: En una implementación completa, tendríamos acceso directo a las relaciones
                    pass
        
        # Agregar conexiones entre entidades de la misma comunidad
        for level, communities in hierarchy.communities_by_level.items():
            for community in communities:
                entities_in_community = list(community.entities)
                
                # Conectar todas las entidades dentro de la comunidad
                for i, entity1 in enumerate(entities_in_community):
                    for entity2 in entities_in_community[i+1:]:
                        if entity1 in entity_dict and entity2 in entity_dict:
                            net.add_edge(
                                entity1,
                                entity2,
                                color="#cccccc",
                                width=1,
                                title=f"Misma comunidad (Nivel {level})"
                            )
        
        # Guardar archivo
        output_path = self.output_dir / filename
        net.save_graph(str(output_path))
        
        logger.info(f"Community graph saved to: {output_path}")
        return str(output_path)
    
    def visualize_networkx_graph(self, entities: List[Entity], relationships: List[Relationship],
                               filename: str = "networkx_graph.html") -> str:
        """
        Crea visualización usando NetworkX como backend
        
        Args:
            entities: Lista de entidades
            relationships: Lista de relaciones
            filename: Nombre del archivo HTML
            
        Returns:
            Ruta del archivo HTML generado
        """
        # Crear grafo NetworkX
        G = nx.DiGraph()
        
        # Agregar nodos (entidades)
        for entity in entities:
            G.add_node(
                entity.id,
                name=entity.name,
                type=entity.type.value,
                confidence=entity.confidence
            )
        
        # Agregar edges (relaciones)
        for rel in relationships:
            if G.has_node(rel.source_entity_id) and G.has_node(rel.target_entity_id):
                G.add_edge(
                    rel.source_entity_id,
                    rel.target_entity_id,
                    relation=rel.relation_type.value,
                    confidence=rel.confidence
                )
        
        # Calcular métricas de centralidad
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        # Crear visualización pyvis desde NetworkX
        net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black")
        
        # Agregar nodos con métricas
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            entity_type = EntityType(node_data.get('type', 'OTHER'))
            color = self.entity_colors.get(entity_type, self.entity_colors[EntityType.OTHER])
            
            # Tamaño basado en centralidad
            size = max(15, centrality.get(node_id, 0) * 100)
            
            tooltip = f"""
            <b>{node_data['name']}</b><br>
            Tipo: {node_data['type']}<br>
            Centralidad: {centrality.get(node_id, 0):.3f}<br>
            Betweenness: {betweenness.get(node_id, 0):.3f}
            """
            
            net.add_node(
                node_id,
                label=node_data['name'],
                color=color,
                size=size,
                title=tooltip
            )
        
        # Agregar edges
        for edge in G.edges(data=True):
            source, target, data = edge
            relation_type = RelationType(data.get('relation', 'OTHER'))
            color = self.relation_colors.get(relation_type, self.relation_colors[RelationType.OTHER])
            
            net.add_edge(
                source,
                target,
                label=data['relation'],
                color=color,
                title=f"Relación: {data['relation']}"
            )
        
        # Guardar archivo
        output_path = self.output_dir / filename
        net.save_graph(str(output_path))
        
        logger.info(f"NetworkX graph saved to: {output_path}")
        return str(output_path)
    
    def create_graph_summary(self, entities: List[Entity], relationships: List[Relationship],
                           communities: CommunityHierarchy = None, reports: List[CommunityReport] = None) -> Dict[str, Any]:
        """
        Crea resumen estadístico del grafo para acompañar visualizaciones
        """
        # Crear grafo NetworkX para análisis
        G = nx.DiGraph()
        
        for entity in entities:
            G.add_node(entity.id, type=entity.type.value)
        
        for rel in relationships:
            if G.has_node(rel.source_entity_id) and G.has_node(rel.target_entity_id):
                G.add_edge(rel.source_entity_id, rel.target_entity_id)
        
        # Calcular métricas
        summary = {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "density": nx.density(G),
            "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            "connected_components": nx.number_weakly_connected_components(G),
            "entity_type_distribution": {}
        }
        
        # Distribución de tipos de entidades
        for entity in entities:
            entity_type = entity.type.value
            summary["entity_type_distribution"][entity_type] = summary["entity_type_distribution"].get(entity_type, 0) + 1
        
        # Información de comunidades si está disponible
        if communities:
            summary["communities"] = {
                "total_communities": communities.total_communities,
                "levels": communities.total_levels,
                "largest_community_size": max([len(comm.entities) for level_comms in communities.communities_by_level.values() for comm in level_comms], default=0)
            }
        
        return summary
    
    def _generate_community_colors(self, hierarchy: CommunityHierarchy) -> Dict[str, str]:
        """Genera colores únicos para cada comunidad"""
        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#F4A460", "#98D8C8", "#F7DC6F", "#BB8FCE",
            "#85C1E9", "#F8C471", "#82E0AA", "#F1948A", "#85929E"
        ]
        
        community_colors = {}
        color_idx = 0
        
        for level, communities in hierarchy.communities_by_level.items():
            for community in communities:
                community_colors[community.id] = colors[color_idx % len(colors)]
                color_idx += 1
        
        return community_colors
    
    def generate_visualization_report(self, entities: List[Entity], relationships: List[Relationship],
                                    communities: CommunityHierarchy = None, reports: List[CommunityReport] = None) -> Dict[str, str]:
        """
        Genera todas las visualizaciones disponibles
        
        Returns:
            Diccionario con rutas de archivos generados
        """
        results = {}
        
        # 1. Knowledge Graph básico
        results["knowledge_graph"] = self.visualize_knowledge_graph(
            entities, relationships, "Knowledge Graph Completo"
        )
        
        # 2. Visualización con NetworkX
        results["networkx_graph"] = self.visualize_networkx_graph(
            entities, relationships
        )
        
        # 3. Visualización de comunidades (si están disponibles)
        if communities:
            results["communities_graph"] = self.visualize_communities(
                communities, entities, reports
            )
        
        # 4. Resumen estadístico
        summary = self.create_graph_summary(entities, relationships, communities, reports)
        summary_path = self.output_dir / "graph_summary.json"
        
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        results["summary"] = str(summary_path)
        
        logger.info(f"Generated {len(results)} visualization files")
        return results 