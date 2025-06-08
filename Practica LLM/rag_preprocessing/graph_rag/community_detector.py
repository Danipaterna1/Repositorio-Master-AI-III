"""
Community Detector for Microsoft Graph RAG
==========================================

Implementa detección de comunidades jerárquicas usando el algoritmo Leiden.
Esta es la pieza central del enfoque Microsoft Graph RAG que permite:

1. Detectar comunidades densamente conectadas
2. Crear jerarquías multinivel (Level 0, 1, 2...)
3. Permitir query-focused summarization

Basado en el paper Leiden de Traag et al. (2019)
"""

import logging
import time
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
try:
    import community.community_louvain as community_louvain  # python-louvain
    HAS_LOUVAIN = True
except ImportError:
    try:
        import community as community_louvain  # fallback import
        HAS_LOUVAIN = True
    except ImportError:
        HAS_LOUVAIN = False
        logging.warning("python-louvain not available")

try:
    import igraph as ig
    import leidenalg
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False
    logging.warning("igraph/leidenalg not available - using Louvain instead of Leiden")

from .entity_extractor import Entity, Relationship, ExtractionResult

logger = logging.getLogger(__name__)

@dataclass
class Community:
    """Representa una comunidad de entidades relacionadas"""
    id: str
    level: int
    entities: Set[str]  # Entity IDs
    relationships: Set[str]  # Relationship IDs
    parent_community_id: Optional[str] = None
    child_communities: Set[str] = field(default_factory=set)
    size: int = 0
    modularity_score: float = 0.0
    description: str = ""
    summary: str = ""  # Se llenará por CommunityReportGenerator
    
    def __post_init__(self):
        self.size = len(self.entities)

@dataclass
class CommunityHierarchy:
    """Jerarquía completa de comunidades detectadas"""
    communities_by_level: Dict[int, List[Community]]
    total_levels: int
    modularity_by_level: Dict[int, float]
    detection_algorithm: str
    processing_time: float
    total_communities: int
    
    def get_level_communities(self, level: int) -> List[Community]:
        """Obtiene todas las comunidades de un nivel específico"""
        return self.communities_by_level.get(level, [])
    
    def get_community_by_id(self, community_id: str) -> Optional[Community]:
        """Busca una comunidad por su ID en todos los niveles"""
        for communities in self.communities_by_level.values():
            for community in communities:
                if community.id == community_id:
                    return community
        return None

class CommunityDetector:
    """
    Detector de comunidades jerárquicas usando algoritmo Leiden (o Louvain fallback)
    
    Implementa el enfoque de Microsoft Graph RAG para detectar estructuras
    modulares en el knowledge graph.
    """
    
    def __init__(self, algorithm: str = "leiden", random_state: int = 42):
        """
        Args:
            algorithm: 'leiden' (preferido) o 'louvain' (fallback)
            random_state: Semilla para reproducibilidad
        """
        # Determinar algoritmo disponible
        if algorithm == "leiden" and HAS_IGRAPH:
            self.algorithm = "leiden"
        elif HAS_LOUVAIN:
            self.algorithm = "louvain"
            if algorithm == "leiden":
                logger.warning("Leiden algorithm requires igraph/leidenalg. Falling back to Louvain.")
        else:
            logger.error("No community detection algorithms available")
            raise ImportError("Install python-louvain or igraph+leidenalg")
        
        self.random_state = random_state
        
        # Métricas de detección
        self.metrics = {
            "graphs_processed": 0,
            "total_communities_detected": 0,
            "avg_modularity": 0.0,
            "avg_processing_time": 0.0,
            "avg_levels_detected": 0.0
        }
        
        logger.info(f"CommunityDetector initialized with {self.algorithm} algorithm")
    
    def detect_communities(self, 
                          entities: List[Entity], 
                          relationships: List[Relationship],
                          max_levels: int = 3,
                          min_community_size: int = 2) -> CommunityHierarchy:
        """
        Detecta comunidades jerárquicas en el knowledge graph
        
        Args:
            entities: Lista de entidades extraídas
            relationships: Lista de relaciones extraídas
            max_levels: Número máximo de niveles jerárquicos
            min_community_size: Tamaño mínimo de comunidad
            
        Returns:
            CommunityHierarchy con comunidades detectadas por nivel
        """
        start_time = time.time()
        
        logger.info(f"Detecting communities with {self.algorithm} algorithm")
        logger.info(f"Input: {len(entities)} entities, {len(relationships)} relationships")
        
        # 1. Construir grafo NetworkX
        graph = self._build_networkx_graph(entities, relationships)
        
        if graph.number_of_nodes() == 0:
            logger.warning("Empty graph - no communities to detect")
            return self._create_empty_hierarchy(start_time)
        
        # 2. Detectar comunidades jerárquicas
        if self.algorithm == "leiden":
            hierarchy = self._detect_leiden_communities(graph, entities, relationships, 
                                                       max_levels, min_community_size)
        else:
            hierarchy = self._detect_louvain_communities(graph, entities, relationships,
                                                        max_levels, min_community_size)
        
        processing_time = time.time() - start_time
        hierarchy.processing_time = processing_time
        
        # 3. Actualizar métricas
        self._update_metrics(hierarchy, processing_time)
        
        logger.info(f"Community detection completed in {processing_time:.2f}s")
        logger.info(f"Detected {hierarchy.total_communities} communities across {hierarchy.total_levels} levels")
        
        return hierarchy
    
    def _build_networkx_graph(self, entities: List[Entity], relationships: List[Relationship]) -> nx.Graph:
        """Construye grafo NetworkX desde entidades y relaciones"""
        G = nx.Graph()
        
        # Añadir nodos (entidades)
        for entity in entities:
            G.add_node(entity.id, 
                      name=entity.name,
                      type=entity.type.value,
                      description=entity.description)
        
        # Añadir aristas (relaciones)
        for relationship in relationships:
            if relationship.source_entity_id in G.nodes and relationship.target_entity_id in G.nodes:
                G.add_edge(relationship.source_entity_id, 
                          relationship.target_entity_id,
                          weight=relationship.confidence,
                          type=relationship.relation_type.value,
                          description=relationship.description)
        
        logger.info(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def _detect_leiden_communities(self, graph: nx.Graph, entities: List[Entity], 
                                  relationships: List[Relationship], max_levels: int,
                                  min_community_size: int) -> CommunityHierarchy:
        """Detecta comunidades usando algoritmo Leiden (más avanzado)"""
        try:
            # Convertir NetworkX a igraph
            ig_graph = self._networkx_to_igraph(graph)
            
            communities_by_level = {}
            modularity_by_level = {}
            
            current_graph = ig_graph
            level = 0
            
            while level < max_levels:
                # Ejecutar Leiden con leidenalg
                partition = leidenalg.find_partition(
                    current_graph, 
                    leidenalg.ModularityVertexPartition,
                    seed=self.random_state
                )
                
                modularity = partition.modularity
                modularity_by_level[level] = modularity
                
                # Crear objetos Community
                level_communities = self._create_communities_from_partition(
                    partition, current_graph, level, min_community_size
                )
                
                if not level_communities or len(level_communities) <= 1:
                    logger.info(f"Stopping at level {level}: no meaningful communities found")
                    break
                
                communities_by_level[level] = level_communities
                
                # Crear grafo para siguiente nivel (super-nodes)
                if level < max_levels - 1:
                    current_graph = self._create_super_graph(current_graph, partition)
                
                level += 1
                
                logger.info(f"Level {level-1}: {len(level_communities)} communities, modularity: {modularity:.3f}")
            
            return CommunityHierarchy(
                communities_by_level=communities_by_level,
                total_levels=level,
                modularity_by_level=modularity_by_level,
                detection_algorithm="leiden",
                processing_time=0.0,  # Se actualizará
                total_communities=sum(len(communities) for communities in communities_by_level.values())
            )
            
        except Exception as e:
            logger.error(f"Leiden detection failed: {e}")
            # Fallback a Louvain
            return self._detect_louvain_communities(graph, entities, relationships, max_levels, min_community_size)
    
    def _detect_louvain_communities(self, graph: nx.Graph, entities: List[Entity],
                                   relationships: List[Relationship], max_levels: int,
                                   min_community_size: int) -> CommunityHierarchy:
        """Detecta comunidades usando algoritmo Louvain (fallback)"""
        communities_by_level = {}
        modularity_by_level = {}
        
        # Ejecutar Louvain (solo un nivel)
        if HAS_LOUVAIN:
            partition = community_louvain.best_partition(graph, random_state=self.random_state)
            modularity = community_louvain.modularity(partition, graph)
        else:
            logger.error("Louvain algorithm not available")
            raise ImportError("Install python-louvain for community detection")
        
        modularity_by_level[0] = modularity
        
        # Convertir partition a objetos Community
        communities_dict = defaultdict(list)
        for node_id, community_id in partition.items():
            communities_dict[community_id].append(node_id)
        
        level_communities = []
        for comm_id, node_ids in communities_dict.items():
            if len(node_ids) >= min_community_size:
                community = Community(
                    id=f"level_0_community_{comm_id}",
                    level=0,
                    entities=set(node_ids),
                    relationships=set(),  # Se puede mejorar
                    size=len(node_ids),
                    modularity_score=modularity
                )
                level_communities.append(community)
        
        communities_by_level[0] = level_communities
        
        logger.info(f"Louvain: {len(level_communities)} communities, modularity: {modularity:.3f}")
        
        return CommunityHierarchy(
            communities_by_level=communities_by_level,
            total_levels=1,
            modularity_by_level=modularity_by_level,
            detection_algorithm="louvain",
            processing_time=0.0,  # Se actualizará
            total_communities=len(level_communities)
        )
    
    def _networkx_to_igraph(self, nx_graph: nx.Graph) -> 'ig.Graph':
        """Convierte grafo NetworkX a igraph"""
        # Crear mapeo de IDs a índices
        node_list = list(nx_graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Crear lista de aristas para igraph
        edges = []
        weights = []
        
        for source, target, data in nx_graph.edges(data=True):
            edges.append((node_to_idx[source], node_to_idx[target]))
            weights.append(data.get('weight', 1.0))
        
        # Crear igraph
        ig_graph = ig.Graph(edges)
        ig_graph.es['weight'] = weights
        
        # Añadir atributos de nodos
        for attr in ['name', 'type', 'description']:
            if attr in nx_graph.nodes[node_list[0]]:
                ig_graph.vs[attr] = [nx_graph.nodes[node].get(attr, '') for node in node_list]
        
        # Guardar mapeo original
        ig_graph.vs['original_id'] = node_list
        
        return ig_graph
    
    def _create_communities_from_partition(self, partition, graph, level: int, 
                                         min_size: int) -> List[Community]:
        """Crea objetos Community desde partición de igraph"""
        communities = []
        
        for i, community_nodes in enumerate(partition):
            if len(community_nodes) >= min_size:
                # Obtener IDs originales de las entidades
                original_ids = [graph.vs[node]['original_id'] for node in community_nodes]
                
                community = Community(
                    id=f"level_{level}_community_{i}",
                    level=level,
                    entities=set(original_ids),
                    relationships=set(),  # TODO: extraer relaciones específicas
                    size=len(original_ids),
                    modularity_score=partition.modularity,
                    description=f"Community at level {level} with {len(original_ids)} entities"
                )
                communities.append(community)
        
        return communities
    
    def _create_super_graph(self, graph, partition) -> 'ig.Graph':
        """Crea super-grafo para siguiente nivel jerárquico"""
        # Crear grafo donde cada comunidad es un nodo
        community_graph = partition.cluster_graph()
        return community_graph
    
    def _create_empty_hierarchy(self, start_time: float) -> CommunityHierarchy:
        """Crea jerarquía vacía cuando no hay datos"""
        return CommunityHierarchy(
            communities_by_level={},
            total_levels=0,
            modularity_by_level={},
            detection_algorithm=self.algorithm,
            processing_time=time.time() - start_time,
            total_communities=0
        )
    
    def _update_metrics(self, hierarchy: CommunityHierarchy, processing_time: float):
        """Actualiza métricas de detección"""
        self.metrics["graphs_processed"] += 1
        self.metrics["total_communities_detected"] += hierarchy.total_communities
        self.metrics["avg_processing_time"] = (
            (self.metrics["avg_processing_time"] * (self.metrics["graphs_processed"] - 1) + processing_time)
            / self.metrics["graphs_processed"]
        )
        self.metrics["avg_levels_detected"] = (
            (self.metrics["avg_levels_detected"] * (self.metrics["graphs_processed"] - 1) + hierarchy.total_levels)
            / self.metrics["graphs_processed"]
        )
        
        # Calcular modularity promedio
        if hierarchy.modularity_by_level:
            avg_modularity = sum(hierarchy.modularity_by_level.values()) / len(hierarchy.modularity_by_level)
            self.metrics["avg_modularity"] = (
                (self.metrics["avg_modularity"] * (self.metrics["graphs_processed"] - 1) + avg_modularity)
                / self.metrics["graphs_processed"]
            )
    
    def get_detection_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de detección de comunidades"""
        return {
            **self.metrics,
            "algorithm_used": self.algorithm,
            "has_igraph": HAS_IGRAPH
        }
    
    def reset_metrics(self):
        """Resetea métricas de detección"""
        self.metrics = {
            "graphs_processed": 0,
            "total_communities_detected": 0,
            "avg_modularity": 0.0,
            "avg_processing_time": 0.0,
            "avg_levels_detected": 0.0
        }
        logger.info("Community detection metrics reset")
    
    def visualize_communities(self, hierarchy: CommunityHierarchy, 
                            entities: List[Entity], 
                            output_file: str = "communities.html") -> str:
        """
        Crea visualización interactiva de las comunidades detectadas
        
        Args:
            hierarchy: Jerarquía de comunidades
            entities: Lista de entidades para contexto
            output_file: Archivo de salida HTML
            
        Returns:
            Path del archivo generado
        """
        try:
            from pyvis.network import Network
            
            # Crear visualización para nivel 0 (más granular)
            level_0_communities = hierarchy.get_level_communities(0)
            
            if not level_0_communities:
                logger.warning("No communities to visualize")
                return None
            
            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
            
            # Mapear entidades por ID
            entity_map = {entity.id: entity for entity in entities}
            
            # Colores para comunidades
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", 
                     "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"]
            
            # Añadir nodos coloreados por comunidad
            for i, community in enumerate(level_0_communities):
                color = colors[i % len(colors)]
                
                for entity_id in community.entities:
                    if entity_id in entity_map:
                        entity = entity_map[entity_id]
                        net.add_node(entity_id, 
                                   label=entity.name,
                                   title=f"Community {i}: {entity.type.value}\n{entity.description}",
                                   color=color,
                                   size=20)
            
            # TODO: Añadir aristas basadas en relaciones
            
            net.barnes_hut()
            net.save_graph(output_file)
            
            logger.info(f"Community visualization saved to {output_file}")
            return output_file
            
        except ImportError:
            logger.error("pyvis not available for visualization")
            return None
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return None 