"""
Microsoft Graph RAG Pipeline
============================

Pipeline principal que orquesta todo el flujo de Graph RAG.
Sprint 3 Update: LLM-enhanced community summarization.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from .entity_extractor import EntityExtractor
from .community_detector import CommunityDetector
from .community_summarizer import SyncCommunitySummarizer, LLMCommunitySummarizer
from .graph_builder import GraphBuilder
from .query_engine import GraphRAGQueryEngine

logger = logging.getLogger(__name__)

@dataclass
class GraphRAGConfig:
    """Configuración para Graph RAG Pipeline"""
    max_community_levels: int = 3
    min_community_size: int = 2
    use_llm_extraction: bool = False
    use_llm_summarization: bool = True  # Nueva opción Sprint 3
    community_algorithm: str = "leiden"

class GraphRAGPipeline:
    """
    Pipeline principal de Microsoft Graph RAG
    Sprint 3: LLM-enhanced community summarization
    """
    
    def __init__(self, config: GraphRAGConfig = None):
        self.config = config or GraphRAGConfig()
        
        # Componentes del pipeline
        self.entity_extractor = EntityExtractor()
        self.community_detector = CommunityDetector(algorithm=self.config.community_algorithm)
        
        # Sprint 3: LLM-enhanced summarizer
        self.community_summarizer = SyncCommunitySummarizer(use_llm=self.config.use_llm_summarization)
        
        self.graph_builder = GraphBuilder()
        self.query_engine = GraphRAGQueryEngine()
        
        summarization_mode = "LLM-enhanced" if self.config.use_llm_summarization else "statistical"
        logger.info(f"GraphRAGPipeline initialized with {summarization_mode} community summarization")
    
    def process_documents(self, documents: List[str]) -> Dict[str, Any]:
        """
        Procesa documentos a través del pipeline completo
        
        Args:
            documents: Lista de textos a procesar
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        logger.info(f"Processing {len(documents)} documents through Graph RAG pipeline")
        
        # Placeholder: procesar solo el primer documento por ahora
        # TODO: Implementar batch processing en Sprint 3
        
        if not documents:
            logger.warning("No documents to process")
            return {"entities": [], "communities": [], "status": "empty"}
        
        text = documents[0]
        
        # 1. Extraer entidades y relaciones
        extraction_result = self.entity_extractor.extract_entities_and_relationships(
            text, use_llm_fallback=self.config.use_llm_extraction
        )
        
        # 2. Detectar comunidades
        hierarchy = self.community_detector.detect_communities(
            entities=extraction_result.entities,
            relationships=extraction_result.relationships,
            max_levels=self.config.max_community_levels,
            min_community_size=self.config.min_community_size
        )
        
        # 3. Generar resúmenes con LLM enhancement
        reports = self.community_summarizer.generate_community_reports(
            hierarchy,
            entities=extraction_result.entities,
            relationships=extraction_result.relationships
        )
        
        # 4. Construir knowledge graph
        knowledge_graph = self.graph_builder.build_knowledge_graph(
            extraction_result.entities, 
            extraction_result.relationships
        )
        
        # 5. Generar resumen jerárquico
        hierarchical_summary = self.community_summarizer.generate_hierarchical_summary(reports)
        
        # 6. Configurar query engine con contexto
        entities_dict = {entity.id: entity for entity in extraction_result.entities}
        self.query_engine.set_context(reports, hierarchical_summary, entities_dict)
        
        # Sprint 3: Estadísticas LLM enhancement
        llm_enhanced_reports = sum(1 for r in reports if hasattr(r, 'llm_enhanced') and r.llm_enhanced)
        
        return {
            "entities": len(extraction_result.entities),
            "relationships": len(extraction_result.relationships),
            "communities": hierarchy.total_communities,
            "levels": hierarchy.total_levels,
            "reports": len(reports),
            "llm_enhanced_reports": llm_enhanced_reports,  # Nueva métrica Sprint 3
            "hierarchical_summary": hierarchical_summary,
            "processing_time": extraction_result.processing_time + hierarchy.processing_time,
            "status": "completed",
            # Detalles adicionales Sprint 2
            "extraction_method": extraction_result.method_used,
            "community_algorithm": hierarchy.detection_algorithm,
            "top_communities": [r.title for r in reports[:3]] if reports else [],
            # Nueva info Sprint 3
            "summarization_mode": "LLM-enhanced" if self.config.use_llm_summarization else "statistical"
        }
    
    def query(self, question: str, search_type: str = "hybrid") -> Dict[str, Any]:
        """
        Realiza consulta usando Graph RAG
        
        Args:
            question: Pregunta del usuario
            search_type: Tipo de búsqueda ('local', 'global', 'hybrid')
            
        Returns:
            Resultado de la consulta
        """
        result = self.query_engine.query(question, search_type)
        
        return {
            "question": result.query,
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
            "search_type": result.search_type
        } 