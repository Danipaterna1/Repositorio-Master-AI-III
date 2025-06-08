"""
Query Engine for Microsoft Graph RAG
====================================

Motor de consultas que combina búsqueda local y global.
Placeholder básico para Sprint 2, se implementará completamente en Sprint 3.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Resultado de una consulta Graph RAG"""
    query: str
    answer: str
    sources: List[str]
    confidence: float
    search_type: str  # 'local', 'global', 'hybrid'

class LocalSearch:
    """Búsqueda local en Graph RAG - busca en comunidades específicas"""
    
    def __init__(self):
        self.community_reports = []
        self.entities = {}
        logger.info("LocalSearch initialized (Sprint 2 functional implementation)")
    
    def set_context(self, community_reports: List, entities: Dict):
        """Establece contexto de comunidades y entidades"""
        self.community_reports = community_reports
        self.entities = entities
    
    def search(self, query: str) -> QueryResult:
        """Búsqueda local funcional en comunidades específicas"""
        query_lower = query.lower()
        
        # Encontrar comunidades relevantes
        relevant_communities = []
        for report in self.community_reports:
            # Buscar en títulos y keywords
            if (any(keyword in query_lower for keyword in report.theme_keywords) or
                any(keyword in report.title.lower() for keyword in query_lower.split())):
                relevant_communities.append(report)
        
        if not relevant_communities:
            # Fallback: usar comunidades más importantes
            relevant_communities = sorted(self.community_reports, 
                                        key=lambda r: r.importance_score, reverse=True)[:2]
        
        # Generar respuesta basada en comunidades relevantes
        if relevant_communities:
            primary_community = relevant_communities[0]
            answer = self._generate_local_answer(query, primary_community, relevant_communities[:3])
            sources = [f"Comunidad: {c.title}" for c in relevant_communities[:3]]
            confidence = min(0.9, 0.6 + primary_community.importance_score * 0.3)
        else:
            answer = "No se encontraron comunidades específicas relevantes para esta consulta."
            sources = []
            confidence = 0.3
        
        return QueryResult(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence,
            search_type="local"
        )
    
    def _generate_local_answer(self, query: str, primary_community, all_communities: List) -> str:
        """Genera respuesta basada en comunidades locales"""
        
        # Analizar tipo de pregunta
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['qué', 'cuál', 'cómo']):
            # Pregunta descriptiva
            answer = f"Según la {primary_community.title}, {primary_community.summary}"
            
            if len(all_communities) > 1:
                answer += f" Además, se identificaron {len(all_communities)-1} comunidades relacionadas."
                
        elif any(word in query_lower for word in ['quién', 'quien']):
            # Pregunta sobre personas/organizaciones
            entities = []
            for community in all_communities:
                for entity_id in community.key_entities[:2]:
                    if entity_id in self.entities:
                        entities.append(self.entities[entity_id].name)
            
            if entities:
                answer = f"Las entidades principales identificadas son: {', '.join(entities[:5])}."
            else:
                answer = f"En la {primary_community.title} se identificaron varias entidades relevantes."
                
        else:
            # Pregunta general
            answer = f"La información está organizada en la {primary_community.title}, que incluye {primary_community.summary}"
        
        return answer

class GlobalSearch:
    """Búsqueda global en Graph RAG - busca en resúmenes jerárquicos"""
    
    def __init__(self):
        self.hierarchical_summary = {}
        self.community_reports = []
        logger.info("GlobalSearch initialized (Sprint 2 functional implementation)")
    
    def set_context(self, hierarchical_summary: Dict, community_reports: List):
        """Establece contexto de resúmenes jerárquicos"""
        self.hierarchical_summary = hierarchical_summary
        self.community_reports = community_reports
    
    def search(self, query: str) -> QueryResult:
        """Búsqueda global funcional en resúmenes jerárquicos"""
        query_lower = query.lower()
        
        # Analizar si la pregunta es sobre tendencias globales o patrones
        is_global_query = any(word in query_lower for word in [
            'general', 'total', 'todos', 'todas', 'conjunto', 'resumen',
            'tendencia', 'patrón', 'distribución', 'estadística'
        ])
        
        if is_global_query and self.hierarchical_summary:
            answer = self._generate_global_summary_answer(query)
            sources = ["Resumen Jerárquico Global", "Estadísticas de Niveles"]
            confidence = 0.85
        else:
            # Buscar en temas globales
            global_themes = self.hierarchical_summary.get('overall_themes', [])
            query_words = query_lower.split()
            
            relevant_themes = [theme for theme in global_themes 
                             if any(word in theme.lower() for word in query_words)]
            
            if relevant_themes:
                answer = self._generate_theme_based_answer(query, relevant_themes)
                sources = [f"Tema Global: {theme}" for theme in relevant_themes[:3]]
                confidence = 0.75
            else:
                answer = self._generate_fallback_global_answer(query)
                sources = ["Análisis Global de Comunidades"]
                confidence = 0.6
        
        return QueryResult(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence,
            search_type="global"
        )
    
    def _generate_global_summary_answer(self, query: str) -> str:
        """Genera respuesta basada en resumen global"""
        stats = self.hierarchical_summary
        
        total_communities = stats.get('total_communities', 0)
        levels = stats.get('levels', 0)
        top_communities = stats.get('top_communities', [])
        overall_themes = stats.get('overall_themes', [])
        
        answer = f"A nivel global se identificaron {total_communities} comunidades organizadas en {levels} niveles jerárquicos."
        
        if top_communities:
            answer += f" Las comunidades más importantes son: {', '.join(top_communities[:3])}."
        
        if overall_themes:
            answer += f" Los temas predominantes incluyen: {', '.join(overall_themes[:3])}."
        
        return answer
    
    def _generate_theme_based_answer(self, query: str, relevant_themes: List[str]) -> str:
        """Genera respuesta basada en temas relevantes"""
        return f"Los temas globales relevantes para su consulta incluyen: {', '.join(relevant_themes)}. Estos temas aparecen consistentemente a través de múltiples comunidades en la jerarquía."
    
    def _generate_fallback_global_answer(self, query: str) -> str:
        """Respuesta global de fallback"""
        if self.community_reports:
            total_reports = len(self.community_reports)
            avg_size = sum(len(r.key_entities) for r in self.community_reports) / total_reports
            return f"El análisis global muestra {total_reports} comunidades con un tamaño promedio de {avg_size:.1f} entidades. La información está distribuida jerárquicamente para facilitar tanto consultas específicas como análisis de patrones generales."
        else:
            return "No hay suficiente información global disponible para responder esta consulta."

class GraphRAGQueryEngine:
    """
    Motor de consultas principal de Graph RAG
    Sprint 2: Implementación funcional completa
    """
    
    def __init__(self):
        self.local_search = LocalSearch()
        self.global_search = GlobalSearch()
        self.is_initialized = False
        logger.info("GraphRAGQueryEngine initialized (Sprint 2 functional implementation)")
    
    def set_context(self, community_reports: List, hierarchical_summary: Dict, entities: Dict):
        """Establece contexto para las búsquedas"""
        self.local_search.set_context(community_reports, entities)
        self.global_search.set_context(hierarchical_summary, community_reports)
        self.is_initialized = True
        logger.info("GraphRAGQueryEngine context set successfully")
    
    def query(self, question: str, search_type: str = "hybrid") -> QueryResult:
        """
        Procesa una consulta usando Graph RAG funcional
        
        Args:
            question: Pregunta del usuario
            search_type: 'local', 'global', o 'hybrid'
            
        Returns:
            QueryResult con la respuesta
        """
        if not self.is_initialized:
            logger.warning("QueryEngine not initialized with context - using fallback")
            return self._fallback_query(question, search_type)
        
        if search_type == "local":
            return self.local_search.search(question)
        elif search_type == "global":
            return self.global_search.search(question)
        else:
            # Hybrid: routing inteligente + combinación
            return self._intelligent_hybrid_search(question)
    
    def _intelligent_hybrid_search(self, question: str) -> QueryResult:
        """Búsqueda híbrida inteligente con routing automático"""
        question_lower = question.lower()
        
        # Determinar si la pregunta es mejor para local o global
        global_indicators = ['total', 'general', 'todos', 'resumen', 'conjunto', 'distribución']
        local_indicators = ['específico', 'particular', 'quién', 'cuál', 'dónde']
        
        is_global_biased = any(indicator in question_lower for indicator in global_indicators)
        is_local_biased = any(indicator in question_lower for indicator in local_indicators)
        
        # Ejecutar ambas búsquedas
        local_result = self.local_search.search(question)
        global_result = self.global_search.search(question)
        
        # Routing inteligente
        if is_global_biased and not is_local_biased:
            # Preferir resultado global
            primary_result = global_result
            secondary_result = local_result
            bias_weight = 0.7
        elif is_local_biased and not is_global_biased:
            # Preferir resultado local
            primary_result = local_result
            secondary_result = global_result
            bias_weight = 0.7
        else:
            # Balance equitativo
            primary_result = local_result if local_result.confidence > global_result.confidence else global_result
            secondary_result = global_result if primary_result == local_result else local_result
            bias_weight = 0.6
        
        # Combinar respuestas de manera inteligente
        combined_answer = self._combine_answers(question, primary_result, secondary_result, bias_weight)
        combined_sources = primary_result.sources + secondary_result.sources
        combined_confidence = (primary_result.confidence * bias_weight + 
                             secondary_result.confidence * (1 - bias_weight))
        
        return QueryResult(
            query=question,
            answer=combined_answer,
            sources=list(set(combined_sources)),  # Eliminar duplicados
            confidence=min(0.95, combined_confidence),
            search_type="hybrid"
        )
    
    def _combine_answers(self, question: str, primary: QueryResult, secondary: QueryResult, weight: float) -> str:
        """Combina respuestas de manera inteligente"""
        
        # Si las respuestas son muy diferentes, combinar
        if len(primary.answer) > 50 and len(secondary.answer) > 50:
            return f"{primary.answer} Complementariamente, {secondary.answer.lower()}"
        elif weight > 0.65:
            # Preferir respuesta primaria con mención secundaria
            return f"{primary.answer} (Información adicional disponible en {secondary.search_type} search)"
        else:
            # Balance equitativo
            return f"Desde perspectiva local: {primary.answer} Desde perspectiva global: {secondary.answer}"
    
    def _fallback_query(self, question: str, search_type: str) -> QueryResult:
        """Query de fallback cuando no hay contexto inicializado"""
        return QueryResult(
            query=question,
            answer=f"Graph RAG no inicializado. Para responder '{question}' se requiere procesar documentos primero.",
            sources=["System Message"],
            confidence=0.1,
            search_type=search_type
        ) 