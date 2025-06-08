"""
Community Summarizer for Microsoft Graph RAG
=============================================

Genera resúmenes jerárquicos funcionales de comunidades detectadas.
Sprint 3 Update: LLM-enhanced summarization con fallback estadístico.
"""

import logging
import asyncio
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
import os
from dotenv import load_dotenv

from .community_detector import Community, CommunityHierarchy
from .entity_extractor import Entity, Relationship, EntityType

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Optional LLM imports with fallback
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
    logger.info("Google Gemini integration available")
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Gemini not available - using statistical fallback only")

@dataclass
class CommunityReport:
    """Reporte de resumen enriquecido de una comunidad"""
    community_id: str
    level: int
    title: str
    summary: str
    key_entities: List[str]
    key_relationships: List[str]
    importance_score: float
    entity_types: Dict[str, int]  # Tipo -> count
    theme_keywords: List[str]
    size_category: str  # 'small', 'medium', 'large'
    llm_enhanced: bool = False  # Flag para indicar si se usó LLM

class LLMCommunitySummarizer:
    """
    Genera resúmenes de comunidades con LLM enhancement
    Sprint 3: Gemini + fallback estadístico para robustez
    """
    
    def __init__(self, use_llm: bool = True):
        self.entity_cache = {}  # Cache de entidades por ID
        self.relationship_cache = {}  # Cache de relaciones por ID
        self.use_llm = use_llm and GEMINI_AVAILABLE
        self.llm_client = None
        
        # Initialize LLM if available and requested
        if self.use_llm:
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key and api_key != 'your_google_api_key_here':
                try:
                    self.llm_client = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        google_api_key=api_key,
                        temperature=0.3,
                        max_tokens=500
                    )
                    logger.info("LLM-enhanced Community Summarizer initialized with Gemini")
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini: {e}. Using statistical fallback.")
                    self.use_llm = False
            else:
                logger.info("Google API key not configured. Using statistical fallback.")
                self.use_llm = False
        else:
            logger.info("LLM Community Summarizer initialized (statistical mode)")
    
    async def generate_community_reports(self, hierarchy: CommunityHierarchy, 
                                       entities: List[Entity] = None, 
                                       relationships: List[Relationship] = None) -> List[CommunityReport]:
        """
        Genera reportes de resumen para todas las comunidades con LLM enhancement
        
        Args:
            hierarchy: Jerarquía de comunidades detectadas
            entities: Lista de entidades para contexto
            relationships: Lista de relaciones para contexto
            
        Returns:
            Lista de reportes de comunidades enriquecidos
        """
        # Actualizar caches
        if entities:
            self.entity_cache = {entity.id: entity for entity in entities}
        if relationships:
            self.relationship_cache = {rel.id: rel for rel in relationships}
        
        reports = []
        
        for level, communities in hierarchy.communities_by_level.items():
            logger.info(f"Generating LLM-enhanced reports for level {level} with {len(communities)} communities")
            
            # Process communities in parallel if using async LLM
            if self.use_llm and self.llm_client:
                level_reports = await self._generate_llm_reports_async(communities, level)
            else:
                level_reports = self._generate_statistical_reports(communities, level)
                
            reports.extend(level_reports)
        
        # Ordenar por importancia
        reports.sort(key=lambda r: r.importance_score, reverse=True)
        
        llm_count = sum(1 for r in reports if r.llm_enhanced)
        logger.info(f"Generated {len(reports)} community reports ({llm_count} LLM-enhanced, {len(reports)-llm_count} statistical)")
        return reports
    
    async def _generate_llm_reports_async(self, communities: List[Community], level: int) -> List[CommunityReport]:
        """Genera reportes usando LLM de forma asíncrona"""
        tasks = []
        for i, community in enumerate(communities):
            task = self._generate_llm_enhanced_report(community, level, i)
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _generate_llm_enhanced_report(self, community: Community, level: int, index: int) -> CommunityReport:
        """Genera reporte mejorado con LLM"""
        try:
            # 1. Generar reporte base estadístico
            base_report = self._generate_statistical_report(community, level, index)
            
            # 2. Mejorar con LLM si está disponible
            if self.use_llm and self.llm_client:
                enhanced_title, enhanced_summary = await self._enhance_with_llm(community, base_report)
                
                return CommunityReport(
                    community_id=base_report.community_id,
                    level=base_report.level,
                    title=enhanced_title,
                    summary=enhanced_summary,
                    key_entities=base_report.key_entities,
                    key_relationships=base_report.key_relationships,
                    importance_score=base_report.importance_score,
                    entity_types=base_report.entity_types,
                    theme_keywords=base_report.theme_keywords,
                    size_category=base_report.size_category,
                    llm_enhanced=True
                )
            else:
                return base_report
                
        except Exception as e:
            logger.warning(f"LLM enhancement failed for community {community.id}: {e}. Using statistical fallback.")
            return self._generate_statistical_report(community, level, index)
    
    async def _enhance_with_llm(self, community: Community, base_report: CommunityReport) -> tuple[str, str]:
        """Mejora título y resumen usando LLM"""
        
        # Construir contexto para el LLM
        entity_details = []
        for entity_id in list(community.entities)[:5]:  # Top 5 entidades
            if entity_id in self.entity_cache:
                entity = self.entity_cache[entity_id]
                entity_details.append(f"- {entity.name} ({entity.type.value})")
        
        relationship_details = []
        for rel_id in list(community.relationships)[:3]:  # Top 3 relaciones
            if rel_id in self.relationship_cache:
                rel = self.relationship_cache[rel_id]
                relationship_details.append(f"- {rel.source} → {rel.target} ({rel.type})")
        
        context = f"""
Analiza la siguiente comunidad de entidades relacionadas:

ENTIDADES PRINCIPALES:
{chr(10).join(entity_details)}

RELACIONES PRINCIPALES:
{chr(10).join(relationship_details)}

ANÁLISIS ESTADÍSTICO:
- Tamaño: {community.size} entidades
- Tipos: {base_report.entity_types}
- Palabras clave: {base_report.theme_keywords}
- Título base: {base_report.title}

Genera:
1. Un título descriptivo y conciso (máximo 8 palabras)
2. Un resumen explicativo (máximo 100 palabras)

Formato:
TÍTULO: [tu título aquí]
RESUMEN: [tu resumen aquí]
"""

        try:
            response = await self.llm_client.ainvoke(context)
            response_text = response.content
            
            # Parsear respuesta
            lines = response_text.strip().split('\n')
            title = base_report.title  # fallback
            summary = base_report.summary  # fallback
            
            for line in lines:
                if line.startswith('TÍTULO:'):
                    title = line.replace('TÍTULO:', '').strip()
                elif line.startswith('RESUMEN:'):
                    summary = line.replace('RESUMEN:', '').strip()
            
            logger.debug(f"LLM enhanced community {community.id}: {title}")
            return title, summary
            
        except Exception as e:
            logger.warning(f"LLM API call failed: {e}")
            return base_report.title, base_report.summary
    
    def _generate_statistical_reports(self, communities: List[Community], level: int) -> List[CommunityReport]:
        """Genera reportes usando solo análisis estadístico"""
        reports = []
        for i, community in enumerate(communities):
            report = self._generate_statistical_report(community, level, i)
            reports.append(report)
        return reports
    
    def _generate_statistical_report(self, community: Community, level: int, index: int) -> CommunityReport:
        """Genera reporte usando análisis estadístico (método original)"""
        
        # 1. Analizar tipos de entidades
        entity_types = self._analyze_entity_types(community)
        
        # 2. Extraer palabras clave temáticas
        theme_keywords = self._extract_theme_keywords(community)
        
        # 3. Generar título inteligente
        title = self._generate_intelligent_title(community, entity_types, theme_keywords, index)
        
        # 4. Generar resumen descriptivo
        summary = self._generate_descriptive_summary(community, entity_types, theme_keywords)
        
        # 5. Determinar categoría de tamaño
        size_category = self._categorize_size(community.size)
        
        # 6. Calcular score de importancia mejorado
        importance_score = self._calculate_enhanced_importance(community, entity_types)
        
        return CommunityReport(
            community_id=community.id,
            level=level,
            title=title,
            summary=summary,
            key_entities=list(community.entities)[:5],
            key_relationships=list(community.relationships)[:5],
            importance_score=importance_score,
            entity_types=entity_types,
            theme_keywords=theme_keywords,
            size_category=size_category,
            llm_enhanced=False
        )
    
    def _analyze_entity_types(self, community: Community) -> Dict[str, int]:
        """Analiza la distribución de tipos de entidades en la comunidad"""
        type_counts = defaultdict(int)
        
        for entity_id in community.entities:
            if entity_id in self.entity_cache:
                entity = self.entity_cache[entity_id]
                type_counts[entity.type.value] += 1
        
        return dict(type_counts)
    
    def _extract_theme_keywords(self, community: Community) -> List[str]:
        """Extrae palabras clave temáticas de las entidades de la comunidad"""
        keywords = []
        
        for entity_id in community.entities:
            if entity_id in self.entity_cache:
                entity = self.entity_cache[entity_id]
                # Extraer palabras importantes del nombre de la entidad
                words = entity.name.lower().split()
                keywords.extend([word for word in words if len(word) > 3])
        
        # Obtener las palabras más frecuentes
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(5)]
    
    def _generate_intelligent_title(self, community: Community, entity_types: Dict[str, int], 
                                   theme_keywords: List[str], index: int) -> str:
        """Genera título inteligente basado en el contenido de la comunidad"""
        
        # Tipo de entidad más común
        dominant_type = max(entity_types.items(), key=lambda x: x[1])[0] if entity_types else "Entity"
        
        # Palabra clave más relevante
        primary_keyword = theme_keywords[0] if theme_keywords else "General"
        
        # Plantillas de títulos basadas en contenido
        if "ORG" in entity_types and entity_types["ORG"] >= 2:
            return f"Cluster Organizacional: {primary_keyword.title()}"
        elif "PERSON" in entity_types and entity_types["PERSON"] >= 2:
            return f"Red de Personas: {primary_keyword.title()}"
        elif "TECH" in entity_types or any(keyword in ['microsoft', 'graph', 'rag', 'algorithm'] for keyword in theme_keywords):
            return f"Cluster Tecnológico: {primary_keyword.title()}"
        elif "LOC" in entity_types:
            return f"Cluster Geográfico: {primary_keyword.title()}"
        else:
            return f"Comunidad {dominant_type}: {primary_keyword.title()}"
    
    def _generate_descriptive_summary(self, community: Community, entity_types: Dict[str, int], 
                                    theme_keywords: List[str]) -> str:
        """Genera resumen descriptivo funcional"""
        
        # Componentes del resumen
        size_desc = f"Comunidad de {community.size} entidades"
        
        # Descripción de tipos
        if entity_types:
            type_desc = "compuesta por " + ", ".join([
                f"{count} {type_name.lower()}{'s' if count > 1 else ''}" 
                for type_name, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)
            ])
        else:
            type_desc = "con entidades diversas"
        
        # Temas principales
        if theme_keywords:
            theme_desc = f"relacionada con {', '.join(theme_keywords[:3])}"
        else:
            theme_desc = "con temas diversos"
        
        # Importancia
        if community.modularity_score > 0.3:
            importance_desc = "Esta comunidad muestra alta cohesión interna."
        elif community.modularity_score > 0.1:
            importance_desc = "Esta comunidad presenta cohesión moderada."
        else:
            importance_desc = "Esta comunidad tiene baja cohesión interna."
        
        return f"{size_desc} {type_desc}, {theme_desc}. {importance_desc}"
    
    def _categorize_size(self, size: int) -> str:
        """Categoriza el tamaño de la comunidad"""
        if size >= 10:
            return "large"
        elif size >= 5:
            return "medium"
        else:
            return "small"
    
    def _calculate_enhanced_importance(self, community: Community, entity_types: Dict[str, int]) -> float:
        """Calcula score de importancia mejorado"""
        base_score = community.modularity_score
        
        # Bonus por tamaño (logarítmico)
        size_bonus = min(0.2, community.size * 0.02)
        
        # Bonus por diversidad de tipos
        diversity_bonus = min(0.15, len(entity_types) * 0.05)
        
        # Bonus por tipos importantes (ORG, PERSON)
        important_types_bonus = 0.0
        if entity_types.get("ORG", 0) >= 2:
            important_types_bonus += 0.1
        if entity_types.get("PERSON", 0) >= 2:
            important_types_bonus += 0.1
        
        enhanced_score = base_score + size_bonus + diversity_bonus + important_types_bonus
        return min(1.0, enhanced_score)  # Cap at 1.0
    
    def generate_hierarchical_summary(self, reports: List[CommunityReport]) -> Dict[str, Any]:
        """Genera resumen jerárquico de todas las comunidades"""
        
        # Agrupar por nivel
        reports_by_level = defaultdict(list)
        for report in reports:
            reports_by_level[report.level].append(report)
        
        # Estadísticas por nivel
        level_stats = {}
        for level, level_reports in reports_by_level.items():
            level_stats[level] = {
                "total_communities": len(level_reports),
                "avg_size": sum(len(r.key_entities) for r in level_reports) / len(level_reports),
                "top_themes": self._extract_level_themes(level_reports),
                "most_important": max(level_reports, key=lambda r: r.importance_score).title
            }
        
        return {
            "total_communities": len(reports),
            "levels": len(reports_by_level),
            "level_statistics": level_stats,
            "overall_themes": self._extract_global_themes(reports),
            "top_communities": [r.title for r in sorted(reports, key=lambda r: r.importance_score, reverse=True)[:5]]
        }
    
    def _extract_level_themes(self, reports: List[CommunityReport]) -> List[str]:
        """Extrae temas principales de un nivel"""
        all_keywords = []
        for report in reports:
            all_keywords.extend(report.theme_keywords)
        
        return [keyword for keyword, count in Counter(all_keywords).most_common(3)]
    
    def _extract_global_themes(self, reports: List[CommunityReport]) -> List[str]:
        """Extrae temas globales de todas las comunidades"""
        all_keywords = []
        for report in reports:
            all_keywords.extend(report.theme_keywords)
        
        return [keyword for keyword, count in Counter(all_keywords).most_common(5)]

# Backward compatibility alias
CommunitySummarizer = LLMCommunitySummarizer

def create_community_summarizer(use_llm: bool = True) -> LLMCommunitySummarizer:
    """Factory function para crear summarizer con LLM opcional"""
    return LLMCommunitySummarizer(use_llm=use_llm)

class SyncCommunitySummarizer:
    """
    Wrapper síncrono para compatibilidad con código existente
    """
    
    def __init__(self, use_llm: bool = True):
        self.async_summarizer = LLMCommunitySummarizer(use_llm=use_llm)
    
    def generate_community_reports(self, hierarchy: CommunityHierarchy, 
                                 entities: List[Entity] = None, 
                                 relationships: List[Relationship] = None) -> List[CommunityReport]:
        """Método síncrono que wrappea la versión async"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.async_summarizer.generate_community_reports(hierarchy, entities, relationships)
        )
    
    def generate_hierarchical_summary(self, reports: List[CommunityReport]) -> Dict[str, Any]:
        """Delegación a métodos síncronos existentes"""
        return self.async_summarizer.generate_hierarchical_summary(reports) 