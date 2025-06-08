"""
Entity Extractor for Microsoft Graph RAG
========================================

Implementa extracción híbrida de entidades y relaciones:
1. spaCy para extracción rápida y eficiente 
2. LLM fallback para casos complejos
3. Schema-free approach (sin constrains fijas)

Basado en el enfoque de Microsoft que usa solo LLMs pero optimizado para eficiencia.
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import hashlib

import spacy
from spacy.tokens import Doc, Span

from ..config.settings import get_config
from ..core.embedding_manager import get_embedding_manager

logger = logging.getLogger(__name__)

class EntityType(Enum):
    """Tipos de entidades detectadas"""
    PERSON = "PERSON"
    ORGANIZATION = "ORG" 
    LOCATION = "LOC"
    TECHNOLOGY = "TECH"
    CONCEPT = "CONCEPT"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    OTHER = "OTHER"

class RelationType(Enum):
    """Tipos de relaciones detectadas"""
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CREATES = "creates"
    USES = "uses"
    LEADS = "leads"
    COLLABORATES = "collaborates"
    COMPETES = "competes"
    OTHER = "other"

@dataclass
class Entity:
    """Entidad extraída del texto"""
    id: str
    name: str
    type: EntityType
    description: str
    confidence: float
    source_text: str
    start_char: int
    end_char: int
    aliases: Set[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = set()
        
        # Generar ID único basado en nombre normalizado
        if not self.id:
            normalized_name = self.name.lower().strip()
            self.id = hashlib.md5(normalized_name.encode()).hexdigest()[:12]

@dataclass  
class Relationship:
    """Relación entre dos entidades"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    description: str
    confidence: float
    source_text: str
    
    def __post_init__(self):
        # Generar ID único para la relación
        if not self.id:
            relation_key = f"{self.source_entity_id}_{self.relation_type.value}_{self.target_entity_id}"
            self.id = hashlib.md5(relation_key.encode()).hexdigest()[:12]

@dataclass
class ExtractionResult:
    """Resultado de la extracción de entidades y relaciones"""
    entities: List[Entity]
    relationships: List[Relationship]
    processing_time: float
    method_used: str  # 'spacy', 'llm', 'hybrid'
    source_text: str
    confidence_score: float

class EntityExtractor:
    """
    Extractor híbrido de entidades siguiendo el enfoque Microsoft Graph RAG
    pero optimizado con spaCy para eficiencia
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # Componentes lazy loading
        self._spacy_nlp = None
        self._embedding_manager = None
        self._llm = None
        
        # Métricas de extracción
        self.metrics = {
            "spacy_extractions": 0,
            "llm_extractions": 0,
            "hybrid_extractions": 0,
            "total_entities": 0,
            "total_relationships": 0,
            "avg_confidence": 0.0,
            "processing_time": 0.0
        }
        
        logger.info("EntityExtractor initialized for Microsoft Graph RAG")
    
    @property
    def spacy_nlp(self):
        """Lazy loading del modelo spaCy"""
        if self._spacy_nlp is None:
            try:
                self._spacy_nlp = spacy.load("es_core_news_lg")
                logger.info("spaCy model es_core_news_lg loaded successfully")
            except OSError:
                try:
                    self._spacy_nlp = spacy.load("es_core_news_sm")
                    logger.warning("Using es_core_news_sm (small model) - install es_core_news_lg for better results")
                except OSError:
                    logger.error("No Spanish spaCy model found. Install: python -m spacy download es_core_news_lg")
                    raise
        return self._spacy_nlp
    
    @property
    def embedding_manager(self):
        """Lazy loading del embedding manager"""
        if self._embedding_manager is None:
            self._embedding_manager = get_embedding_manager()
        return self._embedding_manager
    
    def extract_entities_and_relationships(self, text: str, 
                                         use_llm_fallback: bool = True) -> ExtractionResult:
        """
        Extrae entidades y relaciones del texto usando estrategia híbrida
        
        Args:
            text: Texto a procesar
            use_llm_fallback: Si usar LLM cuando spaCy no es suficiente
            
        Returns:
            ExtractionResult con entidades y relaciones extraídas
        """
        start_time = time.time()
        
        # 1. Extracción primaria con spaCy (rápida y eficiente)
        spacy_result = self._extract_with_spacy(text)
        
        # 2. Decidir si necesitamos LLM fallback
        needs_llm = self._needs_llm_processing(text, spacy_result, use_llm_fallback)
        
        if needs_llm:
            logger.info("Using LLM fallback for complex entity extraction")
            llm_result = self._extract_with_llm(text)
            
            # 3. Fusionar resultados de spaCy y LLM
            final_result = self._merge_extraction_results(spacy_result, llm_result, text)
            method_used = "hybrid"
            self.metrics["hybrid_extractions"] += 1
        else:
            final_result = spacy_result
            method_used = "spacy"
            self.metrics["spacy_extractions"] += 1
        
        processing_time = time.time() - start_time
        
        # 4. Calcular confidence promedio
        avg_confidence = self._calculate_average_confidence(final_result)
        
        # 5. Actualizar métricas
        self._update_metrics(final_result, processing_time, avg_confidence)
        
        return ExtractionResult(
            entities=final_result['entities'],
            relationships=final_result['relationships'],
            processing_time=processing_time,
            method_used=method_used,
            source_text=text,
            confidence_score=avg_confidence
        )
    
    def _extract_with_spacy(self, text: str) -> Dict[str, List]:
        """Extracción rápida con spaCy"""
        doc = self.spacy_nlp(text)
        
        entities = []
        relationships = []
        
        # Extraer entidades nombradas
        for ent in doc.ents:
            entity_type = self._map_spacy_to_entity_type(ent.label_)
            
            entity = Entity(
                id="",  # Se generará automáticamente
                name=ent.text.strip(),
                type=entity_type,
                description=f"{entity_type.value} mencionada en el texto",
                confidence=0.8,  # spaCy confidence base
                source_text=text,
                start_char=ent.start_char,
                end_char=ent.end_char
            )
            entities.append(entity)
        
        # Extraer relaciones básicas usando dependencias sintácticas
        relationships.extend(self._extract_spacy_relationships(doc, entities))
        
        return {
            'entities': entities,
            'relationships': relationships
        }
    
    def _extract_with_llm(self, text: str) -> Dict[str, List]:
        """Extracción avanzada con LLM (simulada por ahora)"""
        # TODO: Implementar extracción real con LLM cuando tengamos API key
        logger.info("LLM extraction would be called here (requires API key)")
        
        # Por ahora retornamos resultado vacío - se implementará cuando configuremos LLM
        return {
            'entities': [],
            'relationships': []
        }
    
    def _extract_spacy_relationships(self, doc: Doc, entities: List[Entity]) -> List[Relationship]:
        """Extrae relaciones usando análisis sintáctico mejorado de spaCy"""
        relationships = []
        
        # Crear mapa de entidades por spans de tokens
        entity_spans = {}
        for entity in entities:
            # Encontrar tokens que corresponden a esta entidad
            entity_tokens = []
            for token in doc:
                if (token.idx >= entity.start_char and 
                    token.idx < entity.end_char):
                    entity_tokens.append(token.i)
                    entity_spans[token.i] = entity
        
        # Buscar relaciones basadas en dependencias sintácticas mejoradas
        for token in doc:
            if token.i in entity_spans:
                source_entity = entity_spans[token.i]
                
                # 1. Relaciones directas (objetos, complementos)
                for child in token.children:
                    if child.i in entity_spans:
                        target_entity = entity_spans[child.i]
                        relation_type = self._infer_relation_type(token, child, doc)
                        
                        relationship = Relationship(
                            id="",
                            source_entity_id=source_entity.id,
                            target_entity_id=target_entity.id,
                            relation_type=relation_type,
                            description=f"{source_entity.name} {relation_type.value} {target_entity.name}",
                            confidence=0.8,
                            source_text=doc.text
                        )
                        relationships.append(relationship)
                
                # 2. Relaciones a través de verbos (sujeto-verbo-objeto)
                if token.dep_ == "nsubj":  # Si es sujeto
                    verb = token.head
                    for other_child in verb.children:
                        if (other_child.dep_ in ["dobj", "pobj"] and 
                            other_child.i in entity_spans):
                            target_entity = entity_spans[other_child.i]
                            relation_type = self._infer_verb_relation(verb)
                            
                            relationship = Relationship(
                                id="",
                                source_entity_id=source_entity.id,
                                target_entity_id=target_entity.id,
                                relation_type=relation_type,
                                description=f"{source_entity.name} {verb.text} {target_entity.name}",
                                confidence=0.9,
                                source_text=doc.text
                            )
                            relationships.append(relationship)
                
                # 3. Relaciones de proximidad (entidades cercanas)
                self._extract_proximity_relationships(doc, entity_spans, source_entity, relationships)
        
        # Eliminar duplicados
        unique_relationships = self._deduplicate_relationships(relationships)
        
        return unique_relationships
    
    def _needs_llm_processing(self, text: str, spacy_result: Dict, use_llm: bool) -> bool:
        """Heurística para decidir si usar LLM fallback"""
        if not use_llm:
            return False
        
        # Criterios para usar LLM:
        entities_found = len(spacy_result['entities'])
        text_length = len(text)
        
        # Texto largo con pocas entidades puede beneficiarse de LLM
        if text_length > 1000 and entities_found < 3:
            return True
        
        # Texto técnico/especializado
        technical_indicators = [
            'algoritmo', 'implementación', 'arquitectura', 'sistema',
            'quantización', 'embedding', 'vectorial', 'tecnología'
        ]
        
        technical_score = sum(1 for term in technical_indicators 
                            if term.lower() in text.lower())
        
        if technical_score > 2:
            return True
        
        return False
    
    def _merge_extraction_results(self, spacy_result: Dict, llm_result: Dict, text: str) -> Dict:
        """Fusiona resultados de spaCy y LLM eliminando duplicados"""
        # Por simplicidad, por ahora solo devolvemos spaCy
        # TODO: Implementar fusión inteligente cuando tengamos LLM
        return spacy_result
    
    def _map_spacy_to_entity_type(self, spacy_label: str) -> EntityType:
        """Mapea etiquetas de spaCy a nuestros tipos de entidad"""
        mapping = {
            "PER": EntityType.PERSON,
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "LOC": EntityType.LOCATION,
            "GPE": EntityType.LOCATION,
            "MISC": EntityType.OTHER,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT
        }
        return mapping.get(spacy_label, EntityType.OTHER)
    
    def _infer_relation_type(self, source_token, target_token, doc: Doc) -> RelationType:
        """Infiere el tipo de relación basado en contexto sintáctico"""
        # Análisis básico del verbo o contexto
        verb_context = source_token.lemma_.lower()
        dependency = target_token.dep_
        
        # Mapeo basado en dependencias sintácticas
        if dependency == "dobj" and verb_context in ['crear', 'desarrollar', 'construir', 'implementar']:
            return RelationType.CREATES
        elif dependency == "dobj" and verb_context in ['usar', 'utilizar', 'emplear']:
            return RelationType.USES
        elif dependency in ["prep", "pobj"] and verb_context in ['trabajar', 'colaborar']:
            return RelationType.WORKS_FOR
        elif dependency in ["prep", "pobj"] and source_token.text.lower() in ['en', 'de']:
            return RelationType.LOCATED_IN
        elif verb_context in ['dirigir', 'liderar', 'gestionar']:
            return RelationType.LEADS
        elif verb_context in ['colaborar', 'trabajar']:
            return RelationType.COLLABORATES
        else:
            return RelationType.RELATED_TO
    
    def _infer_verb_relation(self, verb_token) -> RelationType:
        """Infiere relación basada en el verbo específico"""
        verb_lemma = verb_token.lemma_.lower()
        
        verb_mappings = {
            'desarrollar': RelationType.CREATES,
            'crear': RelationType.CREATES,
            'construir': RelationType.CREATES,
            'implementar': RelationType.CREATES,
            'usar': RelationType.USES,
            'utilizar': RelationType.USES,
            'emplear': RelationType.USES,
            'trabajar': RelationType.WORKS_FOR,
            'colaborar': RelationType.COLLABORATES,
            'liderar': RelationType.LEADS,
            'dirigir': RelationType.LEADS,
            'gestionar': RelationType.LEADS,
            'competir': RelationType.COMPETES,
            'estar': RelationType.LOCATED_IN,
            'ubicar': RelationType.LOCATED_IN
        }
        
        return verb_mappings.get(verb_lemma, RelationType.RELATED_TO)
    
    def _extract_proximity_relationships(self, doc: Doc, entity_spans: Dict, source_entity: Entity, relationships: List[Relationship]):
        """Extrae relaciones basadas en proximidad textual"""
        # Buscar entidades cercanas (dentro de una ventana de 5 tokens)
        source_positions = [i for i, entity in entity_spans.items() if entity.id == source_entity.id]
        
        for pos in source_positions:
            for window_pos in range(max(0, pos-5), min(len(doc), pos+6)):
                if window_pos in entity_spans and entity_spans[window_pos].id != source_entity.id:
                    target_entity = entity_spans[window_pos]
                    
                    # Solo crear relación si hay indicadores contextuales
                    context = doc[max(0, pos-2):min(len(doc), pos+3)].text.lower()
                    if any(indicator in context for indicator in ['y', 'con', 'junto', 'además', 'también']):
                        relationship = Relationship(
                            id="",
                            source_entity_id=source_entity.id,
                            target_entity_id=target_entity.id,
                            relation_type=RelationType.RELATED_TO,
                            description=f"{source_entity.name} relacionado con {target_entity.name} (proximidad)",
                            confidence=0.6,
                            source_text=doc.text
                        )
                        relationships.append(relationship)
                        break  # Solo una relación de proximidad por entidad
    
    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Elimina relaciones duplicadas"""
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            # Crear clave única para la relación
            key = (rel.source_entity_id, rel.target_entity_id, rel.relation_type.value)
            reverse_key = (rel.target_entity_id, rel.source_entity_id, rel.relation_type.value)
            
            if key not in seen and reverse_key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships
    
    def _calculate_average_confidence(self, result: Dict) -> float:
        """Calcula confidence promedio de las extracciones"""
        total_confidence = 0.0
        total_items = 0
        
        for entity in result['entities']:
            total_confidence += entity.confidence
            total_items += 1
        
        for relationship in result['relationships']:
            total_confidence += relationship.confidence
            total_items += 1
        
        return total_confidence / max(total_items, 1)
    
    def _update_metrics(self, result: Dict, processing_time: float, avg_confidence: float):
        """Actualiza métricas de extracción"""
        self.metrics["total_entities"] += len(result['entities'])
        self.metrics["total_relationships"] += len(result['relationships'])
        self.metrics["processing_time"] += processing_time
        
        # Actualizar confidence promedio
        total_extractions = (self.metrics["spacy_extractions"] + 
                           self.metrics["llm_extractions"] + 
                           self.metrics["hybrid_extractions"])
        
        if total_extractions > 0:
            self.metrics["avg_confidence"] = (
                (self.metrics["avg_confidence"] * (total_extractions - 1) + avg_confidence) 
                / total_extractions
            )
    
    def get_extraction_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de extracción"""
        total_extractions = (self.metrics["spacy_extractions"] + 
                           self.metrics["llm_extractions"] + 
                           self.metrics["hybrid_extractions"])
        
        return {
            **self.metrics,
            "total_extractions": total_extractions,
            "spacy_percentage": (self.metrics["spacy_extractions"] / max(total_extractions, 1)) * 100,
            "llm_percentage": (self.metrics["llm_extractions"] / max(total_extractions, 1)) * 100,
            "hybrid_percentage": (self.metrics["hybrid_extractions"] / max(total_extractions, 1)) * 100
        }
    
    def reset_metrics(self):
        """Resetea métricas de extracción"""
        self.metrics = {
            "spacy_extractions": 0,
            "llm_extractions": 0,
            "hybrid_extractions": 0,
            "total_entities": 0,
            "total_relationships": 0,
            "avg_confidence": 0.0,
            "processing_time": 0.0
        }
        logger.info("Entity extraction metrics reset") 