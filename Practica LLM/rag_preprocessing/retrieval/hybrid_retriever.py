"""
Hybrid Retrieval Engine - Vector + Graph + Metadata Fusion

Motor de retrieval que combina múltiples signals para contexto máximo.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

from ..core.smart_chunker import SmartChunk
from ..core.batch_embedder import BatchEmbedder

logger = logging.getLogger(__name__)

class RetrievalMode(str, Enum):
    """Modos de retrieval"""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only" 
    METADATA_ONLY = "metadata_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

@dataclass
class RetrievalResult:
    """Resultado de retrieval con scoring"""
    chunk_id: str
    content: str
    relevance_score: float
    
    # Scoring breakdown
    vector_score: float = 0.0
    graph_score: float = 0.0
    metadata_score: float = 0.0
    
    # Metadata
    chunk_type: str = ""
    document_id: str = ""
    document_title: str = ""
    parent_section: str = ""
    
    # Context
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    related_chunks: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "vector_score": self.vector_score,
            "graph_score": self.graph_score,
            "metadata_score": self.metadata_score,
            "chunk_type": self.chunk_type,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "parent_section": self.parent_section,
            "previous_chunk_id": self.previous_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "related_chunks": self.related_chunks or []
        }

@dataclass
class RetrievalQuery:
    """Query con metadata para retrieval inteligente"""
    text: str
    
    # Filtros
    document_ids: Optional[List[str]] = None
    chunk_types: Optional[List[str]] = None  
    sections: Optional[List[str]] = None
    min_complexity: Optional[float] = None
    max_complexity: Optional[float] = None
    
    # Configuración
    top_k: int = 10
    mode: RetrievalMode = RetrievalMode.HYBRID
    include_context: bool = True
    expand_query: bool = False
    
    # Pesos para fusion
    vector_weight: float = 0.5
    graph_weight: float = 0.3
    metadata_weight: float = 0.2

class HybridRetriever:
    """
    Motor de retrieval que combina vector similarity, graph relationships y metadata filtering.
    
    Features:
    - Vector similarity search (cosine)
    - Graph traversal para related content
    - Metadata filtering avanzado
    - Query expansion automática
    - Adaptive scoring basado en query type
    - Context expansion automático
    """
    
    def __init__(self,
                 embedder: BatchEmbedder,
                 vector_store: Optional[Any] = None,
                 graph_store: Optional[Any] = None,
                 metadata_store: Optional[Any] = None):
        
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.metadata_store = metadata_store
        
        # Stores de ejemplo (simplified)
        self.chunks_by_id: Dict[str, SmartChunk] = {}
        self.embeddings_by_id: Dict[str, np.ndarray] = {}
        self.relationships: Dict[str, Set[str]] = {}  # Graph relationships
        
        logger.info("Hybrid retriever initialized")
    
    def index_chunks(self, 
                    chunks: List[SmartChunk], 
                    embeddings: List[np.ndarray]):
        """Index chunks con embeddings y relationships"""
        
        logger.info(f"Indexing {len(chunks)} chunks")
        
        # Store chunks y embeddings
        for chunk, embedding in zip(chunks, embeddings):
            self.chunks_by_id[chunk.id] = chunk
            self.embeddings_by_id[chunk.id] = embedding
            
            # Build relationships
            if chunk.id not in self.relationships:
                self.relationships[chunk.id] = set()
                
            # Add sequential relationships
            if chunk.previous_chunk_id:
                self.relationships[chunk.id].add(chunk.previous_chunk_id)
                if chunk.previous_chunk_id not in self.relationships:
                    self.relationships[chunk.previous_chunk_id] = set()
                self.relationships[chunk.previous_chunk_id].add(chunk.id)
                
            if chunk.next_chunk_id:
                self.relationships[chunk.id].add(chunk.next_chunk_id)
                if chunk.next_chunk_id not in self.relationships:
                    self.relationships[chunk.next_chunk_id] = set()
                self.relationships[chunk.next_chunk_id].add(chunk.id)
        
        logger.info(f"Indexed {len(self.chunks_by_id)} chunks with {sum(len(rels) for rels in self.relationships.values())} relationships")
    
    async def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieval híbrido principal.
        
        Args:
            query: RetrievalQuery con configuración
            
        Returns:
            Lista ordenada de RetrievalResult
        """
        
        start_time = time.time()
        
        # 1. Query processing y expansion
        processed_query = await self._process_query(query)
        
        # 2. Retrieval por modalidad
        if query.mode == RetrievalMode.VECTOR_ONLY:
            results = await self._vector_retrieve(processed_query)
        elif query.mode == RetrievalMode.GRAPH_ONLY:
            results = await self._graph_retrieve(processed_query)
        elif query.mode == RetrievalMode.METADATA_ONLY:
            results = await self._metadata_retrieve(processed_query)
        else:
            # HYBRID o ADAPTIVE
            results = await self._hybrid_retrieve(processed_query)
        
        # 3. Context expansion si está habilitado
        if query.include_context:
            results = await self._expand_context(results, query)
        
        # 4. Final scoring y ranking
        results = self._final_ranking(results, query)
        
        # 5. Limitar resultados
        results = results[:query.top_k]
        
        retrieval_time = time.time() - start_time
        logger.info(f"Retrieved {len(results)} results in {retrieval_time:.3f}s")
        
        return results
    
    async def _process_query(self, query: RetrievalQuery) -> Dict[str, Any]:
        """Procesar y expandir query"""
        
        processed = {
            "original_text": query.text,
            "embedding": None,
            "expanded_terms": [],
            "intent": self._detect_query_intent(query.text)
        }
        
        # Embed query
        processed["embedding"] = self.embedder.embed_single_text(query.text)
        
        # Query expansion (simplified)
        if query.expand_query:
            processed["expanded_terms"] = self._expand_query_terms(query.text)
        
        return processed
    
    def _detect_query_intent(self, text: str) -> str:
        """Detectar intent del query para adaptive scoring"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["how", "what", "why", "explain"]):
            return "explanatory"
        elif any(word in text_lower for word in ["code", "function", "class", "method"]):
            return "code"
        elif any(word in text_lower for word in ["example", "sample", "demo"]):
            return "example"
        else:
            return "general"
    
    def _expand_query_terms(self, text: str) -> List[str]:
        """Expandir términos del query (simplified)"""
        # Simplified expansion - in real implementation use word2vec/synonyms
        words = text.lower().split()
        expanded = []
        
        # Simple synonym mapping
        synonyms = {
            "create": ["make", "build", "generate"],
            "use": ["utilize", "employ", "apply"],
            "error": ["bug", "issue", "problem"],
            "function": ["method", "procedure", "routine"]
        }
        
        for word in words:
            if word in synonyms:
                expanded.extend(synonyms[word])
        
        return expanded
    
    async def _vector_retrieve(self, processed_query: Dict[str, Any]) -> List[RetrievalResult]:
        """Vector similarity retrieval"""
        
        query_embedding = processed_query["embedding"]
        results = []
        
        # Calculate similarities
        for chunk_id, chunk_embedding in self.embeddings_by_id.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                chunk_embedding.reshape(1, -1)
            )[0][0]
            
            chunk = self.chunks_by_id[chunk_id]
            result = RetrievalResult(
                chunk_id=chunk_id,
                content=chunk.content,
                relevance_score=similarity,
                vector_score=similarity,
                chunk_type=chunk.chunk_type.value,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                parent_section=chunk.parent_section,
                previous_chunk_id=chunk.previous_chunk_id,
                next_chunk_id=chunk.next_chunk_id
            )
            results.append(result)
        
        # Sort by similarity
        results.sort(key=lambda x: x.vector_score, reverse=True)
        
        return results
    
    async def _graph_retrieve(self, processed_query: Dict[str, Any]) -> List[RetrievalResult]:
        """Graph-based retrieval"""
        
        # Start with vector retrieve para seed nodes
        vector_results = await self._vector_retrieve(processed_query)
        seed_chunks = [r.chunk_id for r in vector_results[:5]]  # Top 5 as seeds
        
        # Graph traversal
        related_chunks = set()
        for seed_id in seed_chunks:
            related_chunks.update(self._get_related_chunks(seed_id, depth=2))
        
        # Score based on graph distance
        results = []
        for chunk_id in related_chunks:
            chunk = self.chunks_by_id[chunk_id]
            
            # Calculate graph score (simplified)
            graph_score = 1.0 if chunk_id in seed_chunks else 0.5
            
            result = RetrievalResult(
                chunk_id=chunk_id,
                content=chunk.content,
                relevance_score=graph_score,
                graph_score=graph_score,
                chunk_type=chunk.chunk_type.value,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                parent_section=chunk.parent_section,
                previous_chunk_id=chunk.previous_chunk_id,
                next_chunk_id=chunk.next_chunk_id
            )
            results.append(result)
        
        results.sort(key=lambda x: x.graph_score, reverse=True)
        return results
    
    async def _metadata_retrieve(self, processed_query: Dict[str, Any]) -> List[RetrievalResult]:
        """Metadata-based filtering and retrieval"""
        
        results = []
        query_text = processed_query["original_text"].lower()
        
        for chunk_id, chunk in self.chunks_by_id.items():
            metadata_score = 0.0
            
            # Score based on metadata matching
            if chunk.key_phrases:
                phrase_matches = sum(1 for phrase in chunk.key_phrases if phrase in query_text)
                metadata_score += phrase_matches * 0.2
            
            # Boost based on complexity if query seems technical
            if processed_query["intent"] == "code" and chunk.chunk_type.value == "code_block":
                metadata_score += 0.5
            
            # Boost recent/important chunks (simplified)
            if chunk.parent_section and any(word in chunk.parent_section.lower() for word in ["introduction", "summary", "conclusion"]):
                metadata_score += 0.3
            
            if metadata_score > 0:
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    content=chunk.content,
                    relevance_score=metadata_score,
                    metadata_score=metadata_score,
                    chunk_type=chunk.chunk_type.value,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    parent_section=chunk.parent_section,
                    previous_chunk_id=chunk.previous_chunk_id,
                    next_chunk_id=chunk.next_chunk_id
                )
                results.append(result)
        
        results.sort(key=lambda x: x.metadata_score, reverse=True)
        return results
    
    async def _hybrid_retrieve(self, processed_query: Dict[str, Any]) -> List[RetrievalResult]:
        """Fusion de vector, graph y metadata retrieval"""
        
        # Get results from each modality
        vector_results = await self._vector_retrieve(processed_query)
        graph_results = await self._graph_retrieve(processed_query)
        metadata_results = await self._metadata_retrieve(processed_query)
        
        # Combine results
        all_chunk_ids = set()
        all_chunk_ids.update(r.chunk_id for r in vector_results)
        all_chunk_ids.update(r.chunk_id for r in graph_results)
        all_chunk_ids.update(r.chunk_id for r in metadata_results)
        
        # Create lookup dictionaries
        vector_lookup = {r.chunk_id: r for r in vector_results}
        graph_lookup = {r.chunk_id: r for r in graph_results}
        metadata_lookup = {r.chunk_id: r for r in metadata_results}
        
        # Fusion scoring
        fused_results = []
        for chunk_id in all_chunk_ids:
            chunk = self.chunks_by_id[chunk_id]
            
            # Get individual scores
            vector_score = vector_lookup.get(chunk_id, RetrievalResult("", "", 0.0)).vector_score
            graph_score = graph_lookup.get(chunk_id, RetrievalResult("", "", 0.0)).graph_score
            metadata_score = metadata_lookup.get(chunk_id, RetrievalResult("", "", 0.0)).metadata_score
            
            # Weighted fusion
            fused_score = (
                vector_score * 0.5 +
                graph_score * 0.3 +
                metadata_score * 0.2
            )
            
            result = RetrievalResult(
                chunk_id=chunk_id,
                content=chunk.content,
                relevance_score=fused_score,
                vector_score=vector_score,
                graph_score=graph_score,
                metadata_score=metadata_score,
                chunk_type=chunk.chunk_type.value,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                parent_section=chunk.parent_section,
                previous_chunk_id=chunk.previous_chunk_id,
                next_chunk_id=chunk.next_chunk_id
            )
            fused_results.append(result)
        
        fused_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return fused_results
    
    def _get_related_chunks(self, chunk_id: str, depth: int = 1) -> Set[str]:
        """Get related chunks via graph traversal"""
        if depth <= 0 or chunk_id not in self.relationships:
            return {chunk_id}
        
        related = {chunk_id}
        for related_id in self.relationships[chunk_id]:
            if related_id in self.chunks_by_id:  # Ensure chunk exists
                related.update(self._get_related_chunks(related_id, depth - 1))
        
        return related
    
    async def _expand_context(self, 
                            results: List[RetrievalResult], 
                            query: RetrievalQuery) -> List[RetrievalResult]:
        """Expand context for better understanding"""
        
        expanded_results = []
        
        for result in results:
            expanded_results.append(result)
            
            # Add context chunks
            chunk = self.chunks_by_id[result.chunk_id]
            related_chunk_ids = []
            
            # Add previous/next chunks for continuity
            if chunk.previous_chunk_id and chunk.previous_chunk_id in self.chunks_by_id:
                related_chunk_ids.append(chunk.previous_chunk_id)
            if chunk.next_chunk_id and chunk.next_chunk_id in self.chunks_by_id:
                related_chunk_ids.append(chunk.next_chunk_id)
            
            # Add to related chunks list
            result.related_chunks = related_chunk_ids
        
        return expanded_results
    
    def _final_ranking(self, 
                      results: List[RetrievalResult], 
                      query: RetrievalQuery) -> List[RetrievalResult]:
        """Final ranking con boost factors"""
        
        for result in results:
            boost = 1.0
            
            # Boost based on query intent
            intent = self._detect_query_intent(query.text)
            if intent == "code" and result.chunk_type == "code_block":
                boost *= 1.2
            elif intent == "explanatory" and result.chunk_type == "paragraph":
                boost *= 1.1
            
            # Boost based on document importance (simplified)
            if "readme" in result.document_title.lower():
                boost *= 1.1
            
            result.relevance_score *= boost
        
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "total_chunks": len(self.chunks_by_id),
            "total_embeddings": len(self.embeddings_by_id),
            "total_relationships": sum(len(rels) for rels in self.relationships.values()),
            "model_info": self.embedder.get_model_info()
        }

# Factory function
def create_hybrid_retriever(embedder: BatchEmbedder, 
                           config: Dict[str, Any] = None) -> HybridRetriever:
    """Create hybrid retriever with configuration"""
    config = config or {}
    
    return HybridRetriever(
        embedder=embedder,
        vector_store=config.get('vector_store'),
        graph_store=config.get('graph_store'),
        metadata_store=config.get('metadata_store')
    ) 