"""
Enhanced Triple Processor - Integración con Performance Engines

Pipeline RAG optimizado con Smart Chunker + Batch Embedder + Hybrid Retriever
para procesamiento masivo y contexto total.
"""

import os
import logging
import hashlib
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import traceback

# Import existing configurations
from .pipeline_config import TripleProcessorConfig, ProcessingMode, ErrorStrategy
from .unified_metrics import (
    TripleProcessorMetrics, MetricsCollector, ProcessingStage, 
    ProcessingStatus, ComponentMetrics
)

# Performance engines - lazy loaded to avoid dependency conflicts
PERFORMANCE_ENGINES_AVAILABLE = None  # Will be determined on first use

# Use simple types to avoid import errors
SmartChunk = Any
EmbeddingResult = Any

# Mock classes for components that may not be available
class MockSmartChunk:
    def __init__(self, id: str, content: str, **kwargs):
        self.id = id
        self.content = content
        self.metadata = kwargs
    
    def to_dict(self):
        return {"id": self.id, "content": self.content, **self.metadata}

class MockEmbeddingResult:
    def __init__(self, chunk_id: str, embedding):
        self.chunk_id = chunk_id
        self.embedding = embedding

# Import storage systems
from ..storage.metadata.sqlite_manager import SQLiteManager

# Mock classes for unavailable components
class MockVectorManager:
    def __init__(self):
        pass
    
    def add_documents(self, embeddings_data):
        return {"ids": [f"mock_{i}" for i in range(len(embeddings_data))]}
    
    def delete_documents(self, ids):
        pass
    
    def close(self):
        pass

class MockGraphManager:
    def __init__(self):
        pass
    
    def build_knowledge_graph(self, entities, relationships):
        return {
            "graph": None,
            "node_count": len(entities),
            "edge_count": len(relationships)
        }

class SpacyEntityExtractor:
    def __init__(self, model_name="es_core_news_sm"):
        self.model_name = model_name
    
    def extract_entities_and_relationships(self, text):
        # Mock implementation
        entities = [
            {"text": "inteligencia artificial", "label": "CONCEPT"},
            {"text": "machine learning", "label": "CONCEPT"},
            {"text": "algoritmos", "label": "CONCEPT"}
        ]
        relationships = []
        return entities, relationships

class EnhancedTripleProcessorResult:
    """Resultado del procesamiento enhanced con performance data"""
    
    def __init__(self, processing_id: str):
        self.processing_id = processing_id
        self.success = False
        self.document_id: Optional[int] = None
        
        # Performance metrics
        self.chunks_processed = 0
        self.embeddings_generated = 0
        self.processing_time = 0.0
        self.chunks_per_second = 0.0
        
        # Resultados por componente
        self.vector_result: Optional[Dict[str, Any]] = None
        self.graph_result: Optional[Dict[str, Any]] = None
        self.metadata_result: Optional[Dict[str, Any]] = None
        
        # Performance results
        self.smart_chunks: List[Any] = []
        self.embeddings: List[Any] = []
        self.retrieval_ready = False
        
        # Métricas
        self.metrics: Optional[TripleProcessorMetrics] = None
        
        # Errores
        self.errors: List[str] = []
        self.partial_success = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir resultado a diccionario"""
        return {
            "processing_id": self.processing_id,
            "success": self.success,
            "partial_success": self.partial_success,
            "document_id": self.document_id,
            "performance": {
                "chunks_processed": self.chunks_processed,
                "embeddings_generated": self.embeddings_generated,
                "processing_time": self.processing_time,
                "chunks_per_second": self.chunks_per_second,
                "retrieval_ready": self.retrieval_ready
            },
            "results": {
                "vector": self.vector_result,
                "graph": self.graph_result,
                "metadata": self.metadata_result
            },
            "metrics": self.metrics.get_summary() if self.metrics else None,
            "errors": self.errors
        }

class EnhancedTripleProcessor:
    """
    Enhanced Triple Processor con performance engines integrados.
    
    Features:
    - Smart Semantic Chunking con preservación de contexto
    - Batch Embedding processing para grandes volúmenes
    - Hybrid Retrieval preparado para queries complejas
    - Performance monitoring en tiempo real
    """
    
    def __init__(self, config: Optional[TripleProcessorConfig] = None):
        """Inicializar el procesador enhanced"""
        self.config = config or TripleProcessorConfig.create_default()
        self.config.validate()
        
        # Setup logging
        self.logger = logging.getLogger("EnhancedTripleProcessor")
        self.logger.setLevel(logging.INFO)
        
        # Metrics collector
        self.metrics_collector = MetricsCollector()
        
        # Performance engines
        self._smart_chunker: Optional[SmartSemanticChunker] = None
        self._batch_embedder: Optional[BatchEmbedder] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None
        
        # Storage components (inicializados lazy)
        self._vector_manager = None
        self._graph_manager = None
        self._metadata_manager = None
        self._entity_extractor = None
        
        self.logger.info(f"EnhancedTripleProcessor inicializado con modo: {self.config.processing_mode.value}")
        
    def __enter__(self):
        """Context manager entrada"""
        self._initialize_components()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager salida"""
        self._cleanup()
    
    def _initialize_components(self):
        """Inicializar componentes del sistema"""
        if not PERFORMANCE_ENGINES_AVAILABLE:
            self.logger.error("Performance engines no disponibles. Fallback a versión básica.")
            return
        
        # Inicializar performance engines
        self._initialize_performance_engines()
        
        # Inicializar storage según configuración
        if self.config.vector.enabled:
            self._initialize_vector_components()
        
        if self.config.graph.enabled:
            self._initialize_graph_components()
        
        if self.config.metadata.enabled:
            self._initialize_metadata_components()
    
    def _initialize_performance_engines(self):
        """Inicializar motores de performance"""
        try:
            # Smart Chunker
            chunker_config = {
                'target_chunk_size': self.config.vector.chunk_size,
                'overlap_size': self.config.vector.chunk_overlap,
                'embedding_model': 'all-MiniLM-L6-v2'
            }
            self._smart_chunker = create_smart_chunker(chunker_config)
            self.logger.info("Smart Semantic Chunker inicializado")
            
            # Batch Embedder
            embedder_config = {
                'model_name': 'all-MiniLM-L6-v2',
                'batch_size': 32,
                'device': 'auto',
                'cache_embeddings': True
            }
            self._batch_embedder = create_batch_embedder(embedder_config)
            self.logger.info("Batch Embedder inicializado")
            
            # Hybrid Retriever (se inicializa después con embedder)
            self._hybrid_retriever = create_hybrid_retriever(self._batch_embedder)
            self.logger.info("Hybrid Retriever inicializado")
            
        except Exception as e:
            self.logger.error(f"Error inicializando performance engines: {e}")
            raise
    
    def _initialize_vector_components(self):
        """Inicializar componentes vectoriales"""
        try:
            self.logger.warning("Usando componentes vectoriales mock")
            self._vector_manager = MockVectorManager()
        except Exception as e:
            self.logger.error(f"Error inicializando componentes vectoriales: {e}")
            if self.config.error_strategy == ErrorStrategy.FAIL_FAST:
                raise
    
    def _initialize_graph_components(self):
        """Inicializar componentes de grafo"""
        try:
            self.logger.warning("Usando componentes de grafo mock")
            self._graph_manager = MockGraphManager()
            self._entity_extractor = SpacyEntityExtractor()
        except Exception as e:
            self.logger.error(f"Error inicializando componentes de grafo: {e}")
            if self.config.error_strategy == ErrorStrategy.FAIL_FAST:
                raise
    
    def _initialize_metadata_components(self):
        """Inicializar componentes de metadata"""
        try:
            self._metadata_manager = SQLiteManager(
                db_path=self.config.metadata.db_path,
                enable_logging=True
            )
            self.logger.info("Metadata manager inicializado")
        except Exception as e:
            self.logger.error(f"Error inicializando metadata manager: {e}")
            if self.config.error_strategy == ErrorStrategy.FAIL_FAST:
                raise
    
    def _cleanup(self):
        """Cleanup de recursos"""
        try:
            if self._vector_manager:
                self._vector_manager.close()
            if self._metadata_manager:
                self._metadata_manager.close()
        except Exception as e:
            self.logger.error(f"Error durante cleanup: {e}")
    
    async def process_document_enhanced(self, 
                                       text: str,
                                       document_title: Optional[str] = None,
                                       **kwargs) -> EnhancedTripleProcessorResult:
        """
        Procesamiento enhanced con performance engines.
        
        Args:
            text: Contenido del documento
            document_title: Título del documento
            **kwargs: Argumentos adicionales
            
        Returns:
            EnhancedTripleProcessorResult con métricas y resultados
        """
        start_time = datetime.now()
        processing_id = str(uuid.uuid4())
        result = EnhancedTripleProcessorResult(processing_id)
        
        try:
            self.logger.info(f"Iniciando procesamiento enhanced: {processing_id}")
            
            # Validar entrada
            self._validate_input(text)
            
            # Preprocessing
            text = self._preprocess_text(text)
            content_hash = self._calculate_content_hash(text)
            
            # FASE 1: Smart Chunking
            chunks_start = datetime.now()
            smart_chunks = await self._smart_chunking_phase(
                text, processing_id, document_title or "Unknown"
            )
            result.smart_chunks = smart_chunks
            result.chunks_processed = len(smart_chunks)
            
            chunks_time = (datetime.now() - chunks_start).total_seconds()
            self.logger.info(f"Smart chunking completado: {len(smart_chunks)} chunks en {chunks_time:.2f}s")
            
            # FASE 2: Batch Embedding
            embeddings_start = datetime.now()
            embeddings, batch_stats = await self._batch_embedding_phase(smart_chunks)
            result.embeddings = embeddings
            result.embeddings_generated = len(embeddings)
            
            embeddings_time = (datetime.now() - embeddings_start).total_seconds()
            self.logger.info(f"Batch embedding completado: {len(embeddings)} embeddings en {embeddings_time:.2f}s")
            
            # FASE 3: Index para Retrieval
            indexing_start = datetime.now()
            await self._retrieval_indexing_phase(smart_chunks, [e.embedding for e in embeddings])
            result.retrieval_ready = True
            
            indexing_time = (datetime.now() - indexing_start).total_seconds()
            self.logger.info(f"Retrieval indexing completado en {indexing_time:.2f}s")
            
            # FASE 4: Procesamiento tradicional (storage)
            await self._traditional_processing_phase(result, text, content_hash, document_title, kwargs)
            
            # Calcular métricas finales
            total_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = total_time
            result.chunks_per_second = len(smart_chunks) / total_time if total_time > 0 else 0
            
            result.success = True
            self.logger.info(f"Procesamiento enhanced completado: {processing_id} en {total_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error en procesamiento enhanced: {e}")
            result.errors.append(str(e))
            result.success = False
            
        return result
    
    async def _smart_chunking_phase(self, 
                                   text: str, 
                                   document_id: str, 
                                   document_title: str) -> List[Any]:
        """Fase de chunking semántico inteligente"""
        if not self._smart_chunker:
            raise ValueError("Smart chunker no inicializado")
        
        chunks = self._smart_chunker.chunk_document(
            text=text,
            document_id=document_id,
            document_title=document_title,
            preserve_structure=True
        )
        
        return chunks
    
    async def _batch_embedding_phase(self, 
                                    chunks: List[SmartChunk]) -> Tuple[List[EmbeddingResult], Any]:
        """Fase de embedding batch optimizado"""
        if not self._batch_embedder:
            raise ValueError("Batch embedder no inicializado")
        
        embeddings, stats = await self._batch_embedder.embed_chunks_batch(
            chunks=chunks,
            show_progress=True
        )
        
        return embeddings, stats
    
    async def _retrieval_indexing_phase(self, 
                                       chunks: List[SmartChunk], 
                                       embeddings: List[any]):
        """Fase de indexing para retrieval híbrido"""
        if not self._hybrid_retriever:
            raise ValueError("Hybrid retriever no inicializado")
        
        # Index chunks con embeddings
        self._hybrid_retriever.index_chunks(chunks, embeddings)
    
    async def _traditional_processing_phase(self, 
                                          result: EnhancedTripleProcessorResult,
                                          text: str,
                                          content_hash: str,
                                          document_title: Optional[str],
                                          kwargs: Dict):
        """Fase de procesamiento tradicional (storage)"""
        # Procesar storage tradicional si está habilitado
        if self.config.vector.enabled and self._vector_manager:
            # Usar embeddings ya generados
            vector_data = [
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "embedding": emb.embedding.tolist(),
                    "metadata": chunk.to_dict()
                }
                for chunk, emb in zip(result.smart_chunks, result.embeddings)
            ]
            result.vector_result = self._vector_manager.add_documents(vector_data)
        
        if self.config.graph.enabled and self._graph_manager:
            entities, relationships = self._entity_extractor.extract_entities_and_relationships(text)
            result.graph_result = self._graph_manager.build_knowledge_graph(entities, relationships)
        
        if self.config.metadata.enabled and self._metadata_manager:
            # Store document metadata
            result.metadata_result = {
                "document_id": hash(content_hash),
                "chunks_count": len(result.smart_chunks),
                "embeddings_count": len(result.embeddings)
            }
    
    def _validate_input(self, text: str):
        """Validar entrada"""
        if not text or not text.strip():
            raise ValueError("El texto no puede estar vacío")
    
    def _calculate_content_hash(self, text: str) -> str:
        """Calcular hash del contenido"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocesamiento básico del texto"""
        return text.strip()
    
    async def query_enhanced(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query enhanced usando hybrid retrieval.
        
        Args:
            query_text: Texto de la query
            top_k: Número de resultados
            
        Returns:
            Lista de resultados ordenados por relevancia
        """
        if not self._hybrid_retriever:
            raise ValueError("Hybrid retriever no inicializado")
        
        from ..retrieval.hybrid_retriever import RetrievalQuery, RetrievalMode
        
        query = RetrievalQuery(
            text=query_text,
            top_k=top_k,
            mode=RetrievalMode.HYBRID,
            include_context=True
        )
        
        results = await self._hybrid_retriever.retrieve(query)
        
        return [result.to_dict() for result in results]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de performance"""
        stats = {
            "engines_available": PERFORMANCE_ENGINES_AVAILABLE,
            "components_initialized": {
                "smart_chunker": self._smart_chunker is not None,
                "batch_embedder": self._batch_embedder is not None,
                "hybrid_retriever": self._hybrid_retriever is not None
            }
        }
        
        if self._batch_embedder:
            stats["embedder_info"] = self._batch_embedder.get_model_info()
        
        if self._hybrid_retriever:
            stats["retriever_stats"] = self._hybrid_retriever.get_stats()
        
        return stats

# Factory function
def create_enhanced_triple_processor(config: Optional[TripleProcessorConfig] = None) -> EnhancedTripleProcessor:
    """Create enhanced triple processor with configuration"""
    return EnhancedTripleProcessor(config) 