"""
Triple Processor - Orchestrator principal del pipeline RAG integrado
Versión simplificada para testing y desarrollo
Con integración Smart Chunker + Batch Embedder
"""

import os
import logging
import hashlib
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import traceback

# Performance engines will be imported lazy to avoid dependency conflicts
PERFORMANCE_ENGINES_AVAILABLE = None  # Will be determined on first use

# Import de configuraciones y métricas
from .pipeline_config import TripleProcessorConfig, ProcessingMode, ErrorStrategy
from .unified_metrics import (
    TripleProcessorMetrics, MetricsCollector, ProcessingStage, 
    ProcessingStatus, ComponentMetrics
)

# Import de metadata (único componente requerido)
from ..storage.metadata.sqlite_manager import SQLiteManager

# Setup logging básico
def setup_logger(name, level="INFO", detailed=True):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Mock classes para componentes no disponibles
class MockVectorManager:
    def __init__(self):
        pass
    
    def add_documents(self, embeddings_data):
        return {"ids": [f"mock_{i}" for i in range(len(embeddings_data))]}
    
    def delete_documents(self, ids):
        pass
    
    def close(self):
        pass

class MockEmbeddingManager:
    def __init__(self):
        pass
    
    def get_embedding(self, text):
        # Return a mock embedding vector
        import random
        return [random.random() for _ in range(384)]

class MockGraphManager:
    def __init__(self):
        pass
    
    def build_knowledge_graph(self, entities, relationships):
        return {
            "graph": None,
            "node_count": len(entities),
            "edge_count": len(relationships)
        }

class MockCommunityManager:
    def __init__(self, algorithm="leiden"):
        self.algorithm = algorithm
    
    def detect_communities(self, graph, algorithm=None):
        # Return mock communities
        return [{"id": 0, "nodes": ["node1", "node2"], "size": 2}]

# Smart Chunker - will be lazy loaded to avoid dependency conflicts
SMART_CHUNKER_AVAILABLE = None  # Will be determined on first use

class TextChunker:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text):
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks if chunks else [text]

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

class TripleProcessorResult:
    """Resultado del procesamiento triple"""
    
    def __init__(self, processing_id: str):
        self.processing_id = processing_id
        self.success = False
        self.document_id: Optional[int] = None
        
        # Resultados por componente
        self.vector_result: Optional[Dict[str, Any]] = None
        self.graph_result: Optional[Dict[str, Any]] = None
        self.metadata_result: Optional[Dict[str, Any]] = None
        
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
            "results": {
                "vector": self.vector_result,
                "graph": self.graph_result,
                "metadata": self.metadata_result
            },
            "metrics": self.metrics.get_summary() if self.metrics else None,
            "errors": self.errors
        }

class TripleProcessor:
    """
    Orchestrator principal del pipeline RAG triple integrado
    Versión simplificada para testing
    """
    
    def __init__(self, config: Optional[TripleProcessorConfig] = None):
        """Inicializar el procesador triple"""
        self.config = config or TripleProcessorConfig.create_default()
        self.config.validate()
        
        # Setup logging
        self.logger = setup_logger(
            name="TripleProcessor",
            level=self.config.log_level,
            detailed=self.config.enable_detailed_logging
        )
        
        # Metrics collector
        self.metrics_collector = MetricsCollector()
        
        # Componentes del sistema (inicializados lazy)
        self._vector_manager = None
        self._graph_manager = None
        self._metadata_manager = None
        self._embedding_manager = None
        self._community_manager = None
        self._text_chunker = None
        self._entity_extractor = None
        
        self.logger.info(f"TripleProcessor inicializado con modo: {self.config.processing_mode.value}")
        
    def __enter__(self):
        """Context manager entrada"""
        self._initialize_components()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager salida"""
        self._cleanup()
    
    def _initialize_components(self):
        """Inicializar componentes del sistema"""
        # Inicializar componentes según configuración
        if self.config.vector.enabled:
            self._initialize_vector_components()
        
        if self.config.graph.enabled:
            self._initialize_graph_components()
        
        if self.config.metadata.enabled:
            self._initialize_metadata_components()
    
    def _initialize_vector_components(self):
        """Inicializar componentes vectoriales"""
        try:
            self.logger.warning("Usando componentes vectoriales mock")
            self._vector_manager = MockVectorManager()
            self._embedding_manager = MockEmbeddingManager()
            
            # Try to use Smart Chunker with lazy loading, fallback to basic chunker
            if self._try_load_smart_chunker():
                self.logger.info("Inicializando Smart Semantic Chunker")
                chunker_config = {
                    'target_chunk_size': self.config.vector.chunk_size,
                    'overlap_size': self.config.vector.chunk_overlap,
                    'embedding_model': 'all-MiniLM-L6-v2'
                }
                try:
                    from .smart_chunker import create_smart_chunker
                    self._text_chunker = create_smart_chunker(chunker_config)
                except Exception as e:
                    self.logger.warning(f"Error loading smart chunker: {e}, falling back to basic chunker")
                    self._text_chunker = TextChunker(
                        chunk_size=self.config.vector.chunk_size,
                        chunk_overlap=self.config.vector.chunk_overlap
                    )
            else:
                self.logger.info("Smart Chunker no disponible, usando TextChunker básico")
                self._text_chunker = TextChunker(
                    chunk_size=self.config.vector.chunk_size,
                chunk_overlap=self.config.vector.chunk_overlap
            )
            self.logger.info("Componentes vectoriales inicializados")
        except Exception as e:
            self.logger.error(f"Error inicializando componentes vectoriales: {e}")
            raise
    
    def _initialize_graph_components(self):
        """Inicializar componentes de grafos"""
        try:
            self.logger.warning("Usando componentes de grafos mock")
            self._graph_manager = MockGraphManager()
            self._entity_extractor = SpacyEntityExtractor(
                model_name=self.config.graph.spacy_model
            )
            self._community_manager = MockCommunityManager(
                algorithm=self.config.graph.community_algorithm
            )
            self.logger.info("Componentes de grafos inicializados")
        except Exception as e:
            self.logger.error(f"Error inicializando componentes de grafos: {e}")
            raise
    
    def _initialize_metadata_components(self):
        """Inicializar componentes de metadata"""
        try:
            self._metadata_manager = SQLiteManager(
                database_url=self.config.metadata.database_url
            )
            self.logger.info("Componentes de metadata inicializados")
        except Exception as e:
            self.logger.error(f"Error inicializando componentes de metadata: {e}")
            raise
    
    def _try_load_smart_chunker(self) -> bool:
        """Try to load smart chunker, return True if successful"""
        global SMART_CHUNKER_AVAILABLE
        
        if SMART_CHUNKER_AVAILABLE is not None:
            return SMART_CHUNKER_AVAILABLE
        
        try:
            # Try to import without actually importing at module level
            import importlib.util
            spec = importlib.util.find_spec("rag_preprocessing.core.smart_chunker")
            if spec is None:
                SMART_CHUNKER_AVAILABLE = False
                return False
            
            # Try to actually load it
            from .smart_chunker import SmartSemanticChunker, create_smart_chunker
            SMART_CHUNKER_AVAILABLE = True
            self.logger.info("Smart Chunker successfully loaded")
            return True
            
        except Exception as e:
            SMART_CHUNKER_AVAILABLE = False
            self.logger.warning(f"Smart Chunker not available: {e}")
            return False
    
    def _cleanup(self):
        """Limpiar recursos"""
        if self._vector_manager:
            try:
                self._vector_manager.close()
            except:
                pass
        
        if self._metadata_manager:
            try:
                self._metadata_manager.close()
            except:
                pass
    
    def process_document(self, 
                        text: str,
                        document_title: Optional[str] = None,
                        **kwargs) -> TripleProcessorResult:
        """
        Procesar documento através de pipeline triple integrado
        
        Args:
            text: Texto del documento a procesar
            document_title: Título del documento (opcional)
            **kwargs: Parámetros adicionales
        
        Returns:
            TripleProcessorResult con resultados y métricas
        """
        processing_id = str(uuid.uuid4())
        self.logger.info(f"Iniciando procesamiento {processing_id}")
        
        # Crear métricas
        metrics = self.metrics_collector.create_metrics(
            processing_id=processing_id,
            document_length=len(text)
        )
        
        # Crear resultado
        result = TripleProcessorResult(processing_id)
        result.metrics = metrics
        
        try:
            # STAGE 1: Initialization
            metrics.start_stage(ProcessingStage.INITIALIZATION)
            self._validate_input(text)
            content_hash = self._calculate_content_hash(text)
            metrics.finish_stage(
                ProcessingStage.INITIALIZATION,
                success=True,
                content_hash=content_hash
            )
            
            # STAGE 2: Text Preprocessing  
            metrics.start_stage(ProcessingStage.TEXT_PREPROCESSING)
            processed_text = self._preprocess_text(text)
            metrics.finish_stage(
                ProcessingStage.TEXT_PREPROCESSING,
                success=True,
                processed_length=len(processed_text)
            )
            
            # STAGE 3: Component Processing
            self._process_components(result, processed_text, content_hash, document_title, kwargs)
            
            # STAGE 4: Finalization
            metrics.start_stage(ProcessingStage.FINALIZATION)
            self._finalize_processing(result)
            metrics.finish_stage(ProcessingStage.FINALIZATION, success=True)
            
            # Determinar éxito general
            result.success = self._determine_overall_success(result)
            metrics.finalize(success=result.success)
            
            self.logger.info(f"Procesamiento {processing_id} completado: success={result.success}")
            
        except Exception as e:
            error_msg = f"Error en procesamiento {processing_id}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            result.errors.append(error_msg)
            metrics.add_error(error_msg)
            metrics.finalize(success=False)
        
        finally:
            self.metrics_collector.complete_metrics(processing_id)
        
        return result
    
    def _validate_input(self, text: str):
        """Validar entrada"""
        if not text or not text.strip():
            raise ValueError("El texto no puede estar vacío")
        
        if len(text) > 1_000_000:  # Límite de 1MB
            raise ValueError("El texto es demasiado largo")
    
    def _calculate_content_hash(self, text: str) -> str:
        """Calcular hash del contenido"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocesamiento básico del texto"""
        # Limpiar espacios extra
        text = " ".join(text.split())
        
        # Eliminar caracteres de control
        text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")
        
        return text
    
    def _process_components(self, result: TripleProcessorResult, text: str,
                           content_hash: str, document_title: Optional[str], kwargs: Dict):
        """Procesar componentes según configuración"""
        # Vector processing
        if self.config.vector.enabled:
            try:
                result.vector_result = self._process_vector_component(
                    text, content_hash, document_title, kwargs
                )
            except Exception as e:
                error_msg = f"Error en componente vector: {str(e)}"
                result.errors.append(error_msg)
                if self.config.error_strategy == ErrorStrategy.FAIL_FAST:
                    raise
        
        # Graph processing
        if self.config.graph.enabled:
            try:
                result.graph_result = self._process_graph_component(
                    text, content_hash, document_title, kwargs
                )
            except Exception as e:
                error_msg = f"Error en componente graph: {str(e)}"
                result.errors.append(error_msg)
                if self.config.error_strategy == ErrorStrategy.FAIL_FAST:
                    raise
        
        # Metadata processing
        if self.config.metadata.enabled:
            try:
                result.metadata_result = self._process_metadata_component(
                    text, content_hash, document_title, kwargs
                )
            except Exception as e:
                error_msg = f"Error en componente metadata: {str(e)}"
                result.errors.append(error_msg)
                if self.config.error_strategy == ErrorStrategy.FAIL_FAST:
                    raise
    
    def _process_vector_component(self, text: str, content_hash: str, 
                                 document_title: Optional[str], kwargs: Dict) -> Dict[str, Any]:
        """Procesar componente vectorial"""
        metrics = self.metrics_collector.get_metrics(kwargs.get('processing_id', ''))
        if metrics:
            metrics.start_stage(ProcessingStage.VECTOR_PROCESSING)
            metrics.update_vector_metrics(enabled=True, processed=True)
        
        start_time = datetime.utcnow()
        
        try:
            # Chunking
            chunks = self._text_chunker.chunk_text(text)
            
            # Generate embeddings
            embeddings_data = []
            for i, chunk in enumerate(chunks):
                embedding = self._embedding_manager.get_embedding(chunk)
                embeddings_data.append({
                    "id": f"{content_hash}_{i}",
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": {
                        "chunk_index": i,
                        "content_hash": content_hash,
                        "document_title": document_title or "Untitled"
                    }
                })
            
            # Store in vector DB (mock)
            storage_result = self._vector_manager.add_documents(embeddings_data)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = {
                "chunks_created": len(chunks),
                "embeddings_generated": len(embeddings_data),
                "storage_ids": storage_result.get("ids", []),
                "processing_time": processing_time,
                "quality_score": min(1.0, len(chunks) / 10.0)
            }
            
            if metrics:
                metrics.update_vector_metrics(
                    success=True,
                    processing_time=processing_time,
                    items_created=len(chunks),
                    quality_score=result["quality_score"]
                )
                metrics.finish_stage(ProcessingStage.VECTOR_PROCESSING, success=True,
                                   chunks_created=len(chunks))
            
            return result
            
        except Exception as e:
            if metrics:
                metrics.update_vector_metrics(success=False)
                metrics.finish_stage(ProcessingStage.VECTOR_PROCESSING, success=False,
                                   error_message=str(e))
            raise
    
    def _process_graph_component(self, text: str, content_hash: str,
                               document_title: Optional[str], kwargs: Dict) -> Dict[str, Any]:
        """Procesar componente de grafos"""
        metrics = self.metrics_collector.get_metrics(kwargs.get('processing_id', ''))
        if metrics:
            metrics.start_stage(ProcessingStage.GRAPH_PROCESSING)
            metrics.update_graph_metrics(enabled=True, processed=True)
        
        start_time = datetime.utcnow()
        
        try:
            # Entity extraction
            entities, relationships = self._entity_extractor.extract_entities_and_relationships(text)
            
            # Build graph
            graph_data = self._graph_manager.build_knowledge_graph(entities, relationships)
            
            # Community detection
            communities = []
            if len(entities) >= self.config.graph.min_entities:
                communities = self._community_manager.detect_communities(
                    graph_data["graph"], algorithm=self.config.graph.community_algorithm
                )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = {
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "communities_detected": len(communities),
                "graph_nodes": graph_data.get("node_count", 0),
                "graph_edges": graph_data.get("edge_count", 0),
                "processing_time": processing_time,
                "quality_score": min(1.0, len(entities) / 20.0)
            }
            
            if metrics:
                metrics.update_graph_metrics(
                    success=True,
                    processing_time=processing_time,
                    items_created=len(entities) + len(relationships),
                    quality_score=result["quality_score"]
                )
                metrics.finish_stage(ProcessingStage.GRAPH_PROCESSING, success=True,
                                   entities_count=len(entities),
                                   relationships_count=len(relationships))
            
            return result
            
        except Exception as e:
            if metrics:
                metrics.update_graph_metrics(success=False)
                metrics.finish_stage(ProcessingStage.GRAPH_PROCESSING, success=False,
                                   error_message=str(e))
            raise
    
    def _process_metadata_component(self, text: str, content_hash: str,
                                  document_title: Optional[str], kwargs: Dict) -> Dict[str, Any]:
        """Procesar componente de metadata"""
        metrics = self.metrics_collector.get_metrics(kwargs.get('processing_id', ''))
        if metrics:
            metrics.start_stage(ProcessingStage.METADATA_PROCESSING)
            metrics.update_metadata_metrics(enabled=True, processed=True)
        
        start_time = datetime.utcnow()
        
        try:
            # Create document record
            document_data = {
                "title": document_title or "Untitled",
                "content": text,
                "content_hash": content_hash,
                "file_type": "text",
                "source": "triple_processor",
                "vector_processed": self.config.vector.enabled,
                "graph_processed": self.config.graph.enabled,
                "metadata_processed": True
            }
            
            document_id = self._metadata_manager.create_document(document_data)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = {
                "document_id": document_id,
                "document_created": True,
                "processing_time": processing_time,
                "quality_score": 1.0
            }
            
            if metrics:
                metrics.document_id = document_id
                metrics.update_metadata_metrics(
                    success=True,
                    processing_time=processing_time,
                    items_created=1,
                    quality_score=1.0
                )
                metrics.finish_stage(ProcessingStage.METADATA_PROCESSING, success=True,
                                   document_id=document_id)
            
            return result
            
        except Exception as e:
            if metrics:
                metrics.update_metadata_metrics(success=False)
                metrics.finish_stage(ProcessingStage.METADATA_PROCESSING, success=False,
                                   error_message=str(e))
            raise
    
    def _finalize_processing(self, result: TripleProcessorResult):
        """Finalizar procesamiento"""
        # Extraer document_id si está disponible
        if result.metadata_result and "document_id" in result.metadata_result:
            result.document_id = result.metadata_result["document_id"]
    
    def _determine_overall_success(self, result: TripleProcessorResult) -> bool:
        """Determinar éxito general del procesamiento"""
        enabled_systems = self.config.get_enabled_systems()
        
        success_vector = not enabled_systems["vector"] or (result.vector_result is not None)
        success_graph = not enabled_systems["graph"] or (result.graph_result is not None)
        success_metadata = not enabled_systems["metadata"] or (result.metadata_result is not None)
        
        if self.config.error_strategy == ErrorStrategy.FAIL_FAST:
            return success_vector and success_graph and success_metadata
        elif self.config.error_strategy == ErrorStrategy.PARTIAL_SUCCESS:
            # Al menos uno debe ser exitoso
            any_success = success_vector or success_graph or success_metadata
            if any_success and not (success_vector and success_graph and success_metadata):
                result.partial_success = True
            return any_success
        else:  # ROLLBACK_ALL
            return success_vector and success_graph and success_metadata
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado del sistema"""
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics_collector.get_system_stats(),
            "components_status": {
                "vector": self._vector_manager is not None,
                "graph": self._graph_manager is not None,
                "metadata": self._metadata_manager is not None
            }
        }
    
    def process_batch(self, documents: List[Dict[str, Any]]) -> List[TripleProcessorResult]:
        """Procesar lote de documentos"""
        results = []
        
        for doc_data in documents:
            text = doc_data.get("text", "")
            title = doc_data.get("title")
            
            try:
                result = self.process_document(text, title, **doc_data)
                results.append(result)
            except Exception as e:
                # Crear resultado de error
                error_result = TripleProcessorResult(str(uuid.uuid4()))
                error_result.errors.append(f"Error procesando documento: {str(e)}")
                results.append(error_result)
        
        return results