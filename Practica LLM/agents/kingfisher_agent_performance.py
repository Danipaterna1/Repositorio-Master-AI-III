"""
Kingfisher Performance Agent - A2A Agent with Performance Engines

Google A2A-compliant agent con Smart Chunker + Batch Embedder + Hybrid Retriever
para procesamiento masivo y contexto total.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import uuid

# A2A Framework imports  
from .protocol.task_manager import KingfisherTaskManager, TaskStatus
from .protocol.agent_card import create_agent_card
from .config import Config
from .logging_config import get_logger, create_trace_context

# Performance imports - lazy loaded to avoid dependency conflicts
PERFORMANCE_AVAILABLE = None  # Will be determined on first use
EnhancedTripleProcessor = None
create_enhanced_triple_processor = None
TripleProcessorConfig = None
ProcessingMode = None

def _load_performance_engines():
    """Load performance engines only when needed"""
    global PERFORMANCE_AVAILABLE, EnhancedTripleProcessor, create_enhanced_triple_processor
    global TripleProcessorConfig, ProcessingMode
    
    if PERFORMANCE_AVAILABLE is not None:
        return PERFORMANCE_AVAILABLE
    
    try:
        from rag_preprocessing.core.enhanced_triple_processor import (
            EnhancedTripleProcessor as _ETP, create_enhanced_triple_processor as _CETP
        )
        from rag_preprocessing.core.pipeline_config import (
            TripleProcessorConfig as _TPC, ProcessingMode as _PM
        )
        
        EnhancedTripleProcessor = _ETP
        create_enhanced_triple_processor = _CETP
        TripleProcessorConfig = _TPC
        ProcessingMode = _PM
        
        PERFORMANCE_AVAILABLE = True
        logging.info("Performance engines loaded successfully")
        return True
        
    except ImportError as e:
        logging.warning(f"Performance engines not available: {e}")
        PERFORMANCE_AVAILABLE = False
        return False

@dataclass
class PerformanceMetrics:
    """Métricas de performance del agente"""
    documents_processed: int = 0
    total_chunks_generated: int = 0
    total_embeddings_created: int = 0
    avg_processing_time: float = 0.0
    avg_chunks_per_second: float = 0.0
    queries_processed: int = 0
    avg_query_time: float = 0.0

class KingfisherPerformanceAgent:
    """
    Kingfisher Agent optimizado para performance con Google A2A Protocol.
    
    Capabilities:
    1. process_documents_batch - Procesamiento masivo optimizado
    2. retrieve_knowledge_hybrid - Retrieval híbrido vector+graph+metadata  
    3. analyze_performance - Análisis de métricas y performance
    """
    
    def __init__(self):
        """Initialize the performance-optimized Kingfisher agent"""
        self.logger = get_logger(__name__)
        self.config = Config()
        
        # Agent metadata
        self.agent_id = "kingfisher-performance-agent"
        self.version = "2.0.0"
        self.capabilities = [
            "process_documents_batch",
            "retrieve_knowledge_hybrid", 
            "analyze_performance"
        ]
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
        # Task manager
        self.task_manager = KingfisherTaskManager()
        
        # Performance processor
        self.enhanced_processor: Optional[EnhancedTripleProcessor] = None
        
        # Agent card
        self.agent_card = self._create_performance_agent_card()
        
        self.logger.info(f"KingfisherPerformanceAgent inicializado v{self.version}")
    
    def _create_performance_agent_card(self) -> Dict[str, Any]:
        """Create performance-optimized agent card"""
        return {
            "name": "Kingfisher Performance RAG Agent",
            "version": self.version,
            "description": "High-performance RAG agent with smart chunking, batch embeddings, and hybrid retrieval",
            "protocol": "Google Agent-to-Agent",
            "capabilities": [
                {
                    "name": "process_documents_batch",
                    "description": "Process multiple documents with optimized chunking and batch embeddings",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "documents": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "title": {"type": "string"},
                                        "metadata": {"type": "object"}
                                    },
                                    "required": ["content"]
                                }
                            },
                            "processing_mode": {"type": "string", "enum": ["VECTOR_ONLY", "GRAPH_ONLY", "TRIPLE_FULL"]},
                            "batch_size": {"type": "integer", "default": 32}
                        },
                        "required": ["documents"]
                    }
                },
                {
                    "name": "retrieve_knowledge_hybrid", 
                    "description": "Hybrid retrieval combining vector similarity, graph relationships, and metadata filtering",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 10},
                            "mode": {"type": "string", "enum": ["vector_only", "graph_only", "hybrid", "adaptive"]},
                            "include_context": {"type": "boolean", "default": True},
                            "filters": {
                                "type": "object",
                                "properties": {
                                    "document_ids": {"type": "array", "items": {"type": "string"}},
                                    "chunk_types": {"type": "array", "items": {"type": "string"}},
                                    "min_complexity": {"type": "number"},
                                    "max_complexity": {"type": "number"}
                                }
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "analyze_performance",
                    "description": "Analyze system performance metrics and optimization opportunities",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "include_detailed_stats": {"type": "boolean", "default": True},
                            "time_range": {"type": "string", "enum": ["last_hour", "last_day", "all_time"]}
                        }
                    }
                }
            ],
            "performance_specs": {
                "target_throughput": "50+ documents/minute",
                "target_query_time": "<500ms",
                "batch_processing": "32+ chunks/batch",
                "context_preservation": "90%+ quality"
            },
            "health_endpoint": "/health",
            "metrics_endpoint": "/metrics"
        }
    
    async def initialize(self):
        """Initialize the performance agent"""
        # Load performance engines
        if not _load_performance_engines():
            self.logger.error("Performance engines not available")
            raise RuntimeError("Performance engines required for this agent")
        
        try:
            # Initialize enhanced processor
            config = TripleProcessorConfig.create_default()
            config.processing_mode = ProcessingMode.TRIPLE_FULL
            
            self.enhanced_processor = create_enhanced_triple_processor(config)
            
            with create_trace_context("agent_initialization"):
                self.logger.info("Performance agent inicializado correctamente")
                
        except Exception as e:
            self.logger.error(f"Error inicializando performance agent: {e}")
            raise
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process A2A task with performance optimization.
        
        Args:
            task_data: Task data from A2A request
            
        Returns:
            Task result with performance metrics
        """
        task_id = task_data.get("id", str(uuid.uuid4()))
        capability = task_data.get("capability")
        params = task_data.get("params", {})
        
        trace_id = f"task_{task_id}"
        
        with create_trace_context(trace_id):
            self.logger.info(f"Procesando task {capability}: {task_id}")
            
            try:
                # Update task status
                await self.task_manager.update_task_status(task_id, TaskStatus.WORKING)
                
                # Route to appropriate capability
                if capability == "process_documents_batch":
                    result = await self._process_documents_batch(params)
                elif capability == "retrieve_knowledge_hybrid":
                    result = await self._retrieve_knowledge_hybrid(params)
                elif capability == "analyze_performance":
                    result = await self._analyze_performance(params)
                else:
                    raise ValueError(f"Unknown capability: {capability}")
                
                # Update task status
                await self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED)
                
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": result,
                    "performance_metrics": self._get_current_metrics()
                }
                
            except Exception as e:
                self.logger.error(f"Error procesando task {task_id}: {e}")
                await self.task_manager.update_task_status(task_id, TaskStatus.FAILED)
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(e)
                }
    
    async def _process_documents_batch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process documents with batch optimization"""
        documents = params.get("documents", [])
        processing_mode = params.get("processing_mode", "TRIPLE_FULL")
        batch_size = params.get("batch_size", 32)
        
        if not documents:
            raise ValueError("No documents provided")
        
        start_time = time.time()
        results = []
        total_chunks = 0
        total_embeddings = 0
        
        with self.enhanced_processor:
            for i, doc in enumerate(documents):
                try:
                    content = doc.get("content", "")
                    title = doc.get("title", f"Document_{i+1}")
                    
                    # Process with enhanced processor
                    result = await self.enhanced_processor.process_document_enhanced(
                        text=content,
                        document_title=title
                    )
                    
                    # Accumulate metrics
                    total_chunks += result.chunks_processed
                    total_embeddings += result.embeddings_generated
                    
                    results.append({
                        "document_index": i,
                        "title": title,
                        "success": result.success,
                        "chunks_processed": result.chunks_processed,
                        "embeddings_generated": result.embeddings_generated,
                        "processing_time": result.processing_time,
                        "chunks_per_second": result.chunks_per_second
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error procesando documento {i}: {e}")
                    results.append({
                        "document_index": i,
                        "title": doc.get("title", f"Document_{i+1}"),
                        "success": False,
                        "error": str(e)
                    })
        
        # Update global metrics
        total_time = time.time() - start_time
        self.metrics.documents_processed += len(documents)
        self.metrics.total_chunks_generated += total_chunks
        self.metrics.total_embeddings_created += total_embeddings
        
        # Update averages
        if self.metrics.documents_processed > 0:
            self.metrics.avg_processing_time = (
                (self.metrics.avg_processing_time * (self.metrics.documents_processed - len(documents)) + total_time) 
                / self.metrics.documents_processed
            )
        
        if total_time > 0:
            current_chunks_per_second = total_chunks / total_time
            if self.metrics.documents_processed > 0:
                self.metrics.avg_chunks_per_second = (
                    (self.metrics.avg_chunks_per_second * (self.metrics.documents_processed - len(documents)) + current_chunks_per_second) 
                    / self.metrics.documents_processed
                )
        
        return {
            "documents_processed": len(documents),
            "total_chunks_generated": total_chunks,
            "total_embeddings_created": total_embeddings,
            "total_processing_time": total_time,
            "avg_chunks_per_second": total_chunks / total_time if total_time > 0 else 0,
            "results": results,
            "performance_summary": {
                "target_met": (total_chunks / total_time) > 20 if total_time > 0 else False,  # 20+ chunks/second target
                "quality_score": sum(1 for r in results if r.get("success", False)) / len(results) if results else 0
            }
        }
    
    async def _retrieve_knowledge_hybrid(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Hybrid knowledge retrieval"""
        query_text = params.get("query")
        top_k = params.get("top_k", 10)
        mode = params.get("mode", "hybrid")
        include_context = params.get("include_context", True)
        filters = params.get("filters", {})
        
        if not query_text:
            raise ValueError("Query text required")
        
        start_time = time.time()
        
        try:
            # Use enhanced processor for hybrid retrieval
            results = await self.enhanced_processor.query_enhanced(
                query_text=query_text,
                top_k=top_k
            )
            
            query_time = time.time() - start_time
            
            # Update metrics
            self.metrics.queries_processed += 1
            self.metrics.avg_query_time = (
                (self.metrics.avg_query_time * (self.metrics.queries_processed - 1) + query_time) 
                / self.metrics.queries_processed
            )
            
            return {
                "query": query_text,
                "results_count": len(results),
                "query_time": query_time,
                "results": results,
                "performance_summary": {
                    "target_met": query_time < 0.5,  # <500ms target
                    "relevance_score": self._calculate_relevance_score(results)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error en hybrid retrieval: {e}")
            raise
    
    async def _analyze_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system performance"""
        include_detailed = params.get("include_detailed_stats", True)
        time_range = params.get("time_range", "all_time")
        
        performance_analysis = {
            "overall_metrics": {
                "documents_processed": self.metrics.documents_processed,
                "total_chunks_generated": self.metrics.total_chunks_generated,
                "total_embeddings_created": self.metrics.total_embeddings_created,
                "queries_processed": self.metrics.queries_processed,
                "avg_processing_time": self.metrics.avg_processing_time,
                "avg_chunks_per_second": self.metrics.avg_chunks_per_second,
                "avg_query_time": self.metrics.avg_query_time
            },
            "performance_targets": {
                "throughput_target": "50+ docs/minute",
                "throughput_current": f"{(60 / self.metrics.avg_processing_time):.1f} docs/minute" if self.metrics.avg_processing_time > 0 else "N/A",
                "throughput_met": (60 / self.metrics.avg_processing_time) >= 50 if self.metrics.avg_processing_time > 0 else False,
                
                "query_time_target": "<500ms", 
                "query_time_current": f"{self.metrics.avg_query_time*1000:.0f}ms",
                "query_time_met": self.metrics.avg_query_time < 0.5,
                
                "chunks_per_second_target": "20+",
                "chunks_per_second_current": f"{self.metrics.avg_chunks_per_second:.1f}",
                "chunks_per_second_met": self.metrics.avg_chunks_per_second >= 20
            }
        }
        
        if include_detailed:
            # Get detailed engine stats
            if self.enhanced_processor:
                performance_analysis["engine_stats"] = self.enhanced_processor.get_performance_stats()
        
        # Performance recommendations
        recommendations = []
        if self.metrics.avg_chunks_per_second < 20:
            recommendations.append("Consider increasing batch size for embedding processing")
        if self.metrics.avg_query_time > 0.5:
            recommendations.append("Consider optimizing retrieval indexing or caching")
        if self.metrics.documents_processed < 10:
            recommendations.append("Process more documents to get meaningful performance metrics")
        
        performance_analysis["recommendations"] = recommendations
        
        return performance_analysis
    
    def _calculate_relevance_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate average relevance score of results"""
        if not results:
            return 0.0
        
        total_score = sum(result.get("relevance_score", 0) for result in results)
        return total_score / len(results)
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "documents_processed": self.metrics.documents_processed,
            "total_chunks_generated": self.metrics.total_chunks_generated,
            "total_embeddings_created": self.metrics.total_embeddings_created,
            "queries_processed": self.metrics.queries_processed,
            "avg_processing_time": self.metrics.avg_processing_time,
            "avg_chunks_per_second": self.metrics.avg_chunks_per_second,
            "avg_query_time": self.metrics.avg_query_time
        }
    
    def get_agent_card(self) -> Dict[str, Any]:
        """Get agent card for A2A discovery"""
        return self.agent_card
    
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities"""
        return self.capabilities
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "status": "healthy" if PERFORMANCE_AVAILABLE else "degraded",
            "performance_engines_available": PERFORMANCE_AVAILABLE,
            "enhanced_processor_ready": self.enhanced_processor is not None,
            "metrics": self._get_current_metrics(),
            "timestamp": time.time()
        } 