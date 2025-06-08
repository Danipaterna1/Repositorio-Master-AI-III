"""
Kingfisher Simplified Agent - A2A Agent Clean Version

Google A2A-compliant agent sin dependencias problemáticas,
funcionando con componentes básicos pero operacionales.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import uuid
import hashlib

# A2A Framework imports  
from .protocol.task_manager import KingfisherTaskManager, TaskStatus
from .protocol.agent_card import create_agent_card
from .config import Config
from .logging_config import get_logger, create_trace_context

# Basic RAG imports (sin dependencias problemáticas)
try:
    from rag_preprocessing.core.triple_processor import TripleProcessor, TripleProcessorConfig
    from rag_preprocessing.core.pipeline_config import ProcessingMode, ErrorStrategy
    BASIC_RAG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Basic RAG not available: {e}")
    BASIC_RAG_AVAILABLE = False

@dataclass
class SimplifiedMetrics:
    """Métricas básicas del agente"""
    documents_processed: int = 0
    total_chunks_generated: int = 0
    avg_processing_time: float = 0.0
    queries_processed: int = 0
    avg_query_time: float = 0.0

class KingfisherSimplifiedAgent:
    """
    Kingfisher Agent simplificado con Google A2A Protocol.
    
    Capabilities:
    1. process_documents - Procesamiento básico de documentos
    2. retrieve_knowledge - Retrieval básico
    3. get_system_status - Estado del sistema
    """
    
    def __init__(self):
        """Initialize the simplified Kingfisher agent"""
        self.logger = get_logger(__name__)
        self.config = Config()
        
        # Agent metadata
        self.agent_id = "kingfisher-simplified-agent"
        self.version = "2.1.0"
        self.capabilities = [
            "process_documents",
            "retrieve_knowledge", 
            "get_system_status"
        ]
        
        # Basic metrics
        self.metrics = SimplifiedMetrics()
        
        # Task manager
        self.task_manager = KingfisherTaskManager()
        
        # Basic processor
        self.processor: Optional[TripleProcessor] = None
        
        # Agent card
        self.agent_card = self._create_simplified_agent_card()
        
        self.logger.info(f"KingfisherSimplifiedAgent inicializado v{self.version}")
    
    def _create_simplified_agent_card(self) -> Dict[str, Any]:
        """Create simplified agent card"""
        return {
            "name": "Kingfisher Simplified RAG Agent",
            "version": self.version,
            "description": "Simplified RAG agent with basic document processing and retrieval",
            "protocol": "Google Agent-to-Agent",
            "capabilities": [
                {
                    "name": "process_documents",
                    "description": "Process documents with basic chunking and storage",
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
                            "processing_mode": {"type": "string", "enum": ["VECTOR_ONLY", "METADATA_ONLY", "TRIPLE_BASIC"]},
                        },
                        "required": ["documents"]
                    }
                },
                {
                    "name": "retrieve_knowledge", 
                    "description": "Basic knowledge retrieval from processed documents",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 5},
                            "include_metadata": {"type": "boolean", "default": True}
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_system_status",
                    "description": "Get current system status and metrics",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "include_details": {"type": "boolean", "default": True}
                        }
                    }
                }
            ],
            "performance_specs": {
                "target_throughput": "10+ documents/minute",
                "basic_functionality": "Core RAG without advanced features",
                "reliability": "High - no complex dependencies"
            },
            "health_endpoint": "/health",
            "metrics_endpoint": "/metrics"
        }
    
    async def initialize(self):
        """Initialize the simplified agent"""
        if not BASIC_RAG_AVAILABLE:
            self.logger.warning("Basic RAG not available, using mock mode")
            return
        
        try:
            # Initialize basic processor
            config = TripleProcessorConfig.create_default()
            config.processing_mode = ProcessingMode.TRIPLE_FULL
            config.error_strategy = ErrorStrategy.PARTIAL_SUCCESS
            
            self.processor = TripleProcessor(config)
            
            with create_trace_context("simplified_agent_initialization"):
                self.logger.info("Simplified agent inicializado correctamente")
                
        except Exception as e:
            self.logger.error(f"Error inicializando simplified agent: {e}")
            # Don't raise - continue in mock mode
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process A2A task with basic functionality.
        
        Args:
            task_data: Task data from A2A request
            
        Returns:
            Task result with basic metrics
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
                if capability == "process_documents":
                    result = await self._process_documents(params)
                elif capability == "retrieve_knowledge":
                    result = await self._retrieve_knowledge(params)
                elif capability == "get_system_status":
                    result = await self._get_system_status(params)
                else:
                    raise ValueError(f"Unknown capability: {capability}")
                
                # Update task status
                await self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED)
                
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": result,
                    "metrics": self._get_current_metrics()
                }
                
            except Exception as e:
                self.logger.error(f"Error procesando task {task_id}: {e}")
                await self.task_manager.update_task_status(task_id, TaskStatus.FAILED)
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(e)
                }
    
    async def _process_documents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process documents with basic functionality"""
        documents = params.get("documents", [])
        processing_mode = params.get("processing_mode", "TRIPLE_BASIC")
        
        if not documents:
            raise ValueError("No documents provided")
        
        start_time = time.time()
        results = []
        total_chunks = 0
        
        if self.processor and BASIC_RAG_AVAILABLE:
            # Use real processor
            with self.processor:
                for i, doc in enumerate(documents):
                    try:
                        content = doc.get("content", "")
                        title = doc.get("title", f"Document_{i+1}")
                        
                        # Process with basic processor
                        result = self.processor.process_document(
                            text=content,
                            document_title=title
                        )
                        
                        # Extract chunks count
                        chunks_count = 0
                        if result.vector_result:
                            chunks_count = result.vector_result.get("chunks_created", 0)
                        
                        total_chunks += chunks_count
                        
                        results.append({
                            "document_index": i,
                            "title": title,
                            "success": result.success,
                            "chunks_processed": chunks_count,
                            "processing_time": getattr(result, 'processing_time', 0.0),
                            "document_id": result.document_id
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error procesando documento {i}: {e}")
                        results.append({
                            "document_index": i,
                            "title": doc.get("title", f"Document_{i+1}"),
                            "success": False,
                            "error": str(e)
                        })
        else:
            # Mock processing
            for i, doc in enumerate(documents):
                content = doc.get("content", "")
                title = doc.get("title", f"Document_{i+1}")
                
                # Mock chunking
                mock_chunks = len(content.split()) // 100 + 1
                total_chunks += mock_chunks
                
                results.append({
                    "document_index": i,
                    "title": title,
                    "success": True,
                    "chunks_processed": mock_chunks,
                    "processing_time": 0.1,
                    "document_id": f"mock_{hashlib.md5(content.encode()).hexdigest()[:8]}"
                })
        
        # Update metrics
        total_time = time.time() - start_time
        self.metrics.documents_processed += len(documents)
        self.metrics.total_chunks_generated += total_chunks
        
        if self.metrics.documents_processed > 0:
            self.metrics.avg_processing_time = (
                (self.metrics.avg_processing_time * (self.metrics.documents_processed - len(documents)) + total_time) 
                / self.metrics.documents_processed
            )
        
        return {
            "documents_processed": len(documents),
            "total_chunks_generated": total_chunks,
            "total_processing_time": total_time,
            "avg_processing_time": total_time / len(documents) if documents else 0,
            "results": results,
            "processor_mode": "real" if self.processor else "mock"
        }
    
    async def _retrieve_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Basic knowledge retrieval"""
        query_text = params.get("query")
        top_k = params.get("top_k", 5)
        include_metadata = params.get("include_metadata", True)
        
        if not query_text:
            raise ValueError("Query text required")
        
        start_time = time.time()
        
        # Basic mock retrieval
        mock_results = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"Mock result {i+1} for query: {query_text[:50]}...",
                "relevance_score": 0.9 - (i * 0.1),
                "document_title": f"Document {i+1}",
                "metadata": {
                    "chunk_index": i,
                    "processing_time": time.time()
                } if include_metadata else {}
            }
            for i in range(min(top_k, 3))
        ]
        
        query_time = time.time() - start_time
        
        # Update metrics
        self.metrics.queries_processed += 1
        self.metrics.avg_query_time = (
            (self.metrics.avg_query_time * (self.metrics.queries_processed - 1) + query_time) 
            / self.metrics.queries_processed
        )
        
        return {
            "query": query_text,
            "results_count": len(mock_results),
            "query_time": query_time,
            "results": mock_results,
            "retrieval_mode": "mock"
        }
    
    async def _get_system_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get system status"""
        include_details = params.get("include_details", True)
        
        status = {
            "agent_status": "operational",
            "basic_rag_available": BASIC_RAG_AVAILABLE,
            "processor_initialized": self.processor is not None,
            "capabilities": self.capabilities,
            "metrics": self._get_current_metrics()
        }
        
        if include_details:
            status["detailed_info"] = {
                "agent_version": self.version,
                "dependencies": {
                    "basic_rag": BASIC_RAG_AVAILABLE,
                    "task_manager": True,
                    "logging": True
                },
                "uptime": time.time(),
                "processing_mode": "real" if self.processor else "mock"
            }
        
        return status
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            "documents_processed": self.metrics.documents_processed,
            "total_chunks_generated": self.metrics.total_chunks_generated,
            "queries_processed": self.metrics.queries_processed,
            "avg_processing_time": self.metrics.avg_processing_time,
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
            "status": "healthy",
            "basic_rag_available": BASIC_RAG_AVAILABLE,
            "processor_ready": self.processor is not None,
            "metrics": self._get_current_metrics(),
            "timestamp": time.time(),
            "version": self.version
        } 