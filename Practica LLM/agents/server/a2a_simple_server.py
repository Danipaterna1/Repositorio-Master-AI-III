"""
Kingfisher Simple A2A Server - Servidor limpio sin dependencias problemáticas

Este servidor implementa Google A2A Protocol con funcionalidad básica
pero completamente operacional.
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager

# A2A Protocol imports
from ..protocol.task_manager import KingfisherTaskManager, TaskStatus
from ..protocol.agent_card import create_agent_card
from ..config import Config
from ..logging_config import get_logger, setup_logging

# Basic processor import (sin dependencias problemáticas)
try:
    from rag_preprocessing.core.triple_processor import TripleProcessor, TripleProcessorConfig
    from rag_preprocessing.core.pipeline_config import ProcessingMode, ErrorStrategy
    BASIC_RAG_AVAILABLE = True
    print("BASIC RAG IMPORTS SUCCESS")
except ImportError as e:
    BASIC_RAG_AVAILABLE = False
    print(f"BASIC RAG IMPORTS FAILED: {e}")

# Global app state
class AppState:
    def __init__(self):
        self.logger = None
        self.config = None
        self.task_manager = None
        self.processor = None
        self.agent_card = None
        self.metrics = {
            "requests_processed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "uptime_start": time.time()
        }

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("🚀 Iniciando Kingfisher Simple A2A Server...")
    
    try:
        # Setup logging
        setup_logging()
        app_state.logger = get_logger(__name__)
        app_state.config = Config()
        
        # Initialize task manager
        app_state.task_manager = KingfisherTaskManager()
        
        # Initialize basic processor if available
        if BASIC_RAG_AVAILABLE:
            try:
                config = TripleProcessorConfig.create_default()
                config.processing_mode = ProcessingMode.TRIPLE_FULL
                config.error_strategy = ErrorStrategy.PARTIAL_SUCCESS
                app_state.processor = TripleProcessor(config)
                app_state.logger.info("✅ Basic RAG processor initialized")
            except Exception as e:
                app_state.logger.warning(f"❌ Could not initialize processor: {e}")
        else:
            app_state.logger.warning("⚠️ Basic RAG not available, using mock mode")
        
        # Create agent card
        app_state.agent_card = create_simple_agent_card()
        
        app_state.logger.info("✅ Kingfisher Simple A2A Server started successfully")
        
        yield
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise
    
    # Shutdown
    print("🛑 Shutting down Kingfisher Simple A2A Server...")
    if app_state.processor:
        try:
            app_state.processor._cleanup()
        except:
            pass

def create_simple_agent_card() -> Dict[str, Any]:
    """Create simplified agent card"""
    return {
        "name": "Kingfisher Simple RAG Agent",
        "version": "2.1.0", 
        "description": "Simplified RAG agent for document processing and knowledge retrieval",
        "protocol": "Google Agent-to-Agent v1.0",
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
                                    "title": {"type": "string"}
                                },
                                "required": ["content"]
                            }
                        }
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
                        "top_k": {"type": "integer", "default": 5}
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
                        "include_details": {"type": "boolean", "default": true}
                    }
                }
            }
        ],
        "endpoints": {
            "agent_card": "/.well-known/agent.json",
            "task_submission": "/tasks/send",
            "health": "/health",
            "metrics": "/metrics"
        },
        "contact": {
            "support": "kingfisher@example.com"
        }
    }

# Create FastAPI app
app = FastAPI(
    title="Kingfisher Simple A2A Agent",
    description="Simplified Google A2A compliant RAG agent",
    version="2.1.0",
    lifespan=lifespan
)

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Request logging middleware"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Log request
    process_time = time.time() - start_time
    if app_state.logger:
        app_state.logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
        )
    
    # Update metrics
    app_state.metrics["requests_processed"] += 1
    
    return response

# A2A Protocol Endpoints

@app.get("/.well-known/agent.json")
async def get_agent_card():
    """Get agent card for A2A discovery"""
    if not app_state.agent_card:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return JSONResponse(content=app_state.agent_card)

@app.post("/tasks/send")
async def send_task(request: Request, background_tasks: BackgroundTasks):
    """Send task to agent (A2A Protocol)"""
    try:
        # Parse request
        task_data = await request.json()
        task_id = task_data.get("id", str(uuid.uuid4()))
        capability = task_data.get("capability")
        params = task_data.get("params", {})
        
        if not capability:
            raise HTTPException(status_code=400, detail="Capability required")
        
        app_state.logger.info(f"📨 Received task: {capability} ({task_id})")
        
        # Process task immediately for simple operations
        if capability in ["process_documents", "retrieve_knowledge", "get_system_status"]:
            result = await process_task_simple(task_id, capability, params)
            
            app_state.metrics["tasks_completed"] += 1
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown capability: {capability}")
            
    except Exception as e:
        app_state.logger.error(f"❌ Task processing failed: {e}")
        app_state.metrics["tasks_failed"] += 1
        raise HTTPException(status_code=500, detail=str(e))

async def process_task_simple(task_id: str, capability: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Process task with simplified logic"""
    start_time = time.time()
    
    try:
        if capability == "process_documents":
            result = await process_documents_simple(params)
        elif capability == "retrieve_knowledge":
            result = await retrieve_knowledge_simple(params)
        elif capability == "get_system_status":
            result = await get_system_status_simple(params)
        else:
            raise ValueError(f"Unknown capability: {capability}")
        
        processing_time = time.time() - start_time
        
        return {
            "task_id": task_id,
            "status": "completed",
            "result": result,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }

async def process_documents_simple(params: Dict[str, Any]) -> Dict[str, Any]:
    """Process documents with simplified logic"""
    documents = params.get("documents", [])
    
    if not documents:
        raise ValueError("No documents provided")
    
    results = []
    total_chunks = 0
    
    if app_state.processor and BASIC_RAG_AVAILABLE:
        # Use real processor
        with app_state.processor:
            for i, doc in enumerate(documents):
                try:
                    content = doc.get("content", "")
                    title = doc.get("title", f"Document_{i+1}")
                    
                    # Process document
                    result = app_state.processor.process_document(
                        text=content,
                        document_title=title
                    )
                    
                    chunks_count = 0
                    if result.vector_result:
                        chunks_count = result.vector_result.get("chunks_created", 0)
                    
                    total_chunks += chunks_count
                    
                    results.append({
                        "document_index": i,
                        "title": title,
                        "success": result.success,
                        "chunks_created": chunks_count,
                        "document_id": result.document_id
                    })
                    
                    app_state.logger.info(f"✅ Processed document {i+1}: {chunks_count} chunks")
                    
                except Exception as e:
                    app_state.logger.error(f"❌ Error processing document {i}: {e}")
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
            
            # Simulate chunking
            mock_chunks = len(content.split()) // 100 + 1
            total_chunks += mock_chunks
            
            results.append({
                "document_index": i,
                "title": title,
                "success": True,
                "chunks_created": mock_chunks,
                "document_id": f"mock_{hash(content) % 10000}",
                "mode": "mock"
            })
    
    return {
        "documents_processed": len(documents),
        "total_chunks_generated": total_chunks,
        "results": results,
        "processor_mode": "real" if app_state.processor else "mock"
    }

async def retrieve_knowledge_simple(params: Dict[str, Any]) -> Dict[str, Any]:
    """Simple knowledge retrieval"""
    query = params.get("query")
    top_k = params.get("top_k", 5)
    
    if not query:
        raise ValueError("Query required")
    
    # Basic mock results
    results = [
        {
            "chunk_id": f"chunk_{i}",
            "content": f"Mock result {i+1} for query: {query[:50]}...",
            "relevance_score": 0.9 - (i * 0.1),
            "document_title": f"Document {i+1}",
            "metadata": {
                "chunk_index": i,
                "timestamp": datetime.now().isoformat()
            }
        }
        for i in range(min(top_k, 3))
    ]
    
    return {
        "query": query,
        "results_count": len(results),
        "results": results,
        "retrieval_mode": "mock"
    }

async def get_system_status_simple(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get system status"""
    include_details = params.get("include_details", True)
    
    uptime = time.time() - app_state.metrics["uptime_start"]
    
    status = {
        "status": "healthy",
        "basic_rag_available": BASIC_RAG_AVAILABLE,
        "processor_initialized": app_state.processor is not None,
        "uptime_seconds": uptime,
        "metrics": app_state.metrics.copy()
    }
    
    if include_details:
        status["details"] = {
            "agent_version": "2.1.0",
            "capabilities": ["process_documents", "retrieve_knowledge", "get_system_status"],
            "processor_mode": "real" if app_state.processor else "mock",
            "dependencies_status": {
                "basic_rag": BASIC_RAG_AVAILABLE,
                "task_manager": app_state.task_manager is not None,
                "agent_card": app_state.agent_card is not None
            }
        }
    
    return status

# Health and monitoring endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0",
        "uptime_seconds": time.time() - app_state.metrics["uptime_start"],
        "basic_rag_available": BASIC_RAG_AVAILABLE
    }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "metrics": app_state.metrics,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - app_state.metrics["uptime_start"]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Kingfisher Simple A2A Agent",
        "version": "2.1.0",
        "status": "operational",
        "agent_card_url": "/.well-known/agent.json",
        "health_url": "/health",
        "metrics_url": "/metrics"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 