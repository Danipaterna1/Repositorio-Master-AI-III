"""
Kingfisher A2A HTTP Server - FastAPI Implementation

Servidor HTTP compatible con Google A2A Protocol.
"""

import json
import uuid
from typing import Dict, Any
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from agents.protocol.agent_card import get_agent_card
    from agents.protocol.task_manager import KingfisherTaskManager
except ImportError:
    # Fallback para importaciones relativas
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from protocol.agent_card import get_agent_card
    from protocol.task_manager import KingfisherTaskManager

class KingfisherA2AServer:
    """Servidor HTTP A2A-compliant para Kingfisher Agent"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required but not installed")
        
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Kingfisher RAG Agent",
            description="A2A-compliant document processing agent",
            version="1.0.0"
        )
        
        self._task_manager = None
        self._agent_card = None
        
        self._setup_middleware()
        self._setup_routes()
    
    @property
    def task_manager(self):
        """Lazy-loaded task manager"""
        if self._task_manager is None:
            self._task_manager = KingfisherTaskManager()
        return self._task_manager
    
    @property
    def agent_card(self):
        """Lazy-loaded agent card"""
        if self._agent_card is None:
            self._agent_card = get_agent_card()
        return self._agent_card
    
    def _setup_middleware(self):
        """Configura middleware de CORS"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Configura todos los endpoints A2A"""
        
        @self.app.get("/.well-known/agent.json")
        async def get_agent_card():
            """Endpoint estándar A2A para agent discovery"""
            return JSONResponse(content=self.agent_card)
        
        @self.app.post("/tasks/send")
        async def send_task(task_data: Dict[str, Any]):
            """Endpoint A2A para envío de tasks síncronos"""
            try:
                task_id = await self.task_manager.create_task(task_data)
                result_state = await self.task_manager.process_task(task_id)
                
                return {
                    "id": str(uuid.uuid4()),
                    "jsonrpc": "2.0",
                    "result": result_state
                }
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={
                        "id": task_data.get("id", str(uuid.uuid4())),
                        "jsonrpc": "2.0",
                        "error": {"code": -32000, "message": str(e)}
                    }
                )
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint básico"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        
        @self.app.get("/health/detailed")
        async def detailed_health_check():
            """Health check detallado de todos los componentes"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "task_manager": "operational",
                    "protocol": "A2A compliant",
                    "version": "1.0.0"
                }
            }
        
        @self.app.get("/health/ready")
        async def readiness_check():
            """Readiness probe para Kubernetes"""
            return {
                "status": "ready", 
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Endpoint de métricas del servidor"""
            return {
                "active_tasks": len(self.task_manager.active_tasks),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/")
        async def root():
            """Endpoint raíz con información básica"""
            return {
                "name": "Kingfisher RAG Agent",
                "version": "1.0.0",
                "protocol": "Google A2A",
                "status": "operational"
            }
    
    def get_app(self):
        """Retorna la instancia de FastAPI para deployment"""
        return self.app

# Instancia global para fácil acceso - lazy loaded
kingfisher_server = None

def get_app():
    """Get FastAPI app instance with lazy loading"""
    global kingfisher_server
    if kingfisher_server is None:
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required but not installed")
        kingfisher_server = KingfisherA2AServer()
    return kingfisher_server.get_app()

# Create a simple mock app for import compatibility
if FASTAPI_AVAILABLE:
    from fastapi import FastAPI
    app = FastAPI(title="Kingfisher RAG Agent (Mock)")
    
    @app.get("/health")
    async def mock_health():
        return {"status": "mock", "message": "Use get_app() for full functionality"}
else:
    app = None 