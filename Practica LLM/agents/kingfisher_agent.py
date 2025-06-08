"""
Kingfisher Agent - Google A2A Implementation

Agente principal que implementa las interfaces Google A2A para
proporcionar capacidades especializadas de procesamiento RAG
y gestión de conocimiento en el ecosistema multi-agente.

El agente Kingfisher opera como un especialista en:
- Document Processing: Triple storage pipeline
- Knowledge Retrieval: Vector + Graph + Metadata search
- Metadata Analysis: Knowledge base analytics
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .protocol.agent_card import KINGFISHER_AGENT_CARD

# Lazy loading para task manager
KingfisherTaskManager = None
TaskStatus = None
TaskType = None

def _load_task_manager():
    """Load task manager with lazy loading"""
    global KingfisherTaskManager, TaskStatus, TaskType
    
    if KingfisherTaskManager is None:
        try:
            from .protocol.task_manager import KingfisherTaskManager as _KTM, TaskStatus as _TS, TaskType as _TT
            KingfisherTaskManager = _KTM
            TaskStatus = _TS
            TaskType = _TT
        except ImportError:
            # Fallback to simplified version
            from .protocol.task_manager_simple import KingfisherTaskManager as _KTM, TaskStatus as _TS, TaskType as _TT
            KingfisherTaskManager = _KTM
            TaskStatus = _TS
            TaskType = _TT
    
    return KingfisherTaskManager is not None

# Lazy loading del servidor A2A para evitar dependency conflicts
A2A_SERVER_AVAILABLE = None  # Will be determined on first use
KingfisherA2AServer = None

def _load_a2a_server():
    """Load A2A server with lazy loading"""
    global A2A_SERVER_AVAILABLE, KingfisherA2AServer
    
    if A2A_SERVER_AVAILABLE is None:
        try:
            from .server.a2a_server import KingfisherA2AServer as _KingfisherA2AServer
            KingfisherA2AServer = _KingfisherA2AServer
            A2A_SERVER_AVAILABLE = True
        except ImportError:
            A2A_SERVER_AVAILABLE = False
            KingfisherA2AServer = None
    
    return A2A_SERVER_AVAILABLE

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KingfisherAgent:
    """
    Agente Kingfisher principal compatible con Google A2A Protocol.
    
    Proporciona una interfaz unificada para todas las capabilities
    del sistema Kingfisher, incluyendo procesamiento de documentos,
    recuperación de conocimiento y análisis de metadatos.
    
    Puede operar tanto en modo standalone como servidor HTTP A2A.
    """
    
    def __init__(self, 
                 agent_id: Optional[str] = None,
                 enable_http_server: bool = True,
                 server_host: str = "localhost",
                 server_port: int = 8000):
        
        self.agent_id = agent_id or f"kingfisher-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.agent_card = KINGFISHER_AGENT_CARD
        self._task_manager = None
        
        # HTTP Server A2A (opcional) - lazy loaded
        self.http_server = None
        self._enable_http_server = enable_http_server
        self._server_host = server_host
        self._server_port = server_port
        
        # Métricas del agente
        self.agent_metrics = {
            "created_at": datetime.now(),
            "tasks_processed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "last_activity": None
        }
        
        logger.info(f"Kingfisher Agent {self.agent_id} initialized")
    
    @property
    def task_manager(self):
        """Lazy-loaded task manager"""
        if self._task_manager is None:
            self._task_manager = KingfisherTaskManager()
        return self._task_manager
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa un task A2A usando el pipeline Kingfisher.
        
        Esta es la interfaz principal para procesamiento de tasks,
        compatible con el protocolo Google A2A.
        
        Args:
            task_data: Task en formato A2A con message, parameters, etc.
            
        Returns:
            Resultado del task con artifacts y metadata
        """
        start_time = datetime.now()
        
        try:
            # Crear task en el task manager
            task_id = await self.task_manager.create_task(task_data)
            
            logger.info(f"Processing task {task_id} of type {self.task_manager.get_task_status(task_id)['task_type']}")
            
            # Procesar usando LangGraph workflow
            result_state = await self.task_manager.process_task(task_id)
            
            # Actualizar métricas
            self.agent_metrics["tasks_processed"] += 1
            self.agent_metrics["last_activity"] = datetime.now()
            
            if result_state["processing_status"] == TaskStatus.COMPLETED:
                self.agent_metrics["successful_tasks"] += 1
                logger.info(f"Task {task_id} completed successfully")
            else:
                self.agent_metrics["failed_tasks"] += 1
                logger.error(f"Task {task_id} failed: {result_state.get('error_message')}")
            
            return self._format_agent_response(result_state, start_time)
            
        except Exception as e:
            self.agent_metrics["failed_tasks"] += 1
            logger.error(f"Error processing task: {str(e)}")
            
            return {
                "status": "failed",
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "agent_id": self.agent_id
            }
    
    async def process_document(self, 
                             content: str, 
                             title: Optional[str] = None,
                             processing_mode: str = "TRIPLE_FULL",
                             include_llm: bool = True) -> Dict[str, Any]:
        """
        Conveniencia para procesamiento directo de documentos.
        
        Wrapper simplificado para la capability "process_documents"
        sin necesidad de formatear como task A2A completo.
        """
        task_data = {
            "id": f"doc-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "message": {
                "role": "user",
                "parts": [{
                    "kind": "text",
                    "text": content
                }]
            },
            "parameters": {
                "processing_mode": processing_mode,
                "include_llm": include_llm,
                "title": title or "Direct Document Processing"
            }
        }
        
        return await self.process_task(task_data)
    
    async def retrieve_knowledge(self, 
                               query: str,
                               search_mode: str = "hybrid",
                               top_k: int = 5) -> Dict[str, Any]:
        """
        Conveniencia para recuperación directa de conocimiento.
        
        Wrapper simplificado para la capability "retrieve_knowledge".
        """
        task_data = {
            "id": f"query-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "message": {
                "role": "user",
                "parts": [{
                    "kind": "text",
                    "text": query
                }]
            },
            "parameters": {
                "search_mode": search_mode,
                "top_k": top_k
            }
        }
        
        return await self.process_task(task_data)
    
    async def analyze_metadata(self, 
                             analysis_type: str = "documents") -> Dict[str, Any]:
        """
        Conveniencia para análisis directo de metadatos.
        
        Wrapper simplificado para la capability "analyze_metadata".
        """
        task_data = {
            "id": f"analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "message": {
                "role": "user",
                "parts": [{
                    "kind": "text",
                    "text": f"Analyze {analysis_type} metadata"
                }]
            },
            "parameters": {
                "analysis_type": analysis_type
            }
        }
        
        return await self.process_task(task_data)
    
    def _format_agent_response(self, 
                             task_state: Dict[str, Any], 
                             start_time: datetime) -> Dict[str, Any]:
        """
        Formatea la respuesta del agente con metadata adicional.
        """
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "task_id": task_state["task_id"],
            "status": task_state["processing_status"],
            "artifacts": task_state.get("artifacts", []),
            "metadata": {
                **task_state.get("metadata", {}),
                "agent_id": self.agent_id,
                "processing_time": processing_time,
                "task_type": task_state.get("task_type"),
                "timestamp": datetime.now().isoformat()
            },
            "error_message": task_state.get("error_message")
        }
    
    def get_capabilities(self) -> List[str]:
        """Retorna lista de capabilities disponibles"""
        return [skill["id"] for skill in self.agent_card["skills"]]
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Retorna información completa del agente"""
        return {
            "agent_id": self.agent_id,
            "name": self.agent_card["name"],
            "version": self.agent_card["version"],
            "capabilities": self.get_capabilities(),
            "metrics": self.agent_metrics,
            "http_server_enabled": self.http_server is not None,
            "status": "operational"
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas del agente"""
        uptime = (datetime.now() - self.agent_metrics["created_at"]).total_seconds()
        
        return {
            "agent_id": self.agent_id,
            "uptime_seconds": uptime,
            "tasks": {
                "total_processed": self.agent_metrics["tasks_processed"],
                "successful": self.agent_metrics["successful_tasks"],
                "failed": self.agent_metrics["failed_tasks"],
                "success_rate": (
                    self.agent_metrics["successful_tasks"] / 
                    max(self.agent_metrics["tasks_processed"], 1)
                )
            },
            "last_activity": self.agent_metrics["last_activity"].isoformat() if self.agent_metrics["last_activity"] else None,
            "performance": {
                "tasks_per_hour": (
                    self.agent_metrics["tasks_processed"] / 
                    max(uptime / 3600, 1/3600)  # Mínimo 1 segundo
                )
            }
        }
    
    async def start_http_server(self):
        """
        Inicia el servidor HTTP A2A si está configurado.
        
        Para uso programático cuando se quiere control total del lifecycle.
        """
        if not self._enable_http_server:
            raise RuntimeError("HTTP server disabled. Set enable_http_server=True")
        
        # Lazy load A2A server
        if not _load_a2a_server():
            raise RuntimeError("A2A Server not available. Install FastAPI: pip install fastapi uvicorn")
        
        if not self.http_server:
            self.http_server = KingfisherA2AServer(
                host=self._server_host,
                port=self._server_port
            )
        
        logger.info(f"Starting Kingfisher A2A HTTP Server on {self._server_host}:{self._server_port}")
        
        return self.http_server.get_app()
    
    async def shutdown(self):
        """Cierra el agente limpiamente"""
        logger.info(f"Shutting down Kingfisher Agent {self.agent_id}")
        
        # Limpiar tasks activos
        if self.task_manager:
            active_count = len(self.task_manager.active_tasks)
            if active_count > 0:
                logger.warning(f"Shutting down with {active_count} active tasks")
            
            # Cancelar tasks activos
            for task_id in list(self.task_manager.active_tasks.keys()):
                self.task_manager.cancel_task(task_id)
        
        # Limpiar recursos del HTTP server si es necesario
        if self.http_server:
            # Cleanup si se requiere
            pass
        
        logger.info("Kingfisher Agent shutdown completed")

# Factory function para fácil creación
def create_kingfisher_agent(**kwargs) -> KingfisherAgent:
    """
    Factory para crear instancias de KingfisherAgent.
    
    Args:
        **kwargs: Argumentos para KingfisherAgent constructor
        
    Returns:
        KingfisherAgent configurado y listo para usar
    """
    return KingfisherAgent(**kwargs)

# Instancia global para uso directo
default_agent = None

def get_default_agent() -> KingfisherAgent:
    """Retorna la instancia por defecto del agente"""
    global default_agent
    if default_agent is None:
        default_agent = create_kingfisher_agent(enable_http_server=False)
    return default_agent