"""
Kingfisher Agent Simple - Google A2A Implementation

Versión simplificada del agente Kingfisher que funciona sin servidor HTTP,
ideal para testing y uso directo.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .protocol.agent_card import KINGFISHER_AGENT_CARD
from .protocol.task_manager import KingfisherTaskManager, TaskStatus, TaskType

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KingfisherAgentSimple:
    """
    Agente Kingfisher simplificado compatible con Google A2A Protocol.
    
    Versión sin servidor HTTP para uso directo y testing.
    """
    
    def __init__(self, agent_id: Optional[str] = None):
        self.agent_id = agent_id or f"kingfisher-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.agent_card = KINGFISHER_AGENT_CARD
        self.task_manager = KingfisherTaskManager()
        
        # Métricas del agente
        self.agent_metrics = {
            "created_at": datetime.now(),
            "tasks_processed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "last_activity": None
        }
        
        logger.info(f"Kingfisher Agent Simple {self.agent_id} initialized")
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa un task A2A usando el pipeline Kingfisher"""
        start_time = datetime.now()
        
        try:
            # Crear task en el task manager
            task_id = await self.task_manager.create_task(task_data)
            
            logger.info(f"Processing task {task_id}")
            
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
        """Conveniencia para procesamiento directo de documentos"""
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
        """Conveniencia para recuperación directa de conocimiento"""
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
        """Conveniencia para análisis directo de metadatos"""
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
        """Formatea la respuesta del agente con metadata adicional"""
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
            "http_server_enabled": False,
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

# Factory function
def create_simple_agent(**kwargs) -> KingfisherAgentSimple:
    """Factory para crear instancias de KingfisherAgentSimple"""
    return KingfisherAgentSimple(**kwargs)