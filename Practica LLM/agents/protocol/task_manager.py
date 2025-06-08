"""
Kingfisher Task Manager - LangGraph State Machine

Implementa el manejo de tasks del protocolo A2A usando LangGraph
para workflow management y state transitions.
"""

import uuid
import asyncio
from typing import TypedDict, Annotated, Literal, Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Fallback sin LangGraph
    class StateGraph:
        def __init__(self, *args, **kwargs): pass
        def add_node(self, *args, **kwargs): pass
        def add_conditional_edges(self, *args, **kwargs): pass
        def add_edge(self, *args, **kwargs): pass
        def compile(self): return None

class TaskType(str, Enum):
    """Tipos de tasks que maneja Kingfisher"""
    PROCESS_DOCUMENTS = "process_documents"
    RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
    ANALYZE_METADATA = "analyze_metadata"

class TaskStatus(str, Enum):
    """Estados de un task según protocolo A2A"""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

class KingfisherState(TypedDict):
    """Estado del workflow LangGraph para Kingfisher"""
    task_id: str
    task_type: TaskType
    messages: Annotated[List[Dict], add_messages] if LANGGRAPH_AVAILABLE else List[Dict]
    input_data: Dict[str, Any]
    processing_status: TaskStatus
    artifacts: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    error_message: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]

class KingfisherTaskManager:
    """
    Gestor de tasks A2A con LangGraph state machine.
    
    Maneja el lifecycle completo de tasks desde submission hasta completion,
    incluyendo routing, processing y artifact generation.
    """
    
    def __init__(self):
        self.active_tasks: Dict[str, KingfisherState] = {}
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> Optional[Any]:
        """Construye el workflow LangGraph para procesamiento de tasks"""
        if not LANGGRAPH_AVAILABLE:
            return None
            
        workflow = StateGraph(KingfisherState)
        
        # Agregar nodos de procesamiento (sin error_handler para evitar nodos no alcanzables)
        workflow.add_node("document_processor", self._document_processor_node)
        workflow.add_node("knowledge_retriever", self._knowledge_retriever_node)
        workflow.add_node("metadata_analyzer", self._metadata_analyzer_node)
        
        # Routing condicional basado en task_type
        workflow.add_conditional_edges(
            START,
            self._route_task,
            {
                TaskType.PROCESS_DOCUMENTS: "document_processor",
                TaskType.RETRIEVE_KNOWLEDGE: "knowledge_retriever",
                TaskType.ANALYZE_METADATA: "metadata_analyzer"
            }
        )
        
        # Todos los nodos terminan directamente (manejo de errores interno)
        workflow.add_edge("document_processor", END)
        workflow.add_edge("knowledge_retriever", END)
        workflow.add_edge("metadata_analyzer", END)
        
        return workflow.compile()
    
    def _route_task(self, state: KingfisherState) -> TaskType:
        """Determina qué nodo ejecutar basado en el tipo de task"""
        return state["task_type"]
    
    async def _document_processor_node(self, state: KingfisherState) -> Dict[str, Any]:
        """
        Nodo de procesamiento de documentos usando TripleProcessor existente.
        
        Usa el sistema existente sin modificaciones, envolviendo en formato A2A.
        """
        try:
            # Lazy import para evitar dependency conflicts
            try:
                from rag_preprocessing.core.triple_processor import TripleProcessor
                from rag_preprocessing.core.pipeline_config import ProcessingMode
            except ImportError as e:
                return {
                    "processing_status": TaskStatus.FAILED,
                    "error_message": f"TripleProcessor not available: {str(e)}",
                    "end_time": datetime.now()
                }
            
            # Extraer parámetros del input
            content = state["input_data"].get("content", "")
            files = state["input_data"].get("files", [])
            processing_mode = state["input_data"].get("processing_mode", "TRIPLE_FULL")
            include_llm = state["input_data"].get("include_llm", True)
            
            # Si hay archivos, usar el primer archivo como contenido principal
            if files and len(files) > 0:
                # En implementación real, leer el archivo
                # Por ahora usar el content como fallback
                pass
            
            # Configurar TripleProcessor
            processor = TripleProcessor()
            
            # Procesar documento usando sistema existente
            result = await asyncio.to_thread(
                processor.process_document,
                text=content,
                document_title=state["input_data"].get("title", f"Document {state['task_id'][:8]}"),
                mode=ProcessingMode[processing_mode],
                include_llm=include_llm
            )
            
            # Formatear resultado como artifact A2A
            artifacts = [{
                "name": "processing_result",
                "description": "Document processing completed successfully",
                "type": "data",
                "content": {
                    "document_id": result.document_id,
                    "chunks_created": result.chunks_count,
                    "embeddings_stored": result.embeddings_count,
                    "entities_extracted": result.entities_count,
                    "relationships_found": result.relationships_count,
                    "communities_detected": result.communities_count,
                    "processing_time_seconds": result.total_time,
                    "storage_systems": result.storage_systems_used
                }
            }]
            
            return {
                "processing_status": TaskStatus.COMPLETED,
                "artifacts": artifacts,
                "end_time": datetime.now(),
                "metadata": {
                    "processing_mode": processing_mode,
                    "llm_enhanced": include_llm,
                    "performance_metrics": result.performance_metrics
                }
            }
            
        except Exception as e:
            return {
                "processing_status": TaskStatus.FAILED,
                "error_message": str(e),
                "end_time": datetime.now()
            }
    
    async def _knowledge_retriever_node(self, state: KingfisherState) -> Dict[str, Any]:
        """
        Nodo de recuperación de conocimiento usando sistemas existentes.
        
        Integra vector search + graph traversal + metadata filtering.
        """
        try:
            query = state["input_data"].get("content", "")
            search_mode = state["input_data"].get("search_mode", "hybrid")
            top_k = state["input_data"].get("top_k", 5)
            
            # Lazy import para evitar dependency conflicts
            try:
                from rag_preprocessing.storage.vector.chroma_manager import ChromaManager
            except ImportError as e:
                return {
                    "processing_status": TaskStatus.FAILED,
                    "error_message": f"ChromaManager not available: {str(e)}",
                    "end_time": datetime.now()
                }
            
            chroma_manager = ChromaManager()
            
            # Realizar búsqueda vectorial
            vector_results = await asyncio.to_thread(
                chroma_manager.search_similar,
                query_text=query,
                n_results=top_k
            )
            
            # Formatear resultados como artifacts A2A
            artifacts = [{
                "name": "knowledge_results",
                "description": f"Retrieved {len(vector_results)} relevant documents",
                "type": "data",
                "content": {
                    "query": query,
                    "search_mode": search_mode,
                    "results_count": len(vector_results),
                    "results": [
                        {
                            "content": result.get("documents", [""])[0][:500] + "..." if result.get("documents") else "",
                            "relevance_score": result.get("distances", [0])[0],
                            "metadata": result.get("metadatas", [{}])[0]
                        }
                        for result in [vector_results] if vector_results
                    ]
                }
            }]
            
            return {
                "processing_status": TaskStatus.COMPLETED,
                "artifacts": artifacts,
                "end_time": datetime.now(),
                "metadata": {
                    "search_mode": search_mode,
                    "query_length": len(query),
                    "results_retrieved": len(vector_results) if vector_results else 0
                }
            }
            
        except Exception as e:
            return {
                "processing_status": TaskStatus.FAILED,
                "error_message": str(e),
                "end_time": datetime.now()
            }
    
    async def _metadata_analyzer_node(self, state: KingfisherState) -> Dict[str, Any]:
        """
        Nodo de análisis de metadatos usando SQLiteManager existente.
        
        Proporciona estadísticas y análisis de la base de conocimiento.
        """
        try:
            analysis_type = state["input_data"].get("analysis_type", "documents")
            
            # Lazy import para evitar dependency conflicts
            try:
                from rag_preprocessing.storage.metadata.sqlite_manager import SQLiteManager
            except ImportError as e:
                return {
                    "processing_status": TaskStatus.FAILED,
                    "error_message": f"SQLiteManager not available: {str(e)}",
                    "end_time": datetime.now()
                }
            
            metadata_manager = SQLiteManager()
            
            # Obtener estadísticas según el tipo de análisis
            if analysis_type == "documents":
                stats = await asyncio.to_thread(metadata_manager.get_document_stats)
            elif analysis_type == "entities":
                stats = await asyncio.to_thread(metadata_manager.get_entity_stats)
            elif analysis_type == "relationships":
                stats = await asyncio.to_thread(metadata_manager.get_relationship_stats)
            elif analysis_type == "communities":
                stats = await asyncio.to_thread(metadata_manager.get_community_stats)
            else:  # metrics
                stats = await asyncio.to_thread(metadata_manager.get_processing_metrics)
            
            # Formatear como artifact A2A
            artifacts = [{
                "name": "metadata_analysis",
                "description": f"Knowledge base {analysis_type} analysis",
                "type": "data",
                "content": {
                    "analysis_type": analysis_type,
                    "statistics": stats,
                    "generated_at": datetime.now().isoformat()
                }
            }]
            
            return {
                "processing_status": TaskStatus.COMPLETED,
                "artifacts": artifacts,
                "end_time": datetime.now(),
                "metadata": {
                    "analysis_type": analysis_type,
                    "stats_generated": True
                }
            }
            
        except Exception as e:
            return {
                "processing_status": TaskStatus.FAILED,
                "error_message": str(e),
                "end_time": datetime.now()
            }
    
    def determine_task_type(self, content: str, files: List[Dict] = None) -> TaskType:
        """
        Determina el tipo de task basado en el contenido del mensaje.
        
        Usa keywords y heurísticas para clasificar el intent del usuario.
        """
        content_lower = content.lower()
        files = files or []
        
        # Si hay archivos, probablemente es procesamiento
        if files:
            return TaskType.PROCESS_DOCUMENTS
        
        # Keywords para cada tipo de task
        process_keywords = [
            "process", "upload", "document", "file", "analyze document",
            "store", "index", "chunk", "embed"
        ]
        
        retrieve_keywords = [
            "search", "find", "retrieve", "what", "how", "query", "lookup",
            "show me", "tell me", "explain", "information about"
        ]
        
        metadata_keywords = [
            "metadata", "statistics", "stats", "graph structure", "analyze",
            "summary", "report", "metrics", "performance"
        ]
        
        # Contar matches para cada categoría
        process_score = sum(1 for kw in process_keywords if kw in content_lower)
        retrieve_score = sum(1 for kw in retrieve_keywords if kw in content_lower)
        metadata_score = sum(1 for kw in metadata_keywords if kw in content_lower)
        
        # Retornar el tipo con mayor score
        if process_score > retrieve_score and process_score > metadata_score:
            return TaskType.PROCESS_DOCUMENTS
        elif metadata_score > retrieve_score and metadata_score > process_score:
            return TaskType.ANALYZE_METADATA
        else:
            return TaskType.RETRIEVE_KNOWLEDGE  # default
    
    async def create_task(self, task_data: Dict[str, Any]) -> str:
        """
        Crea un nuevo task A2A y lo registra en el sistema.
        
        Args:
            task_data: Datos del task según protocolo A2A
            
        Returns:
            task_id: ID único del task creado
        """
        task_id = task_data.get("id") or str(uuid.uuid4())
        
        # Extraer mensaje y contenido
        message = task_data.get("message", {})
        parts = message.get("parts", [])
        
        # Procesar parts del mensaje
        content = ""
        files = []
        
        for part in parts:
            if part.get("kind") == "text":
                content += part.get("text", "")
            elif part.get("kind") == "file":
                files.append(part.get("file", {}))
            elif part.get("kind") == "data":
                # Manejar data parts si es necesario
                pass
        
        # Determinar tipo de task
        task_type = self.determine_task_type(content, files)
        
        # Extraer parámetros específicos del skill
        parameters = task_data.get("parameters", {})
        
        # Crear estado inicial del task
        initial_state = KingfisherState(
            task_id=task_id,
            task_type=task_type,
            messages=[message],
            input_data={
                "content": content,
                "files": files,
                **parameters
            },
            processing_status=TaskStatus.SUBMITTED,
            artifacts=[],
            metadata={},
            error_message=None,
            start_time=datetime.now(),
            end_time=None
        )
        
        # Registrar task
        self.active_tasks[task_id] = initial_state
        
        return task_id
    
    async def process_task(self, task_id: str) -> KingfisherState:
        """
        Procesa un task usando el workflow LangGraph.
        
        Args:
            task_id: ID del task a procesar
            
        Returns:
            KingfisherState: Estado final del task procesado
        """
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        state = self.active_tasks[task_id]
        state["processing_status"] = TaskStatus.WORKING
        
        try:
            if self.workflow and LANGGRAPH_AVAILABLE:
                # Usar LangGraph workflow
                result = await self.workflow.ainvoke(state)
                self.active_tasks[task_id].update(result)
            else:
                # Fallback sin LangGraph - routing manual
                if state["task_type"] == TaskType.PROCESS_DOCUMENTS:
                    result = await self._document_processor_node(state)
                elif state["task_type"] == TaskType.RETRIEVE_KNOWLEDGE:
                    result = await self._knowledge_retriever_node(state)
                elif state["task_type"] == TaskType.ANALYZE_METADATA:
                    result = await self._metadata_analyzer_node(state)
                else:
                    result = {"processing_status": TaskStatus.FAILED, "error_message": "Unknown task type"}
                
                self.active_tasks[task_id].update(result)
            
            return self.active_tasks[task_id]
            
        except Exception as e:
            error_state = {
                "processing_status": TaskStatus.FAILED,
                "error_message": str(e),
                "end_time": datetime.now()
            }
            self.active_tasks[task_id].update(error_state)
            return self.active_tasks[task_id]
    
    def get_task_status(self, task_id: str) -> Optional[KingfisherState]:
        """Retorna el estado actual de un task"""
        return self.active_tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancela un task en progreso"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["processing_status"] = TaskStatus.CANCELED
            self.active_tasks[task_id]["end_time"] = datetime.now()
            return True
        return False
    
    def update_task_status(self, task_id: str, status: TaskStatus, **kwargs) -> bool:
        """
        Actualiza el estado de un task.
        
        Args:
            task_id: ID del task a actualizar
            status: Nuevo estado del task
            **kwargs: Campos adicionales a actualizar (error_message, artifacts, metadata, etc.)
            
        Returns:
            bool: True si se actualizó correctamente, False si no se encontró el task
        """
        if task_id not in self.active_tasks:
            return False
        
        # Actualizar status principal
        self.active_tasks[task_id]["processing_status"] = status
        
        # Actualizar timestamp de finalización si es estado terminal
        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED]:
            self.active_tasks[task_id]["end_time"] = datetime.now()
        
        # Actualizar campos adicionales
        for key, value in kwargs.items():
            if key in self.active_tasks[task_id]:
                if key == "metadata" and isinstance(value, dict):
                    # Merge metadata dictionaries
                    self.active_tasks[task_id]["metadata"].update(value)
                elif key == "artifacts" and isinstance(value, list):
                    # Extend artifacts list
                    self.active_tasks[task_id]["artifacts"].extend(value)
                else:
                    self.active_tasks[task_id][key] = value
        
        return True
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Limpia tasks completados más antiguos que max_age_hours"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        tasks_to_remove = []
        for task_id, state in self.active_tasks.items():
            if (state["processing_status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED] and
                state["end_time"] and state["end_time"].timestamp() < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.active_tasks[task_id]
        
        return len(tasks_to_remove)