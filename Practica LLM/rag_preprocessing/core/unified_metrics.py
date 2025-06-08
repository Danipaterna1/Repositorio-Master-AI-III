"""
Unified Metrics para sistema RAG triple integrado
Métricas consolidadas de Vector + Graph + Metadata processing
"""

import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class ProcessingStage(str, Enum):
    """Etapas del procesamiento"""
    INITIALIZATION = "initialization"
    TEXT_PREPROCESSING = "text_preprocessing"
    VECTOR_PROCESSING = "vector_processing"
    GRAPH_PROCESSING = "graph_processing"
    METADATA_PROCESSING = "metadata_processing"
    INTEGRATION = "integration"
    FINALIZATION = "finalization"

class ProcessingStatus(str, Enum):
    """Estados del procesamiento"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"

@dataclass
class StageMetrics:
    """Métricas por etapa de procesamiento"""
    stage: ProcessingStage
    status: ProcessingStatus = ProcessingStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_message: Optional[str] = None
    
    # Métricas específicas por etapa
    stage_specific_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def start(self):
        """Iniciar medición de etapa"""
        self.status = ProcessingStatus.RUNNING
        self.start_time = datetime.utcnow()
        
        # Capturar métricas de sistema al inicio
        process = psutil.Process()
        self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        self.cpu_usage_percent = process.cpu_percent()
    
    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """Finalizar medición de etapa"""
        self.end_time = datetime.utcnow()
        
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        
        self.status = ProcessingStatus.SUCCESS if success else ProcessingStatus.FAILED
        
        if error_message:
            self.error_message = error_message
        
        # Capturar métricas finales de sistema
        process = psutil.Process()
        memory_final = process.memory_info().rss / 1024 / 1024
        self.memory_usage_mb = max(self.memory_usage_mb, memory_final)

@dataclass
class ComponentMetrics:
    """Métricas por componente del sistema"""
    component_name: str
    enabled: bool = False
    processed: bool = False
    success: bool = False
    processing_time: float = 0.0
    
    # Métricas específicas por componente
    items_processed: int = 0
    items_created: int = 0
    items_failed: int = 0
    quality_score: float = 0.0
    
    # Métricas detalladas por componente
    component_details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TripleProcessorMetrics:
    """Métricas completas del pipeline triple"""
    # Identificación del procesamiento
    processing_id: str
    document_id: Optional[int] = None
    content_hash: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    # Estado general
    status: ProcessingStatus = ProcessingStatus.PENDING
    total_duration: float = 0.0
    success: bool = False
    
    # Métricas por etapa
    stages: Dict[ProcessingStage, StageMetrics] = field(default_factory=dict)
    
    # Métricas por componente
    vector_metrics: ComponentMetrics = field(default_factory=lambda: ComponentMetrics("vector"))
    graph_metrics: ComponentMetrics = field(default_factory=lambda: ComponentMetrics("graph"))
    metadata_metrics: ComponentMetrics = field(default_factory=lambda: ComponentMetrics("metadata"))
    
    # Métricas del documento
    document_length: int = 0
    document_complexity: float = 0.0
    
    # Métricas de calidad
    overall_quality_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    
    # Métricas de performance
    throughput_chars_per_second: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_average_percent: float = 0.0
    
    # Errores y warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Inicializar métricas por etapa"""
        for stage in ProcessingStage:
            self.stages[stage] = StageMetrics(stage=stage)
    
    def start_stage(self, stage: ProcessingStage):
        """Iniciar medición de una etapa"""
        if stage in self.stages:
            self.stages[stage].start()
    
    def finish_stage(self, stage: ProcessingStage, success: bool = True, 
                    error_message: Optional[str] = None, **metrics):
        """Finalizar medición de una etapa"""
        if stage in self.stages:
            self.stages[stage].finish(success, error_message)
            
            # Agregar métricas específicas de la etapa
            self.stages[stage].stage_specific_metrics.update(metrics)
    
    def update_vector_metrics(self, **kwargs):
        """Actualizar métricas del componente vectorial"""
        for key, value in kwargs.items():
            if hasattr(self.vector_metrics, key):
                setattr(self.vector_metrics, key, value)
            else:
                self.vector_metrics.component_details[key] = value
    
    def update_graph_metrics(self, **kwargs):
        """Actualizar métricas del componente de grafos"""
        for key, value in kwargs.items():
            if hasattr(self.graph_metrics, key):
                setattr(self.graph_metrics, key, value)
            else:
                self.graph_metrics.component_details[key] = value
    
    def update_metadata_metrics(self, **kwargs):
        """Actualizar métricas del componente de metadata"""
        for key, value in kwargs.items():
            if hasattr(self.metadata_metrics, key):
                setattr(self.metadata_metrics, key, value)
            else:
                self.metadata_metrics.component_details[key] = value
    
    def add_error(self, error: str):
        """Agregar error al procesamiento"""
        self.errors.append(f"{datetime.utcnow().isoformat()}: {error}")
    
    def add_warning(self, warning: str):
        """Agregar warning al procesamiento"""
        self.warnings.append(f"{datetime.utcnow().isoformat()}: {warning}")
    
    def finalize(self, success: bool = True):
        """Finalizar medición completa"""
        self.end_time = datetime.utcnow()
        self.success = success
        self.status = ProcessingStatus.SUCCESS if success else ProcessingStatus.FAILED
        
        if self.start_time and self.end_time:
            self.total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Calcular throughput
        if self.total_duration > 0 and self.document_length > 0:
            self.throughput_chars_per_second = self.document_length / self.total_duration
        
        # Calcular métricas de calidad
        self._calculate_quality_metrics()
        
        # Calcular métricas de performance
        self._calculate_performance_metrics()
    
    def _calculate_quality_metrics(self):
        """Calcular métricas de calidad"""
        # Completeness score: qué porcentaje de componentes se procesaron exitosamente
        enabled_components = [
            self.vector_metrics.enabled,
            self.graph_metrics.enabled, 
            self.metadata_metrics.enabled
        ]
        successful_components = [
            self.vector_metrics.success,
            self.graph_metrics.success,
            self.metadata_metrics.success
        ]
        
        enabled_count = sum(enabled_components)
        successful_count = sum(successful_components)
        
        if enabled_count > 0:
            self.completeness_score = successful_count / enabled_count
        
        # Overall quality score: promedio ponderado de calidad por componente
        quality_scores = []
        weights = []
        
        if self.vector_metrics.enabled:
            quality_scores.append(self.vector_metrics.quality_score)
            weights.append(0.4)  # Vector tiene peso 40%
        
        if self.graph_metrics.enabled:
            quality_scores.append(self.graph_metrics.quality_score)
            weights.append(0.4)  # Graph tiene peso 40%
        
        if self.metadata_metrics.enabled:
            quality_scores.append(self.metadata_metrics.quality_score)
            weights.append(0.2)  # Metadata tiene peso 20%
        
        if quality_scores:
            total_weight = sum(weights)
            weighted_scores = [score * weight for score, weight in zip(quality_scores, weights)]
            self.overall_quality_score = sum(weighted_scores) / total_weight
    
    def _calculate_performance_metrics(self):
        """Calcular métricas de performance"""
        # Memory peak: máximo de todas las etapas
        memory_values = [stage.memory_usage_mb for stage in self.stages.values()]
        self.memory_peak_mb = max(memory_values) if memory_values else 0.0
        
        # CPU average: promedio de todas las etapas con tiempo > 0
        cpu_values = [stage.cpu_usage_percent for stage in self.stages.values() 
                     if stage.duration_seconds > 0]
        self.cpu_average_percent = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen de métricas"""
        return {
            "processing_id": self.processing_id,
            "document_id": self.document_id,
            "status": self.status.value,
            "success": self.success,
            "total_duration": self.total_duration,
            "document_length": self.document_length,
            "components": {
                "vector": {
                    "enabled": self.vector_metrics.enabled,
                    "success": self.vector_metrics.success,
                    "processing_time": self.vector_metrics.processing_time,
                    "items_created": self.vector_metrics.items_created
                },
                "graph": {
                    "enabled": self.graph_metrics.enabled,
                    "success": self.graph_metrics.success,
                    "processing_time": self.graph_metrics.processing_time,
                    "items_created": self.graph_metrics.items_created
                },
                "metadata": {
                    "enabled": self.metadata_metrics.enabled,
                    "success": self.metadata_metrics.success,
                    "processing_time": self.metadata_metrics.processing_time,
                    "items_created": self.metadata_metrics.items_created
                }
            },
            "quality": {
                "overall_score": self.overall_quality_score,
                "completeness_score": self.completeness_score,
                "consistency_score": self.consistency_score
            },
            "performance": {
                "throughput_chars_per_second": self.throughput_chars_per_second,
                "memory_peak_mb": self.memory_peak_mb,
                "cpu_average_percent": self.cpu_average_percent
            },
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings)
        }
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Obtener reporte detallado completo"""
        summary = self.get_summary()
        
        # Agregar detalles por etapa
        stages_detail = {}
        for stage_name, stage_metrics in self.stages.items():
            stages_detail[stage_name.value] = {
                "status": stage_metrics.status.value,
                "duration_seconds": stage_metrics.duration_seconds,
                "memory_usage_mb": stage_metrics.memory_usage_mb,
                "cpu_usage_percent": stage_metrics.cpu_usage_percent,
                "error_message": stage_metrics.error_message,
                "specific_metrics": stage_metrics.stage_specific_metrics
            }
        
        summary["stages"] = stages_detail
        summary["errors"] = self.errors
        summary["warnings"] = self.warnings
        
        # Agregar detalles de componentes
        summary["components"]["vector"]["details"] = self.vector_metrics.component_details
        summary["components"]["graph"]["details"] = self.graph_metrics.component_details
        summary["components"]["metadata"]["details"] = self.metadata_metrics.component_details
        
        return summary

class MetricsCollector:
    """Collector para métricas del sistema"""
    
    def __init__(self):
        self.active_metrics: Dict[str, TripleProcessorMetrics] = {}
        self.completed_metrics: List[TripleProcessorMetrics] = []
        self.system_start_time = datetime.utcnow()
    
    def create_metrics(self, processing_id: str, document_length: int = 0) -> TripleProcessorMetrics:
        """Crear nueva instancia de métricas"""
        metrics = TripleProcessorMetrics(
            processing_id=processing_id,
            document_length=document_length
        )
        self.active_metrics[processing_id] = metrics
        return metrics
    
    def get_metrics(self, processing_id: str) -> Optional[TripleProcessorMetrics]:
        """Obtener métricas por ID"""
        return self.active_metrics.get(processing_id)
    
    def complete_metrics(self, processing_id: str):
        """Marcar métricas como completadas"""
        if processing_id in self.active_metrics:
            metrics = self.active_metrics.pop(processing_id)
            self.completed_metrics.append(metrics)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        total_processed = len(self.completed_metrics)
        successful = sum(1 for m in self.completed_metrics if m.success)
        
        if total_processed > 0:
            avg_duration = sum(m.total_duration for m in self.completed_metrics) / total_processed
            avg_quality = sum(m.overall_quality_score for m in self.completed_metrics) / total_processed
        else:
            avg_duration = 0.0
            avg_quality = 0.0
        
        uptime = (datetime.utcnow() - self.system_start_time).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "total_processed": total_processed,
            "successful_processed": successful,
            "success_rate": successful / total_processed if total_processed > 0 else 0.0,
            "active_processing": len(self.active_metrics),
            "average_duration": avg_duration,
            "average_quality_score": avg_quality,
            "throughput_per_hour": total_processed / (uptime / 3600) if uptime > 0 else 0.0
        } 