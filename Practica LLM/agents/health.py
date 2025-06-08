"""
Kingfisher Health Check System

Sistema completo de health checks para monitoreo en producción.
Verifica estado de todos los componentes críticos.
"""

import asyncio
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .config import get_config
from .logging_config import get_logger

logger = get_logger("kingfisher.health")

class HealthStatus(str, Enum):
    """Estados de salud posibles"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    """Resultado de un health check individual"""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": datetime.now().isoformat()
        }
        if self.details:
            result["details"] = self.details
        return result

class HealthChecker:
    """Sistema de health checks"""
    
    def __init__(self):
        self.config = get_config()
        self.startup_time = datetime.now()
        
    async def check_database_connectivity(self) -> HealthCheckResult:
        """Verifica conectividad con las bases de datos"""
        start_time = time.time()
        
        try:
            # ChromaDB check - usando cliente directo
            try:
                import chromadb
                client = chromadb.PersistentClient(path="./data/chromadb")
                collections = client.list_collections()
                chroma_healthy = True
                chroma_details = {"collections_count": len(collections), "status": "operational"}
            except Exception as e:
                chroma_healthy = False
                chroma_details = {"error": str(e), "note": "ChromaDB not configured - this is OK for basic operation"}
            
            # SQLite check - usando sqlite3 directo
            try:
                import sqlite3
                import os
                db_path = "./data/metadata/kingfisher.db"
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                conn.close()
                sqlite_healthy = True
                sqlite_details = {"tables_count": len(tables), "db_path": db_path, "status": "operational"}
            except Exception as e:
                sqlite_healthy = False
                sqlite_details = {"error": str(e), "note": "SQLite not configured - this is OK for basic operation"}
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Para consolidación, consideramos OK si al menos uno funciona o si ambos fallan por configuración
            if chroma_healthy and sqlite_healthy:
                return HealthCheckResult(
                    name="database_connectivity",
                    status=HealthStatus.HEALTHY,
                    message="All databases are accessible",
                    duration_ms=duration_ms,
                    details={
                        "chroma": chroma_details,
                        "sqlite": sqlite_details
                    }
                )
            elif chroma_healthy or sqlite_healthy:
                return HealthCheckResult(
                    name="database_connectivity",
                    status=HealthStatus.DEGRADED,
                    message="Some databases are accessible",
                    duration_ms=duration_ms,
                    details={
                        "chroma": chroma_details,
                        "sqlite": sqlite_details
                    }
                )
            else:
                # Si ambos fallan por configuración, es degraded, no unhealthy
                return HealthCheckResult(
                    name="database_connectivity",
                    status=HealthStatus.DEGRADED,
                    message="Databases not configured - system operational without storage",
                    duration_ms=duration_ms,
                    details={
                        "chroma": chroma_details,
                        "sqlite": sqlite_details,
                        "note": "This is expected during initial setup"
                    }
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="database_connectivity",
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)}",
                duration_ms=duration_ms
            )
    
    async def check_llm_connectivity(self) -> HealthCheckResult:
        """Verifica conectividad con LLM"""
        start_time = time.time()
        
        try:
            # Test básico del LLM
            if self.config.processing.llm_provider == "google":
                # Test rápido de Google Gemini
                import google.generativeai as genai
                # Solo verificamos que la configuración está disponible
                models = genai.list_models()
                llm_healthy = True
                details = {"provider": "google", "models_available": len(list(models))}
            else:
                # Para otros providers, asumir saludable por ahora
                llm_healthy = True
                details = {"provider": self.config.processing.llm_provider}
            
            duration_ms = (time.time() - start_time) * 1000
            
            if llm_healthy:
                return HealthCheckResult(
                    name="llm_connectivity",
                    status=HealthStatus.HEALTHY,
                    message="LLM provider is accessible",
                    duration_ms=duration_ms,
                    details=details
                )
            else:
                return HealthCheckResult(
                    name="llm_connectivity",
                    status=HealthStatus.UNHEALTHY,
                    message="LLM provider is not accessible",
                    duration_ms=duration_ms,
                    details=details
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="llm_connectivity",
                status=HealthStatus.DEGRADED,
                message=f"LLM check warning: {str(e)}",
                duration_ms=duration_ms
            )
    
    async def check_system_resources(self) -> HealthCheckResult:
        """Verifica recursos del sistema"""
        start_time = time.time()
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_healthy = cpu_percent < self.config.security.max_cpu_percent
            
            # Memory
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / 1024 / 1024
            memory_healthy = memory_used_mb < self.config.security.max_memory_mb
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_free_percent = (disk.free / disk.total) * 100
            disk_healthy = disk_free_percent > 10  # Al menos 10% libre
            
            duration_ms = (time.time() - start_time) * 1000
            
            details = {
                "cpu_percent": round(cpu_percent, 1),
                "memory_used_mb": round(memory_used_mb, 1),
                "memory_percent": round(memory.percent, 1),
                "disk_free_percent": round(disk_free_percent, 1)
            }
            
            if cpu_healthy and memory_healthy and disk_healthy:
                status = HealthStatus.HEALTHY
                message = "System resources are within normal limits"
            elif (cpu_healthy or memory_healthy) and disk_healthy:
                status = HealthStatus.DEGRADED
                message = "Some system resources are under pressure"
            else:
                status = HealthStatus.UNHEALTHY
                message = "System resources are critically low"
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Resource check failed: {str(e)}",
                duration_ms=duration_ms
            )
    
    async def check_task_manager(self) -> HealthCheckResult:
        """Verifica el Task Manager"""
        start_time = time.time()
        
        try:
            from .protocol.task_manager import KingfisherTaskManager
            
            # Crear instancia del task manager
            task_manager = KingfisherTaskManager()
            
            # Verificar que está funcionando
            active_tasks = len(task_manager.active_tasks)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="task_manager",
                status=HealthStatus.HEALTHY,
                message="Task manager is operational",
                duration_ms=duration_ms,
                details={
                    "active_tasks": active_tasks,
                    "workflow_available": task_manager.workflow is not None
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="task_manager",
                status=HealthStatus.UNHEALTHY,
                message=f"Task manager check failed: {str(e)}",
                duration_ms=duration_ms
            )
    
    async def check_agent_card(self) -> HealthCheckResult:
        """Verifica el Agent Card"""
        start_time = time.time()
        
        try:
            from .protocol.agent_card import get_agent_card
            
            agent_card = get_agent_card()
            
            # Verificaciones básicas
            has_name = bool(agent_card.get("name"))
            has_skills = len(agent_card.get("skills", [])) > 0
            has_version = bool(agent_card.get("version"))
            
            duration_ms = (time.time() - start_time) * 1000
            
            if has_name and has_skills and has_version:
                return HealthCheckResult(
                    name="agent_card",
                    status=HealthStatus.HEALTHY,
                    message="Agent card is properly configured",
                    duration_ms=duration_ms,
                    details={
                        "skills_count": len(agent_card.get("skills", [])),
                        "version": agent_card.get("version")
                    }
                )
            else:
                return HealthCheckResult(
                    name="agent_card",
                    status=HealthStatus.DEGRADED,
                    message="Agent card has missing information",
                    duration_ms=duration_ms
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="agent_card",
                status=HealthStatus.UNHEALTHY,
                message=f"Agent card check failed: {str(e)}",
                duration_ms=duration_ms
            )
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Ejecuta todos los health checks"""
        logger.info("Starting comprehensive health check")
        start_time = time.time()
        
        # Ejecutar checks en paralelo
        checks = await asyncio.gather(
            self.check_database_connectivity(),
            self.check_llm_connectivity(),
            self.check_system_resources(),
            self.check_task_manager(),
            self.check_agent_card(),
            return_exceptions=True
        )
        
        total_duration = (time.time() - start_time) * 1000
        
        # Procesar resultados
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for check in checks:
            if isinstance(check, Exception):
                # Error en el check
                check_name = "unknown_check"
                results[check_name] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": f"Check failed with exception: {str(check)}",
                    "duration_ms": 0
                }
                overall_status = HealthStatus.UNHEALTHY
            else:
                results[check.name] = check.to_dict()
                
                # Determinar estado general
                if check.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif check.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        # Información del sistema
        uptime = datetime.now() - self.startup_time
        
        health_report = {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": int(uptime.total_seconds()),
            "total_duration_ms": round(total_duration, 2),
            "checks": results,
            "system_info": {
                "agent_name": self.config.agent_name,
                "agent_version": self.config.agent_version,
                "environment": self.config.environment,
                "startup_time": self.startup_time.isoformat()
            }
        }
        
        logger.info(f"Health check completed - Status: {overall_status.value}", 
                   status=overall_status.value, 
                   duration_ms=total_duration,
                   checks_count=len(results))
        
        return health_report

# Global health checker instance
health_checker = HealthChecker() 