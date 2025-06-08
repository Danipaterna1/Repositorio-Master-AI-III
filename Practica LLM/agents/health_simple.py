"""
Kingfisher Simple Health Check

Health check simplificado para consolidación sin dependencias externas.
"""

import time
import psutil
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

from .config import get_config
from .logging_config import get_logger

logger = get_logger("kingfisher.health.simple")

class HealthStatus(str, Enum):
    """Estados de salud posibles"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class SimpleHealthResult:
    """Resultado simplificado de health check"""
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

class SimpleHealthChecker:
    """Health checker simplificado para consolidación"""
    
    def __init__(self):
        self.config = get_config()
        self.startup_time = datetime.now()
    
    async def check_basic_connectivity(self) -> SimpleHealthResult:
        """Check básico de conectividad"""
        start_time = time.time()
        
        try:
            # Test básico de directorios
            import os
            data_dir = "./data"
            logs_dir = "./logs"
            
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(logs_dir, exist_ok=True)
            
            # Test básico de escritura
            test_file = os.path.join(data_dir, "health_test.txt")
            with open(test_file, "w") as f:
                f.write("health_check_test")
            os.remove(test_file)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return SimpleHealthResult(
                name="basic_connectivity",
                status=HealthStatus.HEALTHY,
                message="Basic file system operations working",
                duration_ms=duration_ms,
                details={
                    "data_directory": data_dir,
                    "logs_directory": logs_dir,
                    "write_test": "passed"
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return SimpleHealthResult(
                name="basic_connectivity",
                status=HealthStatus.UNHEALTHY,
                message=f"Basic connectivity failed: {str(e)}",
                duration_ms=duration_ms
            )
    
    async def check_system_resources(self) -> SimpleHealthResult:
        """Check de recursos del sistema"""
        start_time = time.time()
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_healthy = cpu_percent < 80
            
            # Memory
            memory = psutil.virtual_memory()
            memory_healthy = memory.percent < 85
            
            # Disk
            disk = psutil.disk_usage('.')
            disk_free_percent = (disk.free / disk.total) * 100
            disk_healthy = disk_free_percent > 5
            
            duration_ms = (time.time() - start_time) * 1000
            
            details = {
                "cpu_percent": round(cpu_percent, 1),
                "memory_percent": round(memory.percent, 1),
                "disk_free_percent": round(disk_free_percent, 1)
            }
            
            if cpu_healthy and memory_healthy and disk_healthy:
                status = HealthStatus.HEALTHY
                message = "System resources are healthy"
            elif disk_healthy:
                status = HealthStatus.DEGRADED
                message = "System resources under pressure but operational"
            else:
                status = HealthStatus.UNHEALTHY
                message = "System resources critically low"
            
            return SimpleHealthResult(
                name="system_resources",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return SimpleHealthResult(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Resource check failed: {str(e)}",
                duration_ms=duration_ms
            )
    
    async def check_configuration(self) -> SimpleHealthResult:
        """Check de configuración"""
        start_time = time.time()
        
        try:
            config = self.config
            
            # Verificar configuración básica
            has_agent_name = bool(config.agent_name)
            has_server_config = bool(config.server.host and config.server.port)
            has_environment = bool(config.environment)
            
            duration_ms = (time.time() - start_time) * 1000
            
            details = {
                "agent_name": config.agent_name,
                "environment": config.environment,
                "server_host": config.server.host,
                "server_port": config.server.port
            }
            
            if has_agent_name and has_server_config and has_environment:
                return SimpleHealthResult(
                    name="configuration",
                    status=HealthStatus.HEALTHY,
                    message="Configuration is complete",
                    duration_ms=duration_ms,
                    details=details
                )
            else:
                return SimpleHealthResult(
                    name="configuration",
                    status=HealthStatus.DEGRADED,
                    message="Configuration has missing elements",
                    duration_ms=duration_ms,
                    details=details
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return SimpleHealthResult(
                name="configuration",
                status=HealthStatus.UNHEALTHY,
                message=f"Configuration check failed: {str(e)}",
                duration_ms=duration_ms
            )
    
    async def run_simple_checks(self) -> Dict[str, Any]:
        """Ejecuta checks simplificados"""
        logger.info("Starting simple health check")
        start_time = time.time()
        
        # Ejecutar checks básicos
        try:
            basic_check = await self.check_basic_connectivity()
            resource_check = await self.check_system_resources()
            config_check = await self.check_configuration()
            
            checks = [basic_check, resource_check, config_check]
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            checks = []
        
        total_duration = (time.time() - start_time) * 1000
        
        # Determinar estado general
        if not checks:
            overall_status = HealthStatus.UNHEALTHY
        else:
            statuses = [check.status for check in checks]
            if all(s == HealthStatus.HEALTHY for s in statuses):
                overall_status = HealthStatus.HEALTHY
            elif any(s == HealthStatus.UNHEALTHY for s in statuses):
                overall_status = HealthStatus.UNHEALTHY
            else:
                overall_status = HealthStatus.DEGRADED
        
        # Información del sistema
        uptime = datetime.now() - self.startup_time
        
        results = {check.name: check.to_dict() for check in checks}
        
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
                "startup_time": self.startup_time.isoformat(),
                "health_check_type": "simplified"
            }
        }
        
        logger.info(f"Simple health check completed - Status: {overall_status.value}", 
                   status=overall_status.value, 
                   duration_ms=total_duration,
                   checks_count=len(results))
        
        return health_report

# Global simple health checker
simple_health_checker = SimpleHealthChecker() 