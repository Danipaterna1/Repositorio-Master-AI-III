"""
Logging Middleware for Kingfisher A2A API
=========================================

Request/response logging and monitoring.
"""

import time
import uuid
import logging
import json
from typing import Callable
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from datetime import datetime

logger = logging.getLogger(__name__)

class APILoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware para logging de requests y responses
    """
    
    def __init__(self, app, log_requests: bool = True, log_responses: bool = True):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generar ID único para la request
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Timestamp de inicio
        start_time = time.time()
        timestamp = datetime.utcnow()
        
        # Logging de request entrante
        if self.log_requests:
            await self._log_request(request, request_id, timestamp)
        
        try:
            # Procesar request
            response = await call_next(request)
            
            # Calcular tiempo de procesamiento
            processing_time = time.time() - start_time
            
            # Logging de response
            if self.log_responses:
                await self._log_response(request, response, request_id, processing_time)
            
            # Añadir headers de metadata
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"
            
            return response
            
        except Exception as e:
            # Logging de errores
            processing_time = time.time() - start_time
            await self._log_error(request, e, request_id, processing_time)
            raise
    
    async def _log_request(self, request: Request, request_id: str, timestamp: datetime):
        """Log información de request entrante"""
        
        # Extraer información del cliente
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Extraer API key (si existe)
        auth_header = request.headers.get("authorization", "")
        api_key_prefix = ""
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
            api_key_prefix = api_key[:8] + "..." if len(api_key) > 8 else api_key
        
        # Log estructurado
        log_data = {
            "event": "api_request",
            "request_id": request_id,
            "timestamp": timestamp.isoformat(),
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "api_key_prefix": api_key_prefix,
            "content_type": request.headers.get("content-type", ""),
            "content_length": request.headers.get("content-length", "0")
        }
        
        logger.info(f"API Request: {request.method} {request.url.path}", extra=log_data)
    
    async def _log_response(self, request: Request, response: Response, 
                           request_id: str, processing_time: float):
        """Log información de response"""
        
        log_data = {
            "event": "api_response",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "processing_time": round(processing_time, 3),
            "response_size": len(response.body) if hasattr(response, 'body') else 0
        }
        
        # Clasificar por status code
        if response.status_code < 400:
            logger.info(f"API Success: {request.method} {request.url.path} - {response.status_code}", 
                       extra=log_data)
        elif response.status_code < 500:
            logger.warning(f"API Client Error: {request.method} {request.url.path} - {response.status_code}", 
                          extra=log_data)
        else:
            logger.error(f"API Server Error: {request.method} {request.url.path} - {response.status_code}", 
                        extra=log_data)
    
    async def _log_error(self, request: Request, error: Exception, 
                        request_id: str, processing_time: float):
        """Log errores no manejados"""
        
        log_data = {
            "event": "api_error",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "processing_time": round(processing_time, 3)
        }
        
        logger.error(f"API Unhandled Error: {request.method} {request.url.path} - {type(error).__name__}", 
                    extra=log_data, exc_info=True)

def setup_logging(level: str = "INFO", format_type: str = "json") -> None:
    """
    Configura el sistema de logging para la API
    """
    
    # Configurar nivel de logging
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Formatters
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # Configurar root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Configurar loggers específicos
    api_logger = logging.getLogger(__name__)
    api_logger.setLevel(log_level)
    
    # Reducir verbosidad de librerías externas
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    
    logger.info(f"Logging configured: level={level}, format={format_type}")

class JSONFormatter(logging.Formatter):
    """
    Formatter para logs en formato JSON
    """
    
    def format(self, record: logging.LogRecord) -> str:
        # Crear base del log
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Añadir información extra si existe
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName', 
                              'processName', 'process', 'stack_info', 'exc_info', 'exc_text']:
                    log_entry[key] = value
        
        # Añadir exception info si existe
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)

class PerformanceMonitor:
    """
    Monitor de performance para endpoints
    """
    
    def __init__(self):
        self.endpoint_stats = {}
    
    def record_request(self, endpoint: str, method: str, processing_time: float, 
                      status_code: int):
        """Registra estadísticas de un request"""
        key = f"{method}:{endpoint}"
        
        if key not in self.endpoint_stats:
            self.endpoint_stats[key] = {
                "total_requests": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
                "error_count": 0,
                "success_count": 0
            }
        
        stats = self.endpoint_stats[key]
        stats["total_requests"] += 1
        stats["total_time"] += processing_time
        stats["avg_time"] = stats["total_time"] / stats["total_requests"]
        stats["min_time"] = min(stats["min_time"], processing_time)
        stats["max_time"] = max(stats["max_time"], processing_time)
        
        if status_code >= 400:
            stats["error_count"] += 1
        else:
            stats["success_count"] += 1
    
    def get_stats(self) -> dict:
        """Obtiene estadísticas actuales"""
        return self.endpoint_stats
    
    def reset_stats(self):
        """Resetea estadísticas"""
        self.endpoint_stats = {}

# Instancia global del monitor
performance_monitor = PerformanceMonitor() 