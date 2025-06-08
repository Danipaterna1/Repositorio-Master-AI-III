"""
Kingfisher Logging System

Sistema de logging robusto para el agente A2A Kingfisher.
Incluye structured logging, trace IDs, y múltiples outputs.
"""

import logging
import logging.handlers
import json
import uuid
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import contextvars

from .config import get_config

# Context variable para trace IDs
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('trace_id', default='')

class TraceIDFormatter(logging.Formatter):
    """Formatter que incluye trace ID en los logs"""
    
    def format(self, record):
        trace_id = trace_id_var.get('')
        if trace_id:
            record.trace_id = trace_id
        else:
            record.trace_id = 'no-trace'
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """Formatter para logs estructurados en JSON"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'trace_id': trace_id_var.get('no-trace')
        }
        
        # Agregar campos extra si existen
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        # Agregar exception info si existe
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, ensure_ascii=False)

class KingfisherLogger:
    """Logger centralizado para Kingfisher"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.config = get_config().logging
        self._setup_logger()
    
    def _setup_logger(self):
        """Configura el logger según la configuración"""
        self.logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Limpiar handlers existentes
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if self.config.json_logs:
            console_formatter = JSONFormatter()
        else:
            console_formatter = TraceIDFormatter(
                '%(asctime)s - [%(trace_id)s] - %(name)s - %(levelname)s - %(message)s'
            )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (si está configurado)
        if self.config.file_path:
            file_path = Path(self.config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            
            if self.config.json_logs:
                file_formatter = JSONFormatter()
            else:
                file_formatter = TraceIDFormatter(self.config.format)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info level"""
        self._log(logging.INFO, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug level"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error level"""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical level"""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Log interno con campos extra"""
        extra = {'extra_fields': kwargs} if kwargs else {}
        self.logger.log(level, message, extra=extra)

# Factory function para crear loggers
def get_logger(name: str) -> KingfisherLogger:
    """Obtiene un logger para el módulo especificado"""
    return KingfisherLogger(name)

# Context manager para trace IDs
class trace_context:
    """Context manager para manejar trace IDs"""
    
    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or str(uuid.uuid4())[:8]
        self.token = None
    
    def __enter__(self):
        self.token = trace_id_var.set(self.trace_id)
        return self.trace_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            trace_id_var.reset(self.token)

# Decorador para agregar trace ID automáticamente
def with_trace_id(func):
    """Decorador que agrega trace ID a funciones"""
    def wrapper(*args, **kwargs):
        if not trace_id_var.get(''):
            with trace_context():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

# Logger principal de la aplicación
app_logger = get_logger("kingfisher")
server_logger = get_logger("kingfisher.server")
task_logger = get_logger("kingfisher.tasks")
db_logger = get_logger("kingfisher.database")

# Funciones de conveniencia
def log_task_start(task_id: str, task_type: str, **kwargs):
    """Log inicio de task"""
    task_logger.info(f"Task started: {task_type}", 
                    task_id=task_id, task_type=task_type, **kwargs)

def log_task_complete(task_id: str, task_type: str, duration_ms: float, **kwargs):
    """Log completado de task"""
    task_logger.info(f"Task completed: {task_type}", 
                    task_id=task_id, task_type=task_type, 
                    duration_ms=duration_ms, **kwargs)

def log_task_error(task_id: str, task_type: str, error: str, **kwargs):
    """Log error de task"""
    task_logger.error(f"Task failed: {task_type} - {error}", 
                     task_id=task_id, task_type=task_type, 
                     error=error, **kwargs)

def log_api_request(method: str, path: str, status_code: int, duration_ms: float, **kwargs):
    """Log request de API"""
    server_logger.info(f"{method} {path} - {status_code}", 
                      method=method, path=path, status_code=status_code,
                      duration_ms=duration_ms, **kwargs)

def log_db_operation(operation: str, table: str, duration_ms: float, **kwargs):
    """Log operación de base de datos"""
    db_logger.debug(f"DB {operation} on {table}", 
                   operation=operation, table=table, 
                   duration_ms=duration_ms, **kwargs)

def create_trace_context(trace_id: Optional[str] = None) -> trace_context:
    """
    Crea un contexto de trace para tracking de requests.
    
    Args:
        trace_id: ID de trace opcional. Si no se proporciona, se genera uno nuevo.
        
    Returns:
        Context manager para manejar el trace ID
    """
    return trace_context(trace_id)