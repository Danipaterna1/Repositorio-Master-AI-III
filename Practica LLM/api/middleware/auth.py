"""
Authentication Middleware for Kingfisher A2A API
================================================

Basic API key authentication for production use.
"""

import os
import logging
from typing import Optional
from fastapi import HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

# Default API keys (development)
DEFAULT_API_KEYS = {
    "dev-key-001": {"name": "Development Key", "permissions": ["read", "write"]},
    "admin-key-001": {"name": "Admin Key", "permissions": ["read", "write", "admin"]},
    "readonly-key-001": {"name": "Read Only Key", "permissions": ["read"]}
}

def get_valid_api_keys() -> dict:
    """
    Obtiene las API keys válidas desde variables de entorno o defaults
    """
    # Intentar cargar desde environment variables
    env_keys = os.getenv("KINGFISHER_API_KEYS")
    if env_keys:
        try:
            import json
            return json.loads(env_keys)
        except json.JSONDecodeError:
            logger.warning("Invalid KINGFISHER_API_KEYS format, using defaults")
    
    # Usar defaults para desarrollo
    logger.info("Using default API keys for development")
    return DEFAULT_API_KEYS

def validate_api_key(api_key: str) -> Optional[dict]:
    """
    Valida una API key y retorna información del cliente
    """
    valid_keys = get_valid_api_keys()
    
    if api_key in valid_keys:
        key_info = valid_keys[api_key]
        logger.info(f"Valid API key used: {key_info.get('name', 'Unknown')}")
        return {
            "api_key": api_key,
            "name": key_info.get("name", "Unknown"),
            "permissions": key_info.get("permissions", ["read"]),
            "authenticated_at": datetime.utcnow()
        }
    
    logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
    return None

async def get_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> str:
    """
    Extrae y valida la API key del header Authorization
    """
    if not credentials:
        # En desarrollo permitir requests sin auth
        if os.getenv("KINGFISHER_ENV", "development") == "development":
            logger.info("No API key provided, allowing in development mode")
            return "dev-key-001"  # Default para desarrollo
        
        raise HTTPException(
            status_code=401,
            detail="API key required. Use Authorization: Bearer <your-api-key>",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    api_key = credentials.credentials
    
    if not validate_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return api_key

async def verify_api_key(api_key: str = Depends(get_api_key)) -> dict:
    """
    Dependency para verificar API key y obtener información del cliente
    """
    client_info = validate_api_key(api_key)
    
    if not client_info:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key"
        )
    
    return client_info

def check_permission(required_permission: str):
    """
    Decorator para verificar permisos específicos
    """
    def permission_dependency(client_info: dict = Depends(verify_api_key)):
        permissions = client_info.get("permissions", [])
        
        if required_permission not in permissions and "admin" not in permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{required_permission}' required"
            )
        
        return client_info
    
    return permission_dependency

# Dependencies preconfiguradas
def require_read_permission():
    """Requiere permiso de lectura"""
    return check_permission("read")

def require_write_permission():
    """Requiere permiso de escritura"""
    return check_permission("write")

def require_admin_permission():
    """Requiere permiso de administrador"""
    return check_permission("admin")

class APIKeyInfo:
    """
    Información de la API key para logging y métricas
    """
    def __init__(self, client_info: dict):
        self.api_key = client_info.get("api_key", "unknown")
        self.name = client_info.get("name", "Unknown Client")
        self.permissions = client_info.get("permissions", [])
        self.authenticated_at = client_info.get("authenticated_at")
    
    def has_permission(self, permission: str) -> bool:
        """Verifica si tiene un permiso específico"""
        return permission in self.permissions or "admin" in self.permissions
    
    def to_dict(self) -> dict:
        """Convierte a diccionario para logging"""
        return {
            "client_name": self.name,
            "permissions": self.permissions,
            "api_key_prefix": self.api_key[:8] + "..." if self.api_key else "none"
        } 