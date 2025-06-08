"""
Kingfisher A2A Agent Module

Este módulo implementa el agente Kingfisher como parte del ecosistema Google A2A,
proporcionando capacidades especializadas de procesamiento RAG y gestión de conocimiento.

Capabilities:
- Document Processing: Chunking + Embedding + Triple Storage
- Knowledge Retrieval: Vector + Graph + Metadata Search  
- Metadata Analysis: Knowledge Base Analytics
"""

# Import condicional para evitar errores si FastAPI no está disponible
try:
    from .kingfisher_agent import KingfisherAgent
    KINGFISHER_AGENT_AVAILABLE = True
except ImportError:
    KINGFISHER_AGENT_AVAILABLE = False

try:
    from .server.a2a_server import KingfisherA2AServer
    A2A_SERVER_AVAILABLE = True
except ImportError:
    A2A_SERVER_AVAILABLE = False

# Siempre disponible
from .kingfisher_agent_simple import KingfisherAgentSimple

__all__ = ["KingfisherAgentSimple"]

if KINGFISHER_AGENT_AVAILABLE:
    __all__.append("KingfisherAgent")

if A2A_SERVER_AVAILABLE:
    __all__.append("KingfisherA2AServer")

__version__ = "1.0.0" 