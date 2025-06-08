"""
Kingfisher A2A HTTP Server

Implementa el servidor HTTP que expone los endpoints A2A estándar
para la comunicación con otros agentes en el ecosistema Google A2A.
"""

from .a2a_server import KingfisherA2AServer

__all__ = [
    "KingfisherA2AServer"
] 