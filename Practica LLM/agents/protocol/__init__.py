"""
Google A2A Protocol Implementation

Implementa el protocolo estándar Google Agent-to-Agent para la comunicación
entre agentes especializados en el ecosistema A2A.
"""

from .agent_card import KINGFISHER_AGENT_CARD
# Force use of simplified task manager to avoid import conflicts
from .task_manager_simple import KingfisherTaskManager

__all__ = [
    "KINGFISHER_AGENT_CARD",
    "KingfisherTaskManager"
] 