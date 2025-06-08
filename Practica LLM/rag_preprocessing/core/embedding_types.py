"""
Basic embedding types for compatibility
Minimal types to replace legacy embedding_manager dependencies
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class EmbeddingResult:
    """Resultado de embedding - tipo básico para compatibilidad"""
    embedding: np.ndarray
    text: str
    metadata: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = None
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {} 