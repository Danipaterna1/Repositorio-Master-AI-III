"""
Hybrid Vector Store 2025
Sistema de almacenamiento vectorial que soporta tanto ChromaDB (desarrollo) 
como Qdrant (producción) con una interfaz unificada.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass

from ..config.settings import get_config, VectorStoreType
from ..core.embedding_types import EmbeddingResult

logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    """Resultado de búsqueda vectorial"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

@dataclass
class VectorDocument:
    """Documento para almacenamiento vectorial"""
    id: Optional[str] = None
    content: str = ""
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}

class BaseVectorStore(ABC):
    """Interfaz base para vector stores"""
    
    @abstractmethod
    def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """Añade documentos al vector store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 10, **kwargs) -> List[VectorSearchResult]:
        """Búsqueda por similitud vectorial"""
        pass
    
    @abstractmethod
    def search_with_filters(self, query_embedding: np.ndarray, 
                           filters: Dict[str, Any] = None, k: int = 10) -> List[VectorSearchResult]:
        """Búsqueda con filtros de metadatos"""
        pass
    
    @abstractmethod
    def delete_documents(self, ids: List[str]) -> bool:
        """Elimina documentos por ID"""
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """Información de la colección"""
        pass

class ChromaVectorStore(BaseVectorStore):
    """
    Vector Store usando ChromaDB para desarrollo rápido
    Perfecto para prototipos, desarrollo local y datasets pequeños (<1M vectores)
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.vector_config = self.config.vector_store
        self._client = None
        self._collection = None
        
        logger.info("ChromaDB Vector Store initialized for development")
    
    @property
    def client(self):
        """Lazy loading del cliente ChromaDB"""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                # Configuración para desarrollo local
                self._client = chromadb.PersistentClient(
                    path=self.vector_config.chromadb_path,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logger.info(f"ChromaDB client connected to: {self.vector_config.chromadb_path}")
            except ImportError:
                raise ImportError("ChromaDB not installed. Run: pip install chromadb>=0.5.0")
        return self._client
    
    @property
    def collection(self):
        """Lazy loading de la colección"""
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(
                    name=self.vector_config.chromadb_collection
                )
                logger.info(f"Using existing collection: {self.vector_config.chromadb_collection}")
            except Exception:
                # Crear colección si no existe
                self._collection = self.client.create_collection(
                    name=self.vector_config.chromadb_collection,
                    metadata={"description": "RAG vector store for development"}
                )
                logger.info(f"Created new collection: {self.vector_config.chromadb_collection}")
        return self._collection
    
    def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """Añade documentos a ChromaDB"""
        if not documents:
            return []
        
        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding.tolist() if doc.embedding is not None else None for doc in documents]
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Filtrar documentos sin embeddings
        valid_docs = [(i, e, t, m) for i, e, t, m in zip(ids, embeddings, texts, metadatas) if e is not None]
        
        if not valid_docs:
            logger.warning("No documents with embeddings to add")
            return []
        
        valid_ids, valid_embeddings, valid_texts, valid_metadatas = zip(*valid_docs)
        
        try:
            self.collection.add(
                ids=list(valid_ids),
                embeddings=list(valid_embeddings),
                documents=list(valid_texts),
                metadatas=list(valid_metadatas)
            )
            logger.info(f"Added {len(valid_ids)} documents to ChromaDB")
            return list(valid_ids)
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            return []
    
    def search(self, query_embedding: np.ndarray, k: int = 10, **kwargs) -> List[VectorSearchResult]:
        """Búsqueda vectorial en ChromaDB"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    search_results.append(VectorSearchResult(
                        id=doc_id,
                        content=results['documents'][0][i] if results['documents'] else "",
                        score=1.0 - results['distances'][0][i],  # ChromaDB usa distancia, convertir a score
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                    ))
            
            logger.debug(f"ChromaDB search returned {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"Error searching in ChromaDB: {e}")
            return []
    
    def search_with_filters(self, query_embedding: np.ndarray, 
                           filters: Dict[str, Any] = None, k: int = 10) -> List[VectorSearchResult]:
        """Búsqueda con filtros en ChromaDB"""
        where_clause = None
        if filters:
            # Convertir filtros a formato ChromaDB
            where_clause = {}
            for key, value in filters.items():
                if isinstance(value, list):
                    where_clause[key] = {"$in": value}
                else:
                    where_clause[key] = {"$eq": value}
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    search_results.append(VectorSearchResult(
                        id=doc_id,
                        content=results['documents'][0][i] if results['documents'] else "",
                        score=1.0 - results['distances'][0][i],
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                    ))
            
            logger.debug(f"ChromaDB filtered search returned {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"Error in filtered search: {e}")
            return []
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Elimina documentos de ChromaDB"""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Información de la colección ChromaDB"""
        try:
            count = self.collection.count()
            return {
                "name": self.vector_config.chromadb_collection,
                "count": count,
                "type": "ChromaDB",
                "path": self.vector_config.chromadb_path
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}

class QdrantVectorStore(BaseVectorStore):
    """
    Vector Store usando Qdrant para producción enterprise
    Soporta filtros avanzados, quantización y escalabilidad >10M vectores
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.vector_config = self.config.vector_store
        self._client = None
        
        logger.info("Qdrant Vector Store initialized for production")
    
    @property
    def client(self):
        """Lazy loading del cliente Qdrant"""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams
                
                self._client = QdrantClient(
                    url=self.vector_config.qdrant_url,
                    api_key=self.vector_config.qdrant_api_key
                )
                logger.info(f"Qdrant client connected to: {self.vector_config.qdrant_url}")
                
                # Verificar si la colección existe, si no crearla
                self._ensure_collection_exists()
            except ImportError:
                raise ImportError("Qdrant not installed. Run: pip install qdrant-client==1.12.1")
        return self._client
    
    def _ensure_collection_exists(self):
        """Asegura que la colección existe en Qdrant"""
        try:
            from qdrant_client.models import Distance, VectorParams
            
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.vector_config.qdrant_collection not in collection_names:
                # Crear colección con configuración enterprise
                self.client.create_collection(
                    collection_name=self.vector_config.qdrant_collection,
                    vectors_config=VectorParams(
                        size=768,  # Dimensión por defecto, ajustar según modelo
                        distance=Distance.COSINE
                    ),
                    # Configuración enterprise con quantización
                    optimizers_config={
                        "default_segment_number": 2,
                        "max_segment_size": 20000,
                        "memmap_threshold": 20000,
                        "indexing_threshold": 20000,
                        "flush_interval_sec": 30,
                        "max_optimization_threads": 2
                    }
                )
                logger.info(f"Created Qdrant collection: {self.vector_config.qdrant_collection}")
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection: {e}")
    
    def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """Añade documentos a Qdrant (implementación simplificada)"""
        # TODO: Implementar cuando se active producción
        logger.warning("Qdrant implementation pending - use ChromaDB for development")
        return []
    
    def search(self, query_embedding: np.ndarray, k: int = 10, **kwargs) -> List[VectorSearchResult]:
        """Búsqueda en Qdrant (implementación simplificada)"""
        # TODO: Implementar cuando se active producción
        logger.warning("Qdrant search implementation pending - use ChromaDB for development")
        return []
    
    def search_with_filters(self, query_embedding: np.ndarray, 
                           filters: Dict[str, Any] = None, k: int = 10) -> List[VectorSearchResult]:
        """Búsqueda con filtros en Qdrant (implementación simplificada)"""
        # TODO: Implementar cuando se active producción
        logger.warning("Qdrant filtered search implementation pending")
        return []
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Elimina documentos de Qdrant (implementación simplificada)"""
        # TODO: Implementar cuando se active producción
        return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Información de la colección Qdrant"""
        try:
            info = self.client.get_collection(self.vector_config.qdrant_collection)
            return {
                "name": self.vector_config.qdrant_collection,
                "count": info.points_count,
                "type": "Qdrant",
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting Qdrant collection info: {e}")
            return {"error": str(e)}

class HybridVectorStore:
    """
    Vector Store híbrido que permite cambiar entre ChromaDB y Qdrant
    según la configuración del environment
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.vector_config = self.config.vector_store
        
        # Seleccionar store según configuración
        if self.vector_config.store_type == VectorStoreType.CHROMADB:
            self.store = ChromaVectorStore(config)
            logger.info("Using ChromaDB for development")
        elif self.vector_config.store_type == VectorStoreType.QDRANT:
            self.store = QdrantVectorStore(config)
            logger.info("Using Qdrant for production")
        else:
            raise ValueError(f"Unknown vector store type: {self.vector_config.store_type}")
    
    def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """Proxy para añadir documentos"""
        return self.store.add_documents(documents)
    
    def search(self, query_embedding: np.ndarray, k: int = 10, **kwargs) -> List[VectorSearchResult]:
        """Proxy para búsqueda vectorial"""
        return self.store.search(query_embedding, k, **kwargs)
    
    def search_with_filters(self, query_embedding: np.ndarray, 
                           filters: Dict[str, Any] = None, k: int = 10) -> List[VectorSearchResult]:
        """Proxy para búsqueda con filtros"""
        return self.store.search_with_filters(query_embedding, filters, k)
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Proxy para eliminar documentos"""
        return self.store.delete_documents(ids)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Proxy para información de colección"""
        return self.store.get_collection_info()
    
    def get_store_type(self) -> str:
        """Retorna el tipo de store actual"""
        return self.vector_config.store_type.value
    
    def switch_store_type(self, store_type: VectorStoreType):
        """Cambia dinámicamente el tipo de store"""
        if store_type != self.vector_config.store_type:
            self.vector_config.store_type = store_type
            
            if store_type == VectorStoreType.CHROMADB:
                self.store = ChromaVectorStore(self.config)
                logger.info("Switched to ChromaDB")
            elif store_type == VectorStoreType.QDRANT:
                self.store = QdrantVectorStore(self.config)
                logger.info("Switched to Qdrant")

# Factory functions
def create_vector_store(config=None) -> HybridVectorStore:
    """Factory para crear vector store híbrido"""
    return HybridVectorStore(config)

# Instancia global lazy
_global_vector_store: Optional[HybridVectorStore] = None

def get_vector_store() -> HybridVectorStore:
    """Obtiene la instancia global del vector store"""
    global _global_vector_store
    if _global_vector_store is None:
        _global_vector_store = create_vector_store()
    return _global_vector_store 