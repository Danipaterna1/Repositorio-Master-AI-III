"""
Batch Embedding Engine - Optimized for High Volume Processing

Motor de embeddings que procesa chunks en batches para máxima eficiencia.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

from .smart_chunker import SmartChunk

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Resultado de embedding con metadata"""
    chunk_id: str
    embedding: np.ndarray
    model_name: str
    dimensions: int
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "embedding": self.embedding.tolist(),
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "processing_time": self.processing_time
        }

@dataclass
class BatchStats:
    """Estadísticas de procesamiento batch"""
    total_chunks: int
    successful_embeddings: int
    failed_embeddings: int
    total_time: float
    avg_time_per_chunk: float
    chunks_per_second: float

class BatchEmbedder:
    """
    Motor de embeddings optimizado para procesamiento masivo.
    
    Features:
    - Batch processing para máxima eficiencia
    - Multi-model support
    - GPU optimization cuando disponible
    - Error handling robusto
    - Progress tracking
    - Caching de embeddings
    """
    
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 max_workers: int = 4,
                 device: str = "auto",
                 cache_embeddings: bool = True,
                 cache_dir: str = "./data/embeddings/cache"):
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cache_embeddings = cache_embeddings
        self.cache_dir = Path(cache_dir)
        
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self._load_model()
        
        # Setup cache
        if self.cache_embeddings:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._embedding_cache = {}
            self._load_cache()
    
    def _load_model(self):
        """Load embedding model with optimization"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(f"sentence-transformers/{self.model_name}")
            
            # Move to device if available
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(self.device)
                logger.info("Model moved to GPU")
                
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _load_cache(self):
        """Load embedding cache from disk"""
        cache_file = self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.npy"
        
        if cache_file.exists():
            try:
                cache_data = np.load(cache_file, allow_pickle=True).item()
                self._embedding_cache = cache_data
                logger.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk"""
        if not self.cache_embeddings:
            return
            
        cache_file = self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.npy"
        
        try:
            np.save(cache_file, self._embedding_cache)
            logger.info(f"Saved {len(self._embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, content: str) -> str:
        """Generate cache key for content"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
    
    async def embed_chunks_batch(self, 
                                chunks: List[SmartChunk],
                                show_progress: bool = True) -> Tuple[List[EmbeddingResult], BatchStats]:
        """
        Embed chunks in optimized batches.
        
        Args:
            chunks: List of SmartChunk objects
            show_progress: Whether to show progress
            
        Returns:
            Tuple of (embedding results, batch statistics)
        """
        start_time = time.time()
        results = []
        failed_count = 0
        
        # Split into batches
        batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
        
        logger.info(f"Processing {len(chunks)} chunks in {len(batches)} batches")
        
        for i, batch in enumerate(batches):
            if show_progress:
                logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} chunks)")
            
            try:
                batch_results = await self._process_batch(batch)
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Failed to process batch {i+1}: {e}")
                failed_count += len(batch)
        
        # Calculate statistics
        total_time = time.time() - start_time
        successful_count = len(results)
        
        stats = BatchStats(
            total_chunks=len(chunks),
            successful_embeddings=successful_count,
            failed_embeddings=failed_count,
            total_time=total_time,
            avg_time_per_chunk=total_time / len(chunks) if chunks else 0,
            chunks_per_second=len(chunks) / total_time if total_time > 0 else 0
        )
        
        # Save cache
        if self.cache_embeddings:
            self._save_cache()
        
        logger.info(f"Batch processing complete: {successful_count}/{len(chunks)} successful")
        logger.info(f"Performance: {stats.chunks_per_second:.2f} chunks/second")
        
        return results, stats
    
    async def _process_batch(self, batch: List[SmartChunk]) -> List[EmbeddingResult]:
        """Process a single batch of chunks"""
        results = []
        
        # Prepare texts and check cache
        texts_to_embed = []
        cache_mapping = {}
        
        for chunk in batch:
            cache_key = self._get_cache_key(chunk.content)
            
            if self.cache_embeddings and cache_key in self._embedding_cache:
                # Use cached embedding
                cached_embedding = self._embedding_cache[cache_key]
                result = EmbeddingResult(
                    chunk_id=chunk.id,
                    embedding=cached_embedding,
                    model_name=self.model_name,
                    dimensions=len(cached_embedding),
                    processing_time=0.0  # Cached
                )
                results.append(result)
            else:
                # Need to embed
                texts_to_embed.append(chunk.content)
                cache_mapping[len(texts_to_embed) - 1] = (chunk.id, cache_key)
        
        # Embed uncached texts
        if texts_to_embed:
            embeddings = await self._embed_texts(texts_to_embed)
            
            for idx, embedding in enumerate(embeddings):
                chunk_id, cache_key = cache_mapping[idx]
                
                # Cache the embedding
                if self.cache_embeddings:
                    self._embedding_cache[cache_key] = embedding
                
                result = EmbeddingResult(
                    chunk_id=chunk_id,
                    embedding=embedding,
                    model_name=self.model_name,
                    dimensions=len(embedding),
                    processing_time=0.1  # Approximate
                )
                results.append(result)
        
        return results
    
    async def _embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed texts using the model"""
        def _do_embedding():
            return self.model.encode(
                texts,
                batch_size=min(len(texts), self.batch_size),
                show_progress_bar=False,
                convert_to_numpy=True
            )
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            embeddings = await loop.run_in_executor(executor, _do_embedding)
        
        return [embedding for embedding in embeddings]
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """Embed a single text (synchronous)"""
        cache_key = self._get_cache_key(text)
        
        if self.cache_embeddings and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        embedding = self.model.encode([text])[0]
        
        if self.cache_embeddings:
            self._embedding_cache[cache_key] = embedding
        
        return embedding
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "batch_size": self.batch_size,
            "cache_size": len(self._embedding_cache) if self.cache_embeddings else 0
        }
    
    def clear_cache(self):
        """Clear embedding cache"""
        if self.cache_embeddings:
            self._embedding_cache.clear()
            logger.info("Embedding cache cleared")
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_embedding_cache') and self.cache_embeddings:
            self._save_cache()

# Factory function
def create_batch_embedder(config: Dict[str, Any] = None) -> BatchEmbedder:
    """Create batch embedder with configuration"""
    config = config or {}
    
    return BatchEmbedder(
        model_name=config.get('model_name', 'all-MiniLM-L6-v2'),
        batch_size=config.get('batch_size', 32),
        max_workers=config.get('max_workers', 4),
        device=config.get('device', 'auto'),
        cache_embeddings=config.get('cache_embeddings', True),
        cache_dir=config.get('cache_dir', './data/embeddings/cache')
    ) 