"""
Microsoft Graph RAG Implementation
==================================

Implementación completa del enfoque Graph RAG de Microsoft con:
- Entity Extraction con LLM
- Community Detection (Leiden Algorithm)
- Hierarchical Summarization con LLM Enhancement (Sprint 3)
- Query-Focused Summarization (QFS)
- Dual Search (Local + Global)

Basado en el paper: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
Sprint 3 Update: LLM-enhanced community summarization con Google Gemini
"""

from .entity_extractor import (
    EntityExtractor,
    Entity,
    Relationship,
    ExtractionResult
)

from .community_detector import (
    CommunityDetector,
    Community,
    CommunityHierarchy
)

from .community_summarizer import (
    CommunitySummarizer,
    LLMCommunitySummarizer,
    SyncCommunitySummarizer,
    CommunityReport,
    create_community_summarizer
)

from .graph_builder import (
    GraphBuilder,
    KnowledgeGraph
)

from .query_engine import (
    GraphRAGQueryEngine,
    LocalSearch,
    GlobalSearch,
    QueryResult
)

from .pipeline import (
    GraphRAGPipeline,
    GraphRAGConfig
)

from .graph_visualizer import (
    GraphVisualizer
)

__all__ = [
    # Entity Extraction
    "EntityExtractor",
    "Entity", 
    "Relationship",
    "ExtractionResult",
    
    # Community Detection
    "CommunityDetector",
    "Community",
    "CommunityHierarchy",
    
    # Summarization (Sprint 3: LLM-enhanced)
    "CommunitySummarizer", 
    "LLMCommunitySummarizer",
    "SyncCommunitySummarizer", 
    "CommunityReport",
    "create_community_summarizer",
    
    # Graph Building
    "GraphBuilder",
    "KnowledgeGraph",
    
    # Query Engine
    "GraphRAGQueryEngine",
    "LocalSearch",
    "GlobalSearch", 
    "QueryResult",
    
    # Pipeline
    "GraphRAGPipeline",
    "GraphRAGConfig",
    
    # Visualization
    "GraphVisualizer"
]

# Versioning
__version__ = "1.1.0"  # Sprint 3 update
__author__ = "RAG System 2025" 