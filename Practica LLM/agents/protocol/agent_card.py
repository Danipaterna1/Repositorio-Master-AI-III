"""
Kingfisher Agent Card - Google A2A Protocol

Define la identidad, capabilities y skills del agente Kingfisher
para discovery y comunicación en el ecosistema Google A2A.
"""

import os
from typing import Dict, Any

def get_agent_card() -> Dict[str, Any]:
    """
    Retorna la Agent Card de Kingfisher compatible con Google A2A Protocol.
    
    La Agent Card es servida en /.well-known/agent.json para agent discovery.
    """
    base_url = os.getenv("KINGFISHER_BASE_URL", "http://localhost:8000")
    
    return {
        "name": "Kingfisher RAG Agent",
        "description": "Specialized agent for document preprocessing and knowledge retrieval using triple storage (vector + graph + relational)",
        "version": "1.0.0",
        "provider": {
            "organization": "Kingfisher AI Lab",
            "contact": "support@kingfisher.ai"
        },
        "url": base_url,
        "documentationUrl": f"{base_url}/docs",
        "capabilities": {
            "streaming": True,
            "pushNotifications": True,
            "stateTransitionHistory": True,
            "batchProcessing": True
        },
        "authentication": {
            "schemes": ["bearer", "api-key"]
        },
        "defaultInputModes": ["text", "data", "file"],
        "defaultOutputModes": ["text", "data"],
        "skills": [
            {
                "id": "process_documents",
                "name": "Process Documents",
                "description": "Process documents through chunking, embedding, and triple storage pipeline",
                "inputModes": ["file", "text", "data"],
                "outputModes": ["data", "text"],
                "tags": ["preprocessing", "chunking", "embedding", "storage"],
                "examples": [
                    "Process this PDF document",
                    "Analyze and store this text content",
                    "Extract knowledge from uploaded files",
                    "Process research papers for knowledge extraction"
                ],
                "parameters": {
                    "processing_mode": {
                        "type": "string",
                        "enum": ["TRIPLE_FULL", "VECTOR_ONLY", "GRAPH_ONLY", "METADATA_ONLY"],
                        "default": "TRIPLE_FULL",
                        "description": "Type of processing to apply to the document"
                    },
                    "include_llm": {
                        "type": "boolean", 
                        "default": True,
                        "description": "Whether to use LLM enhancement for community summarization"
                    }
                }
            },
            {
                "id": "retrieve_knowledge",
                "name": "Retrieve Knowledge",
                "description": "Retrieve relevant information using vector, graph, and metadata search",
                "inputModes": ["text", "data"],
                "outputModes": ["data", "text"],
                "tags": ["retrieval", "search", "rag", "knowledge"],
                "examples": [
                    "Find information about machine learning",
                    "What documents mention artificial intelligence?",
                    "Retrieve related concepts to deep learning",
                    "Search for papers about transformer architectures"
                ],
                "parameters": {
                    "search_mode": {
                        "type": "string",
                        "enum": ["vector", "graph", "hybrid", "metadata"],
                        "default": "hybrid",
                        "description": "Type of search to perform"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Number of results to return"
                    }
                }
            },
            {
                "id": "analyze_metadata",
                "name": "Analyze Metadata",
                "description": "Analyze document metadata, relationships, and knowledge graph structure",
                "inputModes": ["text", "data"],
                "outputModes": ["data", "text"],
                "tags": ["metadata", "analysis", "graph", "relationships"],
                "examples": [
                    "Show document statistics",
                    "Analyze knowledge graph structure",
                    "Generate metadata reports",
                    "Provide processing metrics summary"
                ],
                "parameters": {
                    "analysis_type": {
                        "type": "string", 
                        "enum": ["documents", "entities", "relationships", "communities", "metrics"],
                        "default": "documents",
                        "description": "Type of metadata analysis to perform"
                    }
                }
            }
        ],
        "endpoints": {
            "tasks": {
                "send": f"{base_url}/tasks/send",
                "sendSubscribe": f"{base_url}/tasks/sendSubscribe",
                "cancel": f"{base_url}/tasks/cancel",
                "status": f"{base_url}/tasks/status"
            },
            "health": f"{base_url}/health",
            "metrics": f"{base_url}/metrics"
        },
        "rateLimit": {
            "requestsPerMinute": 60,
            "burstSize": 10
        },
        "timeout": {
            "defaultSeconds": 30,
            "maxSeconds": 300
        }
    }

# Constante para fácil acceso
KINGFISHER_AGENT_CARD = get_agent_card()

def create_agent_card(skills: list = None) -> Dict[str, Any]:
    """
    Crea una Agent Card personalizada para agentes performance.
    
    Args:
        skills: Lista de skills opcionales para personalizar el agente
        
    Returns:
        Dict con la configuración del agent card
    """
    card = get_agent_card()
    
    if skills:
        # Agregar skills adicionales o personalizar los existentes
        performance_skills = [
            {
                "id": "performance_monitoring",
                "name": "Performance Monitoring", 
                "description": "Monitor system performance metrics and processing statistics",
                "inputModes": ["data"],
                "outputModes": ["data", "text"],
                "tags": ["performance", "monitoring", "metrics"],
                "examples": [
                    "Show current performance metrics",
                    "Monitor processing times",
                    "Analyze system performance"
                ]
            }
        ]
        card["skills"].extend(performance_skills)
    
    return card 