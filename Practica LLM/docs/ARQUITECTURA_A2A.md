# KINGFISHER A2A ARCHITECTURE

## Google Agent-to-Agent (A2A) Integration Plan

### **¿Qué es A2A?**

**Agent-to-Agent (A2A)** es un protocolo abierto de Google (lanzado abril 2025) que permite **comunicación estándar entre agentes de IA**, sin importar quién los desarrolló. Es **complementary to MCP** (Model Context Protocol):

- **MCP**: Conecta agentes con **herramientas** y recursos  
- **A2A**: Conecta **agentes entre sí** para colaboración

### **Arquitectura A2A**

```
┌─────────────────────────────────────────────────────────────┐
│                     GOOGLE A2A ECOSYSTEM                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Client    │    │   Routing   │    │   Search    │     │
│  │   Agent     │◄──►│   Agent     │◄──►│   Agent     │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                   │                   │          │
│         ▼                   ▼                   ▼          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            KINGFISHER RAG AGENT (A2A)                  │ │
│  │  ┌─────────────────────────────────────────────────┐   │ │
│  │  │              TRIPLE PROCESSOR               │   │ │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────────┐    │   │ │
│  │  │  │ Vector  │ │ Graph   │ │ Relational  │    │   │ │
│  │  │  │   DB    │ │   DB    │ │     DB      │    │   │ │
│  │  │  └─────────┘ └─────────┘ └─────────────┘    │   │ │
│  │  └─────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## **IMPLEMENTACIÓN KINGFISHER A2A**

### **Kingfisher como Agente A2A Especializado**

Kingfisher operará como un **agente especializado en procesamiento RAG** dentro del ecosistema Google A2A, ofreciendo 3 capabilities principales:

1. **Document Processing** - Chunking + Embedding + Triple Storage
2. **Knowledge Retrieval** - Vector + Graph + Metadata Search  
3. **Metadata Analysis** - Knowledge Base Analytics

### **Agent Card - Identidad del Agente**

```json
{
  "name": "Kingfisher RAG Agent",
  "description": "Specialized agent for document preprocessing and knowledge retrieval using triple storage",
  "version": "1.0.0",
  "url": "http://localhost:8000",
  "capabilities": {
    "streaming": true,
    "pushNotifications": true
  },
  "skills": [
    {
      "id": "process_documents",
      "name": "Process Documents", 
      "description": "Process documents through chunking, embedding, and triple storage pipeline"
    },
    {
      "id": "retrieve_knowledge",
      "name": "Retrieve Knowledge",
      "description": "Retrieve relevant information using vector, graph, and metadata search"
    },
    {
      "id": "analyze_metadata", 
      "name": "Analyze Metadata",
      "description": "Analyze document metadata, relationships, and knowledge graph structure"
    }
  ]
}
```

### **LangGraph State Machine Implementation**

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START

class KingfisherState(TypedDict):
    task_id: str
    task_type: Literal["process_documents", "retrieve_knowledge", "analyze_metadata"]
    input_data: dict
    processing_status: Literal["submitted", "working", "completed", "failed"]
    artifacts: list

# Workflow nodes que usan sistemas existentes
workflow = StateGraph(KingfisherState)
workflow.add_node("document_processor", document_processor_node)
workflow.add_node("knowledge_retriever", knowledge_retriever_node)
workflow.add_node("metadata_analyzer", metadata_analyzer_node)

# Routing basado en task_type
workflow.add_conditional_edges(START, route_task, {
    "process_documents": "document_processor",
    "retrieve_knowledge": "knowledge_retriever", 
    "analyze_metadata": "metadata_analyzer"
})

kingfisher_agent = workflow.compile()
```

### **A2A HTTP Server (FastAPI)**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

class KingfisherA2AServer:
    def __init__(self):
        self.app = FastAPI(title="Kingfisher RAG Agent")
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/.well-known/agent.json")
        async def get_agent_card():
            return self.agent_card

        @self.app.post("/tasks/send")
        async def send_task(task_data: dict):
            result = await self.process_task(task_data)
            return {"id": task_data["id"], "result": result}

        @self.app.post("/tasks/sendSubscribe")
        async def send_task_subscribe(task_data: dict):
            return StreamingResponse(
                self.process_task_stream(task_data),
                media_type="text/event-stream"
            )
```

## **CASOS DE USO A2A**

### **Cliente usando Kingfisher Agent**

```python
import httpx

async def use_kingfisher():
    async with httpx.AsyncClient() as client:
        # 1. Discovery
        card = await client.get("http://localhost:8000/.well-known/agent.json")
        
        # 2. Process documents
        task = {
            "id": "task_001",
            "message": {
                "role": "user",
                "parts": [{
                    "kind": "text",
                    "text": "Process this research paper"
                }]
            }
        }
        
        result = await client.post("http://localhost:8000/tasks/send", json=task)
        print("Processing result:", result.json())
```

### **Multi-Agent Workflow**

```python
# Kingfisher (RAG) → Summarizer Agent → Translator Agent
async def multi_agent_workflow():
    # 1. Kingfisher procesa documento
    kingfisher_result = await process_with_kingfisher(document)
    
    # 2. Summarizer resume resultados  
    summary = await summarize_with_agent(kingfisher_result)
    
    # 3. Translator traduce resumen
    translation = await translate_with_agent(summary)
    
    return translation
```

## **INTEGRACIÓN SIN RUPTURAS**

### **Principio de Compatibilidad**

- **Sistema Existente**: TripleProcessor, SQLiteManager, ChromaDB se mantienen intactos
- **Nueva Capa A2A**: Envuelve componentes existentes sin modificarlos
- **Coexistencia**: APIs actuales + nuevos endpoints A2A funcionan simultáneamente

### **Ejemplo de Compatibilidad**

```python
# Antes (sigue funcionando):
from rag_preprocessing.core.triple_processor import TripleProcessor
processor = TripleProcessor()
result = processor.process_document(content)

# Después (nueva opción A2A disponible):
import httpx
async with httpx.AsyncClient() as client:
    result = await client.post("http://localhost:8000/tasks/send", json=task)
```

## **CRITERIOS DE ÉXITO SPRINT 3.2.4**

✅ **Agent Card** servido en `/.well-known/agent.json`  
✅ **HTTP endpoints A2A** (`/tasks/send`, `/tasks/sendSubscribe`)  
✅ **LangGraph workflow** para 3 tipos de tasks  
✅ **Integración sin rupturas** con sistemas existentes  
✅ **Streaming responses** vía SSE  
✅ **Tests A2A** verificando compliance  
✅ **Documentación** de uso como agente A2A  

## **ESTRUCTURA DE ARCHIVOS A CREAR**

```
agents/
├── __init__.py
├── kingfisher_agent.py      # Agente principal A2A
├── capabilities/
│   ├── document_processor.py # Skill: procesar documentos
│   ├── knowledge_retriever.py # Skill: recuperar conocimiento
│   └── metadata_analyzer.py   # Skill: analizar metadatos
├── protocol/
│   ├── agent_card.py        # Agent Card definition
│   └── task_manager.py      # Task lifecycle management
└── server/
    ├── a2a_server.py        # HTTP server A2A-compliant
    └── middleware.py        # Auth, logging, monitoring
```

## **TIMELINE IMPLEMENTACIÓN**

- **Día 1-2**: Estructura base + Agent Card + LangGraph workflow  
- **Día 3-4**: FastAPI server + endpoints A2A + integración  
- **Día 5**: Testing + documentación + refinamiento

**Resultado**: Kingfisher operando como agente especializado en ecosistema Google A2A 🚀 