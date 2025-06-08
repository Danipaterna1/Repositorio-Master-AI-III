# 📋 PLAN TÉCNICO - PREPROCESAMIENTO RAG

> **👉 Para seguimiento completo del proyecto, ver [BITACORA_KINGFISHER.md](BITACORA_KINGFISHER.md)**

## 🎯 OBJETIVO TÉCNICO ESPECÍFICO
```
PIPELINE REUTILIZABLE: DOCUMENTOS → CHUNKING → EMBEDDING → ALMACENAMIENTO TRIPLE → AGENTE A2A
```

---

## 🏗️ **ARQUITECTURA TÉCNICA IMPLEMENTADA**

### **📦 CORE MODULES COMPLETADOS**

#### **1. rag_preprocessing/core/**
```python
├── triple_processor.py          # ✅ Pipeline principal
├── embedding_manager.py         # ✅ Gestión de embeddings
├── smart_chunker.py            # ⚠️ Performance optimization
├── enhanced_triple_processor.py # ⚠️ Performance optimization
└── embedding_types.py          # ✅ Type definitions
```

#### **2. rag_preprocessing/storage/**
```python
├── vector_store.py             # ✅ ChromaDB integration
├── graph_store.py              # ✅ NetworkX + spaCy
├── metadata_store.py           # ✅ SQLite integration
└── hybrid_vector_store.py      # ⚠️ Performance optimization
```

#### **3. agents/**
```python
├── kingfisher_agent.py         # ✅ Main A2A agent
├── kingfisher_agent_performance.py # ⚠️ Performance variant
├── protocol/
│   ├── agent_card.py           # ✅ A2A discovery
│   └── task_manager.py         # ✅ LangGraph workflow
└── server/
    ├── a2a_server.py           # ✅ Production server
    └── a2a_performance_server.py # ❌ Dependency conflicts
```

---

## 🔧 **IMPLEMENTACIÓN TÉCNICA DETALLADA**

### **✅ TRIPLE STORAGE PIPELINE (COMPLETADO)**

#### **Vector Storage (ChromaDB)**
```python
# Configuración actual exitosa
COLLECTION_NAME = "kingfisher_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions
CHUNK_SIZE = 500  # Optimizado para context
CHUNK_OVERLAP = 50  # 10% overlap
```

#### **Graph Storage (NetworkX + spaCy)**
```python
# Configuración actual exitosa
NLP_MODEL = "es_core_news_sm"  # Spanish language model
ENTITY_TYPES = ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]
RELATIONSHIP_EXTRACTION = "dependency_parsing"  # spaCy dep trees
COMMUNITY_ALGORITHM = "leiden"  # Superior to Louvain
```

#### **Metadata Storage (SQLite)**
```python
# Schema implementado
TABLES = {
    "documents": ["id", "title", "content", "timestamp", "source"],
    "chunks": ["id", "document_id", "content", "vector_id", "position"],
    "entities": ["id", "text", "label", "document_id", "start", "end"],
    "relationships": ["id", "source", "target", "relation", "document_id"]
}
```

### **✅ GOOGLE A2A PROTOCOL (COMPLETADO)**

#### **Agent Card Specification**
```json
{
  "name": "Kingfisher RAG Agent",
  "version": "1.0.0",
  "provider": {"name": "Kingfisher", "url": "http://localhost:8000"},
  "skills": [
    {
      "name": "process_documents",
      "description": "Process documents through triple storage pipeline",
      "parameters": {
        "content": {"type": "string", "required": true},
        "processing_mode": {"type": "string", "enum": ["VECTOR_ONLY", "GRAPH_ONLY", "TRIPLE_FULL"]}
      }
    },
    {
      "name": "retrieve_knowledge", 
      "description": "Retrieve information using hybrid search",
      "parameters": {
        "query": {"type": "string", "required": true},
        "search_type": {"type": "string", "enum": ["vector", "graph", "hybrid"]}
      }
    },
    {
      "name": "analyze_metadata",
      "description": "Analyze document metadata and relationships", 
      "parameters": {
        "document_id": {"type": "string", "required": true}
      }
    }
  ]
}
```

#### **LangGraph Workflow Implementation**
```python
# Estado exitoso del workflow
WORKFLOW_NODES = {
    "route_task": "Determinar tipo de skill solicitada",
    "process_documents": "Ejecutar pipeline triple storage", 
    "retrieve_knowledge": "Búsqueda híbrida vector+graph",
    "analyze_metadata": "Análisis relacional metadatos",
    "complete_task": "Finalizar y retornar resultado"
}

WORKFLOW_EDGES = {
    START → "route_task",
    "route_task" → ["process_documents", "retrieve_knowledge", "analyze_metadata"],
    ["process_documents", "retrieve_knowledge", "analyze_metadata"] → "complete_task",
    "complete_task" → END
}
```

---

## ⚠️ **PROBLEMAS TÉCNICOS CRÍTICOS IDENTIFICADOS**

### **🚨 DEPENDENCY CONFLICTS (BLOQUEANTE)**

#### **Conflicto NumPy/SciPy**
```
ERROR: numpy.dtype size changed, may indicate binary incompatibility
Expected 96 from C header, got 88 from PyObject

CAUSA: 
- NumPy version: 2.3.0 (installed)
- SciPy requirement: NumPy >=1.22.4 and <1.29.0
- Binary incompatibility between versions

IMPACTO:
- Performance server no puede iniciar
- Smart chunker + NLTK + SciPy chain fails
- Advanced performance features blocked
```

#### **Import Chain Issues**
```python
# Cadena de imports problemática
smart_chunker.py → nltk → scipy.stats → scipy.spatial → numpy (binary conflict)

SOLUCION INMEDIATA:
pip install numpy==1.26.4 --force-reinstall
pip install scipy==1.12.0 --upgrade
```

### **🔧 ASYNC/SYNC MISMATCH**
```python
# Error en kingfisher_agent_performance.py
await self.task_manager.update_task_status(task_id, status)
#     ^^^^^^^^^^^^^^^^^ No es async function

FIX:
self.task_manager.update_task_status(task_id, status)  # Sin await
```

---

## 🎯 **ESTRATEGIA DE RESOLUCIÓN INMEDIATA**

### **PASO 1: Environment Fix (5 min)**
```bash
# En PowerShell
.\venv\Scripts\activate

# Fix dependencies 
pip install numpy==1.26.4 --force-reinstall
pip install scipy==1.12.0 --upgrade

# Verificar compatibilidad
python -c "import numpy; import scipy; print('OK')"
```

### **PASO 2: Code Fix (10 min)**
```python
# En kingfisher_agent_performance.py línea ~87
# CAMBIAR:
await self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED)

# POR:
self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED)
```

### **PASO 3: Validation Test (5 min)**
```bash
# Test performance server
uvicorn agents.server.a2a_performance_server:app --host 0.0.0.0 --port 8001 --reload

# Verificar ambos servidores
curl http://localhost:8000/health  # Main server
curl http://localhost:8001/health  # Performance server
```

---

## 🎯 **OBJETIVO TÉCNICO INMEDIATO**

### **CRITERIO DE ÉXITO TÉCNICO:**
```bash
✅ Virtual environment activated
✅ NumPy 1.26.4 + SciPy 1.12.0 compatible
✅ Performance server starts successfully
✅ Both servers (8000 + 8001) operational
✅ 7/7 integration tests passing
✅ All performance optimizations functional
```

### **ENTREGABLES TÉCNICOS POST-FIX:**
1. **Performance benchmarks** documentados
2. **Smart chunker** operational
3. **Batch embedder** functional
4. **Hybrid retriever** tested
5. **Dual-server architecture** validated

---

## 📋 **VALIDACIÓN TÉCNICA FINAL**

### **Component Integration Test**
```python
# Test completo post-fix
from rag_preprocessing.core import TripleProcessor
from agents.kingfisher_agent_performance import KingfisherPerformanceAgent

# 1. Core functionality
processor = TripleProcessor()
result = await processor.process_document("test content")
assert result.success

# 2. Performance features  
agent = KingfisherPerformanceAgent()
task_result = await agent.process_task({"skill": "process_documents"})
assert task_result.status == "completed"

# 3. HTTP endpoints
import requests
response = requests.get("http://localhost:8001/health")
assert response.status_code == 200
```

---

**🎯 RESULTADO ESPERADO**: Pipeline RAG completo con performance optimizations funcional y listo para integración como componente reutilizable. 