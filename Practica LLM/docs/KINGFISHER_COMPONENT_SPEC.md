# 🧩 KINGFISHER RAG COMPONENT SPECIFICATION

## 🎯 **OBJECTIVE: REUSABLE RAG COMPONENT**

Kingfisher es un **componente modular de preprocessing RAG** diseñado para integrarse en aplicaciones mayores.

### 📦 **Integration Modes**

1. **🐍 Embedded Library**: Import directo en Python
2. **🌐 HTTP Microservice**: A2A-compliant endpoints  
3. **🔌 Event-Driven**: Queue/PubSub integration

### 🏗️ **Architecture as Component**

```
LARGER APPLICATION
├── frontend/                    
├── backend/                     
├── auth/                        
├── kingfisher/                  # 🎯 RAG COMPONENT
│   ├── rag_preprocessing/       # Core processing
│   ├── agents/                  # A2A interface
│   └── interfaces/              # Integration APIs
└── database/                    
```

## 🎯 **INTERFACES EXPOSED**

### 🐍 **Python Interface**
```python
from rag_preprocessing.core import TripleProcessor
from agents.kingfisher_agent import KingfisherAgent

# Direct processing
processor = TripleProcessor()
result = await processor.process_document(content)

# Agent processing
agent = KingfisherAgent()
task_result = await agent.process_task(task_data)
```

### 🌐 **HTTP Interface**
```http
POST /tasks/send              # A2A task processing
GET /.well-known/agent.json   # Agent discovery
GET /health                   # Health monitoring
GET /metrics                  # Performance data
```

### 🔧 **Configuration**
```python
from agents.config import KingfisherConfig

config = KingfisherConfig(
    database=DatabaseConfig(...),
    llm=LLMConfig(...),
    performance=PerformanceConfig(...)
)
```

## 🚀 **READY FOR INTEGRATION**

| Component | Status | Interface |
|-----------|---------|-----------|
| Core Processing | ✅ Ready | Python + HTTP |
| A2A Protocol | ✅ Ready | Standard compliant |
| Configuration | ✅ Ready | Flexible |
| Error Handling | ✅ Ready | Robust |
| Documentation | ✅ Ready | Complete |

**🎯 Kingfisher is ready to be integrated as a component** ✅ 