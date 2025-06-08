# 🚀 Kingfisher Quick Start Guide

**¡Ponte en marcha con Kingfisher en 5 minutos!**

---

## ⚡ Super Quick Start (1 minuto)

```bash
# 1. Activar entorno virtual
.\venv\Scripts\activate

# 2. Iniciar Kingfisher
python -c "from agents.server.a2a_server import app; import uvicorn; uvicorn.run(app, host='localhost', port=8000)"

# 3. Verificar (en otra terminal)
curl http://localhost:8000/health
```

**¡Ya tienes Kingfisher corriendo en http://localhost:8000!** 🎉

---

## 📋 Checklist de Verificación

### ✅ **Sistema Funcional**
- [ ] Health check responde: `{"status":"healthy"}`
- [ ] Agent Card disponible: `http://localhost:8000/.well-known/agent.json`
- [ ] Dependencias OK: NumPy 1.26.4 + SciPy 1.12.0

### ✅ **Tests Pasando**
```bash
# Test básico (30 segundos)
python tests/test_quick_a2a.py

# Test completo (2 minutos)
python tests/test_completo_a2a.py

# Evaluación final (3 minutos)
python tests/test_final_kingfisher.py
```

---

## 💡 Primeros Pasos

### **1. Procesar tu primer documento**

```bash
curl -X POST http://localhost:8000/tasks/send \
  -H "Content-Type: application/json" \
  -d '{
    "capability": "process_documents",
    "params": {
      "documents": [{
        "content": "Artificial Intelligence is transforming industries worldwide. Machine learning algorithms enable computers to learn from data without explicit programming.",
        "title": "AI Introduction",
        "metadata": {"category": "technology", "author": "AI Researcher"}
      }]
    }
  }'
```

### **2. Buscar conocimiento**

```bash
curl -X POST http://localhost:8000/tasks/send \
  -H "Content-Type: application/json" \
  -d '{
    "capability": "retrieve_knowledge",
    "params": {
      "query": "What is artificial intelligence?",
      "max_results": 3
    }
  }'
```

### **3. Analizar métricas del sistema**

```bash
curl http://localhost:8000/metrics
```

---

## 🎯 Casos de Uso Inmediatos

### **📄 Knowledge Base Personal**
1. Procesa tus documentos importantes
2. Búscalos semánticamente
3. Obtén respuestas inteligentes

### **🤖 Integración con LLMs**
1. Preprocesa documentos corporativos
2. Úsalos como contexto para GPT/Claude
3. Mejora respuestas con conocimiento específico

### **🔍 Sistema de Búsqueda Avanzado**
1. Vector search: similitud semántica
2. Graph search: relaciones entre entidades
3. Metadata filters: búsquedas específicas

---

## ⚙️ Configuración Básica

### **Archivos importantes:**
- `requirements.txt` - Dependencias del proyecto
- `agents/server/a2a_server.py` - Servidor principal
- `rag_preprocessing/core/` - Pipeline de procesamiento
- `data/` - Bases de datos (vector, graph, metadata)

### **Puertos por defecto:**
- **8000**: Servidor A2A principal
- **8001**: Performance server (opcional)

### **Directorios de datos:**
- `data/chromadb/` - Vector database
- `data/graphs/` - Graph database
- `data/metadata/` - Relational database

---

## 🆘 Problemas Comunes

### **Error: "uvicorn no se reconoce"**
```bash
# Solución: usar Python directamente
python -c "from agents.server.a2a_server import app; import uvicorn; uvicorn.run(app, host='localhost', port=8000)"
```

### **Error: NumPy/SciPy incompatibility**
```bash
# Solución: reinstalar dependencias
pip uninstall scipy -y
pip install scipy==1.12.0 --no-cache-dir
```

### **Error: Puerto en uso**
```bash
# Solución: usar otro puerto
python -c "from agents.server.a2a_server import app; import uvicorn; uvicorn.run(app, host='localhost', port=8001)"
```

---

## 📚 Próximos Pasos

1. **Lee la documentación completa**: [README.md](README.md)
2. **Explora los ejemplos**: Carpeta `demos/`
3. **Revisa la arquitectura**: [docs/ARQUITECTURA_A2A.md](docs/ARQUITECTURA_A2A.md)
4. **Prueba tests avanzados**: Carpeta `tests/`

---

## 🎉 ¡Ya estás listo!

**Kingfisher está diseñado para ser**:
- ✅ **Fácil de usar** - API REST simple
- ✅ **Potente** - Triple storage architecture
- ✅ **Escalable** - Production ready
- ✅ **Estándar** - Google A2A compliant

**¡Disfruta construyendo aplicaciones inteligentes con Kingfisher!** 🐦✨ 