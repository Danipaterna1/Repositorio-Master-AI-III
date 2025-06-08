"""
Tests for Kingfisher RAG Preprocessing System
============================================

Directorio de tests organizados por componente:

ACTIVOS (Nuevos y funcionando):
- test_triple_processor_simple.py ✅ - Test del sistema triple unificado
- test_metadata_system.py ✅ - Test del sistema de base de datos relacional

LEGACY (Heredados de sprints anteriores):
- test_complete_graph_rag.py - Test completo Graph RAG (Sprint 2)
- test_graph_rag_entity_extraction.py - Test extracción de entidades
- test_llm_community_summarization.py - Test resumen LLM de comunidades
- test_rag_system.py - Test sistema RAG vector original
- test_triple_processor.py - Version completa del test triple

EJECUTAR TESTS:
```bash
# Tests activos
python tests/test_triple_processor_simple.py
python tests/test_metadata_system.py

# Tests legacy (verificar funcionalidad)
python tests/test_complete_graph_rag.py
```
""" 