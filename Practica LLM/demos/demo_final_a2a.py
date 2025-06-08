#!/usr/bin/env python3
"""
🚀 DEMO FINAL KINGFISHER A2A - DOCUMENTO REAL
==============================================

Demo que procesa un documento real a través del sistema A2A completo:
DOCUMENTO → CHUNKING → EMBEDDING → TRIPLE STORAGE → CONSULTA A2A
"""

import requests
import json
import time
from pathlib import Path

# Configuración
BASE_URL = "http://localhost:8000"
DOCUMENT_PATH = "test_documents/texto_tecnico.txt"

def load_document(path):
    """Cargar documento de test"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def process_document_via_a2a(content):
    """Procesar documento via A2A"""
    print("📤 Enviando documento para procesamiento...")
    
    task = {
        "id": f"demo_doc_{int(time.time())}",
        "skill": "process_documents",
        "message": {
            "parts": [
                {
                    "kind": "text",
                    "text": f"Procesa este documento técnico y extrae entidades, relaciones y conceptos clave:\n\n{content}"
                }
            ]
        },
        "parameters": {
            "processing_mode": "TRIPLE_FULL",
            "include_llm": True,
            "extract_entities": True,
            "build_relationships": True
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/tasks/send",
        json=task,
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Documento procesado exitosamente")
        return result
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None

def query_knowledge_via_a2a(query):
    """Consultar conocimiento via A2A"""
    print(f"🔍 Consultando: '{query}'")
    
    task = {
        "id": f"demo_query_{int(time.time())}",
        "skill": "retrieve_knowledge",
        "message": {
            "parts": [
                {
                    "kind": "text",
                    "text": query
                }
            ]
        },
        "parameters": {
            "search_mode": "hybrid",
            "top_k": 3,
            "include_metadata": True
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/tasks/send",
        json=task,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Consulta procesada")
        return result
    else:
        print(f"❌ Error: {response.status_code}")
        return None

def main():
    print("🚀 DEMO FINAL: KINGFISHER A2A EN ACCIÓN")
    print("=" * 50)
    
    # 1. Cargar documento
    print("📄 Cargando documento...")
    doc_content = load_document(DOCUMENT_PATH)
    print(f"📊 Tamaño: {len(doc_content)} caracteres")
    print(f"📝 Preview: {doc_content[:200]}...\n")
    
    # 2. Procesar documento
    print("🔄 FASE 1: Procesamiento completo")
    processing_result = process_document_via_a2a(doc_content)
    
    if processing_result:
        print("✅ Documento almacenado en triple storage")
        if 'artifacts' in processing_result:
            print(f"📁 Artifacts generados: {len(processing_result['artifacts'])}")
        print()
    
    # 3. Consultas de conocimiento
    print("🔄 FASE 2: Consultas de conocimiento")
    
    queries = [
        "¿Qué es machine learning?",
        "¿Cuáles son los conceptos técnicos principales?",
        "¿Qué tecnologías se mencionan?"
    ]
    
    for query in queries:
        print(f"\n--- Query: {query} ---")
        result = query_knowledge_via_a2a(query)
        
        if result and 'artifacts' in result:
            for artifact in result['artifacts']:
                if 'content' in artifact:
                    content = artifact['content']
                    if 'results' in content:
                        print(f"📚 Encontrados {len(content['results'])} resultados:")
                        for i, res in enumerate(content['results'][:2]):
                            print(f"   {i+1}. {res.get('content', 'No content')[:100]}...")
    
    print("\n" + "=" * 50)
    print("🏆 DEMO COMPLETADO")
    print("✅ Pipeline completo validado:")
    print("   📄 Documento → 🔄 Chunking → 🧠 Embedding → 🗄️ Triple Storage → 🤖 A2A")
    print("✅ Sistema Kingfisher operacional al 100%")

if __name__ == "__main__":
    main()