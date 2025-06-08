#!/usr/bin/env python3
"""
🧩 DEMO: KINGFISHER COMO COMPONENTE
===================================

Demuestra cómo integrar Kingfisher RAG como componente en una aplicación mayor.
Simula una aplicación de chat que usa Kingfisher para procesamiento RAG.
"""

import asyncio
import json
import time
from datetime import datetime

class ChatApplication:
    """Aplicación de chat que usa Kingfisher como componente RAG"""
    
    def __init__(self):
        self.name = "SmartChat Pro"
        self.version = "2.0.0"
        self.users = {}
        
        # 🎯 KINGFISHER COMO COMPONENTE
        self.rag_component = None
        self.rag_service_url = "http://localhost:8000"
        
    async def initialize_rag_component(self):
        """Inicializar componente RAG"""
        print("🔧 Inicializando componente RAG Kingfisher...")
        import requests
        try:
            response = requests.get(f"{self.rag_service_url}/health", timeout=5)
            if response.status_code == 200:
                self.rag_component = {
                    'service_url': self.rag_service_url,
                    'mode': 'http'
                }
                print("✅ RAG Component inicializado en modo HTTP")
                return True
        except:
            pass
        print("❌ No se pudo inicializar componente RAG")
        return False
    
    async def process_document_via_rag(self, content, user_id):
        """Procesar documento usando componente RAG"""
        print(f"📄 Procesando documento para usuario {user_id}...")
        
        if not self.rag_component:
            return {"error": "RAG component not initialized"}
        
        start_time = time.time()
        
        try:
            import requests
            
            task_data = {
                "id": f"chat_task_{int(time.time())}",
                "skill": "process_documents",
                "message": {
                    "parts": [{"kind": "text", "text": content}]
                },
                "parameters": {
                    "processing_mode": "TRIPLE_FULL",
                    "include_llm": True
                }
            }
            
            response = requests.post(
                f"{self.rag_component['service_url']}/tasks/send",
                json=task_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "processing_time": processing_time,
                    "task_id": result.get("task_id"),
                    "artifacts_count": len(result.get("artifacts", [])),
                    "mode": "http"
                }
            else:
                return {"error": f"HTTP request failed: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"HTTP processing failed: {str(e)}"}
    
    async def search_knowledge_via_rag(self, query, user_id):
        """Buscar conocimiento usando componente RAG"""
        print(f"🔍 Buscando: '{query}' para usuario {user_id}")
        
        if not self.rag_component:
            return {"error": "RAG component not initialized"}
        
        start_time = time.time()
        
        try:
            import requests
            
            task_data = {
                "id": f"search_{int(time.time())}",
                "skill": "retrieve_knowledge",
                "message": {"parts": [{"kind": "text", "text": query}]},
                "parameters": {"search_mode": "hybrid", "top_k": 3}
            }
            
            response = requests.post(
                f"{self.rag_component['service_url']}/tasks/send",
                json=task_data,
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            search_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "search_time": search_time,
                    "results": result.get("artifacts", []),
                    "mode": "http"
                }
            else:
                return {"error": f"HTTP search failed: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"HTTP search failed: {str(e)}"}
    
    async def run_integration_demo(self):
        """Ejecutar demo completo de integración"""
        print("🚀 DEMO: KINGFISHER COMO COMPONENTE EN APLICACIÓN MAYOR")
        print("=" * 60)
        
        # 1. Inicializar aplicación
        print(f"🏗️ Inicializando {self.name} v{self.version}...")
        
        # 2. Inicializar componente RAG
        rag_initialized = await self.initialize_rag_component()
        if not rag_initialized:
            print("❌ Demo cancelado: No se pudo inicializar RAG")
            return
        
        # 3. Simular usuario
        user_id = "user_123"
        self.users[user_id] = {
            "name": "Ana García",
            "session_start": datetime.now(),
            "documents_processed": 0,
            "queries_made": 0
        }
        
        print(f"👤 Usuario conectado: {self.users[user_id]['name']}")
        
        # 4. Documento de prueba
        document_content = """
        Inteligencia Artificial y Machine Learning en 2024
        
        La inteligencia artificial ha experimentado avances significativos este año.
        Los modelos de lenguaje grandes (LLMs) como GPT-4, Claude y Gemini han 
        revolucionado el procesamiento de texto. 
        
        En machine learning, las técnicas de aprendizaje por refuerzo y redes 
        neuronales transformers siguen siendo dominantes. Los algoritmos de 
        clustering como K-means y DBSCAN son fundamentales para análisis de datos.
        """
        
        # 5. Procesar documento via RAG
        print("\n🔄 Procesando documento via componente RAG...")
        doc_result = await self.process_document_via_rag(document_content, user_id)
        
        if doc_result.get("status") == "success":
            print(f"✅ Documento procesado exitosamente")
            print(f"   📊 Tiempo: {doc_result['processing_time']:.2f}s")
            print(f"   🧩 Modo: {doc_result['mode']}")
            print(f"   📁 Artifacts: {doc_result.get('artifacts_count', 0)}")
            self.users[user_id]['documents_processed'] += 1
        else:
            print(f"❌ Error procesando documento: {doc_result.get('error')}")
        
        # 6. Realizar búsquedas via RAG
        queries = [
            "¿Qué es machine learning?",
            "¿Cuáles son los algoritmos mencionados?"
        ]
        
        print("\n🔍 Realizando búsquedas via componente RAG...")
        for query in queries:
            search_result = await self.search_knowledge_via_rag(query, user_id)
            
            if search_result.get("status") == "success":
                print(f"✅ '{query}' - {search_result['search_time']:.2f}s")
                results_count = len(search_result.get('results', []))
                print(f"   📚 Resultados: {results_count}")
                self.users[user_id]['queries_made'] += 1
            else:
                print(f"❌ '{query}' - Error: {search_result.get('error')}")
        
        # 7. Resumen
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE INTEGRACIÓN")
        print("=" * 60)
        
        user_stats = self.users[user_id]
        print(f"👤 Usuario: {user_stats['name']}")
        print(f"📄 Documentos procesados: {user_stats['documents_processed']}")
        print(f"🔍 Consultas realizadas: {user_stats['queries_made']}")
        print(f"🧩 Componente RAG: {self.rag_component['mode'].upper()}")
        
        print("\n🎯 RESULTADO: Kingfisher integrado exitosamente como componente")
        print("✅ La aplicación puede usar RAG sin conocer detalles internos")
        print("✅ Interface limpia y reutilizable")

async def main():
    """Función principal del demo"""
    app = ChatApplication()
    await app.run_integration_demo()

if __name__ == "__main__":
    asyncio.run(main()) 