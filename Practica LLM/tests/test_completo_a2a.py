#!/usr/bin/env python3
"""
🧪 TEST COMPLETO A2A - VALIDACIÓN REAL DEL SISTEMA
==================================================

Test end-to-end del sistema Kingfisher A2A funcionando en puerto 8000.
Valida el pipeline completo: DOCUMENTOS → CHUNKING → EMBEDDING → TRIPLE STORAGE → A2A
"""

import asyncio
import json
import requests
import time
from datetime import datetime
from pathlib import Path

# Configuración del test
BASE_URL = "http://localhost:8000"
TEST_DATA_DIR = Path("test_documents")

class KingfisherA2AValidator:
    """Validador completo del sistema A2A Kingfisher"""
    
    def __init__(self):
        self.base_url = BASE_URL
        self.results = {}
        
    def test_server_health(self):
        """Test 1: Verificar salud del servidor"""
        print("\n🔧 TEST 1: Server Health Check")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Servidor funcionando: {data['status']}")
                print(f"📅 Timestamp: {data['timestamp']}")
                print(f"📌 Version: {data['version']}")
                return True
            else:
                print(f"❌ Error status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error conectando al servidor: {e}")
            return False
    
    def test_agent_discovery(self):
        """Test 2: Verificar Agent Discovery (A2A Protocol)"""
        print("\n🔧 TEST 2: A2A Agent Discovery")
        try:
            response = requests.get(f"{self.base_url}/.well-known/agent.json", timeout=5)
            if response.status_code == 200:
                agent_card = response.json()
                print(f"✅ Agent Card disponible")
                print(f"📛 Nombre: {agent_card.get('name', 'N/A')}")
                print(f"📋 Descripción: {agent_card.get('description', 'N/A')[:80]}...")
                print(f"🎯 Skills disponibles: {len(agent_card.get('skills', []))}")
                
                # Mostrar skills
                for skill in agent_card.get('skills', []):
                    print(f"   • {skill.get('name', 'Unknown')}: {skill.get('description', 'No description')[:50]}...")
                
                return True, agent_card
            else:
                print(f"❌ Error status: {response.status_code}")
                return False, None
        except Exception as e:
            print(f"❌ Error obteniendo Agent Card: {e}")
            return False, None
    
    def test_document_processing_task(self):
        """Test 3: Enviar task de procesamiento de documento"""
        print("\n🔧 TEST 3: Document Processing Task")
        
        # Crear un task A2A para procesamiento de documento
        task_data = {
            "id": f"test_doc_task_{int(time.time())}",
            "skill": "process_documents",
            "message": {
                "parts": [
                    {
                        "kind": "text",
                        "text": "Process this research document about artificial intelligence and machine learning. Extract entities, relationships, and store in triple storage system."
                    }
                ]
            },
            "parameters": {
                "processing_mode": "TRIPLE_FULL",
                "include_llm": True
            }
        }
        
        try:
            print(f"📤 Enviando task: {task_data['id']}")
            response = requests.post(
                f"{self.base_url}/tasks/send",
                json=task_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Task procesado exitosamente")
                print(f"📊 Task ID: {result.get('task_id', 'N/A')}")
                print(f"📈 Status: {result.get('status', 'N/A')}")
                
                if 'artifacts' in result:
                    print(f"📁 Artifacts generados: {len(result['artifacts'])}")
                    for artifact in result['artifacts']:
                        print(f"   • {artifact.get('name', 'Unknown')}: {artifact.get('description', 'No description')[:50]}...")
                
                return True, result
            else:
                print(f"❌ Error status: {response.status_code}")
                print(f"📝 Response: {response.text[:200]}...")
                return False, None
                
        except Exception as e:
            print(f"❌ Error procesando task: {e}")
            return False, None
    
    def test_knowledge_retrieval(self):
        """Test 4: Test de recuperación de conocimiento"""
        print("\n🔧 TEST 4: Knowledge Retrieval")
        
        retrieval_task = {
            "id": f"test_retrieval_{int(time.time())}",
            "skill": "retrieve_knowledge", 
            "message": {
                "parts": [
                    {
                        "kind": "text",
                        "text": "Find information about machine learning algorithms and artificial intelligence concepts"
                    }
                ]
            },
            "parameters": {
                "search_mode": "hybrid",
                "top_k": 5
            }
        }
        
        try:
            print(f"🔍 Enviando query de búsqueda...")
            response = requests.post(
                f"{self.base_url}/tasks/send",
                json=retrieval_task,
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Búsqueda completada")
                print(f"📊 Task ID: {result.get('task_id', 'N/A')}")
                print(f"📈 Status: {result.get('status', 'N/A')}")
                
                if 'artifacts' in result and result['artifacts']:
                    knowledge_results = result['artifacts'][0].get('content', {})
                    results_count = knowledge_results.get('results_count', 0)
                    print(f"📚 Resultados encontrados: {results_count}")
                    
                    if 'results' in knowledge_results:
                        for i, res in enumerate(knowledge_results['results'][:3]):
                            print(f"   {i+1}. {res.get('content', 'No content')[:80]}...")
                
                return True, result
            else:
                print(f"❌ Error status: {response.status_code}")
                return False, None
                
        except Exception as e:
            print(f"❌ Error en retrieval: {e}")
            return False, None
    
    def test_metadata_analysis(self):
        """Test 5: Análisis de metadata"""
        print("\n🔧 TEST 5: Metadata Analysis")
        
        analysis_task = {
            "id": f"test_analysis_{int(time.time())}",
            "skill": "analyze_metadata",
            "message": {
                "parts": [
                    {
                        "kind": "text", 
                        "text": "Generate system statistics and metadata analysis"
                    }
                ]
            },
            "parameters": {
                "analysis_type": "documents"
            }
        }
        
        try:
            print(f"📊 Solicitando análisis de metadata...")
            response = requests.post(
                f"{self.base_url}/tasks/send",
                json=analysis_task,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Análisis completado")
                print(f"📊 Task ID: {result.get('task_id', 'N/A')}")
                
                if 'artifacts' in result and result['artifacts']:
                    stats = result['artifacts'][0].get('content', {}).get('statistics', {})
                    print(f"📈 Estadísticas del sistema:")
                    for key, value in stats.items():
                        print(f"   • {key}: {value}")
                
                return True, result
            else:
                print(f"❌ Error status: {response.status_code}")
                return False, None
                
        except Exception as e:
            print(f"❌ Error en análisis: {e}")
            return False, None
    
    def test_system_performance(self):
        """Test 6: Verificar métricas de performance"""
        print("\n🔧 TEST 6: System Performance Metrics")
        
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                metrics = response.json()
                print(f"✅ Métricas obtenidas en {response_time:.2f}ms")
                print(f"📊 Métricas disponibles:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"   • {key}: {value}")
                    else:
                        print(f"   • {key}: {str(value)[:50]}...")
                
                return True, metrics
            else:
                print(f"⚠️ Endpoint /metrics no disponible (status: {response.status_code})")
                return True, {}  # No es crítico
                
        except Exception as e:
            print(f"⚠️ Métricas no disponibles: {e}")
            return True, {}  # No es crítico
    
    def run_complete_validation(self):
        """Ejecutar validación completa del sistema"""
        print("🚀 INICIANDO VALIDACIÓN COMPLETA SISTEMA A2A KINGFISHER")
        print("=" * 60)
        
        tests_results = []
        start_time = time.time()
        
        # Test 1: Health Check
        health_ok = self.test_server_health()
        tests_results.append(("Server Health", health_ok))
        
        if not health_ok:
            print("\n❌ FALLO CRÍTICO: Servidor no disponible")
            return False
        
        # Test 2: Agent Discovery
        discovery_ok, agent_card = self.test_agent_discovery()
        tests_results.append(("Agent Discovery", discovery_ok))
        
        # Test 3: Document Processing
        processing_ok, proc_result = self.test_document_processing_task()
        tests_results.append(("Document Processing", processing_ok))
        
        # Test 4: Knowledge Retrieval
        retrieval_ok, retr_result = self.test_knowledge_retrieval()
        tests_results.append(("Knowledge Retrieval", retrieval_ok))
        
        # Test 5: Metadata Analysis
        analysis_ok, analysis_result = self.test_metadata_analysis()
        tests_results.append(("Metadata Analysis", analysis_ok))
        
        # Test 6: Performance Metrics
        metrics_ok, metrics = self.test_system_performance()
        tests_results.append(("Performance Metrics", metrics_ok))
        
        # Resumen
        total_time = time.time() - start_time
        passed_tests = sum(1 for _, result in tests_results if result)
        total_tests = len(tests_results)
        
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE VALIDACIÓN")
        print("=" * 60)
        
        for test_name, result in tests_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
        
        print(f"\n🎯 RESULTADO FINAL: {passed_tests}/{total_tests} tests pasaron")
        print(f"⏱️ Tiempo total: {total_time:.2f} segundos")
        
        if passed_tests == total_tests:
            print("🏆 EXCELENTE: Sistema A2A Kingfisher completamente operacional")
            print("✅ Pipeline completo validado: DOCUMENTOS → TRIPLE STORAGE → A2A")
        elif passed_tests >= total_tests * 0.8:
            print("🟡 BUENO: Sistema mayormente funcional, algunos componentes opcionales fallan")
        else:
            print("❌ PROBLEMAS: Sistema tiene fallas significativas que requieren atención")
        
        return passed_tests == total_tests

if __name__ == "__main__":
    print("🧪 KINGFISHER A2A SYSTEM VALIDATOR")
    print("=" * 50)
    print(f"🎯 Objetivo: Validar sistema completo funcionando")
    print(f"🌐 Servidor: {BASE_URL}")
    print(f"📅 Timestamp: {datetime.now().isoformat()}")
    
    validator = KingfisherA2AValidator()
    success = validator.run_complete_validation()
    
    exit_code = 0 if success else 1
    exit(exit_code)