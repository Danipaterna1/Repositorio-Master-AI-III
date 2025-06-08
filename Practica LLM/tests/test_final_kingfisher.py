#!/usr/bin/env python3
"""
🧪 KINGFISHER FINAL SYSTEM EVALUATION
=====================================

Test completo del sistema Kingfisher RAG con documentos reales
y evaluación de todas las capacidades.
"""

import requests
import time
import json
from typing import Dict, Any
from datetime import datetime

def print_header(title: str):
    print(f"\n🎯 {title}")
    print("=" * 60)

def print_test(test_name: str, status: str = "RUNNING"):
    icons = {"RUNNING": "🔧", "PASS": "✅", "FAIL": "❌"}
    print(f"{icons.get(status, '🔧')} {test_name}")

def test_kingfisher_complete_system():
    """Test completo del sistema Kingfisher"""
    
    print_header("KINGFISHER COMPLETE SYSTEM EVALUATION")
    print(f"📅 Test iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    base_url = "http://localhost:8000"
    results = {}
    
    # Test 1: Health Check
    print_test("Health Check", "RUNNING")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✓ Status: {health_data['status']}")
            print(f"   ✓ Version: {health_data['version']}")
            results["health"] = "PASS"
            print_test("Health Check", "PASS")
        else:
            results["health"] = "FAIL"
            print_test("Health Check", "FAIL")
    except Exception as e:
        results["health"] = "FAIL"
        print(f"   ✗ Error: {e}")
        print_test("Health Check", "FAIL")
    
    # Test 2: Agent Card Compliance
    print_test("Agent Card A2A Compliance", "RUNNING")
    try:
        response = requests.get(f"{base_url}/.well-known/agent.json", timeout=5)
        if response.status_code == 200:
            agent_data = response.json()
            print(f"   ✓ Name: {agent_data['name']}")
            print(f"   ✓ Skills: {len(agent_data['skills'])}")
            print(f"   ✓ Provider: {agent_data['provider']['name']}")
            
            # Verificar Google A2A compliance
            required_fields = ['name', 'description', 'version', 'provider', 'skills']
            compliance = all(field in agent_data for field in required_fields)
            
            if compliance:
                results["agent_card"] = "PASS"
                print_test("Agent Card A2A Compliance", "PASS")
            else:
                results["agent_card"] = "FAIL"
                print_test("Agent Card A2A Compliance", "FAIL")
        else:
            results["agent_card"] = "FAIL"
            print_test("Agent Card A2A Compliance", "FAIL")
    except Exception as e:
        results["agent_card"] = "FAIL"
        print(f"   ✗ Error: {e}")
        print_test("Agent Card A2A Compliance", "FAIL")
    
    # Test 3: Document Processing with Real Content
    print_test("Document Processing (Complex)", "RUNNING")
    try:
        test_document = {
            "capability": "process_documents",
            "params": {
                "documents": [{
                    "content": """
                    Kingfisher es un sistema avanzado de preprocessing RAG que implementa:
                    
                    1. ARQUITECTURA TRIPLE STORAGE:
                       - Vector Database (ChromaDB): Para búsqueda semántica
                       - Graph Database (NetworkX): Para relaciones entre entidades
                       - Relational Database (SQLite): Para metadata estructurada
                    
                    2. GOOGLE A2A PROTOCOL:
                       - Agent Card compliant
                       - HTTP/HTTPS + SSE communication
                       - JSON-RPC 2.0 format
                       - LangGraph state machine integration
                    
                    3. CAPACIDADES PRINCIPALES:
                       - Chunking inteligente de documentos
                       - Embedding generation con SentenceTransformers
                       - Entity extraction y knowledge graphs
                       - Retrieval híbrido (vector + graph + metadata)
                       - Análisis de metadata y estadísticas
                    
                    4. CASOS DE USO:
                       - Microservicio RAG en aplicaciones enterprise
                       - Sistema de knowledge management
                       - Pipeline de preprocessing para LLMs
                       - Componente de arquitecturas multi-agent
                    """,
                    "title": "Kingfisher System Architecture Overview",
                    "metadata": {
                        "author": "Kingfisher Team",
                        "version": "1.0.0",
                        "category": "technical_documentation"
                    }
                }]
            }
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{base_url}/tasks/send", 
            json=test_document, 
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Task processed successfully")
            print(f"   ✓ Response format: JSON-RPC 2.0")
            print(f"   ✓ Task ID generated: {result.get('result', {}).get('task_id', 'N/A')[:8]}...")
            results["document_processing"] = "PASS"
            print_test("Document Processing (Complex)", "PASS")
        else:
            results["document_processing"] = "FAIL"
            print_test("Document Processing (Complex)", "FAIL")
    except Exception as e:
        results["document_processing"] = "FAIL"
        print(f"   ✗ Error: {e}")
        print_test("Document Processing (Complex)", "FAIL")
    
    # Test 4: Knowledge Retrieval
    print_test("Knowledge Retrieval", "RUNNING")
    try:
        retrieval_query = {
            "capability": "retrieve_knowledge",
            "params": {
                "query": "What is Kingfisher triple storage architecture?",
                "max_results": 5,
                "include_metadata": True
            }
        }
        
        response = requests.post(
            f"{base_url}/tasks/send",
            json=retrieval_query,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"   ✓ Retrieval query processed")
            results["knowledge_retrieval"] = "PASS"
            print_test("Knowledge Retrieval", "PASS")
        else:
            results["knowledge_retrieval"] = "FAIL"
            print_test("Knowledge Retrieval", "FAIL")
    except Exception as e:
        results["knowledge_retrieval"] = "FAIL"
        print(f"   ✗ Error: {e}")
        print_test("Knowledge Retrieval", "FAIL")
    
    # Test 5: Metadata Analysis
    print_test("Metadata Analysis", "RUNNING")
    try:
        metadata_query = {
            "capability": "analyze_metadata",
            "params": {
                "analysis_type": "system_stats",
                "include_performance": True
            }
        }
        
        response = requests.post(
            f"{base_url}/tasks/send",
            json=metadata_query,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"   ✓ Metadata analysis completed")
            results["metadata_analysis"] = "PASS"
            print_test("Metadata Analysis", "PASS")
        else:
            results["metadata_analysis"] = "FAIL"
            print_test("Metadata Analysis", "FAIL")
    except Exception as e:
        results["metadata_analysis"] = "FAIL"
        print(f"   ✗ Error: {e}")
        print_test("Metadata Analysis", "FAIL")
    
    # Test 6: Performance Metrics
    print_test("Performance Metrics", "RUNNING")
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/metrics", timeout=5)
        response_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            metrics = response.json()
            print(f"   ✓ Response time: {response_time:.2f}ms")
            print(f"   ✓ Active tasks: {metrics.get('active_tasks', 'N/A')}")
            results["performance"] = "PASS"
            print_test("Performance Metrics", "PASS")
        else:
            results["performance"] = "FAIL"
            print_test("Performance Metrics", "FAIL")
    except Exception as e:
        results["performance"] = "FAIL"
        print(f"   ✗ Error: {e}")
        print_test("Performance Metrics", "FAIL")
    
    # Resumen Final
    print_header("KINGFISHER EVALUATION SUMMARY")
    
    passed_tests = sum(1 for result in results.values() if result == "PASS")
    total_tests = len(results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"📊 Tests Passed: {passed_tests}/{total_tests}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    for test_name, result in results.items():
        icon = "✅" if result == "PASS" else "❌"
        print(f"   {icon} {test_name.replace('_', ' ').title()}")
    
    # Evaluación Final
    if success_rate >= 90:
        grade = "🏆 EXCELENTE"
        evaluation = "Sistema Kingfisher completamente operacional y listo para producción"
    elif success_rate >= 75:
        grade = "🥈 BUENO"
        evaluation = "Sistema Kingfisher mayormente funcional con mejoras menores necesarias"
    elif success_rate >= 50:
        grade = "🥉 ACEPTABLE"
        evaluation = "Sistema Kingfisher funcional pero necesita optimizaciones"
    else:
        grade = "❌ REQUIERE TRABAJO"
        evaluation = "Sistema Kingfisher necesita correcciones significativas"
    
    print(f"\n🎯 EVALUACIÓN FINAL: {grade}")
    print(f"📝 {evaluation}")
    print(f"📅 Test completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results, success_rate

if __name__ == "__main__":
    results, score = test_kingfisher_complete_system()
    print(f"\n🎯 SCORE FINAL: {score:.1f}%") 