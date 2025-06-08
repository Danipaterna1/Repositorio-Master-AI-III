"""
Test completo para el Triple Processor - Pipeline RAG integrado
Testing simultáneo Vector + Graph + Metadata
"""

import unittest
import os
import tempfile
import shutil
from datetime import datetime

# Configurar paths
import sys
sys.path.insert(0, os.path.abspath('.'))

from rag_preprocessing.core import (
    TripleProcessor, TripleProcessorConfig, 
    ProcessingMode, ErrorStrategy
)

class TestTripleProcessor(unittest.TestCase):
    """Test suite para el Triple Processor"""
    
    @classmethod
    def setUpClass(cls):
        """Setup para toda la clase de tests"""
        cls.temp_dir = tempfile.mkdtemp()
        print(f"\n🚀 INICIANDO TESTS DEL TRIPLE PROCESSOR")
        print(f"📁 Directorio temporal: {cls.temp_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup después de todos los tests"""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
        print(f"\n🧹 Limpieza completada")
    
    def setUp(self):
        """Setup para cada test individual"""
        # Configuración para testing
        self.config = TripleProcessorConfig.create_testing()
        self.config.metadata.database_url = f"sqlite:///{self.temp_dir}/test_metadata.db"
        
        # Texto de prueba
        self.test_text = """
        La inteligencia artificial es una rama de la ciencia de la computación que se centra en la creación de sistemas capaces de realizar tareas que normalmente requieren inteligencia humana. Estos sistemas incluyen el aprendizaje automático, el procesamiento de lenguaje natural y la visión por computadora.

        Las aplicaciones de la inteligencia artificial son vastas y están transformando múltiples industrias. En el sector salud, los algoritmos de IA ayudan en el diagnóstico médico y el descubrimiento de medicamentos. En el transporte, los vehículos autónomos utilizan IA para navegar de forma segura.

        El machine learning, una subcategoría de la IA, permite a las máquinas aprender patrones de los datos sin ser programadas explícitamente. Los algoritmos de deep learning, inspirados en las redes neuronales del cerebro humano, han logrado avances significativos en reconocimiento de imágenes y procesamiento de texto.
        """
        
        self.test_title = "Introducción a la Inteligencia Artificial"
    
    def test_01_config_validation(self):
        """Test validación de configuración"""
        print("\n🔍 Test 1: Validación de configuración")
        
        # Config válida
        self.assertTrue(self.config.validate())
        
        # Config inválida - timeout negativo
        bad_config = TripleProcessorConfig.create_testing()
        bad_config.timeout_seconds = -1
        
        with self.assertRaises(ValueError):
            bad_config.validate()
        
        print("✅ Validación de configuración OK")
    
    def test_02_initialization(self):
        """Test inicialización del procesador"""
        print("\n🔍 Test 2: Inicialización del procesador")
        
        with TripleProcessor(self.config) as processor:
            self.assertIsNotNone(processor)
            
            # Verificar que los componentes se inicializaron según config
            self.assertTrue(hasattr(processor, '_metadata_manager'))
            self.assertTrue(hasattr(processor, '_vector_manager'))
            self.assertTrue(hasattr(processor, '_graph_manager'))
        
        print("✅ Inicialización OK")
    
    def test_03_single_document_processing_full(self):
        """Test procesamiento completo de documento único"""
        print("\n🔍 Test 3: Procesamiento completo - TRIPLE FULL")
        
        # Configurar para TRIPLE_FULL
        config = TripleProcessorConfig.create_testing()
        config.metadata.database_url = f"sqlite:///{self.temp_dir}/test_full.db"
        config.processing_mode = ProcessingMode.TRIPLE_FULL
        config.error_strategy = ErrorStrategy.PARTIAL_SUCCESS
        
        with TripleProcessor(config) as processor:
            result = processor.process_document(
                text=self.test_text,
                document_title=self.test_title
            )
            
            # Verificar resultado general
            self.assertIsNotNone(result)
            print(f"📊 Processing ID: {result.processing_id}")
            print(f"✅ Success: {result.success}")
            print(f"📄 Document ID: {result.document_id}")
            
            # Verificar componentes individuales
            if result.vector_result:
                print(f"🔢 Vector - Chunks: {result.vector_result.get('chunks_created', 0)}")
                self.assertGreater(result.vector_result.get('chunks_created', 0), 0)
            
            if result.graph_result:
                print(f"🕸️ Graph - Entidades: {result.graph_result.get('entities_extracted', 0)}")
                self.assertGreater(result.graph_result.get('entities_extracted', 0), 0)
            
            if result.metadata_result:
                print(f"📋 Metadata - Document ID: {result.metadata_result.get('document_id')}")
                self.assertIsNotNone(result.metadata_result.get('document_id'))
            
            # Verificar métricas
            if result.metrics:
                summary = result.metrics.get_summary()
                print(f"⏱️ Duración total: {summary['total_duration']:.3f}s")
                print(f"🎯 Quality score: {summary['quality']['overall_score']:.2f}")
                
                self.assertGreater(summary['total_duration'], 0)
                self.assertGreaterEqual(summary['quality']['overall_score'], 0)
        
        print("✅ Procesamiento completo OK")
    
    def test_04_vector_only_mode(self):
        """Test modo solo vectorial"""
        print("\n🔍 Test 4: Modo VECTOR_ONLY")
        
        config = TripleProcessorConfig.create_testing()
        config.metadata.database_url = f"sqlite:///{self.temp_dir}/test_vector.db"
        config.processing_mode = ProcessingMode.VECTOR_ONLY
        
        with TripleProcessor(config) as processor:
            result = processor.process_document(
                text=self.test_text,
                document_title=self.test_title
            )
            
            # Solo vector debe estar procesado
            self.assertIsNotNone(result.vector_result)
            self.assertIsNone(result.graph_result)
            self.assertIsNone(result.metadata_result)
            
            print(f"✅ Vector procesado: {result.vector_result.get('chunks_created', 0)} chunks")
        
        print("✅ Modo VECTOR_ONLY OK")
    
    def test_05_error_handling(self):
        """Test manejo de errores"""
        print("\n🔍 Test 5: Manejo de errores")
        
        config = TripleProcessorConfig.create_testing()
        config.metadata.database_url = f"sqlite:///{self.temp_dir}/test_error.db"
        config.error_strategy = ErrorStrategy.PARTIAL_SUCCESS
        
        with TripleProcessor(config) as processor:
            # Test con texto vacío
            result = processor.process_document("")
            
            self.assertFalse(result.success)
            self.assertGreater(len(result.errors), 0)
            print(f"⚠️ Error capturado correctamente: {result.errors[0][:50]}...")
        
        print("✅ Manejo de errores OK")
    
    def test_06_batch_processing(self):
        """Test procesamiento por lotes"""
        print("\n🔍 Test 6: Procesamiento por lotes")
        
        config = TripleProcessorConfig.create_testing()
        config.metadata.database_url = f"sqlite:///{self.temp_dir}/test_batch.db"
        
        documents = [
            {
                "text": "La programación es el arte de crear software.",
                "title": "Programación Básica"
            },
            {
                "text": "Los algoritmos son secuencias de instrucciones para resolver problemas.",
                "title": "Algoritmos"
            },
            {
                "text": "Las bases de datos almacenan información de forma estructurada.",
                "title": "Bases de Datos"
            }
        ]
        
        with TripleProcessor(config) as processor:
            results = processor.process_batch(documents)
            
            self.assertEqual(len(results), 3)
            
            successful_count = sum(1 for r in results if r.success)
            print(f"📊 Procesados: {len(results)}, Exitosos: {successful_count}")
            
            # Al menos 2 de 3 deberían ser exitosos
            self.assertGreaterEqual(successful_count, 2)
        
        print("✅ Procesamiento por lotes OK")
    
    def test_07_parallel_vs_sequential(self):
        """Test comparación paralelo vs secuencial"""
        print("\n🔍 Test 7: Paralelo vs Secuencial")
        
        # Configuración paralela
        config_parallel = TripleProcessorConfig.create_testing()
        config_parallel.metadata.database_url = f"sqlite:///{self.temp_dir}/test_parallel.db"
        config_parallel.enable_parallel_processing = True
        
        # Configuración secuencial  
        config_sequential = TripleProcessorConfig.create_testing()
        config_sequential.metadata.database_url = f"sqlite:///{self.temp_dir}/test_sequential.db"
        config_sequential.enable_parallel_processing = False
        
        # Test paralelo
        start_time = datetime.utcnow()
        with TripleProcessor(config_parallel) as processor:
            result_parallel = processor.process_document(self.test_text, self.test_title)
        time_parallel = (datetime.utcnow() - start_time).total_seconds()
        
        # Test secuencial
        start_time = datetime.utcnow()
        with TripleProcessor(config_sequential) as processor:
            result_sequential = processor.process_document(self.test_text, self.test_title)
        time_sequential = (datetime.utcnow() - start_time).total_seconds()
        
        print(f"⚡ Tiempo paralelo: {time_parallel:.3f}s")
        print(f"🐌 Tiempo secuencial: {time_sequential:.3f}s")
        
        # Ambos deberían ser exitosos
        self.assertTrue(result_parallel.success or result_parallel.partial_success)
        self.assertTrue(result_sequential.success or result_sequential.partial_success)
        
        print("✅ Comparación paralelo/secuencial OK")
    
    def test_08_system_status(self):
        """Test estado del sistema"""
        print("\n🔍 Test 8: Estado del sistema")
        
        with TripleProcessor(self.config) as processor:
            status = processor.get_system_status()
            
            self.assertIn("config", status)
            self.assertIn("metrics", status)
            self.assertIn("components_status", status)
            
            components = status["components_status"]
            print(f"🔧 Componentes activos:")
            print(f"   Vector: {components.get('vector', False)}")
            print(f"   Graph: {components.get('graph', False)}")  
            print(f"   Metadata: {components.get('metadata', False)}")
        
        print("✅ Estado del sistema OK")
    
    def test_09_metrics_collection(self):
        """Test recolección de métricas"""
        print("\n🔍 Test 9: Recolección de métricas")
        
        with TripleProcessor(self.config) as processor:
            result = processor.process_document(self.test_text, self.test_title)
            
            if result.metrics:
                # Test summary
                summary = result.metrics.get_summary()
                self.assertIn("processing_id", summary)
                self.assertIn("components", summary)
                self.assertIn("quality", summary)
                self.assertIn("performance", summary)
                
                # Test detailed report
                detailed = result.metrics.get_detailed_report()
                self.assertIn("stages", detailed)
                self.assertIn("errors", detailed)
                self.assertIn("warnings", detailed)
                
                print(f"📊 Métricas recolectadas correctamente")
                print(f"   Stages: {len(detailed.get('stages', {}))}")
                print(f"   Errors: {len(detailed.get('errors', []))}")
                print(f"   Overall quality: {summary['quality']['overall_score']:.2f}")
        
        print("✅ Recolección de métricas OK")

def run_tests():
    """Ejecutar todos los tests"""
    print("🧪 EJECUTANDO SUITE COMPLETA DE TESTS - TRIPLE PROCESSOR")
    print("=" * 60)
    
    # Crear suite de tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTripleProcessor)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumen final
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("✅ Triple Processor completamente funcional")
        print(f"📊 Tests ejecutados: {result.testsRun}")
        print(f"✅ Exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
        return True
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        print(f"📊 Tests ejecutados: {result.testsRun}")
        print(f"❌ Fallidos: {len(result.failures)}")
        print(f"💥 Errores: {len(result.errors)}")
        
        if result.failures:
            print("\n🔍 DETALLES DE FALLAS:")
            for test, traceback in result.failures:
                print(f"❌ {test}: {traceback}")
        
        if result.errors:
            print("\n💥 DETALLES DE ERRORES:")
            for test, traceback in result.errors:
                print(f"💥 {test}: {traceback}")
        
        return False

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 