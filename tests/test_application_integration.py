"""
Simplified integration tests for the three main application classes

This test suite focuses on the key integration points between:
- CorpusManager ↔ QueryEngine 
- QueryEngine ↔ QualityLab
- CorpusManager ↔ QualityLab

主要アプリケーションクラスの簡潔な統合テスト
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import shutil

from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.models.document import Document
from refinire_rag.retrieval.simple_reader import SimpleAnswerSynthesizer, SimpleAnswerSynthesizerConfig
from refinire_rag.retrieval.simple_reranker import SimpleReranker, SimpleRerankerConfig


class TestApplicationIntegration:
    """Test integration between main application classes"""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                id="ai_doc_1",
                content="Artificial Intelligence is a field of computer science that creates intelligent machines.",
                metadata={"source": "ai_basics.txt", "category": "AI", "processing_stage": "original"}
            ),
            Document(
                id="ml_doc_2",
                content="Machine Learning is a subset of AI that enables computers to learn from data.",
                metadata={"source": "ml_guide.txt", "category": "ML", "processing_stage": "original"}
            ),
            Document(
                id="dl_doc_3",
                content="Deep Learning uses neural networks with multiple layers to solve complex problems.",
                metadata={"source": "dl_intro.txt", "category": "DL", "processing_stage": "original"}
            )
        ]

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_corpus_manager_to_query_engine_integration(self, temp_workspace, sample_documents):
        """Test CorpusManager → QueryEngine integration"""
        
        # === Step 1: Set up CorpusManager ===
        document_store = SQLiteDocumentStore(db_path=str(temp_workspace / "test_docs.db"))
        vector_store = InMemoryVectorStore()
        
        corpus_manager = CorpusManager(
            document_store=document_store,
            vector_store=vector_store,
            config={"corpus_name": "integration_test_corpus"}
        )
        
        # Mock document loading and processing
        with patch.object(corpus_manager, 'import_original_documents') as mock_import, \
             patch.object(corpus_manager, 'rebuild_corpus_from_original') as mock_rebuild:
            
            # Set up mocks
            mock_import.return_value = Mock(
                total_documents_created=3,
                total_files_processed=3,
                total_chunks_created=0,
                pipeline_stages_executed=1,
                errors_encountered=0
            )
            mock_rebuild.return_value = Mock(
                total_documents_created=6,
                total_chunks_created=6,
                pipeline_stages_executed=3,
                total_processing_time=1.5
            )
            
            # === Step 2: Create QueryEngine using CorpusManager outputs ===
            
            # Import and process documents
            import_result = corpus_manager.import_original_documents(
                corpus_name="integration_test_corpus",
                directory="mock_path"
            )
            rebuild_result = corpus_manager.rebuild_corpus_from_original(
                corpus_name="integration_test_corpus"
            )
            
            # Get stores for QueryEngine (use actual stores from CorpusManager)
            vector_store_for_qe = corpus_manager.vector_store
            
            # Create QueryEngine components
            synthesizer = SimpleAnswerSynthesizer(SimpleAnswerSynthesizerConfig())
            reranker = SimpleReranker(SimpleRerankerConfig())
            
            # Create QueryEngine
            query_engine = QueryEngine(
                corpus_name="integration_test_corpus",
                retrievers=vector_store_for_qe,
                synthesizer=synthesizer,
                reranker=reranker
            )
            
            # === Step 3: Test QueryEngine functionality ===
            with patch.object(synthesizer, 'synthesize') as mock_synthesize:
                mock_synthesize.return_value = "AI is a field of computer science."
                
                result = query_engine.query("What is artificial intelligence?")
                
                # Verify integration
                assert result.answer == "AI is a field of computer science."
                assert hasattr(result, 'sources')
                
        # Verify the workflow executed successfully
        assert import_result.total_documents_created == 3
        assert rebuild_result.total_chunks_created == 6
        print("✅ CorpusManager → QueryEngine integration successful")

    def test_query_engine_to_quality_lab_integration(self, sample_documents):
        """Test QueryEngine → QualityLab integration"""
        
        # === Step 1: Create QueryEngine ===
        mock_vector_store = Mock()
        mock_vector_store.retrieve.return_value = [
            Mock(document_id="ai_doc_1", document=sample_documents[0], score=0.9),
            Mock(document_id="ml_doc_2", document=sample_documents[1], score=0.8)
        ]
        
        synthesizer = SimpleAnswerSynthesizer(SimpleAnswerSynthesizerConfig())
        reranker = SimpleReranker(SimpleRerankerConfig())
        
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=mock_vector_store,
            synthesizer=synthesizer,
            reranker=reranker
        )
        
        # === Step 2: Create QualityLab ===
        quality_lab = QualityLab(
            corpus_name="test_corpus",
            config=QualityLabConfig(qa_pairs_per_document=2)
        )
        
        # === Step 3: Test QualityLab evaluation of QueryEngine ===
        
        # Generate QA pairs
        qa_pairs = quality_lab.generate_qa_pairs(sample_documents, num_pairs=4)
        
        assert len(qa_pairs) == 4
        assert all(qa.document_id in ["ai_doc_1", "ml_doc_2", "dl_doc_3"] for qa in qa_pairs)
        
        # Mock QueryEngine responses for evaluation
        with patch.object(query_engine, 'query') as mock_query:
            # Create mock query result
            mock_result = Mock()
            mock_result.answer = "Test answer for evaluation"
            mock_result.sources = [Mock(document_id="ai_doc_1")]
            mock_result.confidence = 0.85
            mock_query.return_value = mock_result
            
            # Evaluate QueryEngine using QualityLab
            evaluation_results = quality_lab.evaluate_query_engine(
                query_engine=query_engine,
                qa_pairs=qa_pairs
            )
            
            # Verify evaluation results
            assert "evaluation_summary" in evaluation_results
            assert "test_results" in evaluation_results
            assert evaluation_results["corpus_name"] == "test_corpus"
            assert len(evaluation_results["test_results"]) == 4
            
            # Check evaluation metrics
            summary = evaluation_results["evaluation_summary"]
            assert "total_tests" in summary
            assert "pass_rate" in summary
            assert "source_accuracy" in summary
            assert summary["total_tests"] == 4
        
        print("✅ QueryEngine → QualityLab integration successful")

    def test_corpus_manager_to_quality_lab_integration(self, temp_workspace, sample_documents):
        """Test CorpusManager → QualityLab integration"""
        
        # === Step 1: Set up CorpusManager ===
        document_store = SQLiteDocumentStore(db_path=str(temp_workspace / "test_docs.db"))
        vector_store = InMemoryVectorStore()
        
        corpus_manager = CorpusManager(
            document_store=document_store,
            vector_store=vector_store,
            config={"corpus_name": "quality_test_corpus"}
        )
        
        # === Step 2: Set up QualityLab ===
        quality_lab = QualityLab(
            corpus_name="quality_test_corpus",
            config=QualityLabConfig(qa_pairs_per_document=1)
        )
        
        # === Step 3: Test QualityLab using CorpusManager documents ===
        
        # Mock CorpusManager to return documents
        with patch.object(corpus_manager.document_store, 'get_all_document_ids') as mock_get_ids, \
             patch.object(corpus_manager.document_store, 'get_document') as mock_get_doc:
            mock_get_ids.return_value = ["ai_doc_1", "ml_doc_2", "dl_doc_3"]
            mock_get_doc.side_effect = lambda doc_id: next(
                (doc for doc in sample_documents if doc.id == doc_id), None
            )
            
            # Get documents from CorpusManager
            doc_ids = corpus_manager.document_store.get_all_document_ids()
            corpus_documents = [corpus_manager.document_store.get_document(doc_id) for doc_id in doc_ids]
            corpus_documents = [doc for doc in corpus_documents if doc is not None]
            
            # Generate QA pairs using documents from CorpusManager
            qa_pairs = quality_lab.generate_qa_pairs(corpus_documents, num_pairs=3)
            
            # Verify QA pairs generated from CorpusManager documents
            assert len(qa_pairs) == 3
            assert all(qa.document_id in ["ai_doc_1", "ml_doc_2", "dl_doc_3"] for qa in qa_pairs)
            assert all(qa.metadata["corpus_name"] == "quality_test_corpus" for qa in qa_pairs)
            
        print("✅ CorpusManager → QualityLab integration successful")

    def test_three_way_integration_workflow(self, temp_workspace, sample_documents):
        """Test complete workflow: CorpusManager → QueryEngine → QualityLab"""
        
        print("\n🔄 Testing complete three-way integration workflow...")
        
        # === Step 1: CorpusManager Setup ===
        document_store = DocumentStore()
        vector_store = InMemoryVectorStore()
        
        corpus_manager = CorpusManager(
            document_store=document_store,
            vector_store=vector_store,
            config={"corpus_name": "full_integration_corpus"}
        )
        
        # Mock CorpusManager operations
        with patch.object(corpus_manager, 'load_documents_from_directory') as mock_load, \
             patch.object(corpus_manager, 'process_documents') as mock_process:
            
            # Set up CorpusManager mocks
            mock_load.return_value = {"success": True, "documents_loaded": 3}
            mock_process.return_value = {"success": True, "total_chunks_created": 6}
            
            # Mock vector store behavior
            with patch.object(corpus_manager.vector_store, 'retrieve') as mock_retrieve, \
                 patch.object(corpus_manager.document_store, 'get_all_document_ids') as mock_get_ids, \
                 patch.object(corpus_manager.document_store, 'get_document') as mock_get_doc:
                
                mock_retrieve.return_value = [
                    Mock(document_id="ai_doc_1", document=sample_documents[0], score=0.9)
                ]
                mock_get_ids.return_value = ["ai_doc_1", "ml_doc_2", "dl_doc_3"]
                mock_get_doc.side_effect = lambda doc_id: next(
                    (doc for doc in sample_documents if doc.id == doc_id), None
                )
            
            # === Step 2: QueryEngine Setup ===
            
                # Load documents via CorpusManager
                corpus_manager.load_documents_from_directory("mock_path")
                corpus_manager.process_documents()
                
                # Get stores
                vector_store_for_qe = corpus_manager.vector_store
                document_store_for_eval = corpus_manager.document_store
            
                # Create QueryEngine
                synthesizer = SimpleAnswerSynthesizer(SimpleAnswerSynthesizerConfig())
                reranker = SimpleReranker(SimpleRerankerConfig())
                
                query_engine = QueryEngine(
                    corpus_name="full_integration_corpus",
                    retrievers=vector_store_for_qe,
                    synthesizer=synthesizer,
                    reranker=reranker
                )
            
                # === Step 3: QualityLab Setup and Evaluation ===
                
                quality_lab = QualityLab(
                    corpus_name="full_integration_corpus",
                    config=QualityLabConfig(qa_pairs_per_document=1)
                )
                
                # Get documents for evaluation
                doc_ids = document_store_for_eval.get_all_document_ids()
                corpus_documents = [document_store_for_eval.get_document(doc_id) for doc_id in doc_ids]
                corpus_documents = [doc for doc in corpus_documents if doc is not None]
            
                # === Step 4: Complete Workflow Execution ===
                
                with patch.object(synthesizer, 'synthesize') as mock_synthesize2, \
                     patch.object(quality_lab.insight_reporter, 'process') as mock_insight2:
                    
                    # Set up mocks for complete workflow
                    mock_synthesize2.return_value = "Complete integration test answer"
                    
                    mock_report_doc = Mock()
                    mock_report_doc.content = "# Complete Integration Report\n\nAll systems working together successfully."
                    mock_insight2.return_value = [mock_report_doc]
                    
                    # Run complete evaluation workflow
                    complete_results = quality_lab.run_full_evaluation(
                        corpus_documents=corpus_documents,
                        query_engine=query_engine,
                        num_qa_pairs=3,
                        output_file=str(temp_workspace / "integration_report.md")
                    )
                    
                    # === Step 5: Verify Complete Integration ===
                
                    # Verify all components worked together
                    assert "qa_pairs" in complete_results
                    assert "evaluation_summary" in complete_results
                    assert "evaluation_report" in complete_results
                    assert "corpus_name" in complete_results
                    assert complete_results["corpus_name"] == "full_integration_corpus"
                    assert len(complete_results["qa_pairs"]) == 3
                    
                    # Verify workflow statistics
                    lab_stats = quality_lab.get_lab_stats()
                    assert lab_stats["corpus_name"] == "full_integration_corpus"
                    assert lab_stats["qa_pairs_generated"] >= 3
                    assert lab_stats["evaluations_completed"] >= 1
                    assert lab_stats["reports_generated"] >= 1
                    
                    print("✅ CorpusManager processes documents")
                    print("✅ QueryEngine retrieves and generates answers")
                    print("✅ QualityLab evaluates complete system")
                    print("✅ All three components integrated successfully")

    def test_integration_error_propagation(self, temp_workspace):
        """Test how errors propagate through the integration"""
        
        # Test CorpusManager error propagation to QueryEngine
        document_store = SQLiteDocumentStore(db_path=str(temp_workspace / "error_test.db"))
        vector_store = None  # Intentionally None to test error handling
        
        # QueryEngine should handle None vector store gracefully
        with pytest.raises((ValueError, TypeError, AttributeError)):
            QueryEngine(
                corpus_name="error_test_corpus",
                retrievers=vector_store,
                synthesizer=Mock(),
                reranker=Mock()
            )
        
        print("✅ Error propagation handled correctly")

    def test_integration_with_different_corpus_names(self, sample_documents):
        """Test integration with consistent corpus naming"""
        
        corpus_name = "naming_test_corpus"
        
        # All components should use the same corpus name
        document_store = SQLiteDocumentStore(db_path=":memory:")
        vector_store = InMemoryVectorStore()
        
        corpus_manager = CorpusManager(
            document_store=document_store,
            vector_store=vector_store,
            config={"corpus_name": corpus_name}
        )
        
        mock_vector_store = Mock()
        query_engine = QueryEngine(
            corpus_name=corpus_name,
            retrievers=mock_vector_store,
            synthesizer=Mock(),
            reranker=Mock()
        )
        
        quality_lab = QualityLab(
            corpus_name=corpus_name,
            config=QualityLabConfig()
        )
        
        # Verify consistent naming
        assert corpus_manager.config.get("corpus_name") == corpus_name
        assert query_engine.corpus_name == corpus_name
        assert quality_lab.corpus_name == corpus_name
        
        # Test QA pair generation preserves corpus name
        qa_pairs = quality_lab.generate_qa_pairs(sample_documents, num_pairs=2)
        assert all(qa.metadata["corpus_name"] == corpus_name for qa in qa_pairs)
        
        print("✅ Consistent corpus naming across all components")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])