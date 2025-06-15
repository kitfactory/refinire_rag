"""
Comprehensive test for QueryEngine

Tests the complete query processing workflow:
1. Setup QueryEngine with all components
2. Test document retrieval and answer generation
3. Test query normalization
4. Test different component configurations
5. Test error handling and edge cases
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List

from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval.simple_reranker import SimpleReranker, SimpleRerankerConfig
from refinire_rag.retrieval.simple_reader import SimpleAnswerSynthesizer, SimpleAnswerSynthesizerConfig
from refinire_rag.models.document import Document
from refinire_rag.retrieval.base import SearchResult


class MockRetriever:
    """Mock retriever for testing"""
    
    def __init__(self, document_store, vector_store):
        self.document_store = document_store
        self.vector_store = vector_store
        self.processing_stats = {
            "queries_processed": 0,
            "processing_time": 0.0,
            "errors_encountered": 0
        }
    
    def retrieve(self, query: str, limit=10, metadata_filter=None) -> List[SearchResult]:
        """Mock retrieval that returns relevant documents"""
        self.processing_stats["queries_processed"] += 1
        
        # Simple mock: return documents based on query keywords
        if "machine learning" in query.lower():
            doc1 = Document(
                id="ml_doc_1",
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                metadata={"source": "ml_guide", "category": "AI"}
            )
            doc2 = Document(
                id="ml_doc_2", 
                content="Neural networks are a key component of deep learning in machine learning.",
                metadata={"source": "ml_advanced", "category": "AI"}
            )
            
            return [
                SearchResult(document_id="ml_doc_1", document=doc1, score=0.9, 
                           metadata={"retrieval_method": "mock"}),
                SearchResult(document_id="ml_doc_2", document=doc2, score=0.8,
                           metadata={"retrieval_method": "mock"})
            ]
        
        elif "data processing" in query.lower():
            doc3 = Document(
                id="data_doc_1",
                content="Data processing involves cleaning, transforming, and analyzing data.",
                metadata={"source": "data_guide", "category": "Data"}
            )
            
            return [
                SearchResult(document_id="data_doc_1", document=doc3, score=0.85,
                           metadata={"retrieval_method": "mock"})
            ]
        
        else:
            # Return empty for unknown queries
            return []
    
    def get_processing_stats(self):
        return self.processing_stats.copy()


class TestQueryEngineComprehensive:
    """Comprehensive test for QueryEngine workflow"""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for the test"""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            
            # Create subdirectories
            refinire_dir = workspace / "refinire"
            refinire_dir.mkdir()
            
            yield {
                "workspace": workspace,
                "refinire_dir": refinire_dir,
                "rag_dir": refinire_dir / "rag",
                "db_path": workspace / "test_corpus.db"
            }

    @pytest.fixture
    def sample_documents(self, temp_workspace):
        """Create sample documents and corpus state"""
        docs_dir = temp_workspace["workspace"] / "documents"
        docs_dir.mkdir()
        
        # Create sample documents
        sample_docs = {
            "ml_basics.md": """# Machine Learning Basics

Machine learning (ML) is a subset of artificial intelligence that enables computers to learn and improve from experience. Neural networks are a key component of deep learning algorithms.

## Key Concepts
- **Supervised Learning**: Training with labeled data
- **Unsupervised Learning**: Finding patterns in unlabeled data  
- **Deep Learning**: Using neural networks with multiple layers
- **Model Training**: Process of teaching algorithms to make predictions

The training process involves feeding data to algorithms and adjusting parameters to minimize prediction errors.""",

            "data_processing.md": """# Data Processing Pipeline

Data processing involves several stages of cleaning, transforming, and preparing data for analysis.

## Processing Steps
- **Data Collection**: Gathering raw data from various sources
- **Data Cleaning**: Removing errors, duplicates, and inconsistencies
- **Data Transformation**: Converting data into suitable formats
- **Feature Engineering**: Creating new features from existing data

Proper data preprocessing is critical for successful machine learning projects."""
        }
        
        # Write documents to files
        for filename, content in sample_docs.items():
            (docs_dir / filename).write_text(content, encoding='utf-8')
        
        return {
            "docs_dir": docs_dir,
            "file_list": list(sample_docs.keys()),
            "total_files": len(sample_docs)
        }

    @pytest.fixture
    def query_engine_components(self, temp_workspace):
        """Create QueryEngine components"""
        db_path = temp_workspace["db_path"]
        document_store = SQLiteDocumentStore(str(db_path))
        vector_store = InMemoryVectorStore()
        
        # Use mock retriever instead of real one for testing
        retriever = MockRetriever(document_store, vector_store)
        
        # Real reranker
        reranker = SimpleReranker(SimpleRerankerConfig(
            top_k=3,
            boost_exact_matches=True
        ))
        
        # Real answer synthesizer (but we'll mock the LLM calls)
        synthesizer = SimpleAnswerSynthesizer(SimpleAnswerSynthesizerConfig(
            max_context_length=1000,
            temperature=0.1
        ))
        
        return {
            "document_store": document_store,
            "vector_store": vector_store,
            "retriever": retriever,
            "reranker": reranker,
            "synthesizer": synthesizer
        }

    def test_comprehensive_query_workflow(self, temp_workspace, sample_documents, query_engine_components):
        """Test complete QueryEngine workflow"""
        refinire_dir = temp_workspace["refinire_dir"]
        
        # Set environment variable for test
        with patch.dict(os.environ, {"REFINIRE_DIR": str(refinire_dir)}):
            
            # Step 1: Initialize QueryEngine
            print("\n=== Step 1: Initialize QueryEngine ===")
            
            config = QueryEngineConfig(
                enable_query_normalization=True,
                auto_detect_corpus_state=True,
                retriever_top_k=5,
                reranker_top_k=3,
                include_sources=True,
                include_confidence=True
            )
            
            query_engine = QueryEngine(
                document_store=query_engine_components["document_store"],
                vector_store=query_engine_components["vector_store"],
                retriever=query_engine_components["retriever"],
                reranker=query_engine_components["reranker"],
                synthesizer=query_engine_components["synthesizer"],
                config=config
            )
            
            print("✓ QueryEngine initialized successfully")
            assert query_engine.config.enable_query_normalization is True
            assert query_engine.corpus_state is not None
            
            # Step 2: Test basic query answering with LLM mocking
            print("\n=== Step 2: Test Basic Query Answering ===")
            
            # Ensure synthesizer has _llm_pipeline attribute or mock it
            synthesizer = query_engine_components["synthesizer"]
            if not hasattr(synthesizer, '_llm_pipeline'):
                synthesizer._llm_pipeline = Mock()
            
            with patch.object(synthesizer, '_llm_pipeline') as mock_llm:
                # Mock LLM response
                mock_result = Mock()
                mock_result.content = """Based on the provided context, machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience. It involves training algorithms with data to make predictions or decisions without being explicitly programmed for every scenario.

Key aspects of machine learning include:
- Supervised learning with labeled data
- Neural networks for deep learning
- Model training to minimize prediction errors

The field encompasses various techniques for processing and learning from data."""
                
                mock_llm.run.return_value = mock_result
                
                # Test machine learning query
                result = query_engine.answer("What is machine learning?")
                
                print(f"✓ Query processed successfully")
                print(f"  - Answer length: {len(result.answer)} characters")
                print(f"  - Sources found: {len(result.sources)}")
                print(f"  - Confidence: {result.confidence:.2f}")
                print(f"  - Processing time: {result.metadata.get('processing_time', 0):.3f}s")
                
                # Verify result structure
                assert result.query == "What is machine learning?"
                assert len(result.answer) > 50  # Meaningful answer
                assert len(result.sources) > 0  # Found relevant sources
                assert result.confidence > 0  # Has confidence score
                assert "processing_time" in result.metadata
                assert "source_count" in result.metadata
                
                # Verify sources are relevant
                for source in result.sources:
                    assert source.document is not None
                    assert source.score > 0
                    assert "machine learning" in source.document.content.lower()
            
            # Step 3: Test different query types
            print("\n=== Step 3: Test Different Query Types ===")
            
            synthesizer = query_engine_components["synthesizer"]
            if not hasattr(synthesizer, '_llm_pipeline'):
                synthesizer._llm_pipeline = Mock()
            
            with patch.object(synthesizer, '_llm_pipeline') as mock_llm:
                mock_result = Mock()
                mock_result.content = "Data processing involves cleaning, transforming, and analyzing data through multiple stages including collection, cleaning, transformation, and feature engineering."
                mock_llm.run.return_value = mock_result
                
                # Test data processing query
                result2 = query_engine.answer("How does data processing work?")
                
                print(f"✓ Data processing query answered")
                print(f"  - Sources found: {len(result2.sources)}")
                
                assert result2.query == "How does data processing work?"
                assert len(result2.answer) > 20
                
                # Test query with no relevant documents
                result3 = query_engine.answer("What is quantum computing?")
                
                print(f"✓ Irrelevant query handled")
                print(f"  - Sources found: {len(result3.sources)}")
                
                assert result3.query == "What is quantum computing?"
                # Should have no sources since mock retriever doesn't handle this query
                assert len(result3.sources) == 0
            
            # Step 4: Test reranking functionality
            print("\n=== Step 4: Test Reranking Functionality ===")
            
            synthesizer = query_engine_components["synthesizer"]
            if not hasattr(synthesizer, '_llm_pipeline'):
                synthesizer._llm_pipeline = Mock()
            
            with patch.object(synthesizer, '_llm_pipeline') as mock_llm:
                mock_result = Mock()
                mock_result.content = "Machine learning and neural networks are closely related fields in artificial intelligence."
                mock_llm.run.return_value = mock_result
                
                # Test query that should trigger reranking
                result4 = query_engine.answer("Tell me about machine learning and neural networks")
                
                print(f"✓ Reranking test completed")
                print(f"  - Sources after reranking: {len(result4.sources)}")
                
                assert len(result4.sources) > 0
                # Check that results are limited by reranker top_k (3)
                assert len(result4.sources) <= 3
                
                # Verify reranking metadata
                for source in result4.sources:
                    if "reranked_by" in source.metadata:
                        assert source.metadata["reranked_by"] == "SimpleReranker"
                        assert "original_score" in source.metadata
            
            # Step 5: Test statistics and monitoring
            print("\n=== Step 5: Test Statistics and Monitoring ===")
            
            stats = query_engine.get_engine_stats()
            
            print(f"✓ Statistics retrieved:")
            print(f"  - Queries processed: {stats['queries_processed']}")
            print(f"  - Total processing time: {stats['total_processing_time']:.3f}s")
            print(f"  - Average response time: {stats['average_response_time']:.3f}s")
            
            # Verify statistics structure
            assert "queries_processed" in stats
            assert "total_processing_time" in stats
            assert "retriever_stats" in stats
            assert "synthesizer_stats" in stats
            assert "reranker_stats" in stats
            assert "corpus_state" in stats
            assert "config" in stats
            
            # Should have processed multiple queries
            assert stats["queries_processed"] >= 4
            assert stats["total_processing_time"] > 0
            
            print(f"\n🎉 Comprehensive QueryEngine workflow completed successfully!")
            print(f"📊 Final Statistics:")
            print(f"   - Total queries processed: {stats['queries_processed']}")
            print(f"   - Average retrieval count: {stats['average_retrieval_count']:.1f}")
            print(f"   - Configuration verified: ✓")

    def test_error_handling_and_edge_cases(self, temp_workspace, query_engine_components):
        """Test error handling and edge cases"""
        refinire_dir = temp_workspace["refinire_dir"]
        
        with patch.dict(os.environ, {"REFINIRE_DIR": str(refinire_dir)}):
            
            query_engine = QueryEngine(
                document_store=query_engine_components["document_store"],
                vector_store=query_engine_components["vector_store"],
                retriever=query_engine_components["retriever"],
                reranker=query_engine_components["reranker"],
                synthesizer=query_engine_components["synthesizer"]
            )
            
            # Test empty query
            result = query_engine.answer("")
            assert result.query == ""
            assert len(result.answer) > 0  # Should handle gracefully
            
            # Test very long query
            long_query = "What is machine learning? " * 100
            result = query_engine.answer(long_query)
            assert result.query == long_query
            
            # Test query with special characters
            special_query = "What is ML? @#$%^&*()"
            result = query_engine.answer(special_query)
            assert result.query == special_query

    def test_configuration_variations(self, temp_workspace, query_engine_components):
        """Test different configuration options"""
        refinire_dir = temp_workspace["refinire_dir"]
        
        with patch.dict(os.environ, {"REFINIRE_DIR": str(refinire_dir)}):
            
            # Test with normalization disabled
            config_no_norm = QueryEngineConfig(
                enable_query_normalization=False,
                auto_detect_corpus_state=False,
                include_sources=False,
                include_confidence=False
            )
            
            query_engine_no_norm = QueryEngine(
                document_store=query_engine_components["document_store"],
                vector_store=query_engine_components["vector_store"],
                retriever=query_engine_components["retriever"],
                reranker=query_engine_components["reranker"],
                synthesizer=query_engine_components["synthesizer"],
                config=config_no_norm
            )
            
            synthesizer = query_engine_components["synthesizer"]
            if not hasattr(synthesizer, '_llm_pipeline'):
                synthesizer._llm_pipeline = Mock()
            
            with patch.object(synthesizer, '_llm_pipeline') as mock_llm:
                mock_result = Mock()
                mock_result.content = "Test answer without normalization."
                mock_llm.run.return_value = mock_result
                
                result = query_engine_no_norm.answer("What is machine learning?")
                
                # Should work without normalization
                assert result.query == "What is machine learning?"
                assert len(result.answer) > 0
                assert len(result.sources) == 0  # Sources disabled
                assert result.confidence == 0.0  # Confidence disabled

    def test_component_integration(self, temp_workspace, query_engine_components):
        """Test integration between different components"""
        refinire_dir = temp_workspace["refinire_dir"]
        
        with patch.dict(os.environ, {"REFINIRE_DIR": str(refinire_dir)}):
            
            query_engine = QueryEngine(
                document_store=query_engine_components["document_store"],
                vector_store=query_engine_components["vector_store"],
                retriever=query_engine_components["retriever"],
                reranker=query_engine_components["reranker"],
                synthesizer=query_engine_components["synthesizer"]
            )
            
            # Test without reranker
            query_engine_no_rerank = QueryEngine(
                document_store=query_engine_components["document_store"],
                vector_store=query_engine_components["vector_store"],
                retriever=query_engine_components["retriever"],
                reranker=None,  # No reranker
                synthesizer=query_engine_components["synthesizer"]
            )
            
            synthesizer = query_engine_components["synthesizer"]
            if not hasattr(synthesizer, '_llm_pipeline'):
                synthesizer._llm_pipeline = Mock()
            
            with patch.object(synthesizer, '_llm_pipeline') as mock_llm:
                mock_result = Mock()
                mock_result.content = "Answer without reranking."
                mock_llm.run.return_value = mock_result
                
                result = query_engine_no_rerank.answer("What is machine learning?")
                
                assert result.query == "What is machine learning?"
                assert len(result.answer) > 0
                
                # Check metadata indicates no reranker used
                assert result.metadata.get("reranker_used") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])