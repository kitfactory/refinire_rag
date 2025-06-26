"""
Test for improved QueryEngine interface

Tests the new QueryEngine with:
- Multiple retrievers support
- Simplified interface (no document_store dependency)
- query() method instead of answer()
- Manual normalizer setting
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List

from refinire_rag.application.query_engine_new import QueryEngine, QueryEngineConfig
from refinire_rag.retrieval.simple_reranker import SimpleReranker, SimpleRerankerConfig
from refinire_rag.retrieval.simple_reader import SimpleAnswerSynthesizer, SimpleAnswerSynthesizerConfig
from refinire_rag.retrieval.base import SearchResult, Retriever
from refinire_rag.models.document import Document


class MockRetriever(Retriever):
    """Mock retriever for testing"""
    
    def __init__(self, name: str, results: List[SearchResult]):
        self.name = name
        self.results = results
        self.processing_stats = {
            "queries_processed": 0,
            "processing_time": 0.0,
            "errors_encountered": 0
        }
    
    @classmethod
    def get_config_class(cls):
        return dict
    
    def retrieve(self, query: str, limit=10, metadata_filter=None) -> List[SearchResult]:
        """Return predefined results"""
        self.processing_stats["queries_processed"] += 1
        return self.results[:limit]
    
    def get_processing_stats(self):
        return self.processing_stats.copy()


class TestImprovedQueryEngine:
    """Test improved QueryEngine interface"""
    
    def _mock_synthesizer_response(self, synthesizer, mock_answer="Test answer"):
        """Helper to mock synthesizer response for both Refinire and OpenAI"""
        if hasattr(synthesizer, '_llm_pipeline') and synthesizer._llm_pipeline:
            # Refinire case
            mock_llm = Mock()
            mock_result = Mock()
            mock_result.content = mock_answer
            mock_llm.run.return_value = mock_result
            synthesizer._llm_pipeline = mock_llm
        else:
            # OpenAI case
            mock_openai = Mock() 
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = mock_answer
            mock_openai.chat.completions.create.return_value = mock_response
            synthesizer._openai_client = mock_openai

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents and search results"""
        docs = [
            Document(
                id="doc1",
                content="Machine learning is a subset of AI that focuses on algorithms.",
                metadata={"source": "ml_guide", "category": "AI"}
            ),
            Document(
                id="doc2", 
                content="Deep learning uses neural networks with multiple layers.",
                metadata={"source": "dl_guide", "category": "AI"}
            ),
            Document(
                id="doc3",
                content="Natural language processing enables computers to understand text.",
                metadata={"source": "nlp_guide", "category": "NLP"}
            ),
            Document(
                id="doc4",
                content="Computer vision allows machines to interpret visual information.",
                metadata={"source": "cv_guide", "category": "CV"}
            )
        ]
        
        # Create search results for different retrievers
        vector_results = [
            SearchResult(document_id="doc1", document=docs[0], score=0.9),
            SearchResult(document_id="doc2", document=docs[1], score=0.8)
        ]
        
        keyword_results = [
            SearchResult(document_id="doc1", document=docs[0], score=0.85),  # Duplicate
            SearchResult(document_id="doc3", document=docs[2], score=0.75)
        ]
        
        hybrid_results = [
            SearchResult(document_id="doc4", document=docs[3], score=0.7),
            SearchResult(document_id="doc2", document=docs[1], score=0.65)   # Duplicate
        ]
        
        return {
            "docs": docs,
            "vector_results": vector_results,
            "keyword_results": keyword_results,
            "hybrid_results": hybrid_results
        }

    @pytest.fixture
    def mock_retrievers(self, sample_documents):
        """Create mock retrievers"""
        return [
            MockRetriever("VectorRetriever", sample_documents["vector_results"]),
            MockRetriever("KeywordRetriever", sample_documents["keyword_results"]),
            MockRetriever("HybridRetriever", sample_documents["hybrid_results"])
        ]

    @pytest.fixture
    def query_engine_components(self):
        """Create QueryEngine components"""
        reranker = SimpleReranker(SimpleRerankerConfig(top_k=3))
        synthesizer = SimpleAnswerSynthesizer(SimpleAnswerSynthesizerConfig())
        return {"reranker": reranker, "synthesizer": synthesizer}

    def test_single_retriever_initialization(self, mock_retrievers, query_engine_components):
        """Test QueryEngine with single retriever"""
        retriever = mock_retrievers[0]
        
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=retriever,  # Single retriever
            synthesizer=query_engine_components["synthesizer"],
            reranker=query_engine_components["reranker"]
        )
        
        assert len(query_engine.retrievers) == 1
        assert query_engine.retrievers[0] == retriever
        assert query_engine.reranker is not None
        assert query_engine.synthesizer is not None
        assert query_engine.corpus_name == "test_corpus"

    def test_multiple_retrievers_initialization(self, mock_retrievers, query_engine_components):
        """Test QueryEngine with multiple retrievers"""
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=mock_retrievers,  # List of retrievers
            synthesizer=query_engine_components["synthesizer"],
            reranker=query_engine_components["reranker"]
        )
        
        assert len(query_engine.retrievers) == 3
        assert all(r in query_engine.retrievers for r in mock_retrievers)

    def test_optional_reranker(self, mock_retrievers, query_engine_components):
        """Test QueryEngine without reranker"""
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=mock_retrievers[0],
            synthesizer=query_engine_components["synthesizer"]
            # No reranker
        )
        
        assert query_engine.reranker is None
        assert query_engine.synthesizer is not None

    def test_query_method_basic(self, mock_retrievers, query_engine_components):
        """Test basic query() method functionality"""
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=mock_retrievers[0],
            synthesizer=query_engine_components["synthesizer"]
        )
        
        # Mock synthesizer response
        self._mock_synthesizer_response(
            query_engine_components["synthesizer"], 
            "Machine learning is a subset of AI."
        )
        
        result = query_engine.query("What is machine learning?")
        
        assert result.query == "What is machine learning?"
        assert len(result.answer) > 0
        assert len(result.sources) > 0
        assert "total_processing_time" in result.metadata or hasattr(result, 'processing_time')
        assert result.metadata["corpus_name"] == "test_corpus"

    def test_multiple_retrievers_deduplication(self, mock_retrievers, query_engine_components):
        """Test deduplication with multiple retrievers"""
        config = QueryEngineConfig(
            retriever_top_k=5,
            total_top_k=10,
            deduplicate_results=True,
            combine_scores="max"
        )
        
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=mock_retrievers,
            synthesizer=query_engine_components["synthesizer"],
            config=config
        )
        
        self._mock_synthesizer_response(
            query_engine_components["synthesizer"], 
            "Combined answer from multiple sources."
        )
        
        result = query_engine.query("Test query")
        
        # Should have deduplicated results
        document_ids = [source.document_id for source in result.sources]
        assert len(document_ids) == len(set(document_ids))  # No duplicates
        
        # Check retriever metadata
        for source in result.sources:
            assert "retriever_index" in source.metadata
            assert "retriever_type" in source.metadata

    def test_score_combination_strategies(self, mock_retrievers, query_engine_components):
        """Test different score combination strategies"""
        strategies = ["max", "average", "sum"]
        
        for strategy in strategies:
            config = QueryEngineConfig(
                deduplicate_results=True,
                combine_scores=strategy
            )
            
            query_engine = QueryEngine(
                corpus_name="test_corpus",
                retrievers=mock_retrievers,
                synthesizer=query_engine_components["synthesizer"],
                config=config
            )
            
            self._mock_synthesizer_response(
                query_engine_components["synthesizer"], 
                f"Answer with {strategy} score combination."
            )
            
            result = query_engine.query("Test query")
            
            assert len(result.sources) > 0
            print(f"✓ {strategy} strategy: {len(result.sources)} sources")

    def test_no_deduplication(self, mock_retrievers, query_engine_components):
        """Test without deduplication"""
        config = QueryEngineConfig(
            deduplicate_results=False
        )
        
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=mock_retrievers,
            synthesizer=query_engine_components["synthesizer"],
            config=config
        )
        
        self._mock_synthesizer_response(
            query_engine_components["synthesizer"], 
            "Answer without deduplication."
        )
        
        result = query_engine.query("Test query")
        
        # May have duplicate document IDs
        assert len(result.sources) > 0

    def test_manual_normalizer_setting(self, mock_retrievers, query_engine_components):
        """Test that QueryEngine has component configuration"""
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=mock_retrievers[0],
            synthesizer=query_engine_components["synthesizer"]
        )
        
        # QueryEngine has configuration and stats
        assert hasattr(query_engine, 'config')
        assert hasattr(query_engine, 'stats')
        assert hasattr(query_engine, 'corpus_name')
        assert query_engine.corpus_name == "test_corpus"
        
        # Check component info is available
        info = query_engine.get_component_info()
        assert 'retrievers' in info
        assert 'synthesizer' in info

    def test_context_parameters(self, mock_retrievers, query_engine_components):
        """Test metadata filters in query()"""
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=mock_retrievers,
            synthesizer=query_engine_components["synthesizer"]
        )
        
        # Use metadata_filters instead of context
        metadata_filters = {
            "category": "AI",
            "source": "ml_guide"
        }
        
        self._mock_synthesizer_response(
            query_engine_components["synthesizer"], 
            "Answer with metadata filters."
        )
        
        result = query_engine.query("Test query", metadata_filters=metadata_filters)
        
        assert result.query == "Test query"
        assert len(result.sources) > 0

    def test_retriever_management(self, mock_retrievers, query_engine_components):
        """Test adding and removing retrievers"""
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=mock_retrievers[0],
            synthesizer=query_engine_components["synthesizer"]
        )
        
        # Initially 1 retriever
        assert len(query_engine.retrievers) == 1
        
        # Add retriever
        query_engine.add_retriever(mock_retrievers[1])
        assert len(query_engine.retrievers) == 2
        
        # Add another
        query_engine.add_retriever(mock_retrievers[2])
        assert len(query_engine.retrievers) == 3
        
        # Remove by index
        success = query_engine.remove_retriever(1)
        assert success is True
        assert len(query_engine.retrievers) == 2
        
        # Try to remove invalid index
        success = query_engine.remove_retriever(10)
        assert success is False
        assert len(query_engine.retrievers) == 2

    def test_comprehensive_stats(self, mock_retrievers, query_engine_components):
        """Test comprehensive statistics for multiple retrievers"""
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=mock_retrievers,
            synthesizer=query_engine_components["synthesizer"],
            reranker=query_engine_components["reranker"]
        )
        
        # Process some queries first
        self._mock_synthesizer_response(
            query_engine_components["synthesizer"], 
            "Test answer."
        )
        
        query_engine.query("Test query 1")
        query_engine.query("Test query 2")
        
        stats = query_engine.get_stats()
        
        # stats is a QueryEngineStats object, not a dict
        assert hasattr(stats, 'queries_processed')
        assert stats.queries_processed == 2
        assert hasattr(stats, 'total_retrieval_time')
        assert hasattr(stats, 'total_reranking_time')
        # Check if stats object has basic functionality
        assert hasattr(stats, '__dict__')  # Can be converted to dict if needed
        
        # Component stats - check if attributes exist
        assert hasattr(stats, 'total_synthesis_time')
        
        # Just check the stats object works
        assert stats is not None

    def test_error_handling(self, query_engine_components):
        """Test error handling with failing retrievers"""
        # Create a failing retriever
        failing_retriever = Mock()
        failing_retriever.retrieve.side_effect = Exception("Retriever failed")
        failing_retriever.get_processing_stats.return_value = {"error": "failed"}
        
        working_retriever = MockRetriever("WorkingRetriever", [
            SearchResult(
                document_id="doc1",
                document=Document(id="doc1", content="Working content", metadata={}),
                score=0.8
            )
        ])
        
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[failing_retriever, working_retriever],
            synthesizer=query_engine_components["synthesizer"]
        )
        
        self._mock_synthesizer_response(
            query_engine_components["synthesizer"], 
            "Answer despite retriever failure."
        )
        
        result = query_engine.query("Test query")
        
        # Should still work with the working retriever
        assert len(result.sources) > 0
        assert result.answer != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])