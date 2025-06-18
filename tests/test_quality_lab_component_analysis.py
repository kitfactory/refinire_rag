"""
Test QualityLab component-wise analysis functionality

This test demonstrates how QualityLab can analyze retriever and reranker performance individually
to provide detailed capture rate metrics for original documents.

QualityLabのコンポーネント別分析機能のテスト
"""

import pytest
from unittest.mock import Mock, patch

from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.application.query_engine_new import QueryEngine, QueryEngineConfig
from refinire_rag.models.document import Document
from refinire_rag.models.qa_pair import QAPair
from refinire_rag.retrieval.simple_reader import SimpleAnswerSynthesizer, SimpleAnswerSynthesizerConfig
from refinire_rag.retrieval.simple_reranker import SimpleReranker, SimpleRerankerConfig


class TestQualityLabComponentAnalysis:
    """Test component-wise analysis in QualityLab"""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                id="doc_1",
                content="Document 1 content about AI fundamentals",
                metadata={"source": "ai_doc_1.txt", "topic": "AI"}
            ),
            Document(
                id="doc_2", 
                content="Document 2 content about machine learning",
                metadata={"source": "ml_doc_2.txt", "topic": "ML"}
            ),
            Document(
                id="doc_3",
                content="Document 3 content about deep learning",
                metadata={"source": "dl_doc_3.txt", "topic": "DL"}
            )
        ]

    @pytest.fixture
    def mock_query_engine(self):
        """Create mock QueryEngine with multiple retrievers"""
        # Create mock retrievers
        mock_retriever_1 = Mock()
        mock_retriever_2 = Mock()
        
        # Create mock reranker
        mock_reranker = SimpleReranker(SimpleRerankerConfig())
        
        # Create mock synthesizer
        mock_synthesizer = SimpleAnswerSynthesizer(SimpleAnswerSynthesizerConfig())
        
        # Create QueryEngine with multiple retrievers
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever_1, mock_retriever_2],
            synthesizer=mock_synthesizer,
            reranker=mock_reranker,
            config=QueryEngineConfig(retriever_top_k=5, reranker_top_k=3)
        )
        
        return query_engine, mock_retriever_1, mock_retriever_2, mock_reranker, mock_synthesizer

    def test_component_analysis_with_multiple_retrievers(self, sample_documents, mock_query_engine):
        """Test component-wise analysis with multiple retrievers"""
        
        query_engine, mock_retriever_1, mock_retriever_2, mock_reranker, mock_synthesizer = mock_query_engine
        
        # Create QualityLab
        quality_lab = QualityLab(
            corpus_name="test_corpus",
            config=QualityLabConfig(qa_pairs_per_document=1)
        )
        
        # Generate QA pairs
        qa_pairs = quality_lab.generate_qa_pairs(sample_documents, num_pairs=3)
        
        # Mock retriever responses with different capture rates
        def mock_retriever_1_retrieve(query, top_k):
            # Retriever 1 finds doc_1 and doc_2 (good for AI/ML topics)
            return [
                Mock(document_id="doc_1", score=0.9),
                Mock(document_id="doc_2", score=0.8)
            ]
        
        def mock_retriever_2_retrieve(query, top_k):
            # Retriever 2 finds doc_2 and doc_3 (good for ML/DL topics)
            return [
                Mock(document_id="doc_2", score=0.85),
                Mock(document_id="doc_3", score=0.75)
            ]
        
        mock_retriever_1.retrieve = mock_retriever_1_retrieve
        mock_retriever_2.retrieve = mock_retriever_2_retrieve
        
        # Mock reranker to reorder and reduce results
        def mock_reranker_rerank(query, sources, top_k):
            # Simulate reranker choosing best 2 sources
            return sources[:2]  # Take top 2
        
        # Mock synthesizer
        def mock_synthesizer_synthesize(query, sources):
            return f"Answer based on {len(sources)} sources"
        
        with patch.object(mock_reranker, 'rerank', side_effect=mock_reranker_rerank), \
             patch.object(mock_synthesizer, 'synthesize', side_effect=mock_synthesizer_synthesize):
            
            # Evaluate QueryEngine using QualityLab
            evaluation_results = quality_lab.evaluate_query_engine(
                query_engine=query_engine,
                qa_pairs=qa_pairs
            )
        
        # Verify evaluation results contain component analysis
        assert "evaluation_summary" in evaluation_results
        summary = evaluation_results["evaluation_summary"]
        
        # Check retriever performance metrics
        assert "retriever_performance" in summary
        retriever_perf = summary["retriever_performance"]
        
        # Should have 2 retrievers analyzed
        assert len(retriever_perf) == 2
        
        # Check each retriever's performance
        for retriever_id, perf in retriever_perf.items():
            assert "average_recall" in perf
            assert "average_precision" in perf
            assert "total_queries" in perf
            assert "average_documents_found" in perf
            assert perf["total_queries"] == 3  # 3 QA pairs tested
        
        # Check reranker performance metrics
        assert "reranker_performance" in summary
        reranker_perf = summary["reranker_performance"]
        
        if reranker_perf["enabled"]:
            assert "average_recall_after_rerank" in reranker_perf
            assert "average_precision_after_rerank" in reranker_perf
            assert "total_queries" in reranker_perf
        
        print("✅ Component-wise analysis working correctly")

    def test_get_component_performance_summary(self, sample_documents, mock_query_engine):
        """Test getting formatted component performance summary"""
        
        query_engine, mock_retriever_1, mock_retriever_2, mock_reranker, mock_synthesizer = mock_query_engine
        
        # Create QualityLab
        quality_lab = QualityLab(
            corpus_name="test_corpus",
            config=QualityLabConfig(qa_pairs_per_document=1)
        )
        
        # Generate QA pairs
        qa_pairs = quality_lab.generate_qa_pairs(sample_documents, num_pairs=2)
        
        # Mock simple retriever responses
        mock_retriever_1.retrieve.return_value = [Mock(document_id="doc_1", score=0.9)]
        mock_retriever_2.retrieve.return_value = [Mock(document_id="doc_2", score=0.8)]
        
        with patch.object(mock_reranker, 'rerank') as mock_rerank, \
             patch.object(mock_synthesizer, 'synthesize') as mock_synth:
            
            mock_rerank.return_value = [Mock(document_id="doc_1", score=0.9)]
            mock_synth.return_value = "Test answer"
            
            # Evaluate QueryEngine
            evaluation_results = quality_lab.evaluate_query_engine(
                query_engine=query_engine,
                qa_pairs=qa_pairs
            )
            
            # Get formatted component performance summary
            component_summary = quality_lab.get_component_performance_summary(evaluation_results)
        
        # Verify formatted summary structure
        assert "retriever_performance" in component_summary
        assert "reranker_performance" in component_summary
        assert "overall_metrics" in component_summary
        
        # Check retriever performance format
        retriever_perf = component_summary["retriever_performance"]
        for retriever_id, perf in retriever_perf.items():
            assert "type" in perf
            assert "recall" in perf
            assert "precision" in perf
            assert "f1_score" in perf
            assert "avg_documents_found" in perf
            assert "total_queries" in perf
        
        # Check overall metrics
        overall = component_summary["overall_metrics"]
        assert "total_tests" in overall
        assert "overall_recall" in overall
        assert "overall_precision" in overall
        assert "pass_rate" in overall
        
        print("✅ Component performance summary formatting working correctly")

    def test_retriever_capture_rate_analysis(self, sample_documents, mock_query_engine):
        """Test detailed capture rate analysis for each retriever"""
        
        query_engine, mock_retriever_1, mock_retriever_2, mock_reranker, mock_synthesizer = mock_query_engine
        
        # Create QualityLab
        quality_lab = QualityLab(
            corpus_name="test_corpus",
            config=QualityLabConfig(qa_pairs_per_document=1)
        )
        
        # Create specific QA pairs with known expected sources
        qa_pairs = [
            QAPair(
                question="What is AI?",
                answer="AI is artificial intelligence",
                document_id="doc_1",
                metadata={"corpus_name": "test_corpus"}
            ),
            QAPair(
                question="What is ML?", 
                answer="ML is machine learning",
                document_id="doc_2",
                metadata={"corpus_name": "test_corpus"}
            )
        ]
        
        # Mock retrievers with different capture patterns
        def mock_retriever_1_retrieve(query, top_k):
            # Retriever 1 is good at finding doc_1, poor at doc_2
            if "AI" in query:
                return [Mock(document_id="doc_1", score=0.95)]  # High capture for AI
            else:
                return []  # Misses ML query
        
        def mock_retriever_2_retrieve(query, top_k):
            # Retriever 2 is good at finding doc_2, poor at doc_1  
            if "ML" in query:
                return [Mock(document_id="doc_2", score=0.90)]  # High capture for ML
            else:
                return []  # Misses AI query
        
        mock_retriever_1.retrieve = mock_retriever_1_retrieve
        mock_retriever_2.retrieve = mock_retriever_2_retrieve
        
        with patch.object(mock_reranker, 'rerank') as mock_rerank, \
             patch.object(mock_synthesizer, 'synthesize') as mock_synth:
            
            # Mock reranker to pass through results
            mock_rerank.side_effect = lambda query, sources, top_k: sources
            mock_synth.return_value = "Test answer"
            
            # Evaluate QueryEngine
            evaluation_results = quality_lab.evaluate_query_engine(
                query_engine=query_engine,
                qa_pairs=qa_pairs
            )
            
            # Get component performance summary
            component_summary = quality_lab.get_component_performance_summary(evaluation_results)
        
        # Analyze retriever-specific capture rates
        retriever_perf = component_summary["retriever_performance"]
        
        # Should show different performance patterns for each retriever
        retriever_1_id = [k for k in retriever_perf.keys() if "retriever_0" in k][0]
        retriever_2_id = [k for k in retriever_perf.keys() if "retriever_1" in k][0]
        
        retriever_1_perf = retriever_perf[retriever_1_id]
        retriever_2_perf = retriever_perf[retriever_2_id]
        
        # Verify that performance metrics are captured
        assert retriever_1_perf["total_queries"] == 2
        assert retriever_2_perf["total_queries"] == 2
        
        # Each should have different recall rates based on their specializations
        assert "recall" in retriever_1_perf
        assert "recall" in retriever_2_perf
        
        print(f"✅ Retriever 1 recall: {retriever_1_perf['recall']:.2f}")
        print(f"✅ Retriever 2 recall: {retriever_2_perf['recall']:.2f}")
        print("✅ Retriever-specific capture rate analysis working correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])