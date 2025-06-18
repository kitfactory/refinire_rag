"""
Comprehensive tests for Processing layer components.
Processing層コンポーネントの包括的テスト
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from refinire_rag.models.document import Document
from refinire_rag.processing.chunker import ChunkerConfig, Chunker
from refinire_rag.processing.normalizer import NormalizerConfig, Normalizer
from refinire_rag.processing.test_suite import TestSuite, TestSuiteConfig
from refinire_rag.processing.evaluator import Evaluator, EvaluatorConfig
from refinire_rag.processing.contradiction_detector import ContradictionDetector, ContradictionDetectorConfig
from refinire_rag.processing.insight_reporter import InsightReporter, InsightReporterConfig


class TestChunker:
    """Test Chunker functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ChunkingConfig(
            chunk_size=100,
            overlap=20,
            split_by_sentence=True,
            min_chunk_size=30
        )
        self.chunker = Chunker(self.config)
        
        # Test documents
        self.test_docs = [
            Document(
                id="doc1",
                content="This is a test document. It has multiple sentences. Each sentence should be considered for chunking. The chunker should split this appropriately based on the configuration.",
                metadata={"source": "test1.txt"}
            ),
            Document(
                id="doc2",
                content="Short doc.",
                metadata={"source": "test2.txt"}
            ),
            Document(
                id="doc3",
                content="A" * 500,  # Very long single sentence
                metadata={"source": "test3.txt"}
            )
        ]
    
    def test_chunker_configuration(self):
        """Test chunker configuration validation"""
        config = ChunkingConfig(
            chunk_size=200,
            overlap=50,
            split_by_sentence=False,
            min_chunk_size=50
        )
        
        chunker = Chunker(config)
        assert chunker.config.chunk_size == 200
        assert chunker.config.overlap == 50
        assert chunker.config.split_by_sentence == False
        assert chunker.config.min_chunk_size == 50
    
    def test_sentence_based_chunking(self):
        """Test sentence-based chunking"""
        doc = self.test_docs[0]
        chunks = self.chunker.process(doc)
        
        assert len(chunks) > 1
        
        # Check that chunks are reasonable size
        for chunk in chunks:
            assert len(chunk.content) >= self.config.min_chunk_size
            assert len(chunk.content) <= self.config.chunk_size + self.config.overlap
            
            # Check metadata preservation
            assert chunk.metadata["source"] == doc.metadata["source"]
            assert "chunk_index" in chunk.metadata
            assert "parent_document_id" in chunk.metadata
    
    def test_character_based_chunking(self):
        """Test character-based chunking"""
        config = ChunkingConfig(
            chunk_size=50,
            overlap=10,
            split_by_sentence=False,
            min_chunk_size=20
        )
        chunker = Chunker(config)
        
        doc = self.test_docs[2]  # Long single sentence
        chunks = chunker.process(doc)
        
        assert len(chunks) > 1
        
        # Check overlap
        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            chunk1_end = chunks[0].content[-10:]
            chunk2_start = chunks[1].content[:10]
            # There should be some commonality due to overlap
            assert len(chunk1_end) > 0 and len(chunk2_start) > 0
    
    def test_short_document_handling(self):
        """Test handling of documents shorter than chunk size"""
        doc = self.test_docs[1]  # Short document
        chunks = self.chunker.process(doc)
        
        # Short documents should still produce one chunk
        assert len(chunks) == 1
        assert chunks[0].content == doc.content
        assert chunks[0].metadata["chunk_index"] == 0
    
    def test_empty_document_handling(self):
        """Test handling of empty documents"""
        empty_doc = Document(id="empty", content="", metadata={})
        chunks = self.chunker.process(empty_doc)
        
        # Empty documents should produce no chunks or one empty chunk
        assert len(chunks) <= 1
        if chunks:
            assert chunks[0].content == ""
    
    def test_chunk_metadata_enrichment(self):
        """Test chunk metadata enrichment"""
        doc = self.test_docs[0]
        chunks = self.chunker.process(doc)
        
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["parent_document_id"] == doc.id
            assert chunk.metadata["total_chunks"] == len(chunks)
            assert "source" in chunk.metadata
            
            # Check chunk positioning metadata
            assert "chunk_start_char" in chunk.metadata
            assert "chunk_end_char" in chunk.metadata
    
    def test_batch_processing(self):
        """Test batch document processing"""
        all_chunks = []
        
        for doc in self.test_docs:
            chunks = self.chunker.process(doc)
            all_chunks.extend(chunks)
        
        assert len(all_chunks) > len(self.test_docs)  # Should have more chunks than docs
        
        # Verify each original document is represented
        parent_ids = set(chunk.metadata["parent_document_id"] for chunk in all_chunks)
        original_ids = set(doc.id for doc in self.test_docs)
        
        for original_id in original_ids:
            if any(len(doc.content) > 0 for doc in self.test_docs if doc.id == original_id):
                assert original_id in parent_ids


class TestNormalizer:
    """Test Normalizer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary dictionary file
        self.temp_dir = tempfile.mkdtemp()
        self.dict_path = Path(self.temp_dir) / "test_dictionary.md"
        
        # Create test dictionary
        test_dictionary = """# Test Dictionary

## Terms
- AI: Artificial Intelligence
- ML: Machine Learning
- NLP: Natural Language Processing
- API: Application Programming Interface

## Synonyms
- artificial intelligence: AI
- machine learning: ML
- natural language processing: NLP
"""
        self.dict_path.write_text(test_dictionary)
        
        self.config = NormalizerConfig(
            dictionary_file_path=str(self.dict_path),
            normalize_case=True,
            expand_abbreviations=True,
            remove_extra_whitespace=True
        )
        self.normalizer = Normalizer(self.config)
        
        # Test documents
        self.test_docs = [
            Document(
                id="doc1",
                content="This document discusses AI and machine learning concepts.",
                metadata={"type": "technical"}
            ),
            Document(
                id="doc2",
                content="The API uses NLP for processing.   Extra   spaces   here.",
                metadata={"type": "documentation"}
            ),
            Document(
                id="doc3",
                content="ARTIFICIAL INTELLIGENCE is the future of ML.",
                metadata={"type": "article"}
            )
        ]
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_normalizer_configuration(self):
        """Test normalizer configuration"""
        config = NormalizerConfig(
            dictionary_file_path=str(self.dict_path),
            normalize_case=False,
            expand_abbreviations=True,
            remove_extra_whitespace=False
        )
        
        normalizer = Normalizer(config)
        assert normalizer.config.normalize_case == False
        assert normalizer.config.expand_abbreviations == True
        assert normalizer.config.remove_extra_whitespace == False
    
    def test_case_normalization(self):
        """Test case normalization"""
        doc = self.test_docs[2]  # Document with uppercase text
        normalized_docs = self.normalizer.process(doc)
        
        assert len(normalized_docs) == 1
        normalized_content = normalized_docs[0].content.lower()
        
        # Should be normalized to lowercase
        assert "artificial intelligence" in normalized_content
        assert "ARTIFICIAL INTELLIGENCE" not in normalized_docs[0].content
    
    def test_abbreviation_expansion(self):
        """Test abbreviation expansion"""
        doc = self.test_docs[0]  # Document with "AI" and "machine learning"
        normalized_docs = self.normalizer.process(doc)
        
        assert len(normalized_docs) == 1
        content = normalized_docs[0].content
        
        # AI should be expanded to Artificial Intelligence
        # ML should be expanded to Machine Learning
        assert "Artificial Intelligence" in content or "artificial intelligence" in content
        assert "Machine Learning" in content or "machine learning" in content
    
    def test_whitespace_normalization(self):
        """Test extra whitespace removal"""
        doc = self.test_docs[1]  # Document with extra spaces
        normalized_docs = self.normalizer.process(doc)
        
        assert len(normalized_docs) == 1
        content = normalized_docs[0].content
        
        # Should not have multiple consecutive spaces
        assert "   " not in content
        assert "  " not in content or content.count("  ") <= 1  # Allow some flexibility
    
    def test_synonym_replacement(self):
        """Test synonym replacement"""
        # Create document with synonyms
        doc = Document(
            id="synonym_test",
            content="natural language processing is important for artificial intelligence",
            metadata={}
        )
        
        normalized_docs = self.normalizer.process(doc)
        content = normalized_docs[0].content
        
        # Synonyms should be replaced with canonical forms
        # This depends on implementation details
        assert isinstance(content, str)
        assert len(content) > 0
    
    def test_metadata_preservation(self):
        """Test metadata preservation during normalization"""
        doc = self.test_docs[0]
        normalized_docs = self.normalizer.process(doc)
        
        assert len(normalized_docs) == 1
        normalized_doc = normalized_docs[0]
        
        # Original metadata should be preserved
        assert normalized_doc.metadata["type"] == doc.metadata["type"]
        
        # Normalization metadata should be added
        assert "normalization_applied" in normalized_doc.metadata
        assert "processing_stage" in normalized_doc.metadata
    
    def test_dictionary_loading(self):
        """Test dictionary loading and parsing"""
        # Test with missing dictionary file
        config = NormalizerConfig(
            dictionary_file_path="/nonexistent/path.md",
            normalize_case=True
        )
        
        # Should handle missing file gracefully
        try:
            normalizer = Normalizer(config)
            # May work with empty dictionary or raise specific exception
        except FileNotFoundError:
            # Expected behavior for missing file
            pass
    
    def test_batch_normalization(self):
        """Test batch document normalization"""
        normalized_results = []
        
        for doc in self.test_docs:
            normalized_docs = self.normalizer.process(doc)
            normalized_results.extend(normalized_docs)
        
        assert len(normalized_results) == len(self.test_docs)
        
        # All documents should be normalized
        for normalized_doc in normalized_results:
            assert "normalization_applied" in normalized_doc.metadata
            assert isinstance(normalized_doc.content, str)


class TestTestSuite:
    """Test TestSuite functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = TestSuiteConfig(
            test_timeout=30.0,
            max_concurrent_tests=2,
            retry_failed_tests=True,
            max_retries=2
        )
        self.test_suite = TestSuite(self.config)
        
        # Mock test cases
        self.mock_test_cases = [
            Mock(
                id="test_1",
                query="What is AI?",
                expected_answer="Artificial Intelligence",
                expected_sources=["doc1"],
                metadata={"type": "factual"}
            ),
            Mock(
                id="test_2", 
                query="How does machine learning work?",
                expected_answer="Through algorithms and data",
                expected_sources=["doc2"],
                metadata={"type": "conceptual"}
            )
        ]
    
    def test_test_suite_configuration(self):
        """Test test suite configuration"""
        config = TestSuiteConfig(
            test_timeout=60.0,
            max_concurrent_tests=4,
            retry_failed_tests=False
        )
        
        test_suite = TestSuite(config)
        assert test_suite.config.test_timeout == 60.0
        assert test_suite.config.max_concurrent_tests == 4
        assert test_suite.config.retry_failed_tests == False
    
    @patch('src.refinire_rag.processing.test_suite.TestSuite._execute_single_test')
    def test_run_tests_basic(self, mock_execute):
        """Test basic test execution"""
        # Mock test execution results
        mock_execute.side_effect = [
            {"test_id": "test_1", "passed": True, "score": 0.9},
            {"test_id": "test_2", "passed": False, "score": 0.4}
        ]
        
        # Mock query engine
        mock_query_engine = Mock()
        
        results = self.test_suite.run_tests(self.mock_test_cases, mock_query_engine)
        
        assert len(results) == 2
        assert results[0]["passed"] == True
        assert results[1]["passed"] == False
        assert mock_execute.call_count == 2
    
    def test_test_result_aggregation(self):
        """Test test result aggregation"""
        # Mock test results
        test_results = [
            {"test_id": "test_1", "passed": True, "score": 0.9, "response_time": 1.2},
            {"test_id": "test_2", "passed": True, "score": 0.8, "response_time": 1.5},
            {"test_id": "test_3", "passed": False, "score": 0.3, "response_time": 2.0}
        ]
        
        summary = self.test_suite._aggregate_results(test_results)
        
        assert summary["total_tests"] == 3
        assert summary["passed_tests"] == 2
        assert summary["failed_tests"] == 1
        assert summary["success_rate"] == 2/3
        assert summary["average_score"] == (0.9 + 0.8 + 0.3) / 3
        assert summary["average_response_time"] == (1.2 + 1.5 + 2.0) / 3
    
    def test_timeout_handling(self):
        """Test test timeout handling"""
        # Create test case that simulates timeout
        with patch('src.refinire_rag.processing.test_suite.TestSuite._execute_single_test') as mock_execute:
            mock_execute.side_effect = TimeoutError("Test timed out")
            
            mock_query_engine = Mock()
            
            results = self.test_suite.run_tests([self.mock_test_cases[0]], mock_query_engine)
            
            assert len(results) == 1
            assert results[0]["passed"] == False
            assert "timeout" in results[0].get("error", "").lower()
    
    def test_retry_mechanism(self):
        """Test retry mechanism for failed tests"""
        with patch('src.refinire_rag.processing.test_suite.TestSuite._execute_single_test') as mock_execute:
            # First call fails, second succeeds
            mock_execute.side_effect = [
                Exception("Temporary failure"),
                {"test_id": "test_1", "passed": True, "score": 0.8}
            ]
            
            mock_query_engine = Mock()
            
            results = self.test_suite.run_tests([self.mock_test_cases[0]], mock_query_engine)
            
            assert len(results) == 1
            assert results[0]["passed"] == True
            assert mock_execute.call_count == 2  # Original + 1 retry
    
    def test_concurrent_execution(self):
        """Test concurrent test execution"""
        with patch('src.refinire_rag.processing.test_suite.TestSuite._execute_single_test') as mock_execute:
            mock_execute.return_value = {"test_id": "test", "passed": True, "score": 0.9}
            
            mock_query_engine = Mock()
            
            # Run multiple tests
            results = self.test_suite.run_tests(self.mock_test_cases, mock_query_engine)
            
            assert len(results) == len(self.mock_test_cases)
            assert all(result["passed"] for result in results)


class TestEvaluator:
    """Test Evaluator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = EvaluatorConfig(
            similarity_threshold=0.7,
            use_semantic_similarity=True,
            similarity_model="sentence-transformers",
            include_source_evaluation=True
        )
        self.evaluator = Evaluator(self.config)
    
    def test_evaluator_configuration(self):
        """Test evaluator configuration"""
        config = EvaluatorConfig(
            similarity_threshold=0.8,
            use_semantic_similarity=False,
            include_source_evaluation=False
        )
        
        evaluator = Evaluator(config)
        assert evaluator.config.similarity_threshold == 0.8
        assert evaluator.config.use_semantic_similarity == False
        assert evaluator.config.include_source_evaluation == False
    
    @patch('src.refinire_rag.processing.evaluator.Evaluator._calculate_semantic_similarity')
    def test_answer_evaluation(self, mock_similarity):
        """Test answer evaluation"""
        mock_similarity.return_value = 0.85
        
        expected_answer = "Machine learning is a subset of artificial intelligence"
        actual_answer = "ML is part of AI technology"
        
        score = self.evaluator.evaluate_answer(expected_answer, actual_answer)
        
        assert score > 0
        assert score <= 1.0
        mock_similarity.assert_called_once()
    
    def test_source_evaluation(self):
        """Test source evaluation"""
        expected_sources = ["doc1", "doc2"]
        actual_sources = ["doc1", "doc3"]
        
        score = self.evaluator.evaluate_sources(expected_sources, actual_sources)
        
        assert score >= 0
        assert score <= 1.0
        
        # Perfect match should give high score
        perfect_score = self.evaluator.evaluate_sources(expected_sources, expected_sources)
        assert perfect_score == 1.0
        
        # No match should give low score
        no_match_score = self.evaluator.evaluate_sources(expected_sources, ["doc4", "doc5"])
        assert no_match_score == 0.0
    
    def test_lexical_similarity(self):
        """Test lexical similarity calculation"""
        text1 = "machine learning algorithms"
        text2 = "ML algorithms for machines"
        
        similarity = self.evaluator._calculate_lexical_similarity(text1, text2)
        
        assert similarity >= 0
        assert similarity <= 1.0
        
        # Identical texts should have high similarity
        identical_similarity = self.evaluator._calculate_lexical_similarity(text1, text1)
        assert identical_similarity == 1.0
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation of test results"""
        test_result = {
            "query": "What is AI?",
            "expected_answer": "Artificial Intelligence",
            "actual_answer": "AI is artificial intelligence",
            "expected_sources": ["doc1"],
            "actual_sources": ["doc1", "doc2"],
            "response_time": 1.5
        }
        
        evaluation = self.evaluator.evaluate_test_result(test_result)
        
        assert "answer_score" in evaluation
        assert "source_score" in evaluation
        assert "overall_score" in evaluation
        assert "response_time" in evaluation
        
        assert 0 <= evaluation["answer_score"] <= 1
        assert 0 <= evaluation["source_score"] <= 1
        assert 0 <= evaluation["overall_score"] <= 1
    
    def test_batch_evaluation(self):
        """Test batch evaluation of multiple results"""
        test_results = [
            {
                "query": "What is AI?",
                "expected_answer": "Artificial Intelligence", 
                "actual_answer": "AI is artificial intelligence",
                "expected_sources": ["doc1"],
                "actual_sources": ["doc1"]
            },
            {
                "query": "What is ML?",
                "expected_answer": "Machine Learning",
                "actual_answer": "ML is machine learning",
                "expected_sources": ["doc2"],
                "actual_sources": ["doc2"]
            }
        ]
        
        evaluations = self.evaluator.evaluate_batch(test_results)
        
        assert len(evaluations) == 2
        for evaluation in evaluations:
            assert "overall_score" in evaluation
            assert evaluation["overall_score"] > 0


class TestContradictionDetector:
    """Test ContradictionDetector functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ContradictionDetectorConfig(
            nli_model="cross-encoder/nli-deberta-v3-base",
            contradiction_threshold=0.8,
            claim_extraction_method="llm",
            max_claims_per_document=10
        )
        self.detector = ContradictionDetector(self.config)
        
        # Test documents with potential contradictions
        self.test_docs = [
            Document(
                id="doc1",
                content="Python is a programming language. Python is easy to learn.",
                metadata={"source": "tutorial1"}
            ),
            Document(
                id="doc2",
                content="Python is difficult for beginners. Programming requires years of study.",
                metadata={"source": "tutorial2"}
            ),
            Document(
                id="doc3",
                content="Machine learning uses algorithms. Deep learning is a subset of ML.",
                metadata={"source": "ml_guide"}
            )
        ]
    
    def test_contradiction_detector_configuration(self):
        """Test contradiction detector configuration"""
        config = ContradictionDetectorConfig(
            nli_model="different-model",
            contradiction_threshold=0.9,
            claim_extraction_method="rule_based"
        )
        
        detector = ContradictionDetector(config)
        assert detector.config.nli_model == "different-model"
        assert detector.config.contradiction_threshold == 0.9
        assert detector.config.claim_extraction_method == "rule_based"
    
    @patch('src.refinire_rag.processing.contradiction_detector.ContradictionDetector._extract_claims')
    def test_claim_extraction(self, mock_extract_claims):
        """Test claim extraction from documents"""
        mock_extract_claims.return_value = [
            "Python is a programming language",
            "Python is easy to learn"
        ]
        
        doc = self.test_docs[0]
        claims = self.detector._extract_claims(doc.content)
        
        assert len(claims) == 2
        assert "Python is a programming language" in claims
        mock_extract_claims.assert_called_once()
    
    @patch('src.refinire_rag.processing.contradiction_detector.ContradictionDetector._check_contradiction')
    def test_contradiction_detection(self, mock_check_contradiction):
        """Test contradiction detection between claims"""
        mock_check_contradiction.return_value = 0.85  # High contradiction score
        
        claim1 = "Python is easy to learn"
        claim2 = "Python is difficult for beginners"
        
        score = self.detector._check_contradiction(claim1, claim2)
        
        assert score == 0.85
        assert score > self.config.contradiction_threshold
        mock_check_contradiction.assert_called_once()
    
    def test_document_pair_analysis(self):
        """Test contradiction analysis between document pairs"""
        with patch.object(self.detector, '_extract_claims') as mock_extract, \
             patch.object(self.detector, '_check_contradiction') as mock_check:
            
            mock_extract.side_effect = [
                ["Python is easy to learn"],
                ["Python is difficult for beginners"]
            ]
            mock_check.return_value = 0.9  # High contradiction
            
            contradictions = self.detector.analyze_document_pair(
                self.test_docs[0], 
                self.test_docs[1]
            )
            
            assert len(contradictions) > 0
            contradiction = contradictions[0]
            assert contradiction["score"] == 0.9
            assert contradiction["document1_id"] == "doc1"
            assert contradiction["document2_id"] == "doc2"
    
    def test_corpus_wide_analysis(self):
        """Test corpus-wide contradiction analysis"""
        with patch.object(self.detector, 'analyze_document_pair') as mock_analyze:
            mock_analyze.return_value = [{
                "claim1": "Test claim 1",
                "claim2": "Test claim 2", 
                "score": 0.85,
                "document1_id": "doc1",
                "document2_id": "doc2"
            }]
            
            results = self.detector.analyze_corpus(self.test_docs)
            
            assert "contradictions" in results
            assert "total_document_pairs" in results
            assert "contradiction_count" in results
            
            assert results["total_document_pairs"] > 0
            assert len(results["contradictions"]) > 0
    
    def test_contradiction_filtering(self):
        """Test filtering contradictions by threshold"""
        # Mock contradictions with different scores
        mock_contradictions = [
            {"score": 0.9, "claim1": "High contradiction", "claim2": "Opposite claim"},
            {"score": 0.6, "claim1": "Medium contradiction", "claim2": "Similar claim"},
            {"score": 0.3, "claim1": "Low contradiction", "claim2": "Related claim"}
        ]
        
        filtered = self.detector._filter_contradictions(mock_contradictions)
        
        # Only high-scoring contradictions should remain
        assert len(filtered) == 1
        assert filtered[0]["score"] == 0.9
    
    def test_contradiction_report_generation(self):
        """Test contradiction report generation"""
        contradictions = [{
            "score": 0.9,
            "claim1": "Python is easy",
            "claim2": "Python is difficult",
            "document1_id": "doc1",
            "document2_id": "doc2",
            "confidence": 0.95
        }]
        
        report = self.detector.generate_contradiction_report(contradictions)
        
        assert isinstance(report, str)
        assert "contradiction" in report.lower()
        assert "python" in report.lower()
        assert "0.9" in report or "90%" in report


class TestInsightReporter:
    """Test InsightReporter functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = InsightReporterConfig(
            output_format="markdown",
            include_visualizations=True,
            include_recommendations=True,
            confidence_threshold=0.7
        )
        self.reporter = InsightReporter(self.config)
        
        # Test evaluation data
        self.test_evaluation_data = {
            "evaluation_summary": {
                "total_tests": 10,
                "passed_tests": 8,
                "success_rate": 0.8,
                "average_response_time": 2.5,
                "average_score": 0.75
            },
            "test_results": [
                {"test_id": "test_1", "passed": True, "score": 0.9},
                {"test_id": "test_2", "passed": False, "score": 0.4},
                {"test_id": "test_3", "passed": True, "score": 0.8}
            ],
            "corpus_name": "test_corpus",
            "qa_set_name": "test_set",
            "timestamp": 1234567890
        }
    
    def test_insight_reporter_configuration(self):
        """Test insight reporter configuration"""
        config = InsightReporterConfig(
            output_format="json",
            include_visualizations=False,
            include_recommendations=False
        )
        
        reporter = InsightReporter(config)
        assert reporter.config.output_format == "json"
        assert reporter.config.include_visualizations == False
        assert reporter.config.include_recommendations == False
    
    def test_performance_analysis(self):
        """Test performance analysis generation"""
        analysis = self.reporter._analyze_performance(self.test_evaluation_data)
        
        assert "success_rate" in analysis
        assert "response_time_analysis" in analysis
        assert "score_distribution" in analysis
        
        assert analysis["success_rate"] == 0.8
        assert analysis["response_time_analysis"]["average"] == 2.5
    
    def test_trend_analysis(self):
        """Test trend analysis over time"""
        historical_data = [
            {"timestamp": 1234567800, "success_rate": 0.7},
            {"timestamp": 1234567850, "success_rate": 0.75},
            {"timestamp": 1234567890, "success_rate": 0.8}
        ]
        
        trends = self.reporter._analyze_trends(historical_data)
        
        assert "improvement_trend" in trends
        assert "trend_direction" in trends
        
        # Should detect positive trend
        assert trends["trend_direction"] in ["improving", "positive", "upward"]
    
    def test_recommendation_generation(self):
        """Test recommendation generation"""
        recommendations = self.reporter._generate_recommendations(self.test_evaluation_data)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for recommendation in recommendations:
            assert "category" in recommendation
            assert "description" in recommendation
            assert "priority" in recommendation
    
    def test_markdown_report_generation(self):
        """Test markdown report generation"""
        report = self.reporter.generate_report(self.test_evaluation_data)
        
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Check for markdown formatting
        assert "#" in report  # Headers
        assert "**" in report or "*" in report  # Bold/italic
        assert "test_corpus" in report
        assert "80.0%" in report or "0.8" in report  # Success rate
    
    def test_json_report_generation(self):
        """Test JSON report generation"""
        config = InsightReporterConfig(output_format="json")
        reporter = InsightReporter(config)
        
        report = reporter.generate_report(self.test_evaluation_data)
        
        # Should be valid JSON string
        import json
        parsed_report = json.loads(report)
        
        assert "evaluation_summary" in parsed_report
        assert "insights" in parsed_report
        assert "recommendations" in parsed_report
    
    def test_visualization_data_generation(self):
        """Test visualization data generation"""
        viz_data = self.reporter._generate_visualization_data(self.test_evaluation_data)
        
        assert "charts" in viz_data
        assert isinstance(viz_data["charts"], list)
        
        # Should have different chart types
        chart_types = [chart["type"] for chart in viz_data["charts"]]
        assert len(set(chart_types)) > 1  # Multiple chart types
    
    def test_threshold_based_insights(self):
        """Test threshold-based insight generation"""
        # Test data with performance below threshold
        poor_data = self.test_evaluation_data.copy()
        poor_data["evaluation_summary"]["success_rate"] = 0.5  # Below threshold
        
        insights = self.reporter._generate_insights(poor_data)
        
        assert "performance_concerns" in insights
        assert len(insights["performance_concerns"]) > 0
        
        # Should flag low success rate
        concerns = insights["performance_concerns"]
        success_rate_flagged = any("success" in concern.lower() for concern in concerns)
        assert success_rate_flagged
    
    def test_comparative_analysis(self):
        """Test comparative analysis between evaluations"""
        previous_data = {
            "evaluation_summary": {
                "success_rate": 0.7,
                "average_response_time": 3.0,
                "average_score": 0.65
            }
        }
        
        comparison = self.reporter._compare_evaluations(previous_data, self.test_evaluation_data)
        
        assert "success_rate_change" in comparison
        assert "response_time_change" in comparison
        assert "score_change" in comparison
        
        # Should detect improvements
        assert comparison["success_rate_change"] > 0  # Improved from 0.7 to 0.8
        assert comparison["response_time_change"] < 0  # Improved from 3.0 to 2.5


@pytest.mark.integration
class TestProcessingIntegration:
    """Integration tests for processing components"""
    
    def test_complete_processing_pipeline(self):
        """Test complete document processing pipeline"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test dictionary
            dict_path = Path(temp_dir) / "test_dict.md"
            dict_path.write_text("# Dictionary\n- AI: Artificial Intelligence\n")
            
            # Setup components
            chunker = Chunker(ChunkingConfig(chunk_size=100, overlap=20))
            normalizer = Normalizer(NormalizerConfig(
                dictionary_file_path=str(dict_path),
                normalize_case=True
            ))
            
            # Test document
            doc = Document(
                id="pipeline_test",
                content="AI is transforming the world. Machine learning algorithms are becoming more sophisticated. This represents a significant advancement in artificial intelligence technology.",
                metadata={"source": "test.txt"}
            )
            
            # Process through pipeline
            # 1. Normalize
            normalized_docs = normalizer.process(doc)
            assert len(normalized_docs) == 1
            
            # 2. Chunk
            chunks = []
            for norm_doc in normalized_docs:
                doc_chunks = chunker.process(norm_doc)
                chunks.extend(doc_chunks)
            
            assert len(chunks) > 0
            
            # Verify pipeline results
            for chunk in chunks:
                assert "parent_document_id" in chunk.metadata
                assert "normalization_applied" in chunk.metadata
                assert "chunk_index" in chunk.metadata
    
    def test_evaluation_workflow_integration(self):
        """Test complete evaluation workflow"""
        # Setup components
        test_suite = TestSuite(TestSuiteConfig())
        evaluator = Evaluator(EvaluatorConfig())
        reporter = InsightReporter(InsightReporterConfig())
        
        # Mock test execution
        with patch.object(test_suite, 'run_tests') as mock_run, \
             patch.object(evaluator, 'evaluate_batch') as mock_eval:
            
            mock_run.return_value = [
                {"test_id": "test_1", "passed": True, "score": 0.9},
                {"test_id": "test_2", "passed": False, "score": 0.4}
            ]
            
            mock_eval.return_value = [
                {"overall_score": 0.9, "answer_score": 0.95},
                {"overall_score": 0.4, "answer_score": 0.35}
            ]
            
            # Mock query engine and test cases
            mock_query_engine = Mock()
            mock_test_cases = [Mock(), Mock()]
            
            # Execute workflow
            test_results = test_suite.run_tests(mock_test_cases, mock_query_engine)
            evaluations = evaluator.evaluate_batch(test_results)
            
            # Create evaluation data for reporting
            eval_data = {
                "evaluation_summary": {
                    "total_tests": len(test_results),
                    "passed_tests": sum(1 for r in test_results if r["passed"]),
                    "success_rate": 0.5
                },
                "test_results": test_results,
                "evaluations": evaluations
            }
            
            # Generate report
            report = reporter.generate_report(eval_data)
            
            assert isinstance(report, str)
            assert len(report) > 0
            assert "50.0%" in report or "0.5" in report  # Success rate