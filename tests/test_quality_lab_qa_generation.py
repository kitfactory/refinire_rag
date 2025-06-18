"""
Comprehensive tests for QualityLab QA pair generation functionality
QualityLabのQAペア生成機能の包括的テスト

This module tests the QA pair generation pipeline of QualityLab.
このモジュールは、QualityLabのQAペア生成パイプラインをテストします。
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any

from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.models.qa_pair import QAPair
from refinire_rag.models.document import Document


class TestQualityLabQAGeneration:
    """
    Test QualityLab QA pair generation functionality
    QualityLabのQAペア生成機能のテスト
    """

    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        # Create mock components
        self.mock_corpus_manager = Mock()
        self.mock_evaluation_store = Mock()
        
        # Create sample documents
        self.sample_documents = [
            Document(
                id="doc1", 
                content="Python is a high-level programming language known for its simplicity and readability.",
                metadata={"source": "python_guide.txt", "type": "programming"}
            ),
            Document(
                id="doc2",
                content="Machine learning is a subset of artificial intelligence that uses algorithms to learn patterns.",
                metadata={"source": "ml_intro.txt", "type": "ai", "difficulty": "beginner"}
            ),
            Document(
                id="doc3",
                content="Data structures like arrays, linked lists, and trees are fundamental to computer science.",
                metadata={"source": "data_structures.txt", "type": "cs_fundamentals"}
            )
        ]
        
        # Create QualityLab instance
        with patch('refinire_rag.application.quality_lab.TestSuite'), \
             patch('refinire_rag.application.quality_lab.Evaluator'), \
             patch('refinire_rag.application.quality_lab.ContradictionDetector'), \
             patch('refinire_rag.application.quality_lab.InsightReporter'):
            
            self.lab = QualityLab(
                corpus_manager=self.mock_corpus_manager,
                evaluation_store=self.mock_evaluation_store
            )

    @patch('refinire_rag.application.quality_lab.RefinireAgent')
    def test_generate_qa_pairs_basic_success(self, mock_agent_class):
        """
        Test basic successful QA pair generation with RefinireAgent
        RefinireAgentを使った基本的な成功QAペア生成テスト
        """
        # Setup mock RefinireAgent
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Mock agent response for QA generation
        mock_llm_result = Mock()
        mock_llm_result.generation = json.dumps({
            "qa_pairs": [
                {
                    "question": "What is Python known for?",
                    "answer": "Python is known for its simplicity and readability.",
                    "question_type": "factual"
                },
                {
                    "question": "How would you describe Python as a programming language?", 
                    "answer": "Python is a high-level programming language.",
                    "question_type": "conceptual"
                }
            ]
        })
        mock_agent.run.return_value = mock_llm_result
        
        # Setup corpus manager to return documents
        self.mock_corpus_manager.get_documents_by_stage.return_value = [self.sample_documents[0]]
        
        # Generate QA pairs
        qa_pairs = self.lab.generate_qa_pairs(
            qa_set_name="test_set",
            corpus_name="test_corpus",
            num_pairs=2
        )
        
        # Verify RefinireAgent was created and used
        mock_agent_class.assert_called_once()
        mock_agent.run.assert_called_once()
        
        # Verify results
        assert isinstance(qa_pairs, list)
        assert len(qa_pairs) == 2  # 2 QA pairs from mock response
        
        # Verify QA pair structure
        for i, qa_pair in enumerate(qa_pairs):
            assert isinstance(qa_pair, QAPair)
            assert qa_pair.question is not None
            assert qa_pair.answer is not None
            assert qa_pair.document_id == "doc1"
            assert qa_pair.metadata is not None
            assert qa_pair.metadata.get("qa_set_name") == "test_set"
            assert qa_pair.metadata.get("corpus_name") == "test_corpus"
            assert "question_type" in qa_pair.metadata

    @patch('refinire_rag.application.quality_lab.RefinireAgent')
    def test_generate_qa_pairs_with_filters(self, mock_agent_class):
        """
        Test QA pair generation with document filters using RefinireAgent
        RefinireAgentを使ったドキュメントフィルターを使用したQAペア生成テスト
        """
        # Setup mock RefinireAgent
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Mock agent response
        mock_llm_result = Mock()
        mock_llm_result.generation = json.dumps({
            "qa_pairs": [{
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence.",
                "question_type": "factual"
            }]
        })
        mock_agent.run.return_value = mock_llm_result
        
        # Setup corpus manager to return filtered documents
        filtered_docs = [doc for doc in self.sample_documents if doc.metadata.get("type") == "ai"]
        self.mock_corpus_manager.get_documents_by_stage.return_value = self.sample_documents
        
        # Generate QA pairs with filters
        filters = {"type": "ai"}
        qa_pairs = self.lab.generate_qa_pairs(
            qa_set_name="filtered_set",
            corpus_name="test_corpus",
            document_filters=filters,
            num_pairs=1
        )
        
        # Verify corpus manager was called
        self.mock_corpus_manager.get_documents_by_stage.assert_called_once()
        
        # Verify RefinireAgent was used
        mock_agent_class.assert_called_once()
        mock_agent.run.assert_called_once()
        
        # Verify results
        assert isinstance(qa_pairs, list)
        assert len(qa_pairs) >= 1
        
        # Verify filter metadata is included
        for qa_pair in qa_pairs:
            assert qa_pair.metadata.get("document_filters") == filters

    def test_retrieve_corpus_documents_basic(self):
        """
        Test _retrieve_corpus_documents basic functionality
        _retrieve_corpus_documents基本機能のテスト
        """
        # Setup corpus manager
        self.mock_corpus_manager.get_documents_by_stage.return_value = self.sample_documents
        
        # Call method
        documents = self.lab._retrieve_corpus_documents("test_corpus")
        
        # Verify call and results
        self.mock_corpus_manager.get_documents_by_stage.assert_called_once_with("test_corpus", stage="original")
        assert documents == self.sample_documents

    def test_retrieve_corpus_documents_with_filters(self):
        """
        Test _retrieve_corpus_documents with filters
        フィルターを使用した_retrieve_corpus_documentsテスト
        """
        # Setup corpus manager
        filtered_docs = [self.sample_documents[1]]  # Only AI document
        self.mock_corpus_manager.get_documents_by_stage.return_value = self.sample_documents
        
        # Call method with filters
        filters = {"type": "ai", "difficulty": "beginner"}
        documents = self.lab._retrieve_corpus_documents(
            corpus_name="test_corpus",
            document_filters=filters
        )
        
        # Verify call and results (documents will be filtered internally)
        self.mock_corpus_manager.get_documents_by_stage.assert_called_once_with("test_corpus", stage="original")
        
        # Should filter documents based on metadata
        assert len(documents) == 1
        assert documents[0].metadata.get("type") == "ai"

    def test_retrieve_corpus_documents_empty_result(self):
        """
        Test _retrieve_corpus_documents with empty result
        空の結果での_retrieve_corpus_documentsテスト
        """
        # Setup corpus manager to return empty list
        self.mock_corpus_manager.get_documents_by_stage.return_value = []
        
        # Call method
        documents = self.lab._retrieve_corpus_documents("test_corpus")
        
        # Verify results
        assert documents == []
        self.mock_corpus_manager.get_documents_by_stage.assert_called_once_with("test_corpus", stage="original")

    def test_retrieve_corpus_documents_no_corpus_manager(self):
        """
        Test _retrieve_corpus_documents with no corpus manager
        コーパスマネージャーがない場合の_retrieve_corpus_documentsテスト
        """
        # Create lab without corpus manager
        with patch('refinire_rag.application.quality_lab.TestSuite'), \
             patch('refinire_rag.application.quality_lab.Evaluator'), \
             patch('refinire_rag.application.quality_lab.ContradictionDetector'), \
             patch('refinire_rag.application.quality_lab.InsightReporter'):
            
            lab_no_corpus = QualityLab(
                corpus_manager=None,
                evaluation_store=self.mock_evaluation_store
            )
        
        # Should handle gracefully
        documents = lab_no_corpus._retrieve_corpus_documents("test_corpus")
        assert documents == []

    def test_matches_filters_in_operator(self):
        """
        Test _matches_filters with $in operator
        $in演算子での_matches_filtersテスト
        """
        document = self.sample_documents[1]  # AI document
        
        # Test $in operator - should match
        filters = {"type": {"$in": ["ai", "programming"]}}
        assert self.lab._matches_filters(document, filters) is True
        
        # Test $in operator - should not match
        filters = {"type": {"$in": ["database", "networking"]}}
        assert self.lab._matches_filters(document, filters) is False

    def test_matches_filters_comparison_operators(self):
        """
        Test _matches_filters with comparison operators
        比較演算子での_matches_filtersテスト
        """
        # Create document with numeric metadata
        doc_with_numbers = Document(
            id="doc_num",
            content="Test content",
            metadata={"score": 85, "year": 2023, "rating": 4.5}
        )
        
        # Test $gte operator
        filters = {"score": {"$gte": 80}}
        assert self.lab._matches_filters(doc_with_numbers, filters) is True
        
        filters = {"score": {"$gte": 90}}
        assert self.lab._matches_filters(doc_with_numbers, filters) is False
        
        # Test $lte operator
        filters = {"year": {"$lte": 2025}}
        assert self.lab._matches_filters(doc_with_numbers, filters) is True
        
        filters = {"year": {"$lte": 2020}}
        assert self.lab._matches_filters(doc_with_numbers, filters) is False
        
        # Test $gt operator
        filters = {"rating": {"$gt": 4.0}}
        assert self.lab._matches_filters(doc_with_numbers, filters) is True
        
        # Test $lt operator
        filters = {"rating": {"$lt": 5.0}}
        assert self.lab._matches_filters(doc_with_numbers, filters) is True

    def test_matches_filters_contains_operator(self):
        """
        Test _matches_filters with $contains operator
        $contains演算子での_matches_filtersテスト
        """
        document = self.sample_documents[0]  # Python document
        
        # Test $contains operator on metadata field - should match
        filters = {"source": {"$contains": "python"}}
        assert self.lab._matches_filters(document, filters) is True
        
        # Test $contains operator - should not match
        filters = {"source": {"$contains": "Java"}}
        assert self.lab._matches_filters(document, filters) is False
        
        # Test case insensitive matching on source field
        filters = {"source": {"$contains": "PYTHON"}}
        assert self.lab._matches_filters(document, filters) is True

    def test_matches_filters_exact_match(self):
        """
        Test _matches_filters with exact match
        完全一致での_matches_filtersテスト
        """
        document = self.sample_documents[0]  # Python document
        
        # Test exact match - should match
        filters = {"type": "programming"}
        assert self.lab._matches_filters(document, filters) is True
        
        # Test exact match - should not match
        filters = {"type": "database"}
        assert self.lab._matches_filters(document, filters) is False

    def test_matches_filters_multiple_conditions(self):
        """
        Test _matches_filters with multiple conditions
        複数条件での_matches_filtersテスト
        """
        document = self.sample_documents[1]  # AI document with difficulty: beginner
        
        # Test multiple conditions - all should match
        filters = {"type": "ai", "difficulty": "beginner"}
        assert self.lab._matches_filters(document, filters) is True
        
        # Test multiple conditions - one doesn't match
        filters = {"type": "ai", "difficulty": "advanced"}
        assert self.lab._matches_filters(document, filters) is False

    def test_matches_filters_missing_metadata(self):
        """
        Test _matches_filters with missing metadata
        メタデータが欠落している場合の_matches_filtersテスト
        """
        document = self.sample_documents[0]  # No difficulty metadata
        
        # Test filter for missing metadata field
        filters = {"difficulty": "beginner"}
        assert self.lab._matches_filters(document, filters) is False

    def test_matches_filters_no_filters(self):
        """
        Test _matches_filters with no filters
        フィルターがない場合の_matches_filtersテスト
        """
        document = self.sample_documents[0]
        
        # Should match when no filters provided
        assert self.lab._matches_filters(document, None) is True
        assert self.lab._matches_filters(document, {}) is True

    @patch('refinire_rag.application.quality_lab.RefinireAgent')
    def test_generate_qa_pairs_for_document(self, mock_agent_class):
        """
        Test _generate_qa_pairs_for_document functionality with RefinireAgent
        RefinireAgentを使った_generate_qa_pairs_for_document機能のテスト
        """
        # Setup mock RefinireAgent
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Mock agent response
        mock_llm_result = Mock()
        mock_llm_result.generation = json.dumps({
            "qa_pairs": [
                {
                    "question": "What is Python?",
                    "answer": "Python is a high-level programming language.",
                    "question_type": "factual"
                },
                {
                    "question": "Why is Python popular?",
                    "answer": "Python is popular for its simplicity and readability.",
                    "question_type": "analytical"
                }
            ]
        })
        mock_agent.run.return_value = mock_llm_result
        
        # Test QA generation for single document
        document = self.sample_documents[0]
        base_metadata = {
            "qa_set_name": "test_set",
            "corpus_name": "test_corpus",
            "generation_timestamp": "2024-01-01 12:00:00"
        }
        
        qa_pairs = self.lab._generate_qa_pairs_for_document(document, base_metadata)
        
        # Verify RefinireAgent was created and used
        mock_agent_class.assert_called_once()
        mock_agent.run.assert_called_once()
        
        # Verify results
        assert isinstance(qa_pairs, list)
        assert len(qa_pairs) == 2  # 2 QA pairs from mock response
        
        for qa_pair in qa_pairs:
            assert isinstance(qa_pair, QAPair)
            assert qa_pair.document_id == "doc1"
            assert qa_pair.metadata.get("qa_set_name") == "test_set"
            assert qa_pair.metadata.get("corpus_name") == "test_corpus"
            assert "question_type" in qa_pair.metadata

    @patch('refinire_rag.application.quality_lab.RefinireAgent')
    def test_generate_qa_pairs_for_document_error_handling(self, mock_agent_class):
        """
        Test _generate_qa_pairs_for_document error handling with RefinireAgent
        RefinireAgentを使った_generate_qa_pairs_for_documentエラーハンドリングテスト
        """
        # Setup mock RefinireAgent to raise exception
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.run.side_effect = Exception("RefinireAgent API error")
        
        # Test with error
        document = self.sample_documents[0]
        base_metadata = {"qa_set_name": "test_set"}
        
        # Should handle error gracefully and return empty list
        qa_pairs = self.lab._generate_qa_pairs_for_document(document, base_metadata)
        
        # Verify RefinireAgent was attempted
        mock_agent_class.assert_called_once()
        mock_agent.run.assert_called_once()
        
        # Should return empty list on error
        assert qa_pairs == []

    @patch('refinire_rag.application.quality_lab.RefinireAgent')
    def test_generate_qa_pairs_for_document_invalid_json(self, mock_agent_class):
        """
        Test _generate_qa_pairs_for_document with invalid JSON response
        無効なJSONレスポンスでの_generate_qa_pairs_for_documentテスト
        """
        # Setup mock RefinireAgent to return invalid JSON
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        mock_llm_result = Mock()
        mock_llm_result.generation = "Invalid JSON response"
        mock_agent.run.return_value = mock_llm_result
        
        # Test QA generation with invalid JSON
        document = self.sample_documents[0]
        base_metadata = {"qa_set_name": "test_set"}
        
        qa_pairs = self.lab._generate_qa_pairs_for_document(document, base_metadata)
        
        # Should handle JSON error gracefully and return empty list
        assert qa_pairs == []

    @patch('refinire_rag.application.quality_lab.RefinireAgent')
    def test_generate_qa_pairs_for_document_with_custom_config(self, mock_agent_class):
        """
        Test _generate_qa_pairs_for_document with custom configuration using RefinireAgent
        RefinireAgentを使ったカスタム設定での_generate_qa_pairs_for_documentテスト
        """
        # Setup mock RefinireAgent
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Mock agent response with 5 QA pairs
        mock_llm_result = Mock()
        mock_llm_result.generation = json.dumps({
            "qa_pairs": [
                {"question": f"Question {i+1}?", "answer": f"Answer {i+1}", "question_type": ["factual", "analytical"][i % 2]}
                for i in range(5)
            ]
        })
        mock_agent.run.return_value = mock_llm_result
        
        # Create lab with custom configuration
        custom_config = QualityLabConfig(
            qa_pairs_per_document=5,
            question_types=["factual", "analytical"]
        )
        
        with patch('refinire_rag.application.quality_lab.TestSuite'), \
             patch('refinire_rag.application.quality_lab.Evaluator'), \
             patch('refinire_rag.application.quality_lab.ContradictionDetector'), \
             patch('refinire_rag.application.quality_lab.InsightReporter'):
            
            custom_lab = QualityLab(
                corpus_manager=self.mock_corpus_manager,
                evaluation_store=self.mock_evaluation_store,
                config=custom_config
            )
        
        document = self.sample_documents[0]
        base_metadata = {"qa_set_name": "custom_test"}
        
        qa_pairs = custom_lab._generate_qa_pairs_for_document(document, base_metadata)
        
        # Verify RefinireAgent was used
        mock_agent_class.assert_called_once()
        mock_agent.run.assert_called_once()
        
        # Should generate 5 pairs as configured
        assert len(qa_pairs) == 5
        
        # Should use custom question types
        question_types = [qa.metadata.get("question_type") for qa in qa_pairs]
        for q_type in question_types:
            assert q_type in ["factual", "analytical"]

    @patch('refinire_rag.application.quality_lab.RefinireAgent')
    def test_generate_qa_pairs_multiple_documents(self, mock_agent_class):
        """
        Test QA pair generation for multiple documents using RefinireAgent
        RefinireAgentを使った複数ドキュメントでのQAペア生成テスト
        """
        # Setup mock RefinireAgent
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Mock agent responses for different documents
        responses = [
            json.dumps({"qa_pairs": [{"question": "Q1", "answer": "A1", "question_type": "factual"}] * 3}),
            json.dumps({"qa_pairs": [{"question": "Q2", "answer": "A2", "question_type": "conceptual"}] * 3}),
        ]
        
        mock_llm_results = []
        for response in responses:
            mock_result = Mock()
            mock_result.generation = response
            mock_llm_results.append(mock_result)
        
        mock_agent.run.side_effect = mock_llm_results
        
        # Setup corpus manager to return multiple documents
        self.mock_corpus_manager.get_documents_by_stage.return_value = self.sample_documents[:2]
        
        # Generate QA pairs
        qa_pairs = self.lab.generate_qa_pairs(
            qa_set_name="multi_doc_test",
            corpus_name="test_corpus"
        )
        
        # Verify RefinireAgent was called for each document
        assert mock_agent.run.call_count == 2
        
        # Verify results
        assert isinstance(qa_pairs, list)
        assert len(qa_pairs) == 6  # 3 QA pairs per document * 2 documents
        
        # Verify document distribution
        doc1_pairs = [qa for qa in qa_pairs if qa.document_id == "doc1"]
        doc2_pairs = [qa for qa in qa_pairs if qa.document_id == "doc2"]
        
        assert len(doc1_pairs) == 3
        assert len(doc2_pairs) == 3
        
        # Verify all pairs have correct metadata
        for qa_pair in qa_pairs:
            assert qa_pair.metadata.get("qa_set_name") == "multi_doc_test"
            assert qa_pair.metadata.get("corpus_name") == "test_corpus"

    @patch('refinire_rag.application.quality_lab.RefinireAgent')
    def test_generate_qa_pairs_no_documents(self, mock_agent_class):
        """
        Test QA pair generation with no documents
        ドキュメントがない場合のQAペア生成テスト
        """
        # Setup corpus manager to return empty list
        self.mock_corpus_manager.get_documents_by_stage.return_value = []
        
        # Generate QA pairs
        qa_pairs = self.lab.generate_qa_pairs(
            qa_set_name="empty_test",
            corpus_name="empty_corpus"
        )
        
        # Should return empty list without calling RefinireAgent
        assert qa_pairs == []
        mock_agent_class.assert_not_called()