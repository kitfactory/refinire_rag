"""
Test module for QualityLab.register_qa_pairs method

QualityLab.register_qa_pairsメソッドのテストモジュール
"""

import pytest
from unittest.mock import MagicMock, patch
from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.models.qa_pair import QAPair


class TestQualityLabRegisterQAPairs:
    """
    Test class for QualityLab register_qa_pairs method
    
    QualityLab register_qa_pairsメソッドのテストクラス
    """
    
    def setup_method(self):
        """
        Set up test fixtures
        
        テストフィクスチャのセットアップ
        """
        self.config = QualityLabConfig(
            qa_pairs_per_document=2,
            output_format="markdown"
        )
        self.quality_lab = QualityLab(config=self.config)
    
    def test_register_qa_pairs_success(self):
        """
        Test successful registration of QA pairs
        
        QAペアの正常な登録をテスト
        """
        # Create sample QA pairs
        qa_pairs = [
            QAPair(
                question="What is artificial intelligence?",
                answer="AI is a broad field of computer science.",
                document_id="ai_basics_001",
                metadata={
                    "qa_id": "qa_001",
                    "question_type": "definition",
                    "topic": "ai_fundamentals"
                }
            ),
            QAPair(
                question="What are the main types of machine learning?",
                answer="Supervised, unsupervised, and reinforcement learning.",
                document_id="ml_concepts_002",
                metadata={
                    "qa_id": "qa_002",
                    "question_type": "categorization",
                    "topic": "machine_learning"
                }
            )
        ]
        
        # Test registration
        result = self.quality_lab.register_qa_pairs(
            qa_pairs=qa_pairs,
            qa_set_name="test_set",
            metadata={"source": "test"}
        )
        
        assert result is True
        assert self.quality_lab.stats["qa_pairs_generated"] == 2
        
        # Verify metadata was added
        for qa_pair in qa_pairs:
            assert qa_pair.metadata["qa_set_name"] == "test_set"
            assert qa_pair.metadata["registration_source"] == "external_registration"
            assert "registration_timestamp" in qa_pair.metadata
    
    def test_register_qa_pairs_empty_list(self):
        """
        Test registration with empty QA pairs list
        
        空のQAペアリストでの登録をテスト
        """
        result = self.quality_lab.register_qa_pairs(
            qa_pairs=[],
            qa_set_name="empty_set"
        )
        
        assert result is False
        assert self.quality_lab.stats["qa_pairs_generated"] == 0
    
    def test_register_qa_pairs_invalid_qa_pair(self):
        """
        Test registration with invalid QA pair
        
        無効なQAペアでの登録をテスト
        """
        invalid_qa_pairs = [
            QAPair(
                question="",  # Empty question
                answer="This should fail",
                document_id="test_doc",
                metadata={"qa_id": "qa_invalid"}
            )
        ]
        
        result = self.quality_lab.register_qa_pairs(
            qa_pairs=invalid_qa_pairs,
            qa_set_name="invalid_set"
        )
        
        assert result is False
        assert self.quality_lab.stats["qa_pairs_generated"] == 0
    
    def test_register_qa_pairs_invalid_type(self):
        """
        Test registration with invalid QA pair type
        
        無効なQAペアタイプでの登録をテスト
        """
        invalid_qa_pairs = [
            {"question": "test", "answer": "test"}  # Dict instead of QAPair
        ]
        
        result = self.quality_lab.register_qa_pairs(
            qa_pairs=invalid_qa_pairs,
            qa_set_name="invalid_type_set"
        )
        
        assert result is False
        assert self.quality_lab.stats["qa_pairs_generated"] == 0
    
    def test_register_qa_pairs_no_answer(self):
        """
        Test registration with QA pair missing answer
        
        回答が欠けているQAペアでの登録をテスト
        """
        invalid_qa_pairs = [
            QAPair(
                question="Valid question?",
                answer="",  # Empty answer
                document_id="test_doc",
                metadata={"qa_id": "qa_no_answer"}
            )
        ]
        
        result = self.quality_lab.register_qa_pairs(
            qa_pairs=invalid_qa_pairs,
            qa_set_name="no_answer_set"
        )
        
        assert result is False
        assert self.quality_lab.stats["qa_pairs_generated"] == 0
    
    def test_register_qa_pairs_with_metadata(self):
        """
        Test registration with additional metadata
        
        追加メタデータでの登録をテスト
        """
        qa_pairs = [
            QAPair(
                question="Test question?",
                answer="Test answer",
                document_id="test_doc",
                metadata={"qa_id": "qa_test"}
            )
        ]
        
        custom_metadata = {
            "source": "manual_creation",
            "domain": "testing",
            "difficulty": "easy"
        }
        
        result = self.quality_lab.register_qa_pairs(
            qa_pairs=qa_pairs,
            qa_set_name="metadata_test_set",
            metadata=custom_metadata
        )
        
        assert result is True
        assert self.quality_lab.stats["qa_pairs_generated"] == 1
        
        # Verify custom metadata was preserved
        qa_pair = qa_pairs[0]
        assert qa_pair.metadata["qa_set_name"] == "metadata_test_set"
        assert qa_pair.metadata["registration_source"] == "external_registration"
    
    @patch('refinire_rag.application.quality_lab.logger')
    def test_register_qa_pairs_exception_handling(self, mock_logger):
        """
        Test exception handling during registration
        
        登録時の例外処理をテスト
        """
        # Mock evaluation store to raise exception
        self.quality_lab.evaluation_store = MagicMock()
        self.quality_lab.evaluation_store.create_evaluation_run.side_effect = Exception("Storage error")
        
        qa_pairs = [
            QAPair(
                question="Test question?",
                answer="Test answer",
                document_id="test_doc",
                metadata={"qa_id": "qa_test"}
            )
        ]
        
        # Should still succeed even if storage fails
        result = self.quality_lab.register_qa_pairs(
            qa_pairs=qa_pairs,
            qa_set_name="exception_test_set"
        )
        
        assert result is True
        assert self.quality_lab.stats["qa_pairs_generated"] == 1
        
        # Verify warning was logged
        mock_logger.warning.assert_called_once()
    
    def test_register_qa_pairs_metadata_initialization(self):
        """
        Test that QA pairs with None metadata are handled correctly
        
        メタデータがNoneのQAペアが正しく処理されることをテスト
        """
        qa_pairs = [
            QAPair(
                question="Test question?",
                answer="Test answer",
                document_id="test_doc",
                metadata=None
            )
        ]
        
        result = self.quality_lab.register_qa_pairs(
            qa_pairs=qa_pairs,
            qa_set_name="none_metadata_test_set"
        )
        
        assert result is True
        assert self.quality_lab.stats["qa_pairs_generated"] == 1
        
        # Verify metadata was initialized
        qa_pair = qa_pairs[0]
        assert qa_pair.metadata is not None
        assert qa_pair.metadata["qa_set_name"] == "none_metadata_test_set"
        assert qa_pair.metadata["registration_source"] == "external_registration"