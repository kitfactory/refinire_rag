"""
Comprehensive tests for DictionaryMaker processor
DictionaryMaker プロセッサーの包括的テスト

This module provides comprehensive coverage for the DictionaryMaker processing module,
testing domain-specific term extraction, LLM integration, file operations, and edge cases.
このモジュールは、DictionaryMakerプロセッシングモジュールの包括的カバレッジを提供し、
ドメイン固有用語抽出、LLM統合、ファイル操作、エッジケースをテストします。
"""

import pytest
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock, patch, mock_open, MagicMock
from dataclasses import dataclass

from refinire_rag.processing.dictionary_maker import DictionaryMaker, DictionaryMakerConfig
from refinire_rag.models.document import Document


class TestDictionaryMakerConfig:
    """
    Test DictionaryMakerConfig configuration class
    DictionaryMakerConfig設定クラスのテスト
    """
    
    def test_config_default_values(self):
        """
        Test default configuration values
        デフォルト設定値のテスト
        """
        config = DictionaryMakerConfig()
        
        # Default file paths (converted to absolute in __post_init__)
        assert config.dictionary_file_path.endswith("domain_dictionary.md")
        assert config.backup_dictionary is True
        
        # LLM settings
        assert config.llm_temperature == 0.3
        assert config.max_tokens == 2000
        
        # Term extraction settings
        assert config.focus_on_technical_terms is True
        assert config.extract_abbreviations is True
        assert config.detect_expression_variations is True
        assert config.min_term_importance == "medium"
        
        # Processing settings
        assert config.skip_if_no_new_terms is False
        assert config.validate_extracted_terms is True
        
        # Output settings
        assert config.update_document_metadata is True
        assert config.preserve_original_document is True
    
    def test_config_custom_values(self):
        """
        Test configuration with custom values
        カスタム値での設定テスト
        """
        config = DictionaryMakerConfig(
            dictionary_file_path="/custom/dict.md",
            backup_dictionary=False,
            llm_temperature=0.7,
            max_tokens=3000,
            focus_on_technical_terms=False,
            extract_abbreviations=False,
            min_term_importance="high",
            skip_if_no_new_terms=True,
            update_document_metadata=False
        )
        
        assert config.dictionary_file_path == "/custom/dict.md"
        assert config.backup_dictionary is False
        assert config.llm_temperature == 0.7
        assert config.max_tokens == 3000
        assert config.focus_on_technical_terms is False
        assert config.extract_abbreviations is False
        assert config.min_term_importance == "high"
        assert config.skip_if_no_new_terms is True
        assert config.update_document_metadata is False
    
    @patch('refinire_rag.processing.dictionary_maker.get_default_llm_model')
    def test_config_post_init_llm_model(self, mock_get_model):
        """
        Test __post_init__ setting default LLM model
        __post_init__でのデフォルトLLMモデル設定テスト
        """
        mock_get_model.return_value = "gpt-4"
        
        config = DictionaryMakerConfig()
        
        assert config.llm_model == "gpt-4"
        mock_get_model.assert_called_once()
    
    def test_config_post_init_absolute_paths(self):
        """
        Test __post_init__ converting relative paths to absolute
        __post_init__での相対パスから絶対パスへの変換テスト
        """
        config = DictionaryMakerConfig(dictionary_file_path="./relative_dict.md")
        
        assert os.path.isabs(config.dictionary_file_path)
        assert config.dictionary_file_path.endswith("relative_dict.md")
    
    def test_config_post_init_already_absolute_paths(self):
        """
        Test __post_init__ preserving already absolute paths
        __post_init__での既に絶対パスの保持テスト
        """
        absolute_path = "/already/absolute/dict.md"
        config = DictionaryMakerConfig(dictionary_file_path=absolute_path)
        
        assert config.dictionary_file_path == absolute_path


class TestDictionaryMakerInitialization:
    """
    Test DictionaryMaker initialization and setup
    DictionaryMakerの初期化とセットアップのテスト
    """
    
    def test_init_with_default_config(self):
        """
        Test initialization with default configuration
        デフォルト設定での初期化テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            assert isinstance(maker.config, DictionaryMakerConfig)
            assert maker._llm_pipeline is None
            
            # Check processing stats initialization
            assert maker.processing_stats["documents_processed"] == 0
            assert maker.processing_stats["terms_extracted"] == 0
            assert maker.processing_stats["variations_detected"] == 0
            assert maker.processing_stats["dictionary_updates"] == 0
            assert maker.processing_stats["llm_api_calls"] == 0
    
    def test_init_with_custom_config(self):
        """
        Test initialization with custom configuration
        カスタム設定での初期化テスト
        """
        custom_config = DictionaryMakerConfig(
            dictionary_file_path="/custom/dict.md",
            llm_temperature=0.5,
            max_tokens=1500
        )
        
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker(custom_config)
            
            assert maker.config == custom_config
            assert maker.config.dictionary_file_path == "/custom/dict.md"
            assert maker.config.llm_temperature == 0.5
    
    @patch('refinire_rag.processing.dictionary_maker.LLMPipeline')
    def test_init_with_refinire_available(self, mock_llm_pipeline):
        """
        Test initialization when Refinire is available
        Refinire利用可能時の初期化テスト
        """
        mock_pipeline_instance = Mock()
        mock_llm_pipeline.return_value = mock_pipeline_instance
        
        maker = DictionaryMaker()
        
        assert maker._llm_pipeline == mock_pipeline_instance
        mock_llm_pipeline.assert_called_once_with(
            name="dictionary_maker",
            generation_instructions="You are a domain expert that extracts technical terms and their variations from documents.",
            model=maker.config.llm_model
        )
    
    @patch('refinire_rag.processing.dictionary_maker.LLMPipeline')
    def test_init_with_refinire_initialization_error(self, mock_llm_pipeline):
        """
        Test initialization when Refinire initialization fails
        Refinire初期化失敗時の初期化テスト
        """
        mock_llm_pipeline.side_effect = Exception("LLM initialization failed")
        
        maker = DictionaryMaker()
        
        assert maker._llm_pipeline is None
    
    def test_get_config_class(self):
        """
        Test get_config_class method
        get_config_classメソッドのテスト
        """
        assert DictionaryMaker.get_config_class() == DictionaryMakerConfig


class TestDictionaryMakerFileOperations:
    """
    Test DictionaryMaker file operations
    DictionaryMakerのファイル操作テスト
    """
    
    def test_read_existing_dictionary_file_exists(self):
        """
        Test reading existing dictionary file
        既存辞書ファイル読み込みテスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            with patch("builtins.open", mock_open(read_data="# Test Dictionary\n## Terms\n- Test: Definition")):
                with patch("pathlib.Path.exists", return_value=True):
                    config = DictionaryMakerConfig(dictionary_file_path="/test/dict.md")
                    
                    result = maker._read_existing_dictionary(config)
                    
                    assert result == "# Test Dictionary\n## Terms\n- Test: Definition"
    
    def test_read_existing_dictionary_file_not_exists(self):
        """
        Test reading dictionary when file doesn't exist
        ファイル非存在時の辞書読み込みテスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            with patch("pathlib.Path.exists", return_value=False):
                with patch("builtins.open", mock_open()) as mock_file:
                    with patch("pathlib.Path.mkdir"):
                        config = DictionaryMakerConfig(dictionary_file_path="/test/dict.md")
                        
                        result = maker._read_existing_dictionary(config)
                        
                        # Should create new file with template
                        mock_file.assert_called()
                        assert "# ドメイン用語辞書" in result
                        assert "## 専門用語" in result
                        assert "## 技術概念" in result
    
    def test_read_existing_dictionary_file_error(self):
        """
        Test reading dictionary when file operation fails
        ファイル操作失敗時の辞書読み込みテスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            with patch("pathlib.Path.exists", side_effect=OSError("File access error")):
                config = DictionaryMakerConfig(dictionary_file_path="/test/dict.md")
                
                result = maker._read_existing_dictionary(config)
                
                # Should return template on error
                assert "# ドメイン用語辞書" in result
    
    def test_create_empty_dictionary_template(self):
        """
        Test creating empty dictionary template
        空の辞書テンプレート作成テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            template = maker._create_empty_dictionary_template()
            
            assert "# ドメイン用語辞書" in template
            assert "## 専門用語" in template
            assert "## 技術概念" in template
            assert "## 表現揺らぎ" in template
            assert "*この辞書は自動生成され" in template
    
    def test_backup_dictionary_file(self):
        """
        Test dictionary file backup creation
        辞書ファイルバックアップ作成テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            with patch("pathlib.Path.exists", return_value=True):
                with patch("shutil.copy2") as mock_copy:
                    config = DictionaryMakerConfig(dictionary_file_path="/test/dict.md")
                    
                    maker._backup_dictionary_file(config)
                    
                    mock_copy.assert_called_once()
    
    def test_backup_dictionary_file_no_existing_file(self):
        """
        Test backup when no existing file
        既存ファイルなし時のバックアップテスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            with patch("pathlib.Path.exists", return_value=False):
                with patch("shutil.copy2") as mock_copy:
                    config = DictionaryMakerConfig(dictionary_file_path="/test/dict.md")
                    
                    maker._backup_dictionary_file(config)
                    
                    mock_copy.assert_not_called()


class TestDictionaryMakerLLMIntegration:
    """
    Test DictionaryMaker LLM integration
    DictionaryMakerのLLM統合テスト
    """
    
    def test_create_extraction_prompt(self):
        """
        Test creating term extraction prompt
        用語抽出プロンプト作成テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            document = Document(
                id="test_doc",
                content="This document contains RAG and vector search terms.",
                metadata={"title": "Test Document"}
            )
            existing_dictionary = "# Test Dictionary\n## Terms"
            config = DictionaryMakerConfig()
            
            prompt = maker._create_extraction_prompt(document, existing_dictionary, config)
            
            assert "以下のドメイン文書から、重要な専門用語と表現揺らぎを抽出してください" in prompt
            assert "Test Dictionary" in prompt
            assert "Test Document" in prompt
            assert "RAG and vector search" in prompt
            assert "JSON形式" in prompt
    
    @patch('refinire_rag.processing.dictionary_maker.LLMPipeline')
    def test_extract_terms_with_llm_success(self, mock_llm_pipeline):
        """
        Test successful term extraction with LLM
        LLMでの用語抽出成功テスト
        """
        mock_result = Mock()
        mock_result.content = json.dumps({
            "has_new_terms": True,
            "new_terms_count": 2,
            "variations_count": 1,
            "extracted_terms": [
                {
                    "term": "RAG",
                    "full_form": "Retrieval-Augmented Generation",
                    "definition": "検索拡張生成",
                    "variations": ["検索拡張生成", "RAGシステム"]
                }
            ]
        })
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.return_value = mock_result
        mock_llm_pipeline.return_value = mock_pipeline_instance
        
        maker = DictionaryMaker()
        document = Document(id="test", content="Test content", metadata={})
        
        result = maker._extract_terms_with_llm(document, "existing dict", maker.config)
        
        assert result["has_new_terms"] is True
        assert result["new_terms_count"] == 2
        assert len(result["extracted_terms"]) == 1
        assert result["extracted_terms"][0]["term"] == "RAG"
    
    @patch('refinire_rag.processing.dictionary_maker.LLMPipeline')
    def test_extract_terms_with_llm_json_with_markers(self, mock_llm_pipeline):
        """
        Test term extraction with JSON wrapped in markdown markers
        マークダウンマーカーで囲まれたJSONでの用語抽出テスト
        """
        mock_result = Mock()
        mock_result.content = "```json\n{\"has_new_terms\": false, \"new_terms_count\": 0}\n```"
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.return_value = mock_result
        mock_llm_pipeline.return_value = mock_pipeline_instance
        
        maker = DictionaryMaker()
        document = Document(id="test", content="Test content", metadata={})
        
        result = maker._extract_terms_with_llm(document, "existing dict", maker.config)
        
        assert result["has_new_terms"] is False
        assert result["new_terms_count"] == 0
    
    @patch('refinire_rag.processing.dictionary_maker.LLMPipeline')
    def test_extract_terms_with_llm_invalid_json(self, mock_llm_pipeline):
        """
        Test term extraction with invalid JSON response
        無効JSON応答での用語抽出テスト
        """
        mock_result = Mock()
        mock_result.content = "Invalid JSON response"
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.return_value = mock_result
        mock_llm_pipeline.return_value = mock_pipeline_instance
        
        maker = DictionaryMaker()
        document = Document(id="test", content="Test content", metadata={})
        
        result = maker._extract_terms_with_llm(document, "existing dict", maker.config)
        
        assert result["has_new_terms"] is False
        assert result["new_terms_count"] == 0
        assert result["extracted_terms"] == []
    
    @patch('refinire_rag.processing.dictionary_maker.LLMPipeline')
    def test_extract_terms_with_llm_error(self, mock_llm_pipeline):
        """
        Test term extraction when LLM call fails
        LLM呼び出し失敗時の用語抽出テスト
        """
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.side_effect = Exception("LLM call failed")
        mock_llm_pipeline.return_value = mock_pipeline_instance
        
        maker = DictionaryMaker()
        document = Document(id="test", content="Test content", metadata={})
        
        result = maker._extract_terms_with_llm(document, "existing dict", maker.config)
        
        assert result["has_new_terms"] is False
        assert result["new_terms_count"] == 0
    
    def test_extract_terms_without_llm_pipeline(self):
        """
        Test term extraction when LLM pipeline is not available
        LLMパイプライン未利用時の用語抽出テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            document = Document(id="test", content="Test content", metadata={})
            
            result = maker._extract_terms_with_llm(document, "existing dict", maker.config)
            
            # Should return mock data
            assert result["has_new_terms"] is True
            assert result["new_terms_count"] == 3
            assert result["variations_count"] == 2
            assert len(result["extracted_terms"]) == 2
            assert result["extracted_terms"][0]["term"] == "RAG"


class TestDictionaryMakerTermManipulation:
    """
    Test DictionaryMaker term manipulation and formatting
    DictionaryMakerの用語操作とフォーマット処理テスト
    """
    
    def test_format_term_entry_with_full_form(self):
        """
        Test formatting term entry with full form
        完全形ありでの用語エントリフォーマットテスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            term_data = {
                "term": "RAG",
                "full_form": "Retrieval-Augmented Generation",
                "definition": "検索拡張生成",
                "variations": ["検索拡張生成", "RAGシステム"]
            }
            
            result = maker._format_term_entry(term_data)
            
            expected = "- **RAG** (Retrieval-Augmented Generation): 検索拡張生成\n  - 表現揺らぎ: 検索拡張生成, RAGシステム"
            assert result == expected
    
    def test_format_term_entry_without_full_form(self):
        """
        Test formatting term entry without full form
        完全形なしでの用語エントリフォーマットテスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            term_data = {
                "term": "ベクトル検索",
                "definition": "類似度に基づく検索手法",
                "variations": ["セマンティック検索"]
            }
            
            result = maker._format_term_entry(term_data)
            
            expected = "- **ベクトル検索**: 類似度に基づく検索手法\n  - 表現揺らぎ: セマンティック検索"
            assert result == expected
    
    def test_format_term_entry_no_variations(self):
        """
        Test formatting term entry without variations
        表現揺らぎなしでの用語エントリフォーマットテスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            term_data = {
                "term": "テスト用語",
                "definition": "テスト用の定義"
            }
            
            result = maker._format_term_entry(term_data)
            
            assert result == "- **テスト用語**: テスト用の定義"
    
    def test_merge_terms_into_dictionary(self):
        """
        Test merging extracted terms into existing dictionary
        既存辞書への抽出用語マージテスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            existing_dictionary = """# ドメイン用語辞書

## 専門用語

## 技術概念

---"""
            extracted_data = {
                "extracted_terms": [
                    {
                        "term": "RAG",
                        "definition": "検索拡張生成"
                    }
                ]
            }
            
            result = maker._merge_terms_into_dictionary(existing_dictionary, extracted_data)
            
            assert "- **RAG**: 検索拡張生成" in result
            assert "## 専門用語" in result
    
    def test_update_dictionary_file(self):
        """
        Test updating dictionary file with extracted terms
        抽出用語での辞書ファイル更新テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            with patch("builtins.open", mock_open()) as mock_file:
                with patch.object(maker, '_backup_dictionary_file'):
                    config = DictionaryMakerConfig(backup_dictionary=True)
                    extracted_data = {
                        "new_terms_count": 2,
                        "extracted_terms": [{"term": "test", "definition": "def"}]
                    }
                    
                    result = maker._update_dictionary_file("existing dict", extracted_data, config)
                    
                    mock_file.assert_called()
    
    def test_add_dictionary_metadata(self):
        """
        Test adding dictionary extraction metadata to document
        文書への辞書抽出メタデータ追加テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            document = Document(
                id="test_doc",
                content="Test content",
                metadata={"original_key": "original_value"}
            )
            extracted_data = {
                "new_terms_count": 3,
                "variations_count": 2,
                "reasoning": "Test reasoning"
            }
            config = DictionaryMakerConfig(dictionary_file_path="/test/dict.md")
            
            result = maker._add_dictionary_metadata(document, extracted_data, config)
            
            assert result.id == "test_doc"
            assert result.content == "Test content"
            assert result.metadata["original_key"] == "original_value"
            
            dict_meta = result.metadata["dictionary_metadata"]
            assert dict_meta["dictionary_extraction_applied"] is True
            assert dict_meta["new_terms_extracted"] == 3
            assert dict_meta["variations_detected"] == 2
            assert dict_meta["dictionary_file_path"] == "/test/dict.md"
            assert dict_meta["extracted_by"] == "DictionaryMaker"


class TestDictionaryMakerProcessing:
    """
    Test DictionaryMaker document processing workflow
    DictionaryMakerの文書処理ワークフローテスト
    """
    
    def test_process_document_success(self):
        """
        Test successful document processing
        文書処理成功テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            with patch.object(maker, '_read_existing_dictionary', return_value="existing dict"):
                with patch.object(maker, '_extract_terms_with_llm', return_value={
                    "has_new_terms": True,
                    "new_terms_count": 2,
                    "variations_count": 1
                }):
                    with patch.object(maker, '_update_dictionary_file', return_value="updated dict"):
                        document = Document(id="test", content="Test content", metadata={})
                        
                        result = maker.process(document)
                        
                        assert len(result) == 1
                        assert result[0].id == "test"
                        assert "dictionary_metadata" in result[0].metadata
                        
                        # Check statistics update
                        assert maker.processing_stats["documents_processed"] == 1
                        assert maker.processing_stats["terms_extracted"] == 2
                        assert maker.processing_stats["variations_detected"] == 1
    
    def test_process_document_no_new_terms_skip(self):
        """
        Test processing when no new terms found and configured to skip
        新用語なし・スキップ設定時の処理テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            config = DictionaryMakerConfig(skip_if_no_new_terms=True)
            maker = DictionaryMaker(config)
            
            with patch.object(maker, '_read_existing_dictionary', return_value="existing dict"):
                with patch.object(maker, '_extract_terms_with_llm', return_value={
                    "has_new_terms": False,
                    "new_terms_count": 0
                }):
                    document = Document(id="test", content="Test content", metadata={})
                    
                    result = maker.process(document)
                    
                    assert len(result) == 1
                    assert result[0] == document  # Should return original
                    assert maker.processing_stats["documents_processed"] == 1
    
    def test_process_document_no_metadata_update(self):
        """
        Test processing without metadata update
        メタデータ更新なしでの処理テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            config = DictionaryMakerConfig(update_document_metadata=False)
            maker = DictionaryMaker(config)
            
            with patch.object(maker, '_read_existing_dictionary', return_value="existing dict"):
                with patch.object(maker, '_extract_terms_with_llm', return_value={
                    "has_new_terms": True,
                    "new_terms_count": 1
                }):
                    with patch.object(maker, '_update_dictionary_file', return_value="updated dict"):
                        document = Document(id="test", content="Test content", metadata={})
                        
                        result = maker.process(document)
                        
                        assert len(result) == 1
                        assert result[0] == document  # Should return original without metadata
    
    def test_process_document_with_error(self):
        """
        Test processing when error occurs
        エラー発生時の処理テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            with patch.object(maker, '_read_existing_dictionary', side_effect=Exception("Test error")):
                document = Document(id="test", content="Test content", metadata={})
                
                result = maker.process(document)
                
                assert len(result) == 1
                assert result[0] == document  # Should return original on error


class TestDictionaryMakerUtilityMethods:
    """
    Test DictionaryMaker utility methods
    DictionaryMakerユーティリティメソッドテスト
    """
    
    def test_get_dictionary_path(self):
        """
        Test getting dictionary file path
        辞書ファイルパス取得テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            config = DictionaryMakerConfig(dictionary_file_path="/test/dict.md")
            maker = DictionaryMaker(config)
            
            result = maker.get_dictionary_path()
            
            assert result == "/test/dict.md"
    
    def test_get_dictionary_content(self):
        """
        Test getting dictionary content
        辞書内容取得テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            with patch.object(maker, '_read_existing_dictionary', return_value="test dictionary content"):
                result = maker.get_dictionary_content()
                
                assert result == "test dictionary content"
    
    def test_get_extraction_stats(self):
        """
        Test getting extraction statistics
        抽出統計取得テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            config = DictionaryMakerConfig(
                dictionary_file_path="/test/dict.md",
                llm_model="gpt-4",
                focus_on_technical_terms=True
            )
            maker = DictionaryMaker(config)
            
            # Set some processing stats
            maker.processing_stats["documents_processed"] = 5
            maker.processing_stats["terms_extracted"] = 15
            
            result = maker.get_extraction_stats()
            
            assert result["documents_processed"] == 5
            assert result["terms_extracted"] == 15
            assert result["dictionary_file"] == "/test/dict.md"
            assert result["llm_model"] == "gpt-4"
            assert result["focus_technical_terms"] is True


class TestDictionaryMakerEdgeCases:
    """
    Test DictionaryMaker edge cases and error scenarios
    DictionaryMakerエッジケースとエラーシナリオテスト
    """
    
    def test_format_term_entry_missing_fields(self):
        """
        Test formatting term entry with missing fields
        フィールド欠損での用語エントリフォーマットテスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            term_data = {
                "term": "Test"
                # Missing definition, variations
            }
            
            result = maker._format_term_entry(term_data)
            
            assert result == "- **Test**: "
    
    def test_merge_terms_unknown_section(self):
        """
        Test merging terms when sections are not found
        セクション見つからない時の用語マージテスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            existing_dictionary = "# Unknown Format Dictionary"
            extracted_data = {
                "extracted_terms": [{"term": "Test", "definition": "Definition"}]
            }
            
            result = maker._merge_terms_into_dictionary(existing_dictionary, extracted_data)
            
            # Should handle gracefully when sections not found
            assert isinstance(result, str)
    
    def test_process_with_config_override(self):
        """
        Test processing with configuration override
        設定オーバーライドでの処理テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            override_config = DictionaryMakerConfig(skip_if_no_new_terms=True)
            
            with patch.object(maker, '_read_existing_dictionary', return_value="dict"):
                with patch.object(maker, '_extract_terms_with_llm', return_value={
                    "has_new_terms": False
                }):
                    document = Document(id="test", content="content", metadata={})
                    
                    result = maker.process(document, override_config)
                    
                    assert len(result) == 1
                    assert result[0] == document
    
    def test_backup_dictionary_file_with_error(self):
        """
        Test backup dictionary file when error occurs
        エラー発生時の辞書ファイルバックアップテスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            with patch("pathlib.Path.exists", return_value=True):
                with patch("shutil.copy2", side_effect=OSError("Backup failed")):
                    config = DictionaryMakerConfig()
                    
                    # Should handle error gracefully
                    maker._backup_dictionary_file(config)
    
    def test_update_dictionary_file_with_error(self):
        """
        Test updating dictionary file when error occurs
        エラー発生時の辞書ファイル更新テスト
        """
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker()
            
            with patch("builtins.open", side_effect=OSError("File write error")):
                config = DictionaryMakerConfig(backup_dictionary=False)
                extracted_data = {"new_terms_count": 1}
                
                result = maker._update_dictionary_file("existing", extracted_data, config)
                
                # Should return original dictionary on error
                assert result == "existing"