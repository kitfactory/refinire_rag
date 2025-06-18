"""
Comprehensive tests for GraphBuilder processor
GraphBuilder プロセッサーの包括的テスト

This module provides comprehensive coverage for the GraphBuilder processing module,
testing knowledge graph construction, LLM integration, file operations, and edge cases.
このモジュールは、GraphBuilderプロセッシングモジュールの包括的カバレッジを提供し、
知識グラフ構築、LLM統合、ファイル操作、エッジケースをテストします。
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

from refinire_rag.processing.graph_builder import GraphBuilder, GraphBuilderConfig
from refinire_rag.models.document import Document


class TestGraphBuilderConfig:
    """
    Test GraphBuilderConfig configuration class
    GraphBuilderConfig設定クラスのテスト
    """
    
    def test_config_default_values(self):
        """
        Test default configuration values
        デフォルト設定値のテスト
        """
        config = GraphBuilderConfig()
        
        # Default file paths (converted to absolute in __post_init__)
        assert config.graph_file_path.endswith("domain_knowledge_graph.md")
        assert config.dictionary_file_path.endswith("domain_dictionary.md")
        assert config.backup_graph is True
        
        # LLM settings
        assert config.llm_temperature == 0.3
        assert config.max_tokens == 3000
        
        # Relationship extraction settings
        assert config.focus_on_important_relationships is True
        assert config.extract_hierarchical_relationships is True
        assert config.extract_causal_relationships is True
        assert config.extract_composition_relationships is True
        assert config.min_relationship_importance == "medium"
        
        # Processing settings
        assert config.use_dictionary_terms is True
        assert config.auto_detect_dictionary_path is True
        assert config.skip_if_no_new_relationships is False
        assert config.validate_extracted_relationships is True
        assert config.deduplicate_relationships is True
        
        # Output settings
        assert config.update_document_metadata is True
        assert config.preserve_original_document is True
    
    def test_config_custom_values(self):
        """
        Test configuration with custom values
        カスタム値での設定テスト
        """
        config = GraphBuilderConfig(
            graph_file_path="/custom/graph.md",
            dictionary_file_path="/custom/dict.md",
            backup_graph=False,
            llm_temperature=0.7,
            max_tokens=4000,
            focus_on_important_relationships=False,
            min_relationship_importance="high",
            skip_if_no_new_relationships=True,
            update_document_metadata=False
        )
        
        assert config.graph_file_path == "/custom/graph.md"
        assert config.dictionary_file_path == "/custom/dict.md"
        assert config.backup_graph is False
        assert config.llm_temperature == 0.7
        assert config.max_tokens == 4000
        assert config.focus_on_important_relationships is False
        assert config.min_relationship_importance == "high"
        assert config.skip_if_no_new_relationships is True
        assert config.update_document_metadata is False
    
    @patch('refinire_rag.processing.graph_builder.get_default_llm_model')
    def test_config_post_init_llm_model(self, mock_get_model):
        """
        Test __post_init__ setting default LLM model
        __post_init__でのデフォルトLLMモデル設定テスト
        """
        mock_get_model.return_value = "gpt-4"
        
        config = GraphBuilderConfig()
        
        assert config.llm_model == "gpt-4"
        mock_get_model.assert_called_once()
    
    @patch('os.path.isabs')
    @patch('os.path.abspath')
    def test_config_post_init_absolute_paths(self, mock_abspath, mock_isabs):
        """
        Test __post_init__ converting paths to absolute
        __post_init__でのパス絶対化テスト
        """
        mock_isabs.side_effect = [False, False]  # Both paths are relative
        mock_abspath.side_effect = ["/abs/graph.md", "/abs/dict.md"]
        
        config = GraphBuilderConfig(
            graph_file_path="./graph.md",
            dictionary_file_path="./dict.md"
        )
        
        assert config.graph_file_path == "/abs/graph.md"
        assert config.dictionary_file_path == "/abs/dict.md"
        assert mock_abspath.call_count == 2
    
    @patch('os.path.isabs')
    def test_config_post_init_already_absolute_paths(self, mock_isabs):
        """
        Test __post_init__ with already absolute paths
        既に絶対パスでの__post_init__テスト
        """
        mock_isabs.return_value = True  # Paths are already absolute
        
        config = GraphBuilderConfig(
            graph_file_path="/abs/graph.md",
            dictionary_file_path="/abs/dict.md"
        )
        
        assert config.graph_file_path == "/abs/graph.md"
        assert config.dictionary_file_path == "/abs/dict.md"


class TestGraphBuilderInitialization:
    """
    Test GraphBuilder initialization and setup
    GraphBuilderの初期化とセットアップのテスト
    """
    
    def test_init_with_default_config(self):
        """
        Test initialization with default configuration
        デフォルト設定での初期化テスト
        """
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder()
            
            assert isinstance(builder.config, GraphBuilderConfig)
            assert builder._llm_pipeline is None
            assert "documents_processed" in builder.processing_stats
            assert "relationships_extracted" in builder.processing_stats
            assert "graph_updates" in builder.processing_stats
            assert "llm_api_calls" in builder.processing_stats
            assert "duplicate_relationships_avoided" in builder.processing_stats
    
    def test_init_with_custom_config(self):
        """
        Test initialization with custom configuration
        カスタム設定での初期化テスト
        """
        custom_config = GraphBuilderConfig(
            graph_file_path="/custom/path.md",
            llm_temperature=0.5
        )
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(custom_config)
            
            assert builder.config.graph_file_path == "/custom/path.md"
            assert builder.config.llm_temperature == 0.5
    
    @patch('refinire_rag.processing.graph_builder.LLMPipeline')
    def test_init_with_refinire_available(self, mock_llm_pipeline_class):
        """
        Test initialization when Refinire is available
        Refinire利用可能時の初期化テスト
        """
        mock_pipeline = Mock()
        mock_llm_pipeline_class.return_value = mock_pipeline
        
        builder = GraphBuilder()
        
        assert builder._llm_pipeline == mock_pipeline
        mock_llm_pipeline_class.assert_called_once_with(
            name="graph_builder",
            generation_instructions="You are a knowledge graph expert that extracts important relationships from documents.",
            model=builder.config.llm_model
        )
    
    @patch('refinire_rag.processing.graph_builder.LLMPipeline')
    def test_init_with_refinire_initialization_error(self, mock_llm_pipeline_class):
        """
        Test initialization when Refinire LLMPipeline fails
        Refinire LLMPipeline初期化失敗時のテスト
        """
        mock_llm_pipeline_class.side_effect = Exception("Pipeline init failed")
        
        builder = GraphBuilder()
        
        assert builder._llm_pipeline is None
    
    def test_get_config_class(self):
        """
        Test get_config_class method
        get_config_classメソッドのテスト
        """
        assert GraphBuilder.get_config_class() == GraphBuilderConfig


class TestGraphBuilderFileOperations:
    """
    Test GraphBuilder file operations
    GraphBuilderファイル操作のテスト
    """
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.graph_path = os.path.join(self.temp_dir, "test_graph.md")
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_read_existing_graph_file_exists(self):
        """
        Test reading existing graph file
        既存グラフファイル読み込みテスト
        """
        self.setUp()
        try:
            # Create test graph file
            test_content = "# Test Graph\n## Relations\n- Test relation"
            with open(self.graph_path, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            config = GraphBuilderConfig(graph_file_path=self.graph_path)
            with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
                builder = GraphBuilder(config)
                
                result = builder._read_existing_graph(config)
                
                assert result == test_content
        finally:
            self.tearDown()
    
    def test_read_existing_graph_file_not_exists(self):
        """
        Test reading graph when file doesn't exist
        ファイル存在しない場合のグラフ読み込みテスト
        """
        self.setUp()
        try:
            config = GraphBuilderConfig(graph_file_path=self.graph_path)
            with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
                builder = GraphBuilder(config)
                
                result = builder._read_existing_graph(config)
                
                # Should create new file with template
                assert "# ドメイン知識グラフ" in result
                assert os.path.exists(self.graph_path)
        finally:
            self.tearDown()
    
    def test_read_existing_graph_file_error(self):
        """
        Test reading graph file with I/O error
        ファイルI/Oエラー時のグラフ読み込みテスト
        """
        config = GraphBuilderConfig(graph_file_path="/invalid/path/graph.md")
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            
            result = builder._read_existing_graph(config)
            
            # Should return template on error
            assert "# ドメイン知識グラフ" in result
    
    def test_create_empty_graph_template(self):
        """
        Test creating empty graph template
        空グラフテンプレート作成テスト
        """
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder()
            
            template = builder._create_empty_graph_template()
            
            assert "# ドメイン知識グラフ" in template
            assert "### 主要概念" in template
            assert "### 技術関係" in template
            assert "### 機能関係" in template
            assert "### 評価関係" in template
    
    def test_read_dictionary_content_from_document_metadata(self):
        """
        Test reading dictionary from document metadata
        文書メタデータからの辞書読み込みテスト
        """
        self.setUp()
        try:
            # Create test dictionary file
            dict_path = os.path.join(self.temp_dir, "test_dict.md")
            dict_content = "# Test Dictionary\n- term1: definition1"
            with open(dict_path, 'w', encoding='utf-8') as f:
                f.write(dict_content)
            
            # Create document with dictionary metadata
            document = Document(
                id="test-doc",
                content="Test content",
                metadata={
                    "dictionary_metadata": {
                        "dictionary_file_path": dict_path
                    }
                }
            )
            
            config = GraphBuilderConfig(auto_detect_dictionary_path=True)
            with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
                builder = GraphBuilder(config)
                
                result = builder._read_dictionary_content(document, config)
                
                assert result == dict_content
        finally:
            self.tearDown()
    
    def test_read_dictionary_content_from_config_path(self):
        """
        Test reading dictionary from config path
        設定パスからの辞書読み込みテスト
        """
        self.setUp()
        try:
            # Create test dictionary file
            dict_path = os.path.join(self.temp_dir, "config_dict.md")
            dict_content = "# Config Dictionary\n- term2: definition2"
            with open(dict_path, 'w', encoding='utf-8') as f:
                f.write(dict_content)
            
            document = Document(id="test-doc", content="Test content", metadata={})
            config = GraphBuilderConfig(dictionary_file_path=dict_path)
            
            with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
                builder = GraphBuilder(config)
                
                result = builder._read_dictionary_content(document, config)
                
                assert result == dict_content
        finally:
            self.tearDown()
    
    def test_read_dictionary_content_no_file(self):
        """
        Test reading dictionary when no file exists
        辞書ファイル存在しない場合のテスト
        """
        document = Document(id="test-doc", content="Test content", metadata={})
        config = GraphBuilderConfig(dictionary_file_path="/nonexistent/path.md")
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            
            result = builder._read_dictionary_content(document, config)
            
            assert result == ""
    
    def test_backup_graph_file(self):
        """
        Test backing up graph file
        グラフファイルバックアップテスト
        """
        self.setUp()
        try:
            # Create test graph file
            with open(self.graph_path, 'w', encoding='utf-8') as f:
                f.write("# Test Graph")
            
            config = GraphBuilderConfig(graph_file_path=self.graph_path)
            with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
                builder = GraphBuilder(config)
                
                with patch('datetime.datetime') as mock_datetime:
                    mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
                    
                    builder._backup_graph_file(config)
                    
                    # Check backup file was created
                    backup_path = self.graph_path.replace('.md', '.backup_20240101_120000.md')
                    assert os.path.exists(backup_path)
        finally:
            self.tearDown()
    
    def test_backup_graph_file_no_existing_file(self):
        """
        Test backing up when no graph file exists
        グラフファイル存在しない場合のバックアップテスト
        """
        config = GraphBuilderConfig(graph_file_path="/nonexistent/graph.md")
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            
            # Should not raise error
            builder._backup_graph_file(config)


class TestGraphBuilderLLMIntegration:
    """
    Test GraphBuilder LLM integration
    GraphBuilder LLM統合のテスト
    """
    
    def test_create_relationship_extraction_prompt(self):
        """
        Test creating relationship extraction prompt
        関係抽出プロンプト作成テスト
        """
        document = Document(
            id="test-doc",
            content="RAG システムは検索機能と生成機能を含む。",
            metadata={"title": "RAG System Overview"}
        )
        existing_graph = "# Existing Graph\n## Relations"
        dictionary_content = "# Dictionary\n- RAG: Retrieval-Augmented Generation"
        config = GraphBuilderConfig(
            min_relationship_importance="high",
            focus_on_important_relationships=True,
            extract_hierarchical_relationships=True
        )
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            
            prompt = builder._create_relationship_extraction_prompt(
                document, existing_graph, dictionary_content, config
            )
            
            assert "RAG システムは検索機能と生成機能を含む。" in prompt
            assert "RAG System Overview" in prompt
            assert "# Existing Graph" in prompt
            assert "# Dictionary" in prompt
            assert "重要度レベル: high" in prompt
            assert "重要関係重視: True" in prompt
            assert "階層関係抽出: True" in prompt
            assert "JSON形式で回答してください" in prompt
    
    def test_extract_relationships_with_llm_success(self):
        """
        Test successful LLM relationship extraction
        LLM関係抽出成功テスト
        """
        document = Document(id="test-doc", content="Test content", metadata={})
        existing_graph = "# Graph"
        dictionary_content = "# Dict"
        config = GraphBuilderConfig()
        
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.content = json.dumps({
            "has_new_relationships": True,
            "new_relationships_count": 2,
            "duplicates_avoided": 1,
            "extracted_relationships": [
                {
                    "subject": "RAG",
                    "predicate": "含む",
                    "object": "検索機能",
                    "category": "機能関係",
                    "importance": "high"
                }
            ],
            "reasoning": "Important relationships found"
        })
        mock_pipeline.run.return_value = mock_result
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            builder._llm_pipeline = mock_pipeline
            
            result = builder._extract_relationships_with_llm(
                document, existing_graph, dictionary_content, config
            )
            
            assert result["has_new_relationships"] is True
            assert result["new_relationships_count"] == 2
            assert result["duplicates_avoided"] == 1
            assert len(result["extracted_relationships"]) == 1
            assert result["extracted_relationships"][0]["subject"] == "RAG"
    
    def test_extract_relationships_with_llm_json_with_markers(self):
        """
        Test LLM response with JSON markdown markers
        JSONマークダウンマーカー付きLLM応答テスト
        """
        document = Document(id="test-doc", content="Test content", metadata={})
        config = GraphBuilderConfig()
        
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.content = '''```json
{
    "has_new_relationships": true,
    "new_relationships_count": 1,
    "duplicates_avoided": 0,
    "extracted_relationships": [],
    "reasoning": "Test"
}
```'''
        mock_pipeline.run.return_value = mock_result
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            builder._llm_pipeline = mock_pipeline
            
            result = builder._extract_relationships_with_llm(
                document, "", "", config
            )
            
            assert result["has_new_relationships"] is True
            assert result["new_relationships_count"] == 1
    
    def test_extract_relationships_with_llm_invalid_json(self):
        """
        Test LLM response with invalid JSON
        無効JSON付きLLM応答テスト
        """
        document = Document(id="test-doc", content="Test content", metadata={})
        config = GraphBuilderConfig()
        
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.content = "Invalid JSON response"
        mock_pipeline.run.return_value = mock_result
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            builder._llm_pipeline = mock_pipeline
            
            result = builder._extract_relationships_with_llm(
                document, "", "", config
            )
            
            assert result["has_new_relationships"] is False
            assert result["new_relationships_count"] == 0
    
    def test_extract_relationships_with_llm_error(self):
        """
        Test LLM call error handling
        LLM呼び出しエラー処理テスト
        """
        document = Document(id="test-doc", content="Test content", metadata={})
        config = GraphBuilderConfig()
        
        mock_pipeline = Mock()
        mock_pipeline.run.side_effect = Exception("LLM API error")
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            builder._llm_pipeline = mock_pipeline
            
            result = builder._extract_relationships_with_llm(
                document, "", "", config
            )
            
            assert result["has_new_relationships"] is False
            assert result["new_relationships_count"] == 0
    
    def test_extract_relationships_without_llm_pipeline(self):
        """
        Test relationship extraction without LLM pipeline (mock data)
        LLMパイプラインなしでの関係抽出テスト（モックデータ）
        """
        document = Document(id="test-doc", content="Test content", metadata={})
        config = GraphBuilderConfig()
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            # _llm_pipeline is None
            
            result = builder._extract_relationships_with_llm(
                document, "", "", config
            )
            
            assert result["has_new_relationships"] is True
            assert result["new_relationships_count"] == 4
            assert result["duplicates_avoided"] == 1
            assert len(result["extracted_relationships"]) == 4
            assert result["extracted_relationships"][0]["subject"] == "RAG システム"


class TestGraphBuilderGraphManipulation:
    """
    Test GraphBuilder graph manipulation methods
    GraphBuilderグラフ操作メソッドのテスト
    """
    
    def test_format_relationship_entry_high_importance(self):
        """
        Test formatting relationship entry with high importance
        高重要度関係エントリフォーマットテスト
        """
        rel_data = {
            "subject": "RAG システム",
            "predicate": "含む",
            "object": "検索機能",
            "importance": "high"
        }
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder()
            
            result = builder._format_relationship_entry(rel_data)
            
            assert result == "- **RAG システム → 含む → 検索機能**"
    
    def test_format_relationship_entry_medium_importance(self):
        """
        Test formatting relationship entry with medium importance
        中重要度関係エントリフォーマットテスト
        """
        rel_data = {
            "subject": "ベクトル検索",
            "predicate": "基づく",
            "object": "類似度計算",
            "importance": "medium"
        }
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder()
            
            result = builder._format_relationship_entry(rel_data)
            
            assert result == "- *ベクトル検索 → 基づく → 類似度計算*"
    
    def test_format_relationship_entry_low_importance(self):
        """
        Test formatting relationship entry with low importance
        低重要度関係エントリフォーマットテスト
        """
        rel_data = {
            "subject": "システム",
            "predicate": "持つ",
            "object": "機能",
            "importance": "low"
        }
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder()
            
            result = builder._format_relationship_entry(rel_data)
            
            assert result == "- システム → 持つ → 機能"
    
    def test_merge_relationships_into_graph(self):
        """
        Test merging relationships into existing graph
        既存グラフへの関係マージテスト
        """
        existing_graph = """# ドメイン知識グラフ

## エンティティ関係

### 主要概念

### 技術関係

### 機能関係

### 評価関係

---"""
        
        extracted_data = {
            "extracted_relationships": [
                {
                    "subject": "RAG",
                    "predicate": "含む",
                    "object": "検索",
                    "category": "機能関係",
                    "importance": "high"
                },
                {
                    "subject": "ベクトル",
                    "predicate": "使用",
                    "object": "埋め込み",
                    "category": "技術関係",
                    "importance": "medium"
                }
            ]
        }
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder()
            
            result = builder._merge_relationships_into_graph(existing_graph, extracted_data)
            
            assert "- **RAG → 含む → 検索**" in result
            assert "- *ベクトル → 使用 → 埋め込み*" in result
    
    def test_update_graph_file(self):
        """
        Test updating graph file
        グラフファイル更新テスト
        """
        temp_dir = tempfile.mkdtemp()
        try:
            graph_path = os.path.join(temp_dir, "test_graph.md")
            
            # Create initial graph file
            initial_content = "# Initial Graph\n### 機能関係\n"
            with open(graph_path, 'w', encoding='utf-8') as f:
                f.write(initial_content)
            
            config = GraphBuilderConfig(
                graph_file_path=graph_path,
                backup_graph=False
            )
            
            extracted_data = {
                "new_relationships_count": 1,
                "extracted_relationships": [
                    {
                        "subject": "Test",
                        "predicate": "has",
                        "object": "relation",
                        "category": "機能関係",
                        "importance": "medium"
                    }
                ]
            }
            
            with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
                builder = GraphBuilder(config)
                
                result = builder._update_graph_file(initial_content, extracted_data, config)
                
                assert "Test → has → relation" in result
                
                # Verify file was updated
                with open(graph_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                assert "Test → has → relation" in file_content
                
        finally:
            shutil.rmtree(temp_dir)
    
    def test_add_graph_metadata(self):
        """
        Test adding graph metadata to document
        文書へのグラフメタデータ追加テスト
        """
        document = Document(
            id="test-doc",
            content="Test content",
            metadata={"title": "Test Document"}
        )
        
        extracted_data = {
            "new_relationships_count": 3,
            "duplicates_avoided": 1,
            "reasoning": "Found important relationships"
        }
        
        config = GraphBuilderConfig(graph_file_path="/test/graph.md")
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            
            result = builder._add_graph_metadata(document, extracted_data, config)
            
            assert result.id == "test-doc"
            assert result.content == "Test content"
            assert result.metadata["title"] == "Test Document"
            assert result.metadata["processing_stage"] == "graph_analyzed"
            
            graph_metadata = result.metadata["graph_metadata"]
            assert graph_metadata["graph_extraction_applied"] is True
            assert graph_metadata["new_relationships_extracted"] == 3
            assert graph_metadata["duplicates_avoided"] == 1
            assert graph_metadata["extraction_reasoning"] == "Found important relationships"
            assert graph_metadata["graph_file_path"] == "/test/graph.md"
            assert graph_metadata["extracted_by"] == "GraphBuilder"


class TestGraphBuilderProcessing:
    """
    Test GraphBuilder document processing
    GraphBuilder文書処理のテスト
    """
    
    def test_process_document_success(self):
        """
        Test successful document processing
        文書処理成功テスト
        """
        document = Document(
            id="test-doc",
            content="RAG システムの概要",
            metadata={"title": "RAG Overview"}
        )
        
        mock_extracted_data = {
            "has_new_relationships": True,
            "new_relationships_count": 2,
            "duplicates_avoided": 0,
            "extracted_relationships": [],
            "reasoning": "Test"
        }
        
        config = GraphBuilderConfig(update_document_metadata=True)
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            
            with patch.object(builder, '_read_existing_graph', return_value="# Graph"):
                with patch.object(builder, '_read_dictionary_content', return_value=""):
                    with patch.object(builder, '_extract_relationships_with_llm', return_value=mock_extracted_data):
                        with patch.object(builder, '_update_graph_file', return_value="# Updated Graph"):
                            
                            result = builder.process(document)
                            
                            assert len(result) == 1
                            assert result[0].id == "test-doc"
                            assert "graph_metadata" in result[0].metadata
                            assert builder.processing_stats["documents_processed"] == 1
                            assert builder.processing_stats["relationships_extracted"] == 2
                            assert builder.processing_stats["graph_updates"] == 1
                            assert builder.processing_stats["llm_api_calls"] == 1
    
    def test_process_document_no_new_relationships_skip(self):
        """
        Test processing when no new relationships and skip enabled
        新関係なし・スキップ有効時の処理テスト
        """
        document = Document(id="test-doc", content="Test", metadata={})
        
        mock_extracted_data = {
            "has_new_relationships": False,
            "new_relationships_count": 0
        }
        
        config = GraphBuilderConfig(skip_if_no_new_relationships=True)
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            
            with patch.object(builder, '_read_existing_graph', return_value="# Graph"):
                with patch.object(builder, '_read_dictionary_content', return_value=""):
                    with patch.object(builder, '_extract_relationships_with_llm', return_value=mock_extracted_data):
                        
                        result = builder.process(document, config)
                        
                        assert len(result) == 1
                        assert result[0] == document  # Original document returned
                        assert builder.processing_stats["documents_processed"] == 1
                        assert builder.processing_stats["graph_updates"] == 0
    
    def test_process_document_no_metadata_update(self):
        """
        Test processing without metadata update
        メタデータ更新なしでの処理テスト
        """
        document = Document(id="test-doc", content="Test", metadata={})
        
        mock_extracted_data = {
            "has_new_relationships": True,
            "new_relationships_count": 1,
            "extracted_relationships": []
        }
        
        config = GraphBuilderConfig(update_document_metadata=False)
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            
            with patch.object(builder, '_read_existing_graph', return_value="# Graph"):
                with patch.object(builder, '_read_dictionary_content', return_value=""):
                    with patch.object(builder, '_extract_relationships_with_llm', return_value=mock_extracted_data):
                        with patch.object(builder, '_update_graph_file', return_value="# Updated"):
                            
                            result = builder.process(document)
                            
                            assert len(result) == 1
                            assert result[0] == document  # Original document returned
    
    def test_process_document_with_error(self):
        """
        Test processing with error handling
        エラー処理付き文書処理テスト
        """
        document = Document(id="test-doc", content="Test", metadata={})
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder()
            
            with patch.object(builder, '_read_existing_graph', side_effect=Exception("File error")):
                
                result = builder.process(document)
                
                assert len(result) == 1
                assert result[0] == document  # Original document returned on error


class TestGraphBuilderUtilityMethods:
    """
    Test GraphBuilder utility methods
    GraphBuilderユーティリティメソッドのテスト
    """
    
    def test_get_graph_path(self):
        """
        Test getting graph file path
        グラフファイルパス取得テスト
        """
        config = GraphBuilderConfig(graph_file_path="/custom/path.md")
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            
            result = builder.get_graph_path()
            
            assert result == "/custom/path.md"
    
    def test_get_graph_content(self):
        """
        Test getting graph content
        グラフ内容取得テスト
        """
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder()
            
            with patch.object(builder, '_read_existing_graph', return_value="# Test Graph Content"):
                
                result = builder.get_graph_content()
                
                assert result == "# Test Graph Content"
    
    def test_get_graph_stats(self):
        """
        Test getting graph statistics
        グラフ統計取得テスト
        """
        config = GraphBuilderConfig(
            graph_file_path="/test/graph.md",
            dictionary_file_path="/test/dict.md",
            llm_model="gpt-4",
            focus_on_important_relationships=True
        )
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            
            # Set some processing stats
            builder.processing_stats["documents_processed"] = 5
            builder.processing_stats["relationships_extracted"] = 20
            
            result = builder.get_graph_stats()
            
            assert result["documents_processed"] == 5
            assert result["relationships_extracted"] == 20
            assert result["graph_file"] == "/test/graph.md"
            assert result["dictionary_file"] == "/test/dict.md"
            assert result["llm_model"] == "gpt-4"
            assert result["focus_important_relationships"] is True


class TestGraphBuilderEdgeCases:
    """
    Test GraphBuilder edge cases and error scenarios
    GraphBuilderエッジケースとエラーシナリオのテスト
    """
    
    def test_config_with_none_dictionary_path(self):
        """
        Test configuration with None dictionary path
        辞書パスNullでの設定テスト
        """
        config = GraphBuilderConfig(dictionary_file_path=None)
        document = Document(id="test", content="test", metadata={})
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            
            result = builder._read_dictionary_content(document, config)
            
            assert result == ""
    
    def test_merge_relationships_unknown_category(self):
        """
        Test merging relationships with unknown category
        未知カテゴリでの関係マージテスト
        """
        existing_graph = "# Graph\n### 主要概念\n"
        
        extracted_data = {
            "extracted_relationships": [
                {
                    "subject": "Test",
                    "predicate": "has",
                    "object": "unknown",
                    "category": "未知カテゴリ",
                    "importance": "medium"
                }
            ]
        }
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder()
            
            result = builder._merge_relationships_into_graph(existing_graph, extracted_data)
            
            # Should still work, relationship just won't be added to any section
            assert "# Graph" in result
    
    def test_format_relationship_entry_missing_fields(self):
        """
        Test formatting relationship entry with missing fields
        フィールド欠損での関係エントリフォーマットテスト
        """
        rel_data = {
            "subject": "Test"
            # Missing predicate, object, importance
        }
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder()
            
            result = builder._format_relationship_entry(rel_data)
            
            assert result == "- *Test →  → *"
    
    def test_process_with_config_override(self):
        """
        Test processing with configuration override
        設定オーバーライドでの処理テスト
        """
        document = Document(id="test", content="test", metadata={})
        builder_config = GraphBuilderConfig(skip_if_no_new_relationships=False)
        override_config = GraphBuilderConfig(skip_if_no_new_relationships=True)
        
        mock_extracted_data = {
            "has_new_relationships": False,
            "new_relationships_count": 0
        }
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(builder_config)
            
            with patch.object(builder, '_read_existing_graph', return_value="# Graph"):
                with patch.object(builder, '_read_dictionary_content', return_value=""):
                    with patch.object(builder, '_extract_relationships_with_llm', return_value=mock_extracted_data):
                        
                        result = builder.process(document, override_config)
                        
                        # Should use override config and skip processing
                        assert len(result) == 1
                        assert result[0] == document
    
    def test_backup_graph_file_with_error(self):
        """
        Test backup graph file with error
        エラー付きグラフファイルバックアップテスト
        """
        config = GraphBuilderConfig(graph_file_path="/invalid/path.md")
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            
            # Should not raise error, just log warning
            builder._backup_graph_file(config)
    
    def test_update_graph_file_with_error(self):
        """
        Test update graph file with error
        エラー付きグラフファイル更新テスト
        """
        config = GraphBuilderConfig(graph_file_path="/invalid/path.md")
        extracted_data = {"new_relationships_count": 1}
        existing_graph = "# Graph"
        
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config)
            
            result = builder._update_graph_file(existing_graph, extracted_data, config)
            
            # Should return original graph on error
            assert result == existing_graph