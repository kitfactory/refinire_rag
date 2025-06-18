"""
Comprehensive tests for CorpusManager core methods
CorpusManagerのコアメソッドの包括的テスト

This module tests the main functionality of CorpusManager including import, rebuild, and corpus management.
このモジュールは、CorpusManagerのインポート、リビルド、コーパス管理の主要機能をテストします。
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from refinire_rag.application.corpus_manager_new import CorpusManager, CorpusStats
from refinire_rag.models.document import Document
from refinire_rag.storage.document_store import DocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore


class TestCorpusManagerMethods:
    """
    Test CorpusManager core methods functionality
    CorpusManagerのコアメソッド機能のテスト
    """

    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        # Create mock components
        self.mock_document_store = Mock(spec=DocumentStore)
        self.mock_vector_store = Mock(spec=InMemoryVectorStore)
        self.mock_retrievers = [self.mock_vector_store]
        
        # Create CorpusManager instance
        self.corpus_manager = CorpusManager(
            document_store=self.mock_document_store,
            retrievers=self.mock_retrievers
        )

    def test_get_corpus_info_empty_corpus(self):
        """
        Test get_corpus_info with empty corpus
        空のコーパスでのget_corpus_info テスト
        """
        # Setup mocks for empty corpus
        self.mock_document_store.count_documents.return_value = 0
        self.mock_document_store.list_documents.return_value = []
        
        # Call get_corpus_info
        info = self.corpus_manager.get_corpus_info()
        
        # Verify results
        assert info['total_documents'] == 0
        assert info['document_types'] == {}
        assert info['sources'] == {}
        assert 'corpus_stats' in info
        assert info['retrievers'] != []  # Should have retriever info

    def test_get_corpus_info_with_documents(self):
        """
        Test get_corpus_info with documents in corpus
        コーパス内にドキュメントがある状態でのget_corpus_info テスト
        """
        # Create sample documents
        docs = [
            Document(id="1", content="test1", metadata={"file_type": "txt", "source": "file1.txt"}),
            Document(id="2", content="test2", metadata={"file_type": "md", "source": "file2.md"}),
            Document(id="3", content="test3", metadata={"file_type": "txt", "source": "file3.txt"})
        ]
        
        # Setup mocks
        self.mock_document_store.get_document_count.return_value = 3
        self.mock_document_store.list_documents.return_value = docs
        
        # Call get_corpus_info
        info = self.corpus_manager.get_corpus_info()
        
        # Verify results
        assert info['total_documents'] == 3
        assert info['document_types']['txt'] == 2
        assert info['document_types']['md'] == 1
        assert info['sources']['file1.txt'] == 1
        assert info['sources']['file2.md'] == 1
        assert info['sources']['file3.txt'] == 1

    def test_clear_corpus(self):
        """
        Test clear_corpus functionality
        clear_corpus機能のテスト
        """
        # Call clear_corpus
        self.corpus_manager.clear_corpus()
        
        # Verify document store was cleared
        self.mock_document_store.clear_all_documents.assert_called_once()
        
        # Verify all retrievers were cleared
        for retriever in self.mock_retrievers:
            retriever.clear_all_embeddings.assert_called_once()

    def test_add_retriever(self):
        """
        Test add_retriever functionality
        add_retriever機能のテスト
        """
        # Create new mock retriever
        new_retriever = Mock()
        initial_count = len(self.corpus_manager.retrievers)
        
        # Add retriever
        self.corpus_manager.add_retriever(new_retriever)
        
        # Verify retriever was added
        assert len(self.corpus_manager.retrievers) == initial_count + 1
        assert new_retriever in self.corpus_manager.retrievers

    def test_remove_retriever_valid_index(self):
        """
        Test remove_retriever with valid index
        有効なインデックスでのremove_retrieverテスト
        """
        # Add extra retriever for removal test
        extra_retriever = Mock()
        self.corpus_manager.retrievers.append(extra_retriever)
        initial_count = len(self.corpus_manager.retrievers)
        
        # Remove retriever
        result = self.corpus_manager.remove_retriever(1)  # Remove second retriever
        
        # Verify removal
        assert result is True
        assert len(self.corpus_manager.retrievers) == initial_count - 1
        assert extra_retriever not in self.corpus_manager.retrievers

    def test_remove_retriever_invalid_index(self):
        """
        Test remove_retriever with invalid index
        無効なインデックスでのremove_retrieverテスト
        """
        initial_count = len(self.corpus_manager.retrievers)
        
        # Try to remove with invalid index
        result = self.corpus_manager.remove_retriever(999)
        
        # Verify no change
        assert result is False
        assert len(self.corpus_manager.retrievers) == initial_count

    def test_get_retrievers_by_type(self):
        """
        Test get_retrievers_by_type functionality
        get_retrievers_by_type機能のテスト
        """
        # Add retrievers of different types
        vector_store = Mock()
        vector_store.__class__.__name__ = "InMemoryVectorStore"
        keyword_store = Mock()
        keyword_store.__class__.__name__ = "KeywordStore"
        
        self.corpus_manager.retrievers = [vector_store, keyword_store]
        
        # Get retrievers by type
        vector_retrievers = self.corpus_manager.get_retrievers_by_type("VectorStore")
        keyword_retrievers = self.corpus_manager.get_retrievers_by_type("KeywordStore")
        nonexistent = self.corpus_manager.get_retrievers_by_type("NonExistent")
        
        # Verify results
        assert len(vector_retrievers) == 1
        assert vector_store in vector_retrievers
        assert len(keyword_retrievers) == 1
        assert keyword_store in keyword_retrievers
        assert len(nonexistent) == 0

    def test_get_vector_store_from_retrievers(self):
        """
        Test _get_vector_store_from_retrievers functionality
        _get_vector_store_from_retrievers機能のテスト
        """
        # Create mock retrievers
        vector_store = Mock()
        vector_store.__class__.__name__ = "InMemoryVectorStore"
        other_retriever = Mock()
        other_retriever.__class__.__name__ = "KeywordStore"
        
        self.corpus_manager.retrievers = [other_retriever, vector_store]
        
        # Get vector store
        result = self.corpus_manager._get_vector_store_from_retrievers()
        
        # Verify correct vector store was returned
        assert result == vector_store

    def test_get_vector_store_from_retrievers_none_found(self):
        """
        Test _get_vector_store_from_retrievers when no vector store exists
        ベクターストアが存在しない場合の_get_vector_store_from_retrieversテスト
        """
        # Create mock retrievers without vector store
        other_retriever = Mock()
        other_retriever.__class__.__name__ = "KeywordStore"
        
        self.corpus_manager.retrievers = [other_retriever]
        
        # Get vector store
        result = self.corpus_manager._get_vector_store_from_retrievers()
        
        # Verify None was returned
        assert result is None

    @patch('refinire_rag.application.corpus_manager_new.Path.exists')
    @patch('refinire_rag.application.corpus_manager_new.Path.mkdir')
    def test_get_refinire_rag_dir_default(self, mock_mkdir, mock_exists):
        """
        Test _get_refinire_rag_dir with default directory
        デフォルトディレクトリでの_get_refinire_rag_dirテスト
        """
        # Setup mocks
        mock_exists.return_value = False
        
        # Call method
        result = CorpusManager._get_refinire_rag_dir()
        
        # Verify result
        expected_path = Path('./refinire')
        assert result == expected_path
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch.dict(os.environ, {'REFINIRE_DIR': '/custom/path'})
    @patch('refinire_rag.application.corpus_manager_new.Path.exists')
    @patch('refinire_rag.application.corpus_manager_new.Path.mkdir')
    def test_get_refinire_rag_dir_custom(self, mock_mkdir, mock_exists):
        """
        Test _get_refinire_rag_dir with custom directory from environment
        環境変数からのカスタムディレクトリでの_get_refinire_rag_dirテスト
        """
        # Setup mocks
        mock_exists.return_value = False
        
        # Call method
        result = CorpusManager._get_refinire_rag_dir()
        
        # Verify result
        expected_path = Path('/custom/path')
        assert result == expected_path
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_get_corpus_file_path(self):
        """
        Test _get_corpus_file_path functionality
        _get_corpus_file_path機能のテスト
        """
        with patch('refinire_rag.application.corpus_manager_new.CorpusManager._get_refinire_rag_dir') as mock_get_dir:
            mock_get_dir.return_value = Path('/test/refinire')
            
            # Test different file types
            track_path = CorpusManager._get_corpus_file_path('mycorpus', 'track')
            dict_path = CorpusManager._get_corpus_file_path('mycorpus', 'dictionary')
            kg_path = CorpusManager._get_corpus_file_path('mycorpus', 'knowledge_graph')
            
            # Verify paths
            assert track_path == Path('/test/refinire/mycorpus_track.json')
            assert dict_path == Path('/test/refinire/mycorpus_dictionary.md')
            assert kg_path == Path('/test/refinire/mycorpus_knowledge_graph.md')

    def test_get_corpus_file_path_custom_directory(self):
        """
        Test _get_corpus_file_path with custom directory
        カスタムディレクトリでの_get_corpus_file_path テスト
        """
        # Test with custom directory
        track_path = CorpusManager._get_corpus_file_path('mycorpus', 'track', '/custom/dir')
        
        # Verify path
        assert track_path == Path('/custom/dir/mycorpus_track.json')

    def test_get_default_output_directory(self):
        """
        Test _get_default_output_directory functionality
        _get_default_output_directory機能のテスト
        """
        with patch('refinire_rag.application.corpus_manager_new.CorpusManager._get_refinire_rag_dir') as mock_get_dir:
            mock_get_dir.return_value = Path('/test/refinire')
            
            # Test default output directory
            result = CorpusManager._get_default_output_directory('TEST_ENV_VAR', 'subdirectory')
            
            # Verify result
            expected_path = Path('/test/refinire/subdirectory')
            assert result == expected_path

    @patch.dict(os.environ, {'TEST_ENV_VAR': '/custom/output'})
    def test_get_default_output_directory_with_env_var(self):
        """
        Test _get_default_output_directory with environment variable
        環境変数での_get_default_output_directoryテスト
        """
        # Test with environment variable
        result = CorpusManager._get_default_output_directory('TEST_ENV_VAR', 'subdirectory')
        
        # Verify result
        expected_path = Path('/custom/output')
        assert result == expected_path

    def test_create_filter_config_from_glob_simple(self):
        """
        Test _create_filter_config_from_glob with simple pattern
        シンプルなパターンでの_create_filter_config_from_globテスト
        """
        # Test simple glob pattern
        config = self.corpus_manager._create_filter_config_from_glob("*.txt")
        
        # Verify config
        assert config is not None
        assert hasattr(config, 'extensions')
        assert 'txt' in config.extensions

    def test_create_filter_config_from_glob_complex(self):
        """
        Test _create_filter_config_from_glob with complex pattern
        複雑なパターンでの_create_filter_config_from_globテスト
        """
        # Test complex glob pattern
        config = self.corpus_manager._create_filter_config_from_glob("**/*.{txt,md}")
        
        # Verify config
        assert config is not None
        assert hasattr(config, 'extensions')
        assert 'txt' in config.extensions
        assert 'md' in config.extensions

    def test_create_filter_config_from_glob_invalid(self):
        """
        Test _create_filter_config_from_glob with invalid pattern
        無効なパターンでの_create_filter_config_from_globテスト
        """
        # Test invalid glob pattern (no extension)
        config = self.corpus_manager._create_filter_config_from_glob("somefile")
        
        # Verify None is returned for invalid patterns
        assert config is None

    def test_get_retriever_capabilities(self):
        """
        Test _get_retriever_capabilities functionality
        _get_retriever_capabilities機能のテスト
        """
        # Create mock retriever with various methods
        mock_retriever = Mock()
        mock_retriever.search = Mock()
        mock_retriever.similarity_search = Mock()
        mock_retriever.store_embedding = Mock()
        mock_retriever.__class__.__name__ = "TestRetriever"
        
        # Get capabilities
        capabilities = self.corpus_manager._get_retriever_capabilities(mock_retriever)
        
        # Verify capabilities
        assert 'search' in capabilities
        assert 'similarity_search' in capabilities
        assert 'store_embedding' in capabilities

    def test_get_retriever_capabilities_minimal(self):
        """
        Test _get_retriever_capabilities with minimal retriever
        最小限のレトリーバーでの_get_retriever_capabilities テスト
        """
        # Create mock retriever with minimal methods
        mock_retriever = Mock()
        mock_retriever.__class__.__name__ = "MinimalRetriever"
        # Remove common methods to test minimal case
        del mock_retriever.search
        del mock_retriever.similarity_search
        del mock_retriever.store_embedding
        
        # Get capabilities
        capabilities = self.corpus_manager._get_retriever_capabilities(mock_retriever)
        
        # Verify minimal capabilities
        assert isinstance(capabilities, list)
        assert len(capabilities) >= 0  # Should not error


class TestCorpusStats:
    """
    Test CorpusStats functionality
    CorpusStats機能のテスト
    """

    def test_corpus_stats_initialization(self):
        """
        Test CorpusStats initialization
        CorpusStats初期化テスト
        """
        stats = CorpusStats()
        
        # Verify default values
        assert stats.total_files_processed == 0
        assert stats.total_documents_created == 0
        assert stats.total_chunks_created == 0
        assert stats.total_processing_time == 0.0
        assert stats.pipeline_stages_executed == 0
        assert stats.documents_by_stage == {}
        assert stats.errors_encountered == 0

    def test_corpus_stats_with_values(self):
        """
        Test CorpusStats with custom values
        カスタム値でのCorpusStatsテスト
        """
        custom_stages = {"loading": 10, "processing": 8}
        stats = CorpusStats(
            total_files_processed=5,
            total_documents_created=10,
            total_chunks_created=50,
            total_processing_time=1.5,
            pipeline_stages_executed=2,
            documents_by_stage=custom_stages,
            errors_encountered=1
        )
        
        # Verify custom values
        assert stats.total_files_processed == 5
        assert stats.total_documents_created == 10
        assert stats.total_chunks_created == 50
        assert stats.total_processing_time == 1.5
        assert stats.pipeline_stages_executed == 2
        assert stats.documents_by_stage == custom_stages
        assert stats.errors_encountered == 1

    def test_corpus_stats_post_init(self):
        """
        Test CorpusStats __post_init__ method
        CorpusStats __post_init__メソッドテスト
        """
        # Test with None documents_by_stage
        stats = CorpusStats(documents_by_stage=None)
        
        # Verify __post_init__ set empty dict
        assert stats.documents_by_stage == {}
        
        # Test with existing documents_by_stage
        existing_stages = {"test": 1}
        stats2 = CorpusStats(documents_by_stage=existing_stages)
        
        # Verify existing value preserved
        assert stats2.documents_by_stage == existing_stages