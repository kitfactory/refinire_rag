"""
Simple tests for CorpusManager to improve coverage
CorpusManagerのカバレッジ向上のためのシンプルなテスト
"""

import pytest
import os
from unittest.mock import Mock, patch
from pathlib import Path

from refinire_rag.application.corpus_manager_new import CorpusManager, CorpusStats
from refinire_rag.models.document import Document


class TestCorpusManagerBasic:
    """
    Basic tests for CorpusManager
    CorpusManagerの基本テスト
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

    @patch('refinire_rag.application.corpus_manager_new.PluginFactory')
    def test_corpus_manager_initialization(self, mock_factory):
        """
        Test CorpusManager initialization
        CorpusManager初期化テスト
        """
        # Setup mock factory
        mock_store = Mock()
        mock_retriever = Mock()
        mock_factory.create_plugin.side_effect = lambda plugin_name, **kwargs: {
            'sqlite': mock_store,
            'inmemory_vector': mock_retriever
        }.get(plugin_name, Mock())
        
        # Create CorpusManager with mock components
        corpus_manager = CorpusManager(
            document_store=mock_store,
            retrievers=[mock_retriever]
        )
        
        # Verify initialization
        assert corpus_manager.document_store == mock_store
        assert mock_retriever in corpus_manager.retrievers
        assert isinstance(corpus_manager.stats, CorpusStats)

    @patch('refinire_rag.application.corpus_manager_new.PluginFactory')
    def test_corpus_manager_add_retriever(self, mock_factory):
        """
        Test add_retriever functionality
        add_retriever機能のテスト
        """
        # Setup mock factory
        mock_store = Mock()
        mock_retriever1 = Mock()
        mock_retriever2 = Mock()
        mock_factory.create_plugin.return_value = mock_store
        
        # Create CorpusManager
        corpus_manager = CorpusManager(
            document_store=mock_store,
            retrievers=[mock_retriever1]
        )
        
        initial_count = len(corpus_manager.retrievers)
        
        # Add retriever
        corpus_manager.add_retriever(mock_retriever2)
        
        # Verify retriever was added
        assert len(corpus_manager.retrievers) == initial_count + 1
        assert mock_retriever2 in corpus_manager.retrievers

    @patch('refinire_rag.application.corpus_manager_new.PluginFactory')
    def test_corpus_manager_remove_retriever(self, mock_factory):
        """
        Test remove_retriever functionality
        remove_retriever機能のテスト
        """
        # Setup mock factory
        mock_store = Mock()
        mock_retriever1 = Mock()
        mock_retriever2 = Mock()
        mock_factory.create_plugin.return_value = mock_store
        
        # Create CorpusManager
        corpus_manager = CorpusManager(
            document_store=mock_store,
            retrievers=[mock_retriever1, mock_retriever2]
        )
        
        # Remove retriever
        result = corpus_manager.remove_retriever(1)
        
        # Verify removal
        assert result is True
        assert len(corpus_manager.retrievers) == 1
        assert mock_retriever2 not in corpus_manager.retrievers

    @patch('refinire_rag.application.corpus_manager_new.PluginFactory')
    def test_corpus_manager_remove_retriever_invalid_index(self, mock_factory):
        """
        Test remove_retriever with invalid index
        無効なインデックスでのremove_retrieverテスト
        """
        # Setup mock factory
        mock_store = Mock()
        mock_retriever = Mock()
        mock_factory.create_plugin.return_value = mock_store
        
        # Create CorpusManager
        corpus_manager = CorpusManager(
            document_store=mock_store,
            retrievers=[mock_retriever]
        )
        
        initial_count = len(corpus_manager.retrievers)
        
        # Try to remove with invalid index
        result = corpus_manager.remove_retriever(999)
        
        # Verify no change
        assert result is False
        assert len(corpus_manager.retrievers) == initial_count

    @patch('refinire_rag.application.corpus_manager_new.PluginFactory')
    def test_corpus_manager_get_retrievers_by_type(self, mock_factory):
        """
        Test get_retrievers_by_type functionality
        get_retrievers_by_type機能のテスト
        """
        # Setup mock factory
        mock_store = Mock()
        mock_factory.create_plugin.return_value = mock_store
        
        # Create mock retrievers with different types
        vector_store = Mock()
        vector_store.__class__.__name__ = "InMemoryVectorStore"
        keyword_store = Mock()
        keyword_store.__class__.__name__ = "KeywordStore"
        
        # Create CorpusManager
        corpus_manager = CorpusManager(
            document_store=mock_store,
            retrievers=[vector_store, keyword_store]
        )
        
        # Get retrievers by type
        vector_retrievers = corpus_manager.get_retrievers_by_type("VectorStore")
        keyword_retrievers = corpus_manager.get_retrievers_by_type("KeywordStore")
        nonexistent = corpus_manager.get_retrievers_by_type("NonExistent")
        
        # Verify results
        assert len(vector_retrievers) == 1
        assert vector_store in vector_retrievers
        assert len(keyword_retrievers) == 1
        assert keyword_store in keyword_retrievers
        assert len(nonexistent) == 0

    @patch('refinire_rag.application.corpus_manager_new.PluginFactory')
    def test_corpus_manager_clear_corpus(self, mock_factory):
        """
        Test clear_corpus functionality
        clear_corpus機能のテスト
        """
        # Setup mock factory
        mock_store = Mock()
        mock_retriever = Mock()
        mock_factory.create_plugin.return_value = mock_store
        
        # Create CorpusManager
        corpus_manager = CorpusManager(
            document_store=mock_store,
            retrievers=[mock_retriever]
        )
        
        # Call clear_corpus
        corpus_manager.clear_corpus()
        
        # Verify document store was cleared (using the real method name)
        mock_store.clear_all_documents.assert_called_once()
        
        # Verify retriever was cleared (check for different possible clear methods)
        # The implementation tries multiple methods, so we check if any was called
        assert (mock_retriever.clear_all_vectors.called or 
                mock_retriever.clear_all_documents.called or 
                mock_retriever.clear.called)

    @patch('refinire_rag.application.corpus_manager_new.Path.exists')
    def test_get_refinire_rag_dir_default(self, mock_exists):
        """
        Test _get_refinire_rag_dir with default directory
        デフォルトディレクトリでの_get_refinire_rag_dirテスト
        """
        # Setup mocks
        mock_exists.return_value = False
        
        with patch('refinire_rag.application.corpus_manager_new.Path.mkdir') as mock_mkdir:
            # Call method
            result = CorpusManager._get_refinire_rag_dir()
            
            # Verify result (implementation returns tests/data subdirectory)
            expected_path = Path('./tests/data')
            assert result == expected_path
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch.dict(os.environ, {'REFINIRE_DIR': '/custom/path'})
    @patch('refinire_rag.application.corpus_manager_new.Path.exists')
    def test_get_refinire_rag_dir_custom(self, mock_exists):
        """
        Test _get_refinire_rag_dir with custom directory from environment
        環境変数からのカスタムディレクトリでの_get_refinire_rag_dirテスト
        """
        # Setup mocks
        mock_exists.return_value = False
        
        with patch('refinire_rag.application.corpus_manager_new.Path.mkdir') as mock_mkdir:
            # Call method
            result = CorpusManager._get_refinire_rag_dir()
            
            # Verify result (implementation returns /custom/path/rag subdirectory)
            expected_path = Path('/custom/path/rag')
            assert result == expected_path
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_get_corpus_file_path(self):
        """
        Test _get_corpus_file_path functionality
        _get_corpus_file_path機能のテスト
        """
        with patch('refinire_rag.application.corpus_manager_new.CorpusManager._get_refinire_rag_dir') as mock_get_dir:
            mock_get_dir.return_value = Path('/test/tests/data')
            
            with patch('refinire_rag.application.corpus_manager_new.Path.mkdir') as mock_mkdir:
                # Test different file types
                track_path = CorpusManager._get_corpus_file_path('mycorpus', 'track')
                dict_path = CorpusManager._get_corpus_file_path('mycorpus', 'dictionary')
                kg_path = CorpusManager._get_corpus_file_path('mycorpus', 'knowledge_graph')
                
                # Verify paths (paths include rag subdirectory)
                assert track_path == Path('/test/tests/data/mycorpus_track.json')
                assert dict_path == Path('/test/tests/data/mycorpus_dictionary.md')
                assert kg_path == Path('/test/tests/data/mycorpus_knowledge_graph.md')

    def test_get_corpus_file_path_custom_directory(self):
        """
        Test _get_corpus_file_path with custom directory
        カスタムディレクトリでの_get_corpus_file_path テスト
        """
        with patch('refinire_rag.application.corpus_manager_new.Path.mkdir') as mock_mkdir:
            # Test with custom directory
            track_path = CorpusManager._get_corpus_file_path('mycorpus', 'track', '/custom/dir')
            
            # Verify path
            assert track_path == Path('/custom/dir/mycorpus_track.json')

    def test_get_default_output_directory(self):
        """
        Test _get_default_output_directory functionality
        _get_default_output_directory機能のテスト
        """
        with patch('refinire_rag.application.corpus_manager_new.Path.home') as mock_home:
            mock_home.return_value = Path('/test/home')
            
            with patch('refinire_rag.application.corpus_manager_new.Path.mkdir') as mock_mkdir:
                # Test default output directory (no environment variable)
                result = CorpusManager._get_default_output_directory('TEST_ENV_VAR', 'subdirectory')
                
                # Verify result (uses home/.refinire/subdirectory when no env var)
                expected_path = Path('/test/home/.refinire/subdirectory')
                assert result == expected_path
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

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

    @patch('refinire_rag.application.corpus_manager_new.PluginFactory')
    def test_create_filter_config_from_glob_simple(self, mock_factory):
        """
        Test _create_filter_config_from_glob with simple pattern
        シンプルなパターンでの_create_filter_config_from_globテスト
        """
        # Setup mock factory
        mock_store = Mock()
        mock_factory.create_plugin.return_value = mock_store
        
        # Create CorpusManager
        corpus_manager = CorpusManager(document_store=mock_store, retrievers=[])
        
        # Test simple glob pattern
        config = corpus_manager._create_filter_config_from_glob("*.txt")
        
        # Verify config
        assert config is not None
        assert hasattr(config, 'extension_filter')
        assert config.extension_filter is not None
        assert '.txt' in config.extension_filter.include_extensions

    @patch('refinire_rag.application.corpus_manager_new.PluginFactory')
    def test_create_filter_config_from_glob_invalid(self, mock_factory):
        """
        Test _create_filter_config_from_glob with invalid pattern
        無効なパターンでの_create_filter_config_from_globテスト
        """
        # Setup mock factory
        mock_store = Mock()
        mock_factory.create_plugin.return_value = mock_store
        
        # Create CorpusManager
        corpus_manager = CorpusManager(document_store=mock_store, retrievers=[])
        
        # Test invalid glob pattern (no extension)
        config = corpus_manager._create_filter_config_from_glob("somefile")
        
        # Verify None is returned for invalid patterns
        assert config is None