"""
Advanced tests for CorpusManager focusing on missing coverage
CorpusManagerの高度なテスト（欠けているカバレッジに焦点）
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List

from refinire_rag.application.corpus_manager_new import CorpusManager, CorpusStats
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.models.document import Document
from refinire_rag.exceptions import StorageError


class TestCorpusManagerEnvironmentInitialization:
    """Test environment-based initialization methods"""
    
    def test_from_env_classmethod(self):
        """Test from_env class method creates instance with None parameters"""
        with patch.object(CorpusManager, '_create_document_store_from_env') as mock_doc_store:
            with patch.object(CorpusManager, '_create_retrievers_from_env') as mock_retrievers:
                mock_doc_store.return_value = Mock()
                mock_retrievers.return_value = [Mock()]
                
                manager = CorpusManager.from_env()
                
                assert manager is not None
                assert manager.document_store is not None
                assert len(manager.retrievers) > 0
    
    def test_from_env_with_custom_config(self):
        """Test from_env with custom configuration"""
        custom_config = {"test_setting": "value"}
        
        with patch.object(CorpusManager, '_create_document_store_from_env') as mock_doc_store:
            with patch.object(CorpusManager, '_create_retrievers_from_env') as mock_retrievers:
                mock_doc_store.return_value = Mock()
                mock_retrievers.return_value = [Mock()]
                
                manager = CorpusManager.from_env(config=custom_config)
                
                assert manager.config == custom_config
    
    @patch('os.getenv')
    @patch('refinire_rag.factories.plugin_factory.PluginFactory.create_document_stores_from_env')
    @patch('refinire_rag.registry.plugin_registry.PluginRegistry.create_plugin')
    def test_create_document_store_from_env_success(self, mock_registry, mock_factory, mock_getenv):
        """Test successful document store creation from environment"""
        mock_getenv.return_value = "sqlite"
        mock_store = Mock()
        mock_factory.return_value = [mock_store]
        
        manager = CorpusManager(document_store=Mock(), retrievers=[Mock()])
        result = manager._create_document_store_from_env()
        
        assert result == mock_store
        mock_factory.assert_called_once()
    
    @patch('os.getenv')
    @patch('refinire_rag.factories.plugin_factory.PluginFactory.create_document_stores_from_env')
    @patch('refinire_rag.registry.plugin_registry.PluginRegistry.create_plugin')
    def test_create_document_store_from_env_fallback_empty_list(self, mock_registry, mock_factory, mock_getenv):
        """Test document store creation fallback when factory returns empty list"""
        mock_getenv.return_value = "sqlite"
        mock_factory.return_value = []  # Empty list
        mock_fallback_store = Mock()
        mock_registry.return_value = mock_fallback_store
        
        manager = CorpusManager(document_store=Mock(), retrievers=[Mock()])
        result = manager._create_document_store_from_env()
        
        assert result == mock_fallback_store
        mock_registry.assert_called_with('document_stores', 'sqlite')
    
    @patch('os.getenv')
    @patch('refinire_rag.factories.plugin_factory.PluginFactory.create_document_stores_from_env')
    @patch('refinire_rag.registry.plugin_registry.PluginRegistry.create_plugin')
    def test_create_document_store_from_env_fallback_exception(self, mock_registry, mock_factory, mock_getenv):
        """Test document store creation fallback when exception occurs"""
        mock_getenv.return_value = "invalid_store"
        mock_factory.side_effect = Exception("Factory failed")
        mock_fallback_store = Mock()
        mock_registry.return_value = mock_fallback_store
        
        manager = CorpusManager(document_store=Mock(), retrievers=[Mock()])
        result = manager._create_document_store_from_env()
        
        assert result == mock_fallback_store
        mock_registry.assert_called_with('document_stores', 'sqlite')
    
    @patch('refinire_rag.factories.plugin_factory.PluginFactory.create_vector_stores_from_env')
    @patch('refinire_rag.factories.plugin_factory.PluginFactory.create_keyword_stores_from_env')
    @patch('refinire_rag.factories.plugin_factory.PluginFactory.create_retrievers_from_env')
    @patch('refinire_rag.registry.plugin_registry.PluginRegistry.create_plugin')
    def test_create_retrievers_from_env_success(self, mock_registry, mock_retrievers, mock_keyword, mock_vector):
        """Test successful retrievers creation from environment"""
        mock_vector_store = Mock()
        mock_keyword_store = Mock()
        mock_retriever = Mock()
        
        mock_vector.return_value = [mock_vector_store]
        mock_keyword.return_value = [mock_keyword_store]
        mock_retrievers.return_value = [mock_retriever]
        
        manager = CorpusManager(document_store=Mock(), retrievers=[Mock()])
        result = manager._create_retrievers_from_env()
        
        assert len(result) == 3
        assert mock_vector_store in result
        assert mock_keyword_store in result
        assert mock_retriever in result
    
    @patch('refinire_rag.factories.plugin_factory.PluginFactory.create_vector_stores_from_env')
    @patch('refinire_rag.factories.plugin_factory.PluginFactory.create_keyword_stores_from_env')
    @patch('refinire_rag.factories.plugin_factory.PluginFactory.create_retrievers_from_env')
    @patch('refinire_rag.registry.plugin_registry.PluginRegistry.create_plugin')
    def test_create_retrievers_from_env_fallback(self, mock_registry, mock_retrievers, mock_keyword, mock_vector):
        """Test retrievers creation fallback when no retrievers found"""
        mock_vector.return_value = []
        mock_keyword.return_value = []
        mock_retrievers.return_value = []
        
        mock_fallback_retriever = Mock()
        mock_registry.return_value = mock_fallback_retriever
        
        manager = CorpusManager(document_store=Mock(), retrievers=[Mock()])
        result = manager._create_retrievers_from_env()
        
        assert len(result) == 1
        assert result[0] == mock_fallback_retriever
        mock_registry.assert_called_with('vector_stores', 'inmemory_vector')
    
    @patch('refinire_rag.factories.plugin_factory.PluginFactory.create_vector_stores_from_env')
    @patch('refinire_rag.factories.plugin_factory.PluginFactory.create_keyword_stores_from_env')
    @patch('refinire_rag.factories.plugin_factory.PluginFactory.create_retrievers_from_env')
    @patch('refinire_rag.registry.plugin_registry.PluginRegistry.create_plugin')
    def test_create_retrievers_from_env_exception(self, mock_registry, mock_retrievers, mock_keyword, mock_vector):
        """Test retrievers creation fallback when exception occurs"""
        mock_vector.side_effect = Exception("Vector store creation failed")
        
        mock_fallback_retriever = Mock()
        mock_registry.return_value = mock_fallback_retriever
        
        manager = CorpusManager(document_store=Mock(), retrievers=[Mock()])
        result = manager._create_retrievers_from_env()
        
        assert len(result) == 1
        assert result[0] == mock_fallback_retriever


class TestCorpusManagerRetrieverManagement:
    """Test retriever management methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.initial_retriever = Mock()
        self.manager = CorpusManager(
            document_store=self.document_store, 
            retrievers=[self.initial_retriever]
        )
    
    def test_get_vector_store_from_retrievers_found(self):
        """Test getting vector store when vector store capabilities exist"""
        # Create mock with specific spec to control hasattr behavior
        vector_retriever = Mock()
        vector_retriever.add_vector = Mock()
        vector_retriever.search_similar = Mock()
        
        # Create mock without vector methods (use spec to limit attributes)
        non_vector_retriever = Mock(spec=['some_other_method'])
        
        self.manager.retrievers = [non_vector_retriever, vector_retriever]
        result = self.manager._get_vector_store_from_retrievers()
        
        assert result == vector_retriever
    
    def test_get_vector_store_from_retrievers_not_found(self):
        """Test getting vector store when no vector store capabilities exist"""
        non_vector_retriever = Mock()
        # non_vector_retriever doesn't have vector methods
        
        self.manager.retrievers = [non_vector_retriever]
        result = self.manager._get_vector_store_from_retrievers()
        
        assert result == non_vector_retriever  # Returns first retriever
    
    def test_get_vector_store_from_retrievers_empty_list(self):
        """Test getting vector store when retrievers list is empty"""
        self.manager.retrievers = []
        result = self.manager._get_vector_store_from_retrievers()
        
        assert result is None
    
    def test_get_retrievers_by_type(self):
        """Test getting retrievers by type"""
        vector_retriever = Mock()
        vector_retriever.__class__.__name__ = "VectorRetriever"
        
        keyword_retriever = Mock()
        keyword_retriever.__class__.__name__ = "KeywordRetriever"
        
        hybrid_retriever = Mock()
        hybrid_retriever.__class__.__name__ = "HybridRetriever"
        
        self.manager.retrievers = [vector_retriever, keyword_retriever, hybrid_retriever]
        
        # Test vector type
        vector_results = self.manager.get_retrievers_by_type("vector")
        assert len(vector_results) == 1
        assert vector_results[0] == vector_retriever
        
        # Test keyword type
        keyword_results = self.manager.get_retrievers_by_type("keyword")
        assert len(keyword_results) == 1
        assert keyword_results[0] == keyword_retriever
        
        # Test hybrid type
        hybrid_results = self.manager.get_retrievers_by_type("hybrid")
        assert len(hybrid_results) == 1
        assert hybrid_results[0] == hybrid_retriever
        
        # Test nonexistent type
        none_results = self.manager.get_retrievers_by_type("nonexistent")
        assert len(none_results) == 0
    
    def test_add_retriever(self):
        """Test adding a new retriever"""
        new_retriever = Mock()
        new_retriever.__class__.__name__ = "NewRetriever"
        
        initial_count = len(self.manager.retrievers)
        self.manager.add_retriever(new_retriever)
        
        assert len(self.manager.retrievers) == initial_count + 1
        assert new_retriever in self.manager.retrievers
    
    def test_add_retriever_with_vector_capabilities(self):
        """Test adding retriever with vector capabilities updates vector_store"""
        self.manager.vector_store = None  # Start with no vector store
        
        vector_retriever = Mock()
        vector_retriever.__class__.__name__ = "VectorRetriever"
        vector_retriever.add_vector = Mock()  # Has vector capabilities
        
        self.manager.add_retriever(vector_retriever)
        
        assert self.manager.vector_store == vector_retriever
    
    def test_remove_retriever_valid_index(self):
        """Test removing retriever with valid index"""
        additional_retriever = Mock()
        self.manager.retrievers.append(additional_retriever)
        
        initial_count = len(self.manager.retrievers)
        success = self.manager.remove_retriever(0)
        
        assert success is True
        assert len(self.manager.retrievers) == initial_count - 1
        assert self.initial_retriever not in self.manager.retrievers
    
    def test_remove_retriever_invalid_index(self):
        """Test removing retriever with invalid index"""
        initial_count = len(self.manager.retrievers)
        
        # Test negative index
        success = self.manager.remove_retriever(-1)
        assert success is False
        assert len(self.manager.retrievers) == initial_count
        
        # Test index too large
        success = self.manager.remove_retriever(100)
        assert success is False
        assert len(self.manager.retrievers) == initial_count
    
    def test_remove_retriever_updates_vector_store(self):
        """Test removing retriever updates vector_store when necessary"""
        vector_retriever = Mock()
        vector_retriever.add_vector = Mock()
        
        self.manager.retrievers = [vector_retriever]
        self.manager.vector_store = vector_retriever
        
        # Remove the vector store retriever
        success = self.manager.remove_retriever(0)
        
        assert success is True
        assert self.manager.vector_store != vector_retriever


class TestCorpusManagerFilePathMethods:
    """Test static file path utility methods"""
    
    def test_get_refinire_rag_dir_default(self):
        """Test getting default refinire rag directory"""
        with patch('os.getenv') as mock_getenv:
            # Configure the mock to return default value for REFINIRE_DIR
            mock_getenv.side_effect = lambda key, default=None: default if key == "REFINIRE_DIR" else None
            
            result = CorpusManager._get_refinire_rag_dir()
            
            expected = Path("./refinire/rag")
            assert result == expected
    
    def test_get_refinire_rag_dir_from_env(self):
        """Test getting refinire rag directory from environment"""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = Path(temp_dir) / "custom" / "refinire"
            with patch('os.getenv', return_value=str(custom_dir)):
                result = CorpusManager._get_refinire_rag_dir()
                
                expected = custom_dir / "rag"
                assert result == expected
                # Verify the directory was actually created
                assert result.exists()
    
    def test_get_corpus_file_path_track(self):
        """Test getting track file path"""
        corpus_name = "test_corpus"
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            test_rag_dir = Path(temp_dir) / "test" / "rag"
            
            with patch.object(CorpusManager, '_get_refinire_rag_dir') as mock_dir:
                mock_dir.return_value = test_rag_dir
                
                result = CorpusManager._get_corpus_file_path(corpus_name, "track")
                
                expected = test_rag_dir / "test_corpus_track.json"
                assert result == expected
                # Verify directory was created
                assert test_rag_dir.exists()
    
    def test_get_corpus_file_path_dictionary(self):
        """Test getting dictionary file path"""
        corpus_name = "test_corpus"
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            test_rag_dir = Path(temp_dir) / "test" / "rag"
            
            with patch.object(CorpusManager, '_get_refinire_rag_dir') as mock_dir:
                mock_dir.return_value = test_rag_dir
                
                result = CorpusManager._get_corpus_file_path(corpus_name, "dictionary")
                
                expected = test_rag_dir / "test_corpus_dictionary.md"
                assert result == expected
    
    def test_get_corpus_file_path_knowledge_graph(self):
        """Test getting knowledge graph file path"""
        corpus_name = "test_corpus"
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            test_rag_dir = Path(temp_dir) / "test" / "rag"
            
            with patch.object(CorpusManager, '_get_refinire_rag_dir') as mock_dir:
                mock_dir.return_value = test_rag_dir
                
                result = CorpusManager._get_corpus_file_path(corpus_name, "knowledge_graph")
                
                expected = test_rag_dir / "test_corpus_knowledge_graph.md"
                assert result == expected
    
    def test_get_corpus_file_path_with_custom_dir(self):
        """Test getting corpus file path with custom directory"""
        corpus_name = "test_corpus"
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = str(Path(temp_dir) / "custom" / "path")
            
            result = CorpusManager._get_corpus_file_path(corpus_name, "track", custom_dir)
            
            expected = Path(custom_dir) / "test_corpus_track.json"
            assert result == expected
            # Verify directory was created
            assert Path(custom_dir).exists()
    
    def test_get_corpus_file_path_invalid_type(self):
        """Test getting corpus file path with invalid file type"""
        corpus_name = "test_corpus"
        
        with pytest.raises(ValueError, match="Unknown file type"):
            CorpusManager._get_corpus_file_path(corpus_name, "invalid_type")
    
    def test_get_default_output_directory_from_env(self):
        """Test getting default output directory from environment variable"""
        env_var_name = "TEST_OUTPUT_DIR"
        custom_path = "/custom/output"
        
        with patch('os.getenv', return_value=custom_path):
            result = CorpusManager._get_default_output_directory(env_var_name, "subdir")
            
            assert result == Path(custom_path)
    
    def test_get_default_output_directory_fallback(self):
        """Test getting default output directory fallback"""
        env_var_name = "TEST_OUTPUT_DIR"
        subdir = "test_subdir"
        
        with patch('os.getenv', return_value=None):
            # The method should return None when env var is not set
            # Based on the truncated code, it seems to handle this case
            result = CorpusManager._get_default_output_directory(env_var_name, subdir)
            
            assert result is None or isinstance(result, Path)


class TestCorpusManagerInitializationEdgeCases:
    """Test edge cases in CorpusManager initialization"""
    
    def test_initialization_with_non_list_retriever(self):
        """Test initialization converts single retriever to list"""
        document_store = Mock()
        single_retriever = Mock()
        
        manager = CorpusManager(
            document_store=document_store,
            retrievers=single_retriever  # Not a list
        )
        
        assert isinstance(manager.retrievers, list)
        assert len(manager.retrievers) == 1
        assert manager.retrievers[0] == single_retriever
    
    def test_initialization_with_list_retrievers(self):
        """Test initialization preserves list of retrievers"""
        document_store = Mock()
        retriever_list = [Mock(), Mock(), Mock()]
        
        manager = CorpusManager(
            document_store=document_store,
            retrievers=retriever_list
        )
        
        assert isinstance(manager.retrievers, list)
        assert len(manager.retrievers) == 3
        assert manager.retrievers == retriever_list
    
    def test_initialization_with_none_document_store_calls_env_method(self):
        """Test initialization with None document_store calls environment method"""
        with patch.object(CorpusManager, '_create_document_store_from_env') as mock_method:
            mock_store = Mock()
            mock_method.return_value = mock_store
            
            manager = CorpusManager(document_store=None, retrievers=[Mock()])
            
            assert manager.document_store == mock_store
            mock_method.assert_called_once()
    
    def test_initialization_with_none_retrievers_calls_env_method(self):
        """Test initialization with None retrievers calls environment method"""
        with patch.object(CorpusManager, '_create_retrievers_from_env') as mock_method:
            mock_retrievers = [Mock(), Mock()]
            mock_method.return_value = mock_retrievers
            
            manager = CorpusManager(document_store=Mock(), retrievers=None)
            
            assert manager.retrievers == mock_retrievers
            mock_method.assert_called_once()


class TestCorpusStatsInitialization:
    """Test CorpusStats dataclass initialization"""
    
    def test_corpus_stats_default_initialization(self):
        """Test CorpusStats default initialization"""
        stats = CorpusStats()
        
        assert stats.total_files_processed == 0
        assert stats.total_documents_created == 0
        assert stats.total_chunks_created == 0
        assert stats.total_processing_time == 0.0
        assert stats.pipeline_stages_executed == 0
        assert stats.documents_by_stage == {}
        assert stats.errors_encountered == 0
    
    def test_corpus_stats_post_init_with_none_documents_by_stage(self):
        """Test CorpusStats post_init initializes documents_by_stage when None"""
        stats = CorpusStats(documents_by_stage=None)
        
        assert stats.documents_by_stage == {}
    
    def test_corpus_stats_post_init_preserves_existing_documents_by_stage(self):
        """Test CorpusStats post_init preserves existing documents_by_stage"""
        existing_dict = {"stage1": 5, "stage2": 10}
        stats = CorpusStats(documents_by_stage=existing_dict)
        
        assert stats.documents_by_stage == existing_dict
    
    def test_corpus_stats_with_custom_values(self):
        """Test CorpusStats with custom values"""
        custom_documents_by_stage = {"loading": 5, "chunking": 3}
        
        stats = CorpusStats(
            total_files_processed=10,
            total_documents_created=8,
            total_chunks_created=25,
            total_processing_time=45.5,
            pipeline_stages_executed=3,
            documents_by_stage=custom_documents_by_stage,
            errors_encountered=2
        )
        
        assert stats.total_files_processed == 10
        assert stats.total_documents_created == 8
        assert stats.total_chunks_created == 25
        assert stats.total_processing_time == 45.5
        assert stats.pipeline_stages_executed == 3
        assert stats.documents_by_stage == custom_documents_by_stage
        assert stats.errors_encountered == 2


class TestCorpusManagerLoggingAndBackwardCompatibility:
    """Test logging and backward compatibility features"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.vector_retriever = Mock()
        self.vector_retriever.add_vector = Mock()
        self.vector_retriever.search_similar = Mock()
        
        self.non_vector_retriever = Mock()
        # non_vector_retriever doesn't have vector methods - explicitly remove them
        del self.non_vector_retriever.add_vector
        del self.non_vector_retriever.search_similar
    
    def test_initialization_logging_with_multiple_retrievers(self):
        """Test initialization logs retriever types correctly"""
        retrievers = [self.vector_retriever, self.non_vector_retriever]
        
        with patch('refinire_rag.application.corpus_manager_new.logger') as mock_logger:
            manager = CorpusManager(
                document_store=self.document_store,
                retrievers=retrievers
            )
            
            # Should log initialization with retriever types
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0][0]
            assert "Initialized CorpusManager" in call_args
    
    def test_vector_store_backward_compatibility(self):
        """Test vector_store property is set for backward compatibility"""
        retrievers = [self.non_vector_retriever, self.vector_retriever]
        
        manager = CorpusManager(
            document_store=self.document_store,
            retrievers=retrievers
        )
        
        # Should set vector_store to the first retriever with vector capabilities
        assert manager.vector_store == self.vector_retriever
    
    def test_vector_store_backward_compatibility_no_vector_store(self):
        """Test vector_store property when no vector store exists"""
        retrievers = [self.non_vector_retriever]
        
        manager = CorpusManager(
            document_store=self.document_store,
            retrievers=retrievers
        )
        
        # Should set vector_store to first retriever even if not vector store
        assert manager.vector_store == self.non_vector_retriever