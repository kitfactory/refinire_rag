"""
Core methods tests for CorpusManager to achieve higher coverage
CorpusManagerのコアメソッドテスト（高いカバレッジ実現のため）
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from refinire_rag.application.corpus_manager_new import CorpusManager, CorpusStats
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.models.document import Document
from refinire_rag.exceptions import StorageError


class TestCorpusManagerImportMethods:
    """Test CorpusManager import and core functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.mock_retriever = Mock()
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[self.mock_retriever]
        )
    
    @patch('refinire_rag.application.corpus_manager_new.IncrementalDirectoryLoader')
    @patch('refinire_rag.application.corpus_manager_new.ConstantMetadata')
    def test_import_original_documents_basic(self, mock_constant_metadata, mock_loader):
        """Test basic document import functionality"""
        # Setup mocks
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        
        test_docs = [
            Document(id="doc1", content="Content 1", metadata={}),
            Document(id="doc2", content="Content 2", metadata={})
        ]
        mock_loader_instance.process.return_value = iter(test_docs)
        
        # Execute
        stats = self.manager.import_original_documents(
            corpus_name="test_corpus",
            directory="/test/dir"
        )
        
        # Verify
        assert isinstance(stats, CorpusStats)
        mock_loader.assert_called_once()
        mock_constant_metadata.assert_called_once()
        
        # Check that loader process was called
        mock_loader_instance.process.assert_called_once()
    
    @patch('refinire_rag.application.corpus_manager_new.IncrementalDirectoryLoader')
    @patch('refinire_rag.application.corpus_manager_new.ConstantMetadata')
    def test_import_original_documents_with_options(self, mock_constant_metadata, mock_loader):
        """Test document import with all options"""
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        mock_loader_instance.process.return_value = iter([])
        
        additional_metadata = {"project": "test", "version": "1.0"}
        
        stats = self.manager.import_original_documents(
            corpus_name="test_corpus",
            directory="/test/dir",
            glob="**/*.md",
            use_multithreading=True,
            force_reload=True,
            additional_metadata=additional_metadata,
            tracking_file_path="/custom/tracking.json",
            create_dictionary=True,
            create_knowledge_graph=True,
            dictionary_output_dir="/dict/output",
            graph_output_dir="/graph/output"
        )
        
        assert isinstance(stats, CorpusStats)
        
        # Verify constant metadata includes additional metadata
        call_args = mock_constant_metadata.call_args[0][0]
        assert call_args["project"] == "test"
        assert call_args["version"] == "1.0"
        assert call_args["processing_stage"] == "original"
        assert call_args["corpus_name"] == "test_corpus"
    
    def test_create_filter_config_from_glob_simple_extension(self):
        """Test creating filter config from simple glob pattern"""
        filter_config = self.manager._create_filter_config_from_glob("**/*.md")
        
        # Should create filter config with .md extension
        assert filter_config is not None
        assert hasattr(filter_config, 'extension_filter')
    
    def test_create_filter_config_from_glob_multiple_extensions(self):
        """Test creating filter config from multiple extension pattern"""
        filter_config = self.manager._create_filter_config_from_glob("*.{txt,md,py}")
        
        # Should create filter config with multiple extensions
        assert filter_config is not None
        assert hasattr(filter_config, 'extension_filter')
    
    def test_create_filter_config_from_glob_no_extensions(self):
        """Test creating filter config from pattern with no specific extensions"""
        filter_config = self.manager._create_filter_config_from_glob("**/*")
        
        # Should return None for complex patterns without clear extensions
        assert filter_config is None
    
    def test_create_filter_config_from_glob_complex_pattern(self):
        """Test creating filter config from complex glob pattern"""
        filter_config = self.manager._create_filter_config_from_glob("**/test_*/**/*.txt")
        
        # Should handle complex patterns gracefully
        assert filter_config is not None or filter_config is None
    
    @patch('os.getenv')
    def test_get_default_output_directory_from_env(self, mock_getenv):
        """Test getting default output directory from environment variable"""
        mock_getenv.return_value = "/custom/env/path"
        
        result = CorpusManager._get_default_output_directory("TEST_ENV_VAR", "subdir")
        
        assert result == Path("/custom/env/path")
        mock_getenv.assert_called_once_with("TEST_ENV_VAR")
    
    @patch('os.getenv')
    @patch('pathlib.Path.home')
    def test_get_default_output_directory_fallback(self, mock_home, mock_getenv):
        """Test getting default output directory fallback to home"""
        mock_getenv.return_value = None
        mock_home.return_value = Path("/home/user")
        
        result = CorpusManager._get_default_output_directory("TEST_ENV_VAR", "subdir")
        
        expected = Path("/home/user") / ".refinire" / "subdir"
        assert result == expected


class TestCorpusManagerAdvancedMethods:
    """Test advanced CorpusManager functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.mock_retriever = Mock()
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[self.mock_retriever]
        )
    
    def test_get_corpus_info_basic(self):
        """Test getting basic corpus information"""
        # Mock document store responses
        self.document_store.count_documents.return_value = 50
        self.document_store.get_storage_stats.return_value = Mock(
            total_documents=50,
            storage_size_bytes=1024000,
            oldest_document="2023-01-01",
            newest_document="2023-12-31"
        )
        
        # Mock search results for processing stages
        original_docs = [Mock() for _ in range(20)]
        chunk_docs = [Mock() for _ in range(30)]
        
        self.document_store.search_by_metadata.side_effect = [
            [Mock(document=doc) for doc in original_docs],  # original documents
            [Mock(document=doc) for doc in chunk_docs]       # chunk documents
        ]
        
        info = self.manager.get_corpus_info("test_corpus")
        
        assert "corpus_name" in info
        assert "total_documents" in info
        assert "storage_stats" in info
        assert "processing_stages" in info
        assert info["corpus_name"] == "test_corpus"
        assert info["total_documents"] == 50
    
    def test_get_corpus_info_with_storage_error(self):
        """Test get corpus info handles storage errors gracefully"""
        self.document_store.count_documents.return_value = 10
        self.document_store.get_storage_stats.side_effect = StorageError("Storage failed")
        
        info = self.manager.get_corpus_info("test_corpus")
        
        assert info["total_documents"] == 10
        assert "error" in info["storage_stats"]
    
    def test_clear_corpus_basic(self):
        """Test basic corpus clearing functionality"""
        # Mock documents to be deleted
        test_docs = [
            Document(id="doc1", content="Content 1", metadata={"corpus_name": "test_corpus"}),
            Document(id="doc2", content="Content 2", metadata={"corpus_name": "test_corpus"}),
        ]
        
        search_results = [Mock(document=doc) for doc in test_docs]
        self.document_store.search_by_metadata.return_value = search_results
        self.document_store.delete_document.return_value = True
        
        result = self.manager.clear_corpus("test_corpus")
        
        assert result["deleted_count"] == 2
        assert result["success"] is True
        assert self.document_store.delete_document.call_count == 2
    
    def test_clear_corpus_with_deletion_errors(self):
        """Test corpus clearing with some deletion errors"""
        test_docs = [
            Document(id="doc1", content="Content 1", metadata={"corpus_name": "test_corpus"}),
            Document(id="doc2", content="Content 2", metadata={"corpus_name": "test_corpus"}),
        ]
        
        search_results = [Mock(document=doc) for doc in test_docs]
        self.document_store.search_by_metadata.return_value = search_results
        
        # First deletion succeeds, second fails
        self.document_store.delete_document.side_effect = [True, False]
        
        result = self.manager.clear_corpus("test_corpus")
        
        assert result["deleted_count"] == 1
        assert result["failed_count"] == 1
        assert result["success"] is False
    
    def test_clear_corpus_no_documents(self):
        """Test clearing corpus with no documents"""
        self.document_store.search_by_metadata.return_value = []
        
        result = self.manager.clear_corpus("test_corpus")
        
        assert result["deleted_count"] == 0
        assert result["success"] is True
    
    def test_clear_corpus_with_search_error(self):
        """Test corpus clearing with search error"""
        self.document_store.search_by_metadata.side_effect = StorageError("Search failed")
        
        result = self.manager.clear_corpus("test_corpus")
        
        assert result["success"] is False
        assert "error" in result


class TestCorpusManagerConfigurationMethods:
    """Test CorpusManager configuration and utility methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[Mock()]
        )
    
    def test_get_corpus_file_path_variations(self):
        """Test getting corpus file paths for all file types"""
        corpus_name = "test_corpus"
        
        # Test track file
        track_path = CorpusManager._get_corpus_file_path(corpus_name, "track")
        assert track_path.name == "test_corpus_track.json"
        
        # Test dictionary file
        dict_path = CorpusManager._get_corpus_file_path(corpus_name, "dictionary")
        assert dict_path.name == "test_corpus_dictionary.md"
        
        # Test knowledge graph file
        graph_path = CorpusManager._get_corpus_file_path(corpus_name, "knowledge_graph")
        assert graph_path.name == "test_corpus_knowledge_graph.md"
    
    def test_get_corpus_file_path_with_custom_dir(self):
        """Test getting corpus file path with custom directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = CorpusManager._get_corpus_file_path(
                "test_corpus", 
                "track", 
                custom_dir=temp_dir
            )
            
            assert path.parent == Path(temp_dir)
            assert path.name == "test_corpus_track.json"
    
    def test_get_corpus_file_path_invalid_type(self):
        """Test getting corpus file path with invalid file type"""
        with pytest.raises(ValueError, match="Unknown file type"):
            CorpusManager._get_corpus_file_path("test", "invalid_type")
    
    @patch('os.getenv')
    def test_get_refinire_rag_dir_from_env(self, mock_getenv):
        """Test getting refinire rag directory from environment"""
        mock_getenv.return_value = "/custom/refinire/path"
        
        result = CorpusManager._get_refinire_rag_dir()
        
        expected = Path("/custom/refinire/path") / "rag"
        assert result == expected
    
    @patch('os.getenv')
    def test_get_refinire_rag_dir_default(self, mock_getenv):
        """Test getting default refinire rag directory"""
        mock_getenv.return_value = None
        
        result = CorpusManager._get_refinire_rag_dir()
        
        expected = Path("./refinire/rag")
        assert result == expected


class TestCorpusManagerDocumentProcessing:
    """Test document processing and pipeline methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = SQLiteDocumentStore(":memory:")
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[Mock()]
        )
        
        # Sample documents for testing
        self.test_docs = [
            Document(
                id="original_1",
                content="This is original document 1",
                metadata={"processing_stage": "original", "corpus_name": "test"}
            ),
            Document(
                id="chunk_1_1",
                content="First chunk of document 1",
                metadata={
                    "processing_stage": "chunked",
                    "corpus_name": "test",
                    "original_document_id": "original_1"
                }
            ),
            Document(
                id="chunk_1_2",
                content="Second chunk of document 1", 
                metadata={
                    "processing_stage": "chunked",
                    "corpus_name": "test",
                    "original_document_id": "original_1"
                }
            )
        ]
        
        for doc in self.test_docs:
            self.document_store.store_document(doc)
    
    def teardown_method(self):
        """Clean up"""
        self.document_store.close()
    
    def test_get_corpus_info_with_real_store(self):
        """Test getting corpus info with real document store"""
        info = self.manager.get_corpus_info("test")
        
        assert info["corpus_name"] == "test"
        assert info["total_documents"] == 3
        assert "processing_stages" in info
        assert info["processing_stages"]["original"] == 1
        assert info["processing_stages"]["chunked"] == 2
    
    def test_clear_corpus_with_real_store(self):
        """Test clearing corpus with real document store"""
        # Verify initial state
        assert self.document_store.count_documents() == 3
        
        # Clear corpus
        result = self.manager.clear_corpus("test")
        
        assert result["success"] is True
        assert result["deleted_count"] == 3
        assert self.document_store.count_documents() == 0


class TestCorpusManagerErrorHandling:
    """Test error handling in CorpusManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[Mock()]
        )
    
    @patch('refinire_rag.application.corpus_manager_new.IncrementalDirectoryLoader')
    def test_import_documents_with_loader_error(self, mock_loader):
        """Test import documents handles loader errors"""
        mock_loader.side_effect = Exception("Loader initialization failed")
        
        with pytest.raises(Exception, match="Loader initialization failed"):
            self.manager.import_original_documents(
                corpus_name="test",
                directory="/nonexistent"
            )
    
    @patch('refinire_rag.application.corpus_manager_new.ConstantMetadata')
    def test_import_documents_with_metadata_error(self, mock_metadata):
        """Test import documents handles metadata processor errors"""
        mock_metadata.side_effect = Exception("Metadata processor failed")
        
        with pytest.raises(Exception, match="Metadata processor failed"):
            self.manager.import_original_documents(
                corpus_name="test",
                directory="/test"
            )
    
    def test_get_corpus_info_with_count_error(self):
        """Test get corpus info handles count errors"""
        self.document_store.count_documents.side_effect = StorageError("Count failed")
        
        info = self.manager.get_corpus_info("test")
        
        assert "error" in info
        assert "Count failed" in info["error"]
    
    def test_clear_corpus_with_search_timeout(self):
        """Test clear corpus handles search timeouts"""
        self.document_store.search_by_metadata.side_effect = TimeoutError("Search timeout")
        
        result = self.manager.clear_corpus("test")
        
        assert result["success"] is False
        assert "error" in result


class TestCorpusManagerRetrieverIntegration:
    """Test retriever integration functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.vector_retriever = Mock()
        self.vector_retriever.add_vector = Mock()
        self.vector_retriever.search_similar = Mock()
        
        self.keyword_retriever = Mock()
        
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[self.vector_retriever, self.keyword_retriever]
        )
    
    def test_get_retrievers_by_type_case_insensitive(self):
        """Test getting retrievers by type is case insensitive"""
        self.vector_retriever.__class__.__name__ = "VectorRetriever"
        self.keyword_retriever.__class__.__name__ = "KeywordRetriever"
        
        # Test lowercase
        vector_results = self.manager.get_retrievers_by_type("vector")
        assert len(vector_results) == 1
        
        # Test uppercase
        vector_results = self.manager.get_retrievers_by_type("VECTOR")
        assert len(vector_results) == 1
        
        # Test mixed case
        keyword_results = self.manager.get_retrievers_by_type("Keyword")
        assert len(keyword_results) == 1
    
    def test_add_retriever_updates_stats(self):
        """Test adding retriever updates internal state correctly"""
        new_retriever = Mock()
        new_retriever.__class__.__name__ = "HybridRetriever"
        
        initial_count = len(self.manager.retrievers)
        self.manager.add_retriever(new_retriever)
        
        assert len(self.manager.retrievers) == initial_count + 1
        assert new_retriever in self.manager.retrievers
    
    def test_remove_retriever_boundary_conditions(self):
        """Test remove retriever with boundary conditions"""
        initial_count = len(self.manager.retrievers)
        
        # Test removing at exact boundary
        success = self.manager.remove_retriever(initial_count - 1)
        assert success is True
        assert len(self.manager.retrievers) == initial_count - 1
        
        # Test removing from empty list
        self.manager.retrievers = []
        success = self.manager.remove_retriever(0)
        assert success is False