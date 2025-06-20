"""
Complete tests for CorpusManager to achieve maximum coverage
CorpusManagerの最大カバレッジを達成するための完全テスト
"""

import pytest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime

from refinire_rag.application.corpus_manager_new import CorpusManager, CorpusStats
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.models.document import Document
from refinire_rag.exceptions import StorageError


class TestCorpusManagerImportDocumentsFull:
    """Test import_original_documents method comprehensively"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[Mock()]
        )
    
    @patch('refinire_rag.application.corpus_manager_new.IncrementalDirectoryLoader')
    @patch('refinire_rag.application.corpus_manager_new.ConstantMetadata')
    @patch('refinire_rag.application.corpus_manager_new.logger')
    def test_import_original_documents_full_workflow(self, mock_logger, mock_constant_metadata, mock_loader):
        """Test complete import workflow with all features"""
        # Setup mocks
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        
        # Mock sync result
        mock_sync_result = Mock()
        mock_sync_result.added_documents = ["doc1", "doc2", "doc3"]
        mock_sync_result.updated_documents = ["doc4"]
        mock_sync_result.has_errors = False
        mock_sync_result.errors = []
        mock_loader_instance.sync_with_store.return_value = mock_sync_result
        
        # Mock file tracker
        mock_file_tracker = Mock()
        mock_loader_instance.file_tracker = mock_file_tracker
        
        # Execute with force_reload
        stats = self.manager.import_original_documents(
            corpus_name="test_corpus",
            directory="/test/dir",
            glob="**/*.md",
            force_reload=True,
            additional_metadata={"project": "test"}
        )
        
        # Verify
        assert isinstance(stats, CorpusStats)
        assert stats.total_files_processed == 4  # 3 added + 1 updated
        assert stats.total_documents_created == 4
        assert stats.pipeline_stages_executed == 1
        assert stats.documents_by_stage["original"] == 4
        assert stats.errors_encountered == 0
        
        # Verify force reload called
        mock_file_tracker.clear_tracking_data.assert_called_once()
        
        # Verify logging
        mock_logger.info.assert_any_call("Importing documents from: /test/dir")
        mock_logger.info.assert_any_call("Using glob pattern: **/*.md")
        mock_logger.warning.assert_any_call(
            "Multithreading not yet supported by IncrementalDirectoryLoader, processing sequentially"
        )
    
    @patch('refinire_rag.application.corpus_manager_new.IncrementalDirectoryLoader')
    @patch('refinire_rag.application.corpus_manager_new.ConstantMetadata')
    def test_import_original_documents_with_errors(self, mock_constant_metadata, mock_loader):
        """Test import with sync errors"""
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        
        # Mock sync result with errors
        mock_sync_result = Mock()
        mock_sync_result.added_documents = ["doc1"]
        mock_sync_result.updated_documents = []
        mock_sync_result.has_errors = True
        mock_sync_result.errors = ["Error 1", "Error 2"]
        mock_loader_instance.sync_with_store.return_value = mock_sync_result
        
        mock_file_tracker = Mock()
        mock_loader_instance.file_tracker = mock_file_tracker
        
        stats = self.manager.import_original_documents(
            corpus_name="test_corpus",
            directory="/test/dir"
        )
        
        assert stats.total_documents_created == 1
        assert stats.errors_encountered == 2
    
    @patch('refinire_rag.application.corpus_manager_new.IncrementalDirectoryLoader')
    def test_import_original_documents_loader_exception(self, mock_loader):
        """Test import when loader raises exception"""
        mock_loader.side_effect = Exception("Loader creation failed")
        
        with pytest.raises(Exception, match="Loader creation failed"):
            self.manager.import_original_documents(
                corpus_name="test_corpus",
                directory="/test/dir"
            )
    
    @patch('refinire_rag.application.corpus_manager_new.IncrementalDirectoryLoader')
    @patch('refinire_rag.application.corpus_manager_new.ConstantMetadata')
    def test_import_original_documents_sync_exception(self, mock_constant_metadata, mock_loader):
        """Test import when sync raises exception"""
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        mock_loader_instance.sync_with_store.side_effect = Exception("Sync failed")
        
        with pytest.raises(Exception, match="Sync failed"):
            self.manager.import_original_documents(
                corpus_name="test_corpus",
                directory="/test/dir"
            )
    
    @patch('refinire_rag.application.corpus_manager_new.IncrementalDirectoryLoader')
    @patch('refinire_rag.application.corpus_manager_new.ConstantMetadata')
    def test_import_with_knowledge_creation(self, mock_constant_metadata, mock_loader):
        """Test import with dictionary and knowledge graph creation"""
        # Setup basic mocks
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        
        mock_sync_result = Mock()
        mock_sync_result.added_documents = ["doc1"]
        mock_sync_result.updated_documents = []
        mock_sync_result.has_errors = False
        mock_sync_result.errors = []
        mock_loader_instance.sync_with_store.return_value = mock_sync_result
        
        mock_file_tracker = Mock()
        mock_loader_instance.file_tracker = mock_file_tracker
        
        # Mock knowledge creation
        with patch.object(self.manager, '_create_knowledge_artifacts') as mock_create_knowledge:
            stats = self.manager.import_original_documents(
                corpus_name="test_corpus",
                directory="/test/dir",
                create_dictionary=True,
                create_knowledge_graph=True,
                dictionary_output_dir="/dict/output",
                graph_output_dir="/graph/output"
            )
            
            mock_create_knowledge.assert_called_once_with(
                corpus_name="test_corpus",
                create_dictionary=True,
                create_knowledge_graph=True,
                dictionary_output_dir="/dict/output",
                graph_output_dir="/graph/output",
                stats=stats
            )
    
    def test_import_default_tracking_file_path(self):
        """Test that default tracking file path is generated correctly"""
        with patch('refinire_rag.application.corpus_manager_new.IncrementalDirectoryLoader') as mock_loader:
            with patch('refinire_rag.application.corpus_manager_new.ConstantMetadata'):
                mock_loader_instance = Mock()
                mock_loader.return_value = mock_loader_instance
                
                mock_sync_result = Mock()
                mock_sync_result.added_documents = []
                mock_sync_result.updated_documents = []
                mock_sync_result.has_errors = False
                mock_sync_result.errors = []
                mock_loader_instance.sync_with_store.return_value = mock_sync_result
                
                mock_file_tracker = Mock()
                mock_loader_instance.file_tracker = mock_file_tracker
                
                with patch.object(self.manager, '_get_corpus_file_path') as mock_get_path:
                    mock_get_path.return_value = Path("/default/test_corpus_track.json")
                    
                    self.manager.import_original_documents(
                        corpus_name="test_corpus",
                        directory="/test/dir"
                    )
                    
                    mock_get_path.assert_called_once_with("test_corpus", "track")


class TestCorpusManagerKnowledgeArtifacts:
    """Test knowledge artifacts creation methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[Mock()]
        )
    
    @patch('refinire_rag.application.corpus_manager_new.DictionaryMaker')
    @patch('refinire_rag.application.corpus_manager_new.DictionaryMakerConfig')
    @patch('refinire_rag.application.corpus_manager_new.DocumentPipeline')
    @patch('refinire_rag.application.corpus_manager_new.DocumentStoreLoader')
    def test_create_knowledge_artifacts_dictionary_only(self, mock_loader_class, mock_pipeline, mock_dict_config, mock_dict_maker):
        """Test creating dictionary only"""
        # Setup mocks
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        mock_dict_maker_instance = Mock()
        mock_dict_maker.return_value = mock_dict_maker_instance
        
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.process_document.return_value = []
        
        # Mock file path
        with patch.object(self.manager, '_get_corpus_file_path') as mock_get_path:
            mock_get_path.return_value = Path("/output/test_corpus_dictionary.md")
            
            stats = CorpusStats()
            self.manager._create_knowledge_artifacts(
                corpus_name="test_corpus",
                create_dictionary=True,
                create_knowledge_graph=False,
                dictionary_output_dir="/dict/output",
                graph_output_dir=None,
                stats=stats
            )
            
            # Verify dictionary configuration
            mock_dict_config.assert_called_once()
            config_call = mock_dict_config.call_args[1]
            assert config_call["dictionary_file_path"] == "/output/test_corpus_dictionary.md"
            assert config_call["focus_on_technical_terms"] is True
            assert config_call["extract_abbreviations"] is True
            assert config_call["detect_expression_variations"] is True
            
            # Verify pipeline execution
            mock_pipeline.assert_called_once_with([mock_loader, mock_dict_maker_instance])
            mock_pipeline_instance.process_document.assert_called_once()
            
            assert stats.pipeline_stages_executed == 1
    
    @patch('refinire_rag.application.corpus_manager_new.GraphBuilder')
    @patch('refinire_rag.application.corpus_manager_new.GraphBuilderConfig')
    @patch('refinire_rag.application.corpus_manager_new.DocumentPipeline')
    @patch('refinire_rag.application.corpus_manager_new.DocumentStoreLoader')
    def test_create_knowledge_artifacts_graph_only(self, mock_loader_class, mock_pipeline, mock_graph_config, mock_graph_builder):
        """Test creating knowledge graph only"""
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        mock_graph_builder_instance = Mock()
        mock_graph_builder.return_value = mock_graph_builder_instance
        
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.process_document.return_value = []
        
        with patch.object(self.manager, '_get_corpus_file_path') as mock_get_path:
            mock_get_path.return_value = Path("/output/test_corpus_knowledge_graph.md")
            
            stats = CorpusStats()
            self.manager._create_knowledge_artifacts(
                corpus_name="test_corpus",
                create_dictionary=False,
                create_knowledge_graph=True,
                dictionary_output_dir=None,
                graph_output_dir="/graph/output",
                stats=stats
            )
            
            # Verify graph configuration
            mock_graph_config.assert_called_once()
            config_call = mock_graph_config.call_args[1]
            assert config_call["graph_file_path"] == "/output/test_corpus_knowledge_graph.md"
            assert config_call["focus_on_important_relationships"] is True
            assert config_call["extract_hierarchical_relationships"] is True
            assert config_call["extract_causal_relationships"] is True
            
            assert stats.pipeline_stages_executed == 1
    
    @patch('refinire_rag.application.corpus_manager_new.DictionaryMaker')
    @patch('refinire_rag.application.corpus_manager_new.GraphBuilder')
    @patch('refinire_rag.application.corpus_manager_new.DocumentPipeline')
    @patch('refinire_rag.application.corpus_manager_new.DocumentStoreLoader')
    def test_create_knowledge_artifacts_both(self, mock_loader_class, mock_pipeline, mock_graph_builder, mock_dict_maker):
        """Test creating both dictionary and knowledge graph"""
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        mock_dict_maker_instance = Mock()
        mock_dict_maker.return_value = mock_dict_maker_instance
        
        mock_graph_builder_instance = Mock()
        mock_graph_builder.return_value = mock_graph_builder_instance
        
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.process_document.return_value = []
        
        with patch.object(self.manager, '_get_corpus_file_path') as mock_get_path:
            mock_get_path.side_effect = [
                Path("/output/test_corpus_dictionary.md"),
                Path("/output/test_corpus_knowledge_graph.md")
            ]
            
            stats = CorpusStats()
            self.manager._create_knowledge_artifacts(
                corpus_name="test_corpus",
                create_dictionary=True,
                create_knowledge_graph=True,
                dictionary_output_dir="/output",
                graph_output_dir="/output",
                stats=stats
            )
            
            # Should create two pipelines and execute both
            assert mock_pipeline.call_count == 2
            assert mock_pipeline_instance.process_document.call_count == 2
            assert stats.pipeline_stages_executed == 2
    
    @patch('refinire_rag.application.corpus_manager_new.DocumentStoreLoader')
    def test_create_knowledge_artifacts_with_exception(self, mock_loader_class):
        """Test knowledge artifacts creation with exception"""
        mock_loader_class.side_effect = Exception("Loader creation failed")
        
        stats = CorpusStats()
        
        # Should handle exception gracefully
        self.manager._create_knowledge_artifacts(
            corpus_name="test_corpus",
            create_dictionary=True,
            create_knowledge_graph=False,
            dictionary_output_dir=None,
            graph_output_dir=None,
            stats=stats
        )
        
        assert stats.errors_encountered == 1


class TestCorpusManagerRebuildCorpus:
    """Test rebuild_corpus_from_original method"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[Mock()]
        )
    
    def test_rebuild_corpus_no_original_documents(self):
        """Test rebuild when no original documents exist"""
        # Mock get_documents_by_stage to return empty list
        with patch.object(self.manager, '_get_documents_by_stage') as mock_get_docs:
            mock_get_docs.return_value = iter([])
            
            with pytest.raises(ValueError, match="No original documents found"):
                self.manager.rebuild_corpus_from_original("test_corpus")
    
    @patch('pathlib.Path.exists')
    def test_rebuild_corpus_with_dictionary_file_not_found(self, mock_exists):
        """Test rebuild with dictionary file that doesn't exist"""
        # Mock original documents exist
        with patch.object(self.manager, '_get_documents_by_stage') as mock_get_docs:
            mock_get_docs.return_value = iter([Mock(), Mock()])
            
            mock_exists.return_value = False
            
            with pytest.raises(FileNotFoundError, match="Dictionary file not found"):
                self.manager.rebuild_corpus_from_original(
                    corpus_name="test_corpus",
                    use_dictionary=True,
                    dictionary_file_path="/nonexistent/dict.md"
                )
    
    @patch('pathlib.Path.exists')
    def test_rebuild_corpus_with_graph_file_not_found(self, mock_exists):
        """Test rebuild with knowledge graph file that doesn't exist"""
        with patch.object(self.manager, '_get_documents_by_stage') as mock_get_docs:
            mock_get_docs.return_value = iter([Mock(), Mock()])
            
            mock_exists.return_value = False
            
            with pytest.raises(FileNotFoundError, match="Knowledge graph file not found"):
                self.manager.rebuild_corpus_from_original(
                    corpus_name="test_corpus",
                    use_knowledge_graph=True,
                    graph_file_path="/nonexistent/graph.md"
                )
    
    @patch('pathlib.Path.exists')
    def test_rebuild_corpus_finds_default_files(self, mock_exists):
        """Test rebuild that finds default corpus files"""
        with patch.object(self.manager, '_get_documents_by_stage') as mock_get_docs:
            mock_get_docs.return_value = iter([Mock(), Mock()])
            
            # Mock file existence for default paths
            mock_exists.return_value = True
            
            with patch.object(self.manager, '_get_corpus_file_path') as mock_get_path:
                mock_dict_path = Path("/default/test_corpus_dictionary.md")
                mock_graph_path = Path("/default/test_corpus_knowledge_graph.md")
                mock_get_path.side_effect = [mock_dict_path, mock_graph_path]
                
                # Mock the processing pipeline
                with patch('refinire_rag.application.corpus_manager_new.DocumentPipeline') as mock_pipeline:
                    with patch('refinire_rag.application.corpus_manager_new.DocumentStoreLoader'):
                        with patch('refinire_rag.application.corpus_manager_new.NormalizerConfig'):
                            mock_pipeline_instance = Mock()
                            mock_pipeline.return_value = mock_pipeline_instance
                            mock_pipeline_instance.process_documents.return_value = iter([])
                            
                            stats = self.manager.rebuild_corpus_from_original(
                                corpus_name="test_corpus",
                                use_dictionary=True,
                                use_knowledge_graph=True
                            )
                            
                            assert isinstance(stats, CorpusStats)
                            mock_get_path.assert_any_call("test_corpus", "dictionary")
                            mock_get_path.assert_any_call("test_corpus", "knowledge_graph")
    
    @patch('pathlib.Path.exists')
    def test_rebuild_corpus_missing_default_files_warning(self, mock_exists):
        """Test rebuild when default corpus files are missing"""
        with patch.object(self.manager, '_get_documents_by_stage') as mock_get_docs:
            mock_get_docs.return_value = iter([Mock(), Mock()])
            
            # Mock file doesn't exist
            mock_exists.return_value = False
            
            with patch.object(self.manager, '_get_corpus_file_path') as mock_get_path:
                mock_dict_path = Path("/default/test_corpus_dictionary.md")
                mock_get_path.return_value = mock_dict_path
                
                with patch('refinire_rag.application.corpus_manager_new.logger') as mock_logger:
                    # Mock the processing pipeline for the case where dictionary is disabled
                    with patch('refinire_rag.application.corpus_manager_new.DocumentPipeline') as mock_pipeline:
                        with patch('refinire_rag.application.corpus_manager_new.DocumentStoreLoader'):
                            mock_pipeline_instance = Mock()
                            mock_pipeline.return_value = mock_pipeline_instance
                            mock_pipeline_instance.process_documents.return_value = iter([])
                            
                            stats = self.manager.rebuild_corpus_from_original(
                                corpus_name="test_corpus",
                                use_dictionary=True  # Will be disabled due to missing file
                            )
                            
                            # Should log warning about missing file
                            mock_logger.warning.assert_any_call(
                                f"No dictionary file specified and corpus dictionary not found: {mock_dict_path}"
                            )


class TestCorpusManagerUtilityMethods:
    """Test utility and helper methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[Mock()]
        )
    
    def test_get_documents_by_stage(self):
        """Test getting documents by processing stage"""
        # Mock search results
        test_docs = [
            Document(id="doc1", content="Content 1", metadata={"processing_stage": "original"}),
            Document(id="doc2", content="Content 2", metadata={"processing_stage": "original"})
        ]
        search_results = [Mock(document=doc) for doc in test_docs]
        self.document_store.search_by_metadata.return_value = search_results
        
        # Test the method
        docs = list(self.manager._get_documents_by_stage("original"))
        
        assert len(docs) == 2
        assert docs[0].id == "doc1"
        assert docs[1].id == "doc2"
        
        # Verify search was called with correct filters
        self.document_store.search_by_metadata.assert_called_once_with(
            {"processing_stage": "original"}
        )
    
    def test_get_documents_by_stage_with_corpus_filter(self):
        """Test getting documents by stage with corpus name filter"""
        test_docs = [
            Document(id="doc1", content="Content 1", metadata={
                "processing_stage": "chunked",
                "corpus_name": "test_corpus"
            })
        ]
        search_results = [Mock(document=doc) for doc in test_docs]
        self.document_store.search_by_metadata.return_value = search_results
        
        docs = list(self.manager._get_documents_by_stage("chunked", corpus_name="test_corpus"))
        
        assert len(docs) == 1
        assert docs[0].id == "doc1"
        
        # Verify search was called with both filters
        expected_filters = {
            "processing_stage": "chunked",
            "corpus_name": "test_corpus"
        }
        self.document_store.search_by_metadata.assert_called_once_with(expected_filters)
    
    def test_get_documents_by_stage_empty_result(self):
        """Test getting documents by stage with no results"""
        self.document_store.search_by_metadata.return_value = []
        
        docs = list(self.manager._get_documents_by_stage("nonexistent"))
        
        assert len(docs) == 0
    
    def test_get_documents_by_stage_with_search_error(self):
        """Test getting documents by stage with search error"""
        self.document_store.search_by_metadata.side_effect = StorageError("Search failed")
        
        with pytest.raises(StorageError, match="Search failed"):
            list(self.manager._get_documents_by_stage("original"))


class TestCorpusManagerFilterCreation:
    """Test filter creation and glob handling"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.manager = CorpusManager(
            document_store=Mock(),
            retrievers=[Mock()]
        )
    
    def test_create_filter_config_simple_extensions(self):
        """Test creating filter config for various simple extension patterns"""
        # Test .md extension
        filter_config = self.manager._create_filter_config_from_glob("**/*.md")
        assert filter_config is not None
        
        # Test .txt extension
        filter_config = self.manager._create_filter_config_from_glob("*.txt")
        assert filter_config is not None
        
        # Test .py extension
        filter_config = self.manager._create_filter_config_from_glob("src/**/*.py")
        assert filter_config is not None
    
    def test_create_filter_config_brace_expansion(self):
        """Test creating filter config for brace expansion patterns"""
        # Test multiple extensions
        filter_config = self.manager._create_filter_config_from_glob("*.{txt,md,py}")
        assert filter_config is not None
        
        # Test with spaces
        filter_config = self.manager._create_filter_config_from_glob("*.{txt, md, py}")
        assert filter_config is not None
        
        # Test single extension in braces
        filter_config = self.manager._create_filter_config_from_glob("*.{md}")
        assert filter_config is not None
    
    def test_create_filter_config_no_extension_patterns(self):
        """Test creating filter config for patterns without extensions"""
        # Test wildcard without extensions
        filter_config = self.manager._create_filter_config_from_glob("**/*")
        assert filter_config is None
        
        # Test complex pattern
        filter_config = self.manager._create_filter_config_from_glob("**/test_*/**/*")
        assert filter_config is None
        
        # Test name-based pattern
        filter_config = self.manager._create_filter_config_from_glob("README*")
        assert filter_config is None
    
    def test_create_filter_config_edge_cases(self):
        """Test creating filter config for edge cases"""
        # Test empty pattern
        filter_config = self.manager._create_filter_config_from_glob("")
        assert filter_config is None
        
        # Test pattern with no asterisk
        filter_config = self.manager._create_filter_config_from_glob("test.txt")
        assert filter_config is None
        
        # Test malformed brace pattern
        filter_config = self.manager._create_filter_config_from_glob("*.{txt,}")
        assert filter_config is not None  # Should still extract txt
        
        # Test pattern with multiple dots
        filter_config = self.manager._create_filter_config_from_glob("**/*.test.md")
        assert filter_config is not None


class TestCorpusManagerRealIntegration:
    """Integration tests with real document store"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = SQLiteDocumentStore(":memory:")
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[Mock()]
        )
        
        # Add test documents
        self.test_docs = [
            Document(
                id="original_1",
                content="Original document 1",
                metadata={"processing_stage": "original", "corpus_name": "test"}
            ),
            Document(
                id="original_2", 
                content="Original document 2",
                metadata={"processing_stage": "original", "corpus_name": "test"}
            ),
            Document(
                id="chunk_1",
                content="Chunk from document 1",
                metadata={"processing_stage": "chunked", "corpus_name": "test"}
            )
        ]
        
        for doc in self.test_docs:
            self.document_store.store_document(doc)
    
    def teardown_method(self):
        """Clean up"""
        self.document_store.close()
    
    def test_get_documents_by_stage_real_store(self):
        """Test getting documents by stage with real store"""
        # Get original documents
        original_docs = list(self.manager._get_documents_by_stage("original"))
        assert len(original_docs) == 2
        
        # Get chunked documents
        chunked_docs = list(self.manager._get_documents_by_stage("chunked"))
        assert len(chunked_docs) == 1
        
        # Get documents with corpus filter
        corpus_docs = list(self.manager._get_documents_by_stage("original", corpus_name="test"))
        assert len(corpus_docs) == 2
        
        # Get documents with non-matching corpus filter
        other_docs = list(self.manager._get_documents_by_stage("original", corpus_name="other"))
        assert len(other_docs) == 0
    
    def test_get_corpus_info_real_data(self):
        """Test getting corpus info with real document store"""
        info = self.manager.get_corpus_info("test")
        
        assert info["corpus_name"] == "test"
        assert info["total_documents"] == 3
        assert "processing_stages" in info
        assert info["processing_stages"]["original"] == 2
        assert info["processing_stages"]["chunked"] == 1
        assert "storage_stats" in info
    
    def test_clear_corpus_real_data(self):
        """Test clearing corpus with real document store"""
        # Verify initial state
        assert self.document_store.count_documents() == 3
        
        # Clear corpus
        result = self.manager.clear_corpus("test")
        
        assert result["success"] is True
        assert result["deleted_count"] == 3
        assert result["failed_count"] == 0
        assert self.document_store.count_documents() == 0
    
    def test_clear_corpus_partial_failure_real_data(self):
        """Test clearing corpus with some deletion failures"""
        # Mock delete_document to fail for one document
        original_delete = self.document_store.delete_document
        
        def mock_delete(doc_id):
            if doc_id == "original_1":
                return False  # Simulate failure
            return original_delete(doc_id)
        
        self.document_store.delete_document = mock_delete
        
        result = self.manager.clear_corpus("test")
        
        assert result["success"] is False
        assert result["deleted_count"] == 2
        assert result["failed_count"] == 1
        assert self.document_store.count_documents() == 1