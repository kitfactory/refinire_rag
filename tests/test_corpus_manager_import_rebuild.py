"""
Comprehensive tests for CorpusManager import and rebuild functionality
CorpusManagerのインポートとリビルド機能の包括的テスト

This module tests the complex import and rebuild operations of CorpusManager.
このモジュールは、CorpusManagerの複雑なインポートとリビルド操作をテストします。
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any

from refinire_rag.application.corpus_manager_new import CorpusManager, CorpusStats
from refinire_rag.models.document import Document
from refinire_rag.storage.document_store import DocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.loader.document_store_loader import DocumentStoreLoader


class TestCorpusManagerImport:
    """
    Test CorpusManager import functionality
    CorpusManagerのインポート機能のテスト
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

    @patch('refinire_rag.application.corpus_manager_new.DocumentStoreLoader')
    @patch('refinire_rag.application.corpus_manager_new.Path.exists')
    def test_import_original_documents_basic(self, mock_exists, mock_loader_class):
        """
        Test basic import_original_documents functionality
        基本的なimport_original_documents機能のテスト
        """
        # Setup mocks
        mock_exists.return_value = True
        mock_loader = Mock(spec=DocumentStoreLoader)
        mock_loader_class.return_value = mock_loader
        
        # Mock sync results
        mock_sync_result = Mock()
        mock_sync_result.added_documents = [
            Document(id="1", content="test1", metadata={"source": "file1.txt"}),
            Document(id="2", content="test2", metadata={"source": "file2.txt"})
        ]
        mock_sync_result.get_stats.return_value = {"total_files": 2, "total_documents": 2}
        mock_loader.sync_from_directory.return_value = mock_sync_result
        
        # Setup mock for knowledge artifacts creation
        with patch.object(self.corpus_manager, '_create_knowledge_artifacts') as mock_create_artifacts:
            mock_create_artifacts.return_value = []
            
            # Call import method
            result = self.corpus_manager.import_original_documents(
                source_directory="./test_docs",
                corpus_name="test_corpus"
            )
        
        # Verify loader was created and called
        mock_loader_class.assert_called_once()
        mock_loader.sync_from_directory.assert_called_once()
        
        # Verify artifacts creation was called
        mock_create_artifacts.assert_called_once()
        
        # Verify result
        assert result is not None
        assert hasattr(result, 'added_documents')

    @patch('refinire_rag.application.corpus_manager_new.DocumentStoreLoader')
    @patch('refinire_rag.application.corpus_manager_new.Path.exists')
    def test_import_original_documents_with_glob(self, mock_exists, mock_loader_class):
        """
        Test import_original_documents with glob pattern
        globパターンでのimport_original_documentsテスト
        """
        # Setup mocks
        mock_exists.return_value = True
        mock_loader = Mock(spec=DocumentStoreLoader)
        mock_loader_class.return_value = mock_loader
        
        # Mock sync results
        mock_sync_result = Mock()
        mock_sync_result.added_documents = [
            Document(id="1", content="test1", metadata={"source": "file1.txt"})
        ]
        mock_sync_result.get_stats.return_value = {"total_files": 1, "total_documents": 1}
        mock_loader.sync_from_directory.return_value = mock_sync_result
        
        # Setup mock for knowledge artifacts creation
        with patch.object(self.corpus_manager, '_create_knowledge_artifacts') as mock_create_artifacts:
            mock_create_artifacts.return_value = []
            
            # Call import method with glob pattern
            result = self.corpus_manager.import_original_documents(
                source_directory="./test_docs",
                corpus_name="test_corpus",
                glob_pattern="*.txt"
            )
        
        # Verify loader configuration included filter
        call_args = mock_loader_class.call_args
        assert call_args is not None
        
        # Verify result
        assert result is not None

    @patch('refinire_rag.application.corpus_manager_new.DocumentStoreLoader')
    @patch('refinire_rag.application.corpus_manager_new.Path.exists')
    def test_import_original_documents_nonexistent_directory(self, mock_exists, mock_loader_class):
        """
        Test import_original_documents with nonexistent directory
        存在しないディレクトリでのimport_original_documentsテスト
        """
        # Setup mocks
        mock_exists.return_value = False
        
        # Call import method and expect exception
        with pytest.raises(ValueError, match="Source directory does not exist"):
            self.corpus_manager.import_original_documents(
                source_directory="./nonexistent",
                corpus_name="test_corpus"
            )
        
        # Verify loader was not created
        mock_loader_class.assert_not_called()

    @patch('refinire_rag.application.corpus_manager_new.DocumentStoreLoader')
    @patch('refinire_rag.application.corpus_manager_new.Path.exists')
    def test_import_original_documents_with_error_handling(self, mock_exists, mock_loader_class):
        """
        Test import_original_documents error handling
        import_original_documentsのエラーハンドリングテスト
        """
        # Setup mocks
        mock_exists.return_value = True
        mock_loader = Mock(spec=DocumentStoreLoader)
        mock_loader_class.return_value = mock_loader
        
        # Mock loader to raise exception
        mock_loader.sync_from_directory.side_effect = Exception("Sync failed")
        
        # Call import method and expect exception
        with pytest.raises(Exception, match="Sync failed"):
            self.corpus_manager.import_original_documents(
                source_directory="./test_docs",
                corpus_name="test_corpus"
            )

    def test_create_knowledge_artifacts_basic(self):
        """
        Test _create_knowledge_artifacts basic functionality
        _create_knowledge_artifacts基本機能のテスト
        """
        # Create sample documents
        documents = [
            Document(id="1", content="Python is a programming language", metadata={"source": "file1.txt"}),
            Document(id="2", content="Machine learning uses algorithms", metadata={"source": "file2.txt"})
        ]
        
        # Mock the knowledge artifact processors
        with patch('refinire_rag.application.corpus_manager_new.PluginFactory') as mock_factory:
            mock_dictionary_maker = Mock()
            mock_graph_builder = Mock()
            
            # Setup factory to return mock processors
            mock_factory.create_plugin.side_effect = lambda plugin_type, **kwargs: {
                'dictionary_maker': mock_dictionary_maker,
                'graph_builder': mock_graph_builder
            }.get(plugin_type)
            
            # Mock processor outputs
            mock_dictionary_maker.process.return_value = iter([])
            mock_graph_builder.process.return_value = iter([])
            
            # Call method
            result = self.corpus_manager._create_knowledge_artifacts(
                documents=documents,
                corpus_name="test_corpus",
                source_directory="./test"
            )
        
        # Verify processors were called
        mock_dictionary_maker.process.assert_called_once()
        mock_graph_builder.process.assert_called_once()
        
        # Verify result is a list
        assert isinstance(result, list)

    def test_create_knowledge_artifacts_with_output_directories(self):
        """
        Test _create_knowledge_artifacts with custom output directories
        カスタム出力ディレクトリでの_create_knowledge_artifacts テスト
        """
        # Create sample documents
        documents = [
            Document(id="1", content="Test content", metadata={"source": "file1.txt"})
        ]
        
        # Mock the knowledge artifact processors
        with patch('refinire_rag.application.corpus_manager_new.PluginFactory') as mock_factory:
            mock_dictionary_maker = Mock()
            mock_graph_builder = Mock()
            
            # Setup factory to return mock processors
            mock_factory.create_plugin.side_effect = lambda plugin_type, **kwargs: {
                'dictionary_maker': mock_dictionary_maker,
                'graph_builder': mock_graph_builder
            }.get(plugin_type)
            
            # Mock processor outputs
            mock_dictionary_maker.process.return_value = iter([])
            mock_graph_builder.process.return_value = iter([])
            
            # Call method with custom directories
            result = self.corpus_manager._create_knowledge_artifacts(
                documents=documents,
                corpus_name="test_corpus",
                source_directory="./test",
                dictionary_output_directory="./custom_dict",
                knowledge_graph_output_directory="./custom_kg"
            )
        
        # Verify processors were called with custom paths
        mock_factory.create_plugin.assert_called()
        assert isinstance(result, list)


class TestCorpusManagerRebuild:
    """
    Test CorpusManager rebuild functionality
    CorpusManagerのリビルド機能のテスト
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

    @patch('refinire_rag.application.corpus_manager_new.Path.exists')
    def test_rebuild_corpus_from_original_basic(self, mock_exists):
        """
        Test basic rebuild_corpus_from_original functionality
        基本的なrebuild_corpus_from_original機能のテスト
        """
        # Setup mocks
        mock_exists.return_value = True
        
        # Mock document store to return original documents
        original_docs = [
            Document(id="orig_1", content="Original content 1", metadata={"source": "file1.txt", "processing_stage": "original"}),
            Document(id="orig_2", content="Original content 2", metadata={"source": "file2.txt", "processing_stage": "original"})
        ]
        
        with patch.object(self.corpus_manager, '_get_documents_by_stage') as mock_get_docs:
            mock_get_docs.return_value = original_docs
            
            with patch.object(self.corpus_manager, '_create_knowledge_artifacts') as mock_create_artifacts:
                mock_create_artifacts.return_value = []
                
                # Call rebuild method
                result = self.corpus_manager.rebuild_corpus_from_original("test_corpus")
        
        # Verify original documents were retrieved
        mock_get_docs.assert_called_with("original")
        
        # Verify knowledge artifacts were created
        mock_create_artifacts.assert_called_once()
        
        # Verify result
        assert result is not None

    @patch('refinire_rag.application.corpus_manager_new.Path.exists')
    def test_rebuild_corpus_from_original_no_documents(self, mock_exists):
        """
        Test rebuild_corpus_from_original with no original documents
        元ドキュメントがない場合のrebuild_corpus_from_originalテスト
        """
        # Setup mocks
        mock_exists.return_value = True
        
        # Mock document store to return no original documents
        with patch.object(self.corpus_manager, '_get_documents_by_stage') as mock_get_docs:
            mock_get_docs.return_value = []
            
            # Call rebuild method and expect exception
            with pytest.raises(ValueError, match="No original documents found"):
                self.corpus_manager.rebuild_corpus_from_original("test_corpus")

    def test_get_documents_by_stage(self):
        """
        Test _get_documents_by_stage functionality
        _get_documents_by_stage機能のテスト
        """
        # Mock documents with different stages
        all_docs = [
            Document(id="1", content="test1", metadata={"processing_stage": "original"}),
            Document(id="2", content="test2", metadata={"processing_stage": "normalized"}),
            Document(id="3", content="test3", metadata={"processing_stage": "original"}),
            Document(id="4", content="test4", metadata={"processing_stage": "chunked"})
        ]
        
        # Setup mock document store
        self.mock_document_store.list_documents.return_value = all_docs
        
        # Call method
        original_docs = self.corpus_manager._get_documents_by_stage("original")
        
        # Verify correct documents returned
        assert len(original_docs) == 2
        assert all(doc.metadata["processing_stage"] == "original" for doc in original_docs)

    def test_get_documents_by_stage_no_matches(self):
        """
        Test _get_documents_by_stage with no matching documents
        マッチするドキュメントがない場合の_get_documents_by_stageテスト
        """
        # Mock documents with different stages
        all_docs = [
            Document(id="1", content="test1", metadata={"processing_stage": "normalized"}),
            Document(id="2", content="test2", metadata={"processing_stage": "chunked"})
        ]
        
        # Setup mock document store
        self.mock_document_store.list_documents.return_value = all_docs
        
        # Call method for non-existent stage
        result_docs = self.corpus_manager._get_documents_by_stage("original")
        
        # Verify empty result
        assert len(result_docs) == 0

    def test_get_documents_by_stage_missing_metadata(self):
        """
        Test _get_documents_by_stage with documents missing processing_stage metadata
        processing_stageメタデータが欠けているドキュメントでの_get_documents_by_stageテスト
        """
        # Mock documents with missing metadata
        all_docs = [
            Document(id="1", content="test1", metadata={"processing_stage": "original"}),
            Document(id="2", content="test2", metadata={"source": "file2.txt"}),  # Missing processing_stage
            Document(id="3", content="test3", metadata={"processing_stage": "original"})
        ]
        
        # Setup mock document store
        self.mock_document_store.list_documents.return_value = all_docs
        
        # Call method
        original_docs = self.corpus_manager._get_documents_by_stage("original")
        
        # Verify only documents with correct stage are returned
        assert len(original_docs) == 2
        assert all(doc.metadata.get("processing_stage") == "original" for doc in original_docs)


class TestCorpusManagerFactoryMethods:
    """
    Test CorpusManager factory methods
    CorpusManagerのファクトリメソッドのテスト
    """

    @patch.dict(os.environ, {
        'REFINIRE_RAG_DOCUMENT_STORES': 'sqlite',
        'REFINIRE_RAG_VECTOR_STORES': 'inmemory_vector'
    })
    @patch('refinire_rag.application.corpus_manager_new.PluginFactory')
    def test_create_document_store_from_env(self, mock_factory):
        """
        Test _create_document_store_from_env functionality
        _create_document_store_from_env機能のテスト
        """
        # Setup mock factory
        mock_store = Mock()
        mock_factory.create_plugin.return_value = mock_store
        
        # Create CorpusManager to trigger env loading
        corpus_manager = CorpusManager()
        
        # Verify factory was called for document store
        mock_factory.create_plugin.assert_any_call('sqlite', plugin_type='document_store')

    @patch.dict(os.environ, {
        'REFINIRE_RAG_DOCUMENT_STORES': 'sqlite',
        'REFINIRE_RAG_VECTOR_STORES': 'inmemory_vector'
    })
    @patch('refinire_rag.application.corpus_manager_new.PluginFactory')
    def test_create_retrievers_from_env(self, mock_factory):
        """
        Test _create_retrievers_from_env functionality
        _create_retrievers_from_env機能のテスト
        """
        # Setup mock factory
        mock_store = Mock()
        mock_retriever = Mock()
        mock_factory.create_plugin.side_effect = lambda plugin_name, **kwargs: {
            'sqlite': mock_store,
            'inmemory_vector': mock_retriever
        }.get(plugin_name)
        
        # Create CorpusManager to trigger env loading
        corpus_manager = CorpusManager()
        
        # Verify factory was called for retrievers
        mock_factory.create_plugin.assert_any_call('inmemory_vector', plugin_type='vector_store')

    @patch.dict(os.environ, {
        'REFINIRE_RAG_DOCUMENT_STORES': 'sqlite',
        'REFINIRE_RAG_VECTOR_STORES': 'inmemory_vector,keyword_store'
    })
    @patch('refinire_rag.application.corpus_manager_new.PluginFactory')
    def test_create_multiple_retrievers_from_env(self, mock_factory):
        """
        Test _create_retrievers_from_env with multiple retrievers
        複数リトリーバーでの_create_retrievers_from_envテスト
        """
        # Setup mock factory
        mock_store = Mock()
        mock_vector_retriever = Mock()
        mock_keyword_retriever = Mock()
        mock_factory.create_plugin.side_effect = lambda plugin_name, **kwargs: {
            'sqlite': mock_store,
            'inmemory_vector': mock_vector_retriever,
            'keyword_store': mock_keyword_retriever
        }.get(plugin_name)
        
        # Create CorpusManager to trigger env loading
        corpus_manager = CorpusManager()
        
        # Verify multiple retrievers were created
        assert len(corpus_manager.retrievers) == 2
        mock_factory.create_plugin.assert_any_call('inmemory_vector', plugin_type='vector_store')
        mock_factory.create_plugin.assert_any_call('keyword_store', plugin_type='keyword_store')

    @patch('refinire_rag.application.corpus_manager_new.PluginFactory')
    def test_from_env_classmethod(self, mock_factory):
        """
        Test from_env class method
        from_envクラスメソッドのテスト
        """
        # Setup mock factory
        mock_store = Mock()
        mock_retriever = Mock()
        mock_factory.create_plugin.side_effect = lambda plugin_name, **kwargs: {
            'sqlite': mock_store,
            'inmemory_vector': mock_retriever
        }.get(plugin_name)
        
        # Call from_env method
        corpus_manager = CorpusManager.from_env()
        
        # Verify instance was created correctly
        assert isinstance(corpus_manager, CorpusManager)
        assert corpus_manager.document_store == mock_store
        assert mock_retriever in corpus_manager.retrievers