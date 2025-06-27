"""
Comprehensive tests for TFIDFKeywordStore implementation
TFIDFKeywordStore実装の包括的テスト

This module provides comprehensive coverage for the TFIDFKeywordStore module,
testing TF-IDF keyword search, document indexing, metadata filtering, and edge cases.
このモジュールは、TFIDFKeywordStoreモジュールの包括的カバレッジを提供し、
TF-IDFキーワード検索、文書インデックス、メタデータフィルタリング、エッジケースをテストします。
"""

import pytest
import time
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock

from refinire_rag.keywordstore.tfidf_keyword_store import TFIDFKeywordStore
from refinire_rag.models.document import Document
from refinire_rag.retrieval.base import SearchResult


class TestTFIDFKeywordStoreInitialization:
    """
    Test TFIDFKeywordStore initialization and configuration
    TFIDFKeywordStoreの初期化と設定のテスト
    """
    
    def test_init_with_default_config(self):
        """
        Test initialization with default configuration
        デフォルト設定での初期化テスト
        """
        store = TFIDFKeywordStore()
        
        assert store.config.top_k == 10
        assert store.config.similarity_threshold == 0.0
        assert store.config.enable_filtering is True
        assert store.documents == {}
        assert store.vectorizer is None
        assert store.tfidf_matrix is None
        assert store.doc_ids == []
        assert store.index_built is False
        
        # Check inherited processing stats from DocumentProcessor 
        assert "documents_processed" in store.processing_stats
        assert "total_processing_time" in store.processing_stats
        assert "errors_encountered" in store.processing_stats
        assert "documents_indexed" in store.processing_stats
        assert "searches_performed" in store.processing_stats
    
    def test_init_with_custom_config(self):
        """
        Test initialization with custom configuration
        カスタム設定での初期化テスト
        """
        store = TFIDFKeywordStore(
            top_k=20,
            similarity_threshold=0.1,
            enable_filtering=False
        )
        
        assert store.config.top_k == 20
        assert store.config.similarity_threshold == 0.1
        assert store.config.enable_filtering is False
    
    def test_init_with_partial_config(self):
        """
        Test initialization with partial configuration
        部分設定での初期化テスト
        """
        store = TFIDFKeywordStore(top_k=15)
        
        assert store.config.top_k == 15
        assert store.config.similarity_threshold == 0.0  # Default
        assert store.config.enable_filtering is True     # Default
    
    def test_get_config(self):
        """
        Test get_config method
        get_configメソッドのテスト
        """
        store = TFIDFKeywordStore(top_k=20, similarity_threshold=0.1)
        config = store.get_config()
        
        assert config['top_k'] == 20
        assert config['similarity_threshold'] == 0.1
        assert config['enable_filtering'] is True


class TestTFIDFKeywordStoreDocumentManagement:
    """
    Test TFIDFKeywordStore document management operations
    TFIDFKeywordStoreの文書管理操作テスト
    """
    
    def test_index_document(self):
        """
        Test indexing a single document
        単一文書のインデックステスト
        """
        store = TFIDFKeywordStore()
        document = Document(
            id="doc1",
            content="This is a test document about machine learning.",
            metadata={"category": "tech"}
        )
        
        store.index_document(document)
        
        assert "doc1" in store.documents
        assert store.documents["doc1"] == document
        assert store.index_built is False  # Marking for rebuild
    
    def test_index_documents_multiple(self):
        """
        Test indexing multiple documents
        複数文書のインデックステスト
        """
        store = TFIDFKeywordStore()
        documents = [
            Document(id="doc1", content="Machine learning algorithms", metadata={}),
            Document(id="doc2", content="Natural language processing", metadata={}),
            Document(id="doc3", content="Computer vision techniques", metadata={})
        ]
        
        store.index_documents(documents)
        
        assert len(store.documents) == 3
        assert "doc1" in store.documents
        assert "doc2" in store.documents
        assert "doc3" in store.documents
        assert store.index_built is False
    
    def test_remove_document_exists(self):
        """
        Test removing an existing document
        既存文書の削除テスト
        """
        store = TFIDFKeywordStore()
        document = Document(id="doc1", content="Test content", metadata={})
        store.index_document(document)
        
        result = store.remove_document("doc1")
        
        assert result is True
        assert "doc1" not in store.documents
        assert store.index_built is False
    
    def test_remove_document_not_exists(self):
        """
        Test removing a non-existent document
        存在しない文書の削除テスト
        """
        store = TFIDFKeywordStore()
        
        result = store.remove_document("nonexistent")
        
        assert result is False
        assert len(store.documents) == 0
    
    def test_update_document_exists(self):
        """
        Test updating an existing document
        既存文書の更新テスト
        """
        store = TFIDFKeywordStore()
        original_doc = Document(id="doc1", content="Original content", metadata={})
        store.index_document(original_doc)
        
        updated_doc = Document(id="doc1", content="Updated content", metadata={"updated": True})
        result = store.update_document(updated_doc)
        
        assert result is True
        assert store.documents["doc1"].content == "Updated content"
        assert store.documents["doc1"].metadata["updated"] is True
        assert store.index_built is False
    
    def test_update_document_not_exists(self):
        """
        Test updating a non-existent document
        存在しない文書の更新テスト
        """
        store = TFIDFKeywordStore()
        document = Document(id="nonexistent", content="New content", metadata={})
        
        result = store.update_document(document)
        
        assert result is False
        assert "nonexistent" not in store.documents
    
    def test_clear_index(self):
        """
        Test clearing the entire index
        インデックス全体のクリアテスト
        """
        store = TFIDFKeywordStore()
        documents = [
            Document(id="doc1", content="Content 1", metadata={}),
            Document(id="doc2", content="Content 2", metadata={})
        ]
        store.index_documents(documents)
        
        store.clear_index()
        
        assert len(store.documents) == 0
        assert store.vectorizer is None
        assert store.tfidf_matrix is None
        assert store.doc_ids == []
        assert store.index_built is False
    
    def test_get_document_count(self):
        """
        Test getting document count
        文書数取得テスト
        """
        store = TFIDFKeywordStore()
        
        assert store.get_document_count() == 0
        
        documents = [
            Document(id="doc1", content="Content 1", metadata={}),
            Document(id="doc2", content="Content 2", metadata={}),
            Document(id="doc3", content="Content 3", metadata={})
        ]
        store.index_documents(documents)
        
        assert store.get_document_count() == 3
    
    def test_add_document_alias(self):
        """
        Test add_document method (alias for index_document)
        add_documentメソッドのテスト（index_documentのエイリアス）
        """
        store = TFIDFKeywordStore()
        document = Document(id="doc1", content="Test content", metadata={})
        
        store.add_document(document)
        
        assert "doc1" in store.documents
        assert store.documents["doc1"] == document


class TestTFIDFKeywordStoreIndexBuilding:
    """
    Test TFIDFKeywordStore index building functionality
    TFIDFKeywordStoreのインデックス構築機能テスト
    """
    
    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    def test_build_index_success(self, mock_vectorizer_class):
        """
        Test successful index building
        インデックス構築成功テスト
        """
        # Setup mock
        mock_vectorizer = Mock()
        mock_tfidf_matrix = Mock()
        mock_vectorizer.fit_transform.return_value = mock_tfidf_matrix
        mock_vectorizer_class.return_value = mock_vectorizer
        
        store = TFIDFKeywordStore()
        documents = [
            Document(id="doc1", content="Machine learning algorithms", metadata={}),
            Document(id="doc2", content="Natural language processing", metadata={})
        ]
        store.index_documents(documents)
        
        store._build_index()
        
        assert store.vectorizer == mock_vectorizer
        assert store.tfidf_matrix == mock_tfidf_matrix
        assert store.doc_ids == ["doc1", "doc2"]
        mock_vectorizer_class.assert_called_once_with(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        mock_vectorizer.fit_transform.assert_called_once()
    
    def test_build_index_no_documents(self):
        """
        Test building index with no documents
        文書なしでのインデックス構築テスト
        """
        store = TFIDFKeywordStore()
        
        store._build_index()
        
        assert store.vectorizer is None
        assert store.tfidf_matrix is None
        assert store.doc_ids == []
    
    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    def test_build_index_import_error(self, mock_vectorizer_class):
        """
        Test index building with import error
        インポートエラーでのインデックス構築テスト
        """
        mock_vectorizer_class.side_effect = ImportError("scikit-learn not found")
        
        store = TFIDFKeywordStore()
        document = Document(id="doc1", content="Test content", metadata={})
        store.index_document(document)
        
        with pytest.raises(ImportError):
            store._build_index()
    
    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    def test_build_index_generic_error(self, mock_vectorizer_class):
        """
        Test index building with generic error
        一般的エラーでのインデックス構築テスト
        """
        mock_vectorizer = Mock()
        mock_vectorizer.fit_transform.side_effect = RuntimeError("Processing error")
        mock_vectorizer_class.return_value = mock_vectorizer
        
        store = TFIDFKeywordStore()
        document = Document(id="doc1", content="Test content", metadata={})
        store.index_document(document)
        
        with pytest.raises(RuntimeError):
            store._build_index()
    
    def test_clear_index_internal(self):
        """
        Test internal index clearing
        内部インデックスクリアテスト
        """
        store = TFIDFKeywordStore()
        # Set some mock data
        store.vectorizer = Mock()
        store.tfidf_matrix = Mock()
        store.doc_ids = ["doc1", "doc2"]
        
        store._clear_index()
        
        assert store.vectorizer is None
        assert store.tfidf_matrix is None
        assert store.doc_ids == []


class TestTFIDFKeywordStoreSearch:
    """
    Test TFIDFKeywordStore search functionality
    TFIDFKeywordStoreの検索機能テスト
    """
    
    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    @patch('sklearn.metrics.pairwise.cosine_similarity')
    def test_search_index_success(self, mock_cosine_similarity, mock_vectorizer_class):
        """
        Test successful index search
        インデックス検索成功テスト
        """
        # Setup mocks
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = Mock()
        mock_vectorizer_class.return_value = mock_vectorizer
        
        # Mock similarity scores: [0.8, 0.6, 0.3, 0.1] - Need numpy array behavior
        import numpy as np
        mock_cosine_similarity.return_value = np.array([[0.8, 0.6, 0.3, 0.1]])
        
        store = TFIDFKeywordStore()
        store.vectorizer = mock_vectorizer
        store.tfidf_matrix = Mock()
        store.doc_ids = ["doc1", "doc2", "doc3", "doc4"]
        
        results = store._search_index("test query", limit=3)
        
        # Should return top 3 results with scores > 0
        assert len(results) == 3
        assert results[0] == ("doc1", 0.8)
        assert results[1] == ("doc2", 0.6)
        assert results[2] == ("doc3", 0.3)
        
        mock_vectorizer.transform.assert_called_once_with(["test query"])
        mock_cosine_similarity.assert_called_once()
    
    def test_search_index_no_vectorizer(self):
        """
        Test search with no vectorizer
        ベクトライザーなしでの検索テスト
        """
        store = TFIDFKeywordStore()
        
        results = store._search_index("test query", limit=5)
        
        assert results == []
    
    @patch('sklearn.metrics.pairwise.cosine_similarity')
    def test_search_index_error(self, mock_cosine_similarity):
        """
        Test search with error
        エラーでの検索テスト
        """
        mock_cosine_similarity.side_effect = RuntimeError("Search error")
        
        store = TFIDFKeywordStore()
        store.vectorizer = Mock()
        store.tfidf_matrix = Mock()
        store.doc_ids = ["doc1"]
        
        results = store._search_index("test query", limit=5)
        
        assert results == []
    
    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    @patch('sklearn.metrics.pairwise.cosine_similarity')
    def test_search_index_zero_scores(self, mock_cosine_similarity, mock_vectorizer_class):
        """
        Test search with zero similarity scores
        類似度スコア0での検索テスト
        """
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = Mock()
        mock_vectorizer_class.return_value = mock_vectorizer
        
        # All zero similarities
        import numpy as np
        mock_cosine_similarity.return_value = np.array([[0.0, 0.0, 0.0]])
        
        store = TFIDFKeywordStore()
        store.vectorizer = mock_vectorizer
        store.tfidf_matrix = Mock()
        store.doc_ids = ["doc1", "doc2", "doc3"]
        
        results = store._search_index("test query", limit=3)
        
        assert results == []


class TestTFIDFKeywordStoreRetrieve:
    """
    Test TFIDFKeywordStore retrieve functionality
    TFIDFKeywordStoreの取得機能テスト
    """
    
    def test_retrieve_builds_index_if_needed(self):
        """
        Test that retrieve builds index if needed
        必要時にインデックスを構築することのテスト
        """
        store = TFIDFKeywordStore()
        document = Document(id="doc1", content="Test content", metadata={})
        store.index_document(document)
        
        with patch.object(store, '_build_index') as mock_build:
            with patch.object(store, '_search_index', return_value=[]):
                store.retrieve("test query")
                
                mock_build.assert_called_once()
                assert store.index_built is True
    
    def test_retrieve_skips_build_if_built(self):
        """
        Test that retrieve skips building if index already built
        インデックス構築済み時はスキップすることのテスト
        """
        store = TFIDFKeywordStore()
        store.index_built = True
        
        with patch.object(store, '_build_index') as mock_build:
            with patch.object(store, '_search_index', return_value=[]):
                store.retrieve("test query")
                
                mock_build.assert_not_called()
    
    def test_retrieve_with_results(self):
        """
        Test retrieve with successful results
        結果ありでの取得テスト
        """
        store = TFIDFKeywordStore()
        document = Document(
            id="doc1", 
            content="Machine learning algorithms", 
            metadata={"category": "tech"}
        )
        store.index_document(document)
        store.index_built = True
        
        with patch.object(store, '_search_index', return_value=[("doc1", 0.8)]):
            results = store.retrieve("machine learning", limit=5)
            
            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            assert results[0].document_id == "doc1"
            assert results[0].document == document
            assert results[0].score == 0.8
            assert results[0].metadata["retrieval_method"] == "keyword_search"
            assert results[0].metadata["algorithm"] == "tfidf"
            assert results[0].metadata["keyword_store"] == "TFIDFKeywordStore"
    
    def test_retrieve_with_similarity_threshold_filtering(self):
        """
        Test retrieve with similarity threshold filtering
        類似度閾値フィルタリングでの取得テスト
        """
        store = TFIDFKeywordStore(similarity_threshold=0.5, enable_filtering=True)
        document = Document(id="doc1", content="Test content", metadata={})
        store.index_document(document)
        store.index_built = True
        
        # Score below threshold should be filtered out
        with patch.object(store, '_search_index', return_value=[("doc1", 0.3)]):
            results = store.retrieve("test query")
            
            assert len(results) == 0
    
    def test_retrieve_with_metadata_filtering(self):
        """
        Test retrieve with metadata filtering
        メタデータフィルタリングでの取得テスト
        """
        store = TFIDFKeywordStore()
        document = Document(
            id="doc1", 
            content="Test content", 
            metadata={"category": "tech", "level": "advanced"}
        )
        store.index_document(document)
        store.index_built = True
        
        with patch.object(store, '_search_index', return_value=[("doc1", 0.8)]):
            # Should match
            results = store.retrieve(
                "test query", 
                metadata_filter={"category": "tech"}
            )
            assert len(results) == 1
            
            # Should not match
            results = store.retrieve(
                "test query", 
                metadata_filter={"category": "science"}
            )
            assert len(results) == 0
    
    def test_retrieve_with_filtering_disabled(self):
        """
        Test retrieve with filtering disabled
        フィルタリング無効での取得テスト
        """
        store = TFIDFKeywordStore(enable_filtering=False, similarity_threshold=0.5)
        document = Document(id="doc1", content="Test content", metadata={})
        store.index_document(document)
        store.index_built = True
        
        # Even with low score, should not be filtered
        with patch.object(store, '_search_index', return_value=[("doc1", 0.3)]):
            results = store.retrieve("test query")
            
            assert len(results) == 1
    
    def test_retrieve_missing_document(self):
        """
        Test retrieve when document is missing from store
        ストアから文書が欠落している場合の取得テスト
        """
        store = TFIDFKeywordStore()
        store.index_built = True
        
        with patch.object(store, '_search_index', return_value=[("nonexistent", 0.8)]):
            results = store.retrieve("test query")
            
            assert len(results) == 0
    
    def test_retrieve_with_limit(self):
        """
        Test retrieve with result limit
        結果制限での取得テスト
        """
        store = TFIDFKeywordStore()
        documents = [
            Document(id="doc1", content="Content 1", metadata={}),
            Document(id="doc2", content="Content 2", metadata={}),
            Document(id="doc3", content="Content 3", metadata={})
        ]
        store.index_documents(documents)
        store.index_built = True
        
        search_results = [("doc1", 0.8), ("doc2", 0.7), ("doc3", 0.6)]
        with patch.object(store, '_search_index', return_value=search_results):
            results = store.retrieve("test query", limit=2)
            
            assert len(results) == 2
            assert results[0].document_id == "doc1"
            assert results[1].document_id == "doc2"
    
    def test_retrieve_uses_config_limit(self):
        """
        Test retrieve uses config limit when not specified
        未指定時に設定制限を使用することのテスト
        """
        store = TFIDFKeywordStore(top_k=3)
        store.index_built = True
        
        with patch.object(store, '_search_index') as mock_search:
            store.retrieve("test query")
            
            # Should request limit * 2 for filtering
            mock_search.assert_called_once_with("test query", 6)
    
    def test_retrieve_updates_stats(self):
        """
        Test that retrieve updates processing statistics
        取得時に処理統計を更新することのテスト
        """
        store = TFIDFKeywordStore()
        store.index_built = True
        
        # Add custom stats fields that retrieve method updates
        store.processing_stats["queries_processed"] = 0
        
        initial_queries = store.processing_stats["queries_processed"]
        initial_time = store.processing_stats["total_processing_time"]
        
        with patch.object(store, '_search_index', return_value=[]):
            store.retrieve("test query")
            
            assert store.processing_stats["queries_processed"] == initial_queries + 1
            assert store.processing_stats["total_processing_time"] > initial_time
    
    def test_retrieve_handles_error(self):
        """
        Test retrieve handles errors gracefully
        エラー処理の適切な処理テスト
        """
        store = TFIDFKeywordStore()
        store.index_built = True
        
        # Add custom error stats field
        store.processing_stats["errors_encountered"] = 0
        initial_errors = store.processing_stats["errors_encountered"]
        
        with patch.object(store, '_search_index', side_effect=RuntimeError("Search failed")):
            results = store.retrieve("test query")
            
            assert results == []
            assert store.processing_stats["errors_encountered"] == initial_errors + 1
    
    def test_search_alias(self):
        """
        Test search method (alias for retrieve)
        searchメソッドのテスト（retrieveのエイリアス）
        """
        store = TFIDFKeywordStore()
        store.index_built = True
        
        with patch.object(store, 'retrieve') as mock_retrieve:
            store.search("test query", limit=5)
            
            mock_retrieve.assert_called_once_with("test query", limit=5)


class TestTFIDFKeywordStoreMetadataFiltering:
    """
    Test TFIDFKeywordStore metadata filtering functionality
    TFIDFKeywordStoreのメタデータフィルタリング機能テスト
    """
    
    def test_matches_metadata_filter_exact_match(self):
        """
        Test exact match metadata filtering
        完全一致メタデータフィルタリングテスト
        """
        store = TFIDFKeywordStore()
        metadata = {"category": "tech", "level": "beginner"}
        filter_dict = {"category": "tech"}
        
        result = store._matches_metadata_filter(metadata, filter_dict)
        
        assert result is True
    
    def test_matches_metadata_filter_no_match(self):
        """
        Test no match metadata filtering
        不一致メタデータフィルタリングテスト
        """
        store = TFIDFKeywordStore()
        metadata = {"category": "tech", "level": "beginner"}
        filter_dict = {"category": "science"}
        
        result = store._matches_metadata_filter(metadata, filter_dict)
        
        assert result is False
    
    def test_matches_metadata_filter_missing_key(self):
        """
        Test metadata filtering with missing key
        キー欠損でのメタデータフィルタリングテスト
        """
        store = TFIDFKeywordStore()
        metadata = {"category": "tech"}
        filter_dict = {"level": "beginner"}
        
        result = store._matches_metadata_filter(metadata, filter_dict)
        
        assert result is False
    
    def test_matches_metadata_filter_list_condition(self):
        """
        Test metadata filtering with list condition (OR)
        リスト条件（OR）でのメタデータフィルタリングテスト
        """
        store = TFIDFKeywordStore()
        metadata = {"category": "tech", "level": "advanced"}
        filter_dict = {"category": ["tech", "science"]}
        
        result = store._matches_metadata_filter(metadata, filter_dict)
        
        assert result is True
        
        # Test no match with list
        filter_dict = {"category": ["science", "math"]}
        result = store._matches_metadata_filter(metadata, filter_dict)
        
        assert result is False
    
    def test_matches_metadata_filter_range_conditions(self):
        """
        Test metadata filtering with range conditions
        範囲条件でのメタデータフィルタリングテスト
        """
        store = TFIDFKeywordStore()
        metadata = {"score": 85, "year": 2023}
        
        # Test $gte condition
        filter_dict = {"score": {"$gte": 80}}
        result = store._matches_metadata_filter(metadata, filter_dict)
        assert result is True
        
        filter_dict = {"score": {"$gte": 90}}
        result = store._matches_metadata_filter(metadata, filter_dict)
        assert result is False
        
        # Test $lte condition
        filter_dict = {"score": {"$lte": 90}}
        result = store._matches_metadata_filter(metadata, filter_dict)
        assert result is True
        
        filter_dict = {"score": {"$lte": 80}}
        result = store._matches_metadata_filter(metadata, filter_dict)
        assert result is False
        
        # Test $ne condition
        filter_dict = {"year": {"$ne": 2022}}
        result = store._matches_metadata_filter(metadata, filter_dict)
        assert result is True
        
        filter_dict = {"year": {"$ne": 2023}}
        result = store._matches_metadata_filter(metadata, filter_dict)
        assert result is False
    
    def test_matches_metadata_filter_complex_conditions(self):
        """
        Test metadata filtering with complex conditions
        複雑条件でのメタデータフィルタリングテスト
        """
        store = TFIDFKeywordStore()
        metadata = {"category": "tech", "score": 85, "year": 2023}
        filter_dict = {
            "category": "tech",
            "score": {"$gte": 80, "$lte": 90},
            "year": {"$ne": 2022}
        }
        
        result = store._matches_metadata_filter(metadata, filter_dict)
        
        assert result is True


class TestTFIDFKeywordStoreProcessingStats:
    """
    Test TFIDFKeywordStore processing statistics
    TFIDFKeywordStoreの処理統計テスト
    """
    
    def test_get_processing_stats(self):
        """
        Test getting processing statistics
        処理統計取得テスト
        """
        store = TFIDFKeywordStore(top_k=15, similarity_threshold=0.2)
        document = Document(id="doc1", content="Test content", metadata={})
        store.index_document(document)
        
        stats = store.get_processing_stats()
        
        # Check inherited stats
        assert "queries_processed" in stats
        assert "total_processing_time" in stats  
        assert "errors_encountered" in stats
        
        # Check TF-IDF specific stats
        assert stats["retriever_type"] == "TFIDFKeywordStore"
        assert stats["algorithm"] == "tfidf"
        assert stats["similarity_threshold"] == 0.2
        assert stats["top_k"] == 15
        assert stats["document_count"] == 1
        assert stats["index_built"] is False


class TestTFIDFKeywordStoreErrorHandling:
    """
    Test TFIDFKeywordStore error handling scenarios
    TFIDFKeywordStoreのエラー処理シナリオテスト
    """
    
    def test_index_document_error(self):
        """
        Test error handling in document indexing
        文書インデックス時のエラー処理テスト
        """
        store = TFIDFKeywordStore()
        document = Document(id="doc1", content="Test content", metadata={})
        
        # Mock documents dictionary to raise error on assignment
        with patch.object(store, 'documents', spec=dict) as mock_dict:
            mock_dict.__setitem__ = Mock(side_effect=RuntimeError("Index error"))
            
            with pytest.raises(RuntimeError):
                store.index_document(document)
    
    def test_index_documents_error(self):
        """
        Test error handling in multiple document indexing
        複数文書インデックス時のエラー処理テスト
        """
        store = TFIDFKeywordStore()
        documents = [Document(id="doc1", content="Content", metadata={})]
        
        # Mock documents dictionary to raise error
        with patch.object(store, 'documents', spec=dict) as mock_dict:
            mock_dict.__setitem__ = Mock(side_effect=RuntimeError("Batch index error"))
            
            with pytest.raises(RuntimeError):
                store.index_documents(documents)
    
    def test_remove_document_error(self):
        """
        Test error handling in document removal
        文書削除時のエラー処理テスト
        """
        store = TFIDFKeywordStore()
        document = Document(id="doc1", content="Test content", metadata={})
        store.index_document(document)
        
        # Mock documents dictionary to raise error on deletion
        with patch.object(store, 'documents', spec=dict) as mock_dict:
            mock_dict.__contains__ = Mock(return_value=True)
            mock_dict.__delitem__ = Mock(side_effect=RuntimeError("Remove error"))
            
            result = store.remove_document("doc1")
            
            assert result is False
    
    def test_update_document_error(self):
        """
        Test error handling in document update
        文書更新時のエラー処理テスト
        """
        store = TFIDFKeywordStore()
        document = Document(id="doc1", content="Original content", metadata={})
        store.index_document(document)
        
        updated_doc = Document(id="doc1", content="Updated content", metadata={})
        
        # Mock documents dictionary to raise error on update
        with patch.object(store, 'documents', spec=dict) as mock_dict:
            mock_dict.__contains__ = Mock(return_value=True)
            mock_dict.__setitem__ = Mock(side_effect=RuntimeError("Update error"))
            
            result = store.update_document(updated_doc)
            
            assert result is False
    
    def test_clear_index_error(self):
        """
        Test error handling in index clearing
        インデックスクリア時のエラー処理テスト
        """
        store = TFIDFKeywordStore()
        
        # Mock documents dictionary to raise error on clear
        with patch.object(store, 'documents', spec=dict) as mock_dict:
            mock_dict.clear = Mock(side_effect=RuntimeError("Clear error"))
            
            with pytest.raises(RuntimeError):
                store.clear_index()


class TestTFIDFKeywordStoreEdgeCases:
    """
    Test TFIDFKeywordStore edge cases and corner scenarios
    TFIDFKeywordStoreのエッジケースとコーナーシナリオテスト
    """
    
    def test_retrieve_empty_query(self):
        """
        Test retrieve with empty query
        空クエリでの取得テスト
        """
        store = TFIDFKeywordStore()
        document = Document(id="doc1", content="Test content", metadata={})
        store.index_document(document)
        store.index_built = True
        
        with patch.object(store, '_search_index', return_value=[]):
            results = store.retrieve("")
            
            assert results == []
    
    def test_retrieve_very_long_query(self):
        """
        Test retrieve with very long query
        非常に長いクエリでの取得テスト
        """
        store = TFIDFKeywordStore()
        document = Document(id="doc1", content="Test content", metadata={})
        store.index_document(document)
        store.index_built = True
        
        long_query = "word " * 1000  # Very long query
        
        with patch.object(store, '_search_index', return_value=[("doc1", 0.5)]):
            results = store.retrieve(long_query)
            
            assert len(results) == 1
            assert results[0].metadata["query_length"] == len(long_query)
    
    def test_retrieve_unicode_query(self):
        """
        Test retrieve with unicode query
        Unicode クエリでの取得テスト
        """
        store = TFIDFKeywordStore()
        document = Document(id="doc1", content="日本語の文書です", metadata={})
        store.index_document(document)
        store.index_built = True
        
        with patch.object(store, '_search_index', return_value=[("doc1", 0.8)]):
            results = store.retrieve("日本語")
            
            assert len(results) == 1
            assert results[0].document_id == "doc1"
    
    def test_retrieve_with_zero_limit(self):
        """
        Test retrieve with zero limit
        制限0での取得テスト
        """
        store = TFIDFKeywordStore()
        document = Document(id="doc1", content="Test content", metadata={})
        store.index_document(document)
        store.index_built = True
        
        with patch.object(store, '_search_index', return_value=[("doc1", 0.8)]):
            results = store.retrieve("test", limit=0)
            
            assert results == []
    
    def test_document_with_empty_content(self):
        """
        Test handling document with empty content
        空コンテンツ文書の処理テスト
        """
        store = TFIDFKeywordStore()
        document = Document(id="doc1", content="", metadata={})
        
        store.index_document(document)
        
        assert "doc1" in store.documents
        assert store.documents["doc1"].content == ""
    
    def test_document_with_special_characters(self):
        """
        Test handling document with special characters
        特殊文字文書の処理テスト
        """
        store = TFIDFKeywordStore()
        document = Document(
            id="doc1", 
            content="Special chars: !@#$%^&*()_+{}[]|\\:;\"'<>?,./", 
            metadata={}
        )
        
        store.index_document(document)
        
        assert "doc1" in store.documents
        assert store.documents["doc1"] == document
    
    def test_duplicate_document_ids(self):
        """
        Test handling duplicate document IDs
        重複文書IDの処理テスト
        """
        store = TFIDFKeywordStore()
        doc1 = Document(id="doc1", content="Original content", metadata={})
        doc2 = Document(id="doc1", content="Updated content", metadata={})
        
        store.index_document(doc1)
        store.index_document(doc2)  # Should overwrite
        
        assert len(store.documents) == 1
        assert store.documents["doc1"].content == "Updated content"
    
    def test_retrieve_performance_tracking(self):
        """
        Test that retrieve properly tracks performance
        取得時の適切なパフォーマンス追跡テスト
        """
        store = TFIDFKeywordStore()
        store.index_built = True
        
        start_time = store.processing_stats["total_processing_time"]
        
        with patch.object(store, '_search_index', return_value=[]):
            with patch('time.time', side_effect=[1000.0, 1001.5]):  # 1.5 second duration
                store.retrieve("test query")
        
        # Should have added 1.5 seconds to processing time
        assert store.processing_stats["total_processing_time"] >= start_time + 1.5