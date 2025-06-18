"""
Comprehensive tests for VectorStore base class functionality
VectorStore基底クラス機能の包括的テスト

This module provides comprehensive coverage for the VectorStore abstract class,
testing all concrete methods, DocumentProcessor integration, and interface implementations.
このモジュールは、VectorStoreの抽象クラスの包括的カバレッジを提供し、
全ての具象メソッド、DocumentProcessor統合、インターフェース実装をテストします。
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional
import logging

from refinire_rag.storage.vector_store import VectorStore, VectorEntry, VectorSearchResult, VectorStoreStats
from refinire_rag.models.document import Document
from refinire_rag.exceptions import StorageError


class MockVectorStore(VectorStore):
    """Mock implementation of VectorStore for testing concrete methods"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._vectors: Dict[str, VectorEntry] = {}
        self._fail_operations = False
    
    @classmethod
    def get_config_class(cls):
        return Dict
    
    def add_vector(self, entry: VectorEntry) -> str:
        if self._fail_operations:
            raise StorageError("Mock storage error")
        self._vectors[entry.document_id] = entry
        return entry.document_id
    
    def add_vectors(self, entries: List[VectorEntry]) -> List[str]:
        if self._fail_operations:
            raise StorageError("Mock storage error")
        ids = []
        for entry in entries:
            self._vectors[entry.document_id] = entry
            ids.append(entry.document_id)
        return ids
    
    def get_vector(self, document_id: str) -> Optional[VectorEntry]:
        return self._vectors.get(document_id)
    
    def update_vector(self, entry: VectorEntry) -> bool:
        if entry.document_id in self._vectors:
            self._vectors[entry.document_id] = entry
            return True
        return False
    
    def delete_vector(self, document_id: str) -> bool:
        if document_id in self._vectors:
            del self._vectors[document_id]
            return True
        return False
    
    def search_similar(
        self, 
        query_vector: np.ndarray, 
        limit: int = 10,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        if self._fail_operations:
            raise StorageError("Mock search error")
        
        results = []
        for entry in self._vectors.values():
            # Simple cosine similarity calculation
            dot_product = np.dot(query_vector, entry.embedding)
            norms = np.linalg.norm(query_vector) * np.linalg.norm(entry.embedding)
            score = dot_product / norms if norms > 0 else 0.0
            
            if threshold and score < threshold:
                continue
            
            # Apply metadata filters
            if filters:
                match = True
                for key, value in filters.items():
                    if entry.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
                
            result = VectorSearchResult(
                document_id=entry.document_id,
                content=entry.content,
                metadata=entry.metadata,
                score=score,
                embedding=entry.embedding
            )
            results.append(result)
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 100
    ) -> List[VectorSearchResult]:
        results = []
        for entry in self._vectors.values():
            # Simple metadata matching
            match = True
            for key, value in filters.items():
                if entry.metadata.get(key) != value:
                    match = False
                    break
            
            if match:
                result = VectorSearchResult(
                    document_id=entry.document_id,
                    content=entry.content,
                    metadata=entry.metadata,
                    score=1.0,
                    embedding=entry.embedding
                )
                results.append(result)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def count_vectors(self, filters: Optional[Dict[str, Any]] = None) -> int:
        if not filters:
            return len(self._vectors)
        
        count = 0
        for entry in self._vectors.values():
            match = True
            for key, value in filters.items():
                if entry.metadata.get(key) != value:
                    match = False
                    break
            if match:
                count += 1
        return count
    
    def get_stats(self) -> VectorStoreStats:
        if not self._vectors:
            return VectorStoreStats(
                total_vectors=0,
                vector_dimension=0,
                storage_size_bytes=0,
                index_type="mock"
            )
        
        first_entry = next(iter(self._vectors.values()))
        dimension = len(first_entry.embedding)
        size = sum(len(e.content.encode('utf-8')) + e.embedding.nbytes for e in self._vectors.values())
        
        return VectorStoreStats(
            total_vectors=len(self._vectors),
            vector_dimension=dimension,
            storage_size_bytes=size,
            index_type="mock"
        )
    
    def clear(self) -> bool:
        self._vectors.clear()
        return True


class TestVectorStoreDataClasses:
    """
    Test VectorStore data classes functionality
    VectorStoreデータクラス機能のテスト
    """

    def test_vector_entry_initialization(self):
        """
        Test VectorEntry initialization and numpy conversion
        VectorEntry初期化とnumpy変換テスト
        """
        # Test with list embedding
        entry = VectorEntry(
            document_id="doc1",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"category": "test"}
        )
        
        assert entry.document_id == "doc1"
        assert entry.content == "Test content"
        assert isinstance(entry.embedding, np.ndarray)
        assert np.array_equal(entry.embedding, np.array([0.1, 0.2, 0.3]))
        assert entry.metadata == {"category": "test"}

    def test_vector_entry_numpy_embedding(self):
        """
        Test VectorEntry with pre-existing numpy array
        既存のnumpy配列でのVectorEntryテスト
        """
        embedding = np.array([0.4, 0.5, 0.6])
        entry = VectorEntry(
            document_id="doc2",
            content="Test content 2",
            embedding=embedding,
            metadata={}
        )
        
        assert isinstance(entry.embedding, np.ndarray)
        assert np.array_equal(entry.embedding, embedding)

    def test_vector_search_result_creation(self):
        """
        Test VectorSearchResult creation
        VectorSearchResult作成テスト
        """
        result = VectorSearchResult(
            document_id="doc1",
            content="Result content",
            metadata={"category": "result"},
            score=0.85,
            embedding=np.array([0.1, 0.2])
        )
        
        assert result.document_id == "doc1"
        assert result.content == "Result content"
        assert result.metadata == {"category": "result"}
        assert result.score == 0.85
        assert isinstance(result.embedding, np.ndarray)

    def test_vector_search_result_optional_embedding(self):
        """
        Test VectorSearchResult with optional embedding
        オプションの埋め込みでのVectorSearchResultテスト
        """
        result = VectorSearchResult(
            document_id="doc2",
            content="Content",
            metadata={},
            score=0.75
        )
        
        assert result.embedding is None

    def test_vector_store_stats_creation(self):
        """
        Test VectorStoreStats creation
        VectorStoreStats作成テスト
        """
        stats = VectorStoreStats(
            total_vectors=100,
            vector_dimension=512,
            storage_size_bytes=1024,
            index_type="faiss"
        )
        
        assert stats.total_vectors == 100
        assert stats.vector_dimension == 512
        assert stats.storage_size_bytes == 1024
        assert stats.index_type == "faiss"

    def test_vector_store_stats_default_index_type(self):
        """
        Test VectorStoreStats with default index type
        デフォルトインデックスタイプでのVectorStoreStatsテスト
        """
        stats = VectorStoreStats(
            total_vectors=50,
            vector_dimension=256,
            storage_size_bytes=512
        )
        
        assert stats.index_type == "exact"


class TestVectorStoreInitialization:
    """
    Test VectorStore initialization and configuration
    VectorStore初期化と設定のテスト
    """

    def test_default_initialization(self):
        """
        Test default VectorStore initialization
        デフォルトVectorStore初期化テスト
        """
        store = MockVectorStore()
        
        # Check DocumentProcessor initialization
        assert hasattr(store, 'processing_stats')
        assert 'vectors_stored' in store.processing_stats
        assert 'vectors_retrieved' in store.processing_stats
        assert 'searches_performed' in store.processing_stats
        assert 'embedding_errors' in store.processing_stats
        
        # Check initial values
        assert store.processing_stats['vectors_stored'] == 0
        assert store.processing_stats['vectors_retrieved'] == 0
        assert store.processing_stats['searches_performed'] == 0
        assert store.processing_stats['embedding_errors'] == 0
        
        # Check embedder is None initially
        assert store._embedder is None

    def test_custom_config_initialization(self):
        """
        Test VectorStore initialization with custom config
        カスタム設定でのVectorStore初期化テスト
        """
        config = {"custom_param": "value", "batch_size": 10}
        store = MockVectorStore(config)
        
        assert store.config == config
        assert hasattr(store, 'processing_stats')

    def test_set_embedder(self):
        """
        Test setting embedder
        Embedder設定テスト
        """
        store = MockVectorStore()
        mock_embedder = Mock()
        
        store.set_embedder(mock_embedder)
        assert store._embedder == mock_embedder

    def test_get_config_class(self):
        """
        Test get_config_class method
        get_config_classメソッドテスト
        """
        assert MockVectorStore.get_config_class() == Dict


class TestVectorStoreConvenienceMethods:
    """
    Test VectorStore convenience methods
    VectorStore便利メソッドのテスト
    """

    def setup_method(self):
        """Setup test environment"""
        self.store = MockVectorStore()
        self.mock_embedder = Mock()
        self.store.set_embedder(self.mock_embedder)

    def test_add_documents_with_embeddings_success(self):
        """
        Test successful addition of documents with embeddings
        成功した文書と埋め込みの追加テスト
        """
        documents = [
            Document(id="doc1", content="Content 1", metadata={"category": "test"}),
            Document(id="doc2", content="Content 2", metadata={"category": "test"})
        ]
        embeddings = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6])
        ]
        
        result_ids = self.store.add_documents_with_embeddings(documents, embeddings)
        
        assert result_ids == ["doc1", "doc2"]
        assert len(self.store._vectors) == 2
        assert "doc1" in self.store._vectors
        assert "doc2" in self.store._vectors
        
        # Verify VectorEntry creation
        entry1 = self.store._vectors["doc1"]
        assert entry1.document_id == "doc1"
        assert entry1.content == "Content 1"
        assert np.array_equal(entry1.embedding, np.array([0.1, 0.2, 0.3]))
        assert entry1.metadata["category"] == "test"  # Check specific field instead of entire dict

    def test_add_documents_with_embeddings_mismatch_error(self):
        """
        Test error when documents and embeddings count mismatch
        文書と埋め込み数が不一致のエラーテスト
        """
        documents = [Document(id="doc1", content="Content 1", metadata={})]
        embeddings = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]  # More embeddings than documents
        
        with pytest.raises(ValueError, match="Number of documents must match number of embeddings"):
            self.store.add_documents_with_embeddings(documents, embeddings)

    def test_get_vector_dimension_with_vectors(self):
        """
        Test getting vector dimension when vectors exist
        ベクトル存在時のベクトル次元取得テスト
        """
        # Add a vector first
        entry = VectorEntry(
            document_id="doc1",
            content="Content",
            embedding=np.array([0.1, 0.2, 0.3, 0.4]),
            metadata={}
        )
        self.store.add_vector(entry)
        
        dimension = self.store.get_vector_dimension()
        assert dimension == 4

    def test_get_vector_dimension_empty_store(self):
        """
        Test getting vector dimension from empty store
        空のストアからのベクトル次元取得テスト
        """
        dimension = self.store.get_vector_dimension()
        assert dimension is None


class TestVectorStoreSearchMethods:
    """
    Test VectorStore search-related methods
    VectorStore検索関連メソッドのテスト
    """

    def setup_method(self):
        """Setup test environment with sample data"""
        self.store = MockVectorStore()
        self.mock_embedder = Mock()
        self.store.set_embedder(self.mock_embedder)
        
        # Add sample vectors
        self.sample_entries = [
            VectorEntry(
                document_id="doc1",
                content="Machine learning content",
                embedding=np.array([1.0, 0.0, 0.0]),
                metadata={"topic": "AI", "difficulty": "easy"}
            ),
            VectorEntry(
                document_id="doc2",
                content="Deep learning content", 
                embedding=np.array([0.9, 0.1, 0.0]),
                metadata={"topic": "AI", "difficulty": "hard"}
            ),
            VectorEntry(
                document_id="doc3",
                content="Cooking recipes",
                embedding=np.array([0.0, 0.0, 1.0]),
                metadata={"topic": "food", "difficulty": "easy"}
            )
        ]
        
        for entry in self.sample_entries:
            self.store.add_vector(entry)

    def test_search_similar_to_document_success(self):
        """
        Test successful document similarity search
        成功した文書類似度検索テスト
        """
        results = self.store.search_similar_to_document("doc1", limit=2)
        
        assert len(results) <= 2
        assert self.store.processing_stats["searches_performed"] > 0
        
        # Should not include doc1 itself (exclude_self=True by default)
        result_ids = [r.document_id for r in results]
        assert "doc1" not in result_ids

    def test_search_similar_to_document_include_self(self):
        """
        Test document similarity search including self
        自身を含む文書類似度検索テスト
        """
        results = self.store.search_similar_to_document("doc1", limit=3, exclude_self=False)
        
        result_ids = [r.document_id for r in results]
        assert "doc1" in result_ids

    def test_search_similar_to_document_nonexistent(self):
        """
        Test similarity search for non-existent document
        存在しない文書の類似度検索テスト
        """
        results = self.store.search_similar_to_document("nonexistent", limit=5)
        assert len(results) == 0

    def test_search_similar_to_document_with_threshold(self):
        """
        Test similarity search with threshold
        閾値付き類似度検索テスト
        """
        results = self.store.search_similar_to_document("doc1", limit=5, threshold=0.5)
        
        # All results should have score >= 0.5
        for result in results:
            assert result.score >= 0.5

    def test_search_with_text_success(self):
        """
        Test successful text-based search
        成功したテキストベース検索テスト
        """
        self.mock_embedder.embed_text.return_value = np.array([1.0, 0.0, 0.0])
        
        results = self.store.search_with_text("machine learning", limit=2)
        
        assert len(results) <= 2
        assert self.store.processing_stats["searches_performed"] > 0
        self.mock_embedder.embed_text.assert_called_once_with("machine learning")

    def test_search_with_text_no_embedder(self):
        """
        Test text search without embedder
        Embedderなしのテキスト検索テスト
        """
        self.store._embedder = None
        
        results = self.store.search_with_text("test query")
        assert len(results) == 0

    def test_search_with_text_with_filters(self):
        """
        Test text search with metadata filters
        メタデータフィルタ付きテキスト検索テスト
        """
        self.mock_embedder.embed_text.return_value = np.array([1.0, 0.0, 0.0])
        
        results = self.store.search_with_text(
            "machine learning", 
            limit=5,
            filters={"topic": "AI"}
        )
        
        # All results should have topic="AI"
        for result in results:
            assert result.metadata["topic"] == "AI"

    def test_search_with_text_embedder_error(self):
        """
        Test text search with embedder error
        Embedderエラーでのテキスト検索テスト
        """
        self.mock_embedder.embed_text.side_effect = Exception("Embedder error")
        
        results = self.store.search_with_text("test query")
        assert len(results) == 0

    def test_search_similar_to_document_storage_error(self):
        """
        Test similarity search with storage error
        ストレージエラーでの類似度検索テスト
        """
        self.store._fail_operations = True
        
        results = self.store.search_similar_to_document("doc1")
        assert len(results) == 0


class TestVectorStoreDocumentProcessorIntegration:
    """
    Test VectorStore integration with DocumentProcessor
    VectorStoreとDocumentProcessorの統合テスト
    """

    def setup_method(self):
        """Setup test environment"""
        self.store = MockVectorStore()
        self.mock_embedder = Mock()
        self.store.set_embedder(self.mock_embedder)

    def test_process_documents_with_embedder(self):
        """
        Test processing documents with embedder
        Embedderでの文書処理テスト
        """
        documents = [
            Document(id="doc1", content="Content 1", metadata={"category": "test"}),
            Document(id="doc2", content="Content 2", metadata={"category": "test"})
        ]
        
        # Mock embedder responses
        self.mock_embedder.embed_text.side_effect = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6])
        ]
        
        # Process documents
        processed_docs = list(self.store.process(documents))
        
        assert len(processed_docs) == 2
        assert processed_docs[0].id == "doc1"
        assert processed_docs[1].id == "doc2"
        
        # Verify vectors were stored
        assert len(self.store._vectors) == 2
        assert self.store.processing_stats["vectors_stored"] == 2
        
        # Verify embedder was called
        assert self.mock_embedder.embed_text.call_count == 2

    def test_process_documents_without_embedder(self):
        """
        Test processing documents without embedder
        Embedderなしでの文書処理テスト
        """
        self.store._embedder = None
        
        documents = [
            Document(id="doc1", content="Content 1", metadata={})
        ]
        
        with patch('logging.getLogger') as mock_logger:
            processed_docs = list(self.store.process(documents))
            
            assert len(processed_docs) == 1
            assert processed_docs[0].id == "doc1"
            
            # No vectors should be stored
            assert len(self.store._vectors) == 0
            assert self.store.processing_stats["vectors_stored"] == 0

    def test_process_documents_with_fitting_embedder(self):
        """
        Test processing documents with unfitted embedder
        未フィットEmbedderでの文書処理テスト
        """
        # Create embedder that needs fitting
        self.mock_embedder.is_fitted.return_value = False
        self.mock_embedder.fit = Mock()
        self.mock_embedder.embed_text.side_effect = [
            np.array([0.1, 0.2])
        ]
        
        documents = [
            Document(id="doc1", content="Content for fitting", metadata={})
        ]
        
        # Process documents
        processed_docs = list(self.store.process(documents))
        
        assert len(processed_docs) == 1
        
        # Verify fitting was called
        self.mock_embedder.fit.assert_called_once()
        
        # Verify document was processed after fitting
        assert len(self.store._vectors) == 1

    def test_process_documents_fitting_failure(self):
        """
        Test processing documents when embedder fitting fails
        Embedderフィッティング失敗時の文書処理テスト
        """
        self.mock_embedder.is_fitted.return_value = False
        self.mock_embedder.fit.side_effect = Exception("Fitting failed")
        
        documents = [
            Document(id="doc1", content="Content", metadata={})
        ]
        
        with patch('logging.getLogger'):
            processed_docs = list(self.store.process(documents))
            
            assert len(processed_docs) == 1
            # No vectors should be stored due to fitting failure
            assert len(self.store._vectors) == 0

    def test_process_documents_no_valid_texts(self):
        """
        Test processing documents with no valid texts for fitting
        フィッティング用の有効テキストがない文書処理テスト
        """
        self.mock_embedder.is_fitted.return_value = False
        
        documents = [
            Document(id="doc1", content="   ", metadata={}),  # Only whitespace
            Document(id="doc2", content="", metadata={})       # Empty
        ]
        
        with patch('logging.getLogger'):
            processed_docs = list(self.store.process(documents))
            
            assert len(processed_docs) == 2
            # No vectors should be stored
            assert len(self.store._vectors) == 0

    def test_process_documents_embedding_error(self):
        """
        Test processing documents with embedding error
        埋め込みエラーでの文書処理テスト
        """
        self.mock_embedder.embed_text.side_effect = Exception("Embedding failed")
        
        documents = [
            Document(id="doc1", content="Content", metadata={})
        ]
        
        with patch('logging.getLogger'):
            processed_docs = list(self.store.process(documents))
            
            assert len(processed_docs) == 1
            # Document should still be yielded despite error
            assert processed_docs[0].id == "doc1"
            
            # No vectors stored, but error count increased
            assert len(self.store._vectors) == 0
            assert self.store.processing_stats["embedding_errors"] == 1

    def test_process_documents_storage_error(self):
        """
        Test processing documents with storage error
        ストレージエラーでの文書処理テスト
        """
        self.mock_embedder.embed_text.return_value = np.array([0.1, 0.2])
        self.store._fail_operations = True
        
        documents = [
            Document(id="doc1", content="Content", metadata={})
        ]
        
        with patch('logging.getLogger'):
            processed_docs = list(self.store.process(documents))
            
            assert len(processed_docs) == 1
            # Document should still be yielded despite storage error
            assert processed_docs[0].id == "doc1"
            
            # Error count should be increased
            assert self.store.processing_stats["embedding_errors"] == 1


class TestVectorStoreIndexerInterface:
    """
    Test VectorStore Indexer interface implementation
    VectorStoreのIndexerインターフェース実装テスト
    """

    def setup_method(self):
        """Setup test environment"""
        self.store = MockVectorStore()
        self.mock_embedder = Mock()
        self.store.set_embedder(self.mock_embedder)

    def test_index_document_success(self):
        """
        Test successful document indexing
        成功した文書インデックス化テスト
        """
        document = Document(id="doc1", content="Content to index", metadata={"category": "test"})
        self.mock_embedder.embed_text.return_value = np.array([0.1, 0.2, 0.3])
        
        self.store.index_document(document)
        
        # Verify document was indexed
        assert len(self.store._vectors) == 1
        assert "doc1" in self.store._vectors
        
        entry = self.store._vectors["doc1"]
        assert entry.document_id == "doc1"
        assert entry.content == "Content to index"
        assert np.array_equal(entry.embedding, np.array([0.1, 0.2, 0.3]))
        assert entry.metadata["category"] == "test"

    def test_index_document_no_embedder(self):
        """
        Test document indexing without embedder
        Embedderなしの文書インデックス化テスト
        """
        self.store._embedder = None
        document = Document(id="doc1", content="Content", metadata={})
        
        with pytest.raises(ValueError, match="No embedder set for document indexing"):
            self.store.index_document(document)

    def test_index_document_embedder_error(self):
        """
        Test document indexing with embedder error
        Embedderエラーでの文書インデックス化テスト
        """
        document = Document(id="doc1", content="Content", metadata={})
        self.mock_embedder.embed_text.side_effect = Exception("Embedding error")
        
        with pytest.raises(Exception):
            self.store.index_document(document)

    def test_index_documents_success(self):
        """
        Test successful multiple document indexing
        成功した複数文書インデックス化テスト
        """
        documents = [
            Document(id="doc1", content="Content 1", metadata={"category": "test"}),
            Document(id="doc2", content="Content 2", metadata={"category": "test"})
        ]
        
        self.mock_embedder.embed_texts.return_value = [
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4])
        ]
        
        self.store.index_documents(documents)
        
        # Verify all documents were indexed
        assert len(self.store._vectors) == 2
        assert "doc1" in self.store._vectors
        assert "doc2" in self.store._vectors

    def test_index_documents_no_embedder(self):
        """
        Test multiple document indexing without embedder
        Embedderなしの複数文書インデックス化テスト
        """
        self.store._embedder = None
        documents = [Document(id="doc1", content="Content", metadata={})]
        
        with pytest.raises(ValueError, match="No embedder set for document indexing"):
            self.store.index_documents(documents)

    def test_index_documents_embedder_error(self):
        """
        Test multiple document indexing with embedder error
        Embedderエラーでの複数文書インデックス化テスト
        """
        documents = [Document(id="doc1", content="Content", metadata={})]
        self.mock_embedder.embed_texts.side_effect = Exception("Embedding error")
        
        with pytest.raises(Exception):
            self.store.index_documents(documents)

    def test_remove_document_success(self):
        """
        Test successful document removal
        成功した文書削除テスト
        """
        # Add a document first
        entry = VectorEntry(
            document_id="doc1",
            content="Content",
            embedding=np.array([0.1, 0.2]),
            metadata={}
        )
        self.store.add_vector(entry)
        
        # Remove the document
        result = self.store.remove_document("doc1")
        
        assert result is True
        assert len(self.store._vectors) == 0

    def test_remove_document_not_found(self):
        """
        Test document removal when document not found
        文書が見つからない場合の文書削除テスト
        """
        result = self.store.remove_document("nonexistent")
        assert result is False

    def test_update_document_success(self):
        """
        Test successful document update
        成功した文書更新テスト
        """
        # Add initial document
        initial_entry = VectorEntry(
            document_id="doc1",
            content="Original content",
            embedding=np.array([0.1, 0.2]),
            metadata={"version": 1}
        )
        self.store.add_vector(initial_entry)
        
        # Update the document
        updated_document = Document(
            id="doc1", 
            content="Updated content", 
            metadata={"version": 2}
        )
        self.mock_embedder.embed_text.return_value = np.array([0.3, 0.4])
        
        result = self.store.update_document(updated_document)
        
        assert result is True
        
        # Verify the update
        updated_entry = self.store._vectors["doc1"]
        assert updated_entry.content == "Updated content"
        assert updated_entry.metadata["version"] == 2
        assert np.array_equal(updated_entry.embedding, np.array([0.3, 0.4]))

    def test_update_document_not_found(self):
        """
        Test document update when document not found
        文書が見つからない場合の文書更新テスト
        """
        document = Document(id="nonexistent", content="Content", metadata={})
        self.mock_embedder.embed_text.return_value = np.array([0.1, 0.2])
        
        result = self.store.update_document(document)
        assert result is False

    def test_update_document_indexing_error(self):
        """
        Test document update with indexing error
        インデックス化エラーでの文書更新テスト
        """
        # Add initial document
        initial_entry = VectorEntry(
            document_id="doc1",
            content="Content",
            embedding=np.array([0.1, 0.2]),
            metadata={}
        )
        self.store.add_vector(initial_entry)
        
        # Update with error
        document = Document(id="doc1", content="Updated content", metadata={})
        self.mock_embedder.embed_text.side_effect = Exception("Indexing error")
        
        result = self.store.update_document(document)
        assert result is False

    def test_clear_index(self):
        """
        Test clearing the index
        インデックスクリアテスト
        """
        # Add some documents
        entry = VectorEntry(
            document_id="doc1",
            content="Content",
            embedding=np.array([0.1, 0.2]),
            metadata={}
        )
        self.store.add_vector(entry)
        
        # Clear index
        self.store.clear_index()
        
        assert len(self.store._vectors) == 0

    def test_get_document_count(self):
        """
        Test getting document count
        文書数取得テスト
        """
        # Initially empty
        assert self.store.get_document_count() == 0
        
        # Add documents
        entries = [
            VectorEntry("doc1", "Content 1", np.array([0.1]), {}),
            VectorEntry("doc2", "Content 2", np.array([0.2]), {})
        ]
        
        for entry in entries:
            self.store.add_vector(entry)
        
        assert self.store.get_document_count() == 2


class TestVectorStoreRetrieverInterface:
    """
    Test VectorStore Retriever interface implementation
    VectorStoreのRetrieverインターフェース実装テスト
    """

    def setup_method(self):
        """Setup test environment"""
        self.store = MockVectorStore()
        self.mock_embedder = Mock()
        self.store.set_embedder(self.mock_embedder)
        
        # Add sample data
        entries = [
            VectorEntry(
                document_id="doc1",
                content="Machine learning content",
                embedding=np.array([1.0, 0.0]),
                metadata={"topic": "AI"}
            ),
            VectorEntry(
                document_id="doc2", 
                content="Deep learning content",
                embedding=np.array([0.9, 0.1]),
                metadata={"topic": "AI"}
            )
        ]
        
        for entry in entries:
            self.store.add_vector(entry)

    def test_retrieve_success(self):
        """
        Test successful document retrieval
        成功した文書検索テスト
        """
        self.mock_embedder.embed_text.return_value = np.array([1.0, 0.0])
        
        results = self.store.retrieve("machine learning query", limit=2)
        
        assert len(results) <= 2
        
        # Verify SearchResult structure
        for result in results:
            assert hasattr(result, 'document_id')
            assert hasattr(result, 'document')
            assert hasattr(result, 'score')
            assert hasattr(result, 'metadata')
            
            # Verify Document structure
            assert isinstance(result.document, Document)
            assert result.document.id == result.document_id

    def test_retrieve_with_metadata_filter(self):
        """
        Test document retrieval with metadata filter
        メタデータフィルタ付き文書検索テスト
        """
        self.mock_embedder.embed_text.return_value = np.array([1.0, 0.0])
        
        results = self.store.retrieve(
            "test query", 
            limit=5,
            metadata_filter={"topic": "AI"}
        )
        
        # All results should have topic="AI"
        for result in results:
            assert result.metadata["topic"] == "AI"

    def test_retrieve_default_limit(self):
        """
        Test document retrieval with default limit
        デフォルト制限での文書検索テスト
        """
        self.mock_embedder.embed_text.return_value = np.array([1.0, 0.0])
        
        results = self.store.retrieve("test query")
        
        # Should use default limit of 10
        assert len(results) <= 10

    def test_retrieve_no_embedder(self):
        """
        Test document retrieval without embedder
        Embedderなしの文書検索テスト
        """
        self.store._embedder = None
        
        results = self.store.retrieve("test query")
        assert len(results) == 0


class TestVectorStoreErrorHandling:
    """
    Test VectorStore error handling and edge cases
    VectorStoreエラーハンドリングとエッジケースのテスト
    """

    def setup_method(self):
        """Setup test environment"""
        self.store = MockVectorStore()

    def test_abstract_methods_not_implemented(self):
        """
        Test that abstract methods raise NotImplementedError
        抽象メソッドがNotImplementedErrorを発生させることのテスト
        """
        # This test verifies VectorStore is properly abstract
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            VectorStore()

    def test_logging_in_error_scenarios(self):
        """
        Test logging behavior in error scenarios
        エラーシナリオでのログ動作テスト
        """
        self.mock_embedder = Mock()
        self.store.set_embedder(self.mock_embedder)
        
        # Mock the get_vector method to raise an exception to trigger error logging
        with patch.object(self.store, 'get_vector', side_effect=Exception("Test error")):
            with patch('logging.getLogger') as mock_get_logger:
                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger
                
                results = self.store.search_similar_to_document("doc1")
                
                assert len(results) == 0
                mock_logger.error.assert_called()

    def test_stats_integration(self):
        """
        Test processing stats integration
        処理統計統合テスト
        """
        mock_embedder = Mock()
        self.store.set_embedder(mock_embedder)
        mock_embedder.embed_text.return_value = np.array([0.1, 0.2])
        
        # Add a vector
        entry = VectorEntry("doc1", "Content", np.array([0.1, 0.2]), {})
        self.store.add_vector(entry)
        
        # Perform search
        self.store.search_similar_to_document("doc1")
        
        # Check stats were updated
        assert self.store.processing_stats["searches_performed"] > 0

    def test_embedder_integration_edge_cases(self):
        """
        Test embedder integration edge cases
        Embedder統合のエッジケーステスト
        """
        # Test with embedder that has embed_texts method
        mock_embedder = Mock()
        mock_embedder.embed_texts.return_value = [np.array([0.1]), np.array([0.2])]
        self.store.set_embedder(mock_embedder)
        
        documents = [
            Document(id="doc1", content="Content 1", metadata={}),
            Document(id="doc2", content="Content 2", metadata={})
        ]
        
        self.store.index_documents(documents)
        
        assert len(self.store._vectors) == 2
        mock_embedder.embed_texts.assert_called_once_with(["Content 1", "Content 2"])