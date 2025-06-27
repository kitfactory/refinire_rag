"""
Comprehensive tests for InMemoryVectorStore functionality
InMemoryVectorStoreの機能の包括的テスト

This module provides comprehensive coverage for the InMemoryVectorStore class,
testing all core operations, search functionality, and error handling.
このモジュールは、InMemoryVectorStoreクラスの包括的カバレッジを提供し、
全てのコア操作、検索機能、エラーハンドリングをテストします。
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.storage.vector_store import VectorEntry, VectorStoreStats
from refinire_rag.exceptions import StorageError


class TestInMemoryVectorStoreConfiguration:
    """
    Test InMemoryVectorStore configuration and initialization
    InMemoryVectorStoreの設定と初期化のテスト
    """

    def test_default_configuration(self):
        """
        Test default configuration initialization
        デフォルト設定の初期化テスト
        """
        store = InMemoryVectorStore()
        assert store.similarity_metric == "cosine"
        assert len(store._vectors) == 0
        assert store._needs_rebuild is True

    def test_custom_configuration(self):
        """
        Test custom configuration settings
        カスタム設定のテスト
        """
        store = InMemoryVectorStore(similarity_metric="euclidean")
        assert store.similarity_metric == "euclidean"

    def test_store_initialization_with_config(self):
        """
        Test store initialization with different configurations
        異なる設定でのストア初期化テスト
        """
        # Test with cosine similarity
        store_cosine = InMemoryVectorStore(similarity_metric="cosine")
        assert store_cosine.similarity_metric == "cosine"
        assert len(store_cosine._vectors) == 0
        assert store_cosine._needs_rebuild is True

        # Test with euclidean similarity
        store_euclidean = InMemoryVectorStore(similarity_metric="euclidean")
        assert store_euclidean.similarity_metric == "euclidean"

        # Test with dot product similarity
        store_dot = InMemoryVectorStore(similarity_metric="dot")
        assert store_dot.similarity_metric == "dot"

    def test_get_config(self):
        """
        Test get_config method
        get_configメソッドのテスト
        """
        store = InMemoryVectorStore(similarity_metric="euclidean")
        config = store.get_config()
        
        assert config['similarity_metric'] == "euclidean"


class TestInMemoryVectorStoreCRUD:
    """
    Test basic CRUD operations for InMemoryVectorStore
    InMemoryVectorStoreの基本CRUD操作のテスト
    """

    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        self.store = InMemoryVectorStore(similarity_metric="cosine")
        
        # Create sample vector entries
        self.sample_vectors = [
            VectorEntry(
                document_id="doc1",
                content="First document content",
                embedding=[0.1, 0.2, 0.3, 0.4],
                metadata={"category": "science", "author": "Alice", "year": 2023}
            ),
            VectorEntry(
                document_id="doc2", 
                content="Second document content",
                embedding=[0.2, 0.3, 0.4, 0.5],
                metadata={"category": "technology", "author": "Bob", "year": 2022}
            ),
            VectorEntry(
                document_id="doc3",
                content="Third document content", 
                embedding=[0.3, 0.4, 0.5, 0.6],
                metadata={"category": "science", "author": "Alice", "year": 2024}
            )
        ]

    def test_add_vector_basic(self):
        """
        Test basic vector addition
        基本的なベクトル追加テスト
        """
        vector_entry = self.sample_vectors[0]
        
        # Add vector
        self.store.add_vector(vector_entry)
        
        # Verify vector was added
        assert len(self.store._vectors) == 1
        assert "doc1" in self.store._vectors
        assert self.store._vectors["doc1"] == vector_entry
        assert self.store._needs_rebuild is True

    def test_add_vector_invalid_embedding(self):
        """
        Test adding vector with invalid embedding
        無効な埋め込みでのベクトル追加テスト
        """
        # Test with None embedding
        invalid_vector = VectorEntry(
            document_id="invalid1",
            content="Invalid content",
            embedding=None,
            metadata={}
        )
        
        with pytest.raises(StorageError):
            self.store.add_vector(invalid_vector)

        # Test with empty embedding
        invalid_vector.embedding = []
        with pytest.raises(StorageError):
            self.store.add_vector(invalid_vector)

    def test_add_vectors_batch(self):
        """
        Test batch vector addition
        バッチベクトル追加テスト
        """
        # Add multiple vectors
        self.store.add_vectors(self.sample_vectors)
        
        # Verify all vectors were added
        assert len(self.store._vectors) == 3
        for vector in self.sample_vectors:
            assert vector.document_id in self.store._vectors
            assert self.store._vectors[vector.document_id] == vector

    def test_add_vectors_partial_failure(self):
        """
        Test batch addition with some invalid vectors
        一部無効なベクトルでのバッチ追加テスト
        """
        # Create mixed valid and invalid vectors
        mixed_vectors = [
            self.sample_vectors[0],  # Valid
            VectorEntry(
                document_id="invalid",
                content="Invalid content",
                embedding=None,
                metadata={}
            ),  # Invalid
            self.sample_vectors[1]   # Valid
        ]
        
        # The implementation throws an error with None embeddings
        with pytest.raises(StorageError):
            self.store.add_vectors(mixed_vectors)
        
        # The first valid vector might have been added before the error
        assert len(self.store._vectors) <= 1

    def test_get_vector(self):
        """
        Test vector retrieval
        ベクトル取得テスト
        """
        # Add vector first
        vector_entry = self.sample_vectors[0]
        self.store.add_vector(vector_entry)
        
        # Retrieve vector
        retrieved_vector = self.store.get_vector("doc1")
        assert retrieved_vector == vector_entry
        
        # Test non-existent vector
        assert self.store.get_vector("nonexistent") is None

    def test_update_vector(self):
        """
        Test vector update
        ベクトル更新テスト
        """
        # Add initial vector
        original_vector = self.sample_vectors[0]
        self.store.add_vector(original_vector)
        
        # Create updated vector
        updated_vector = VectorEntry(
            document_id="doc1",
            content="Updated content",
            embedding=[0.5, 0.6, 0.7, 0.8],
            metadata={"category": "updated", "author": "Updated Author"}
        )
        
        # Update vector
        success = self.store.update_vector(updated_vector)
        assert success is True
        
        # Verify update
        retrieved_vector = self.store.get_vector("doc1")
        assert retrieved_vector.content == "Updated content"
        assert np.array_equal(retrieved_vector.embedding, np.array([0.5, 0.6, 0.7, 0.8]))
        assert retrieved_vector.metadata["category"] == "updated"
        assert self.store._needs_rebuild is True

    def test_update_nonexistent_vector(self):
        """
        Test updating non-existent vector
        存在しないベクトルの更新テスト
        """
        vector_entry = self.sample_vectors[0]
        
        result = self.store.update_vector(vector_entry)
        assert result is False

    def test_delete_vector(self):
        """
        Test vector deletion
        ベクトル削除テスト
        """
        # Add vector first
        vector_entry = self.sample_vectors[0]
        self.store.add_vector(vector_entry)
        assert len(self.store._vectors) == 1
        
        # Delete vector
        result = self.store.delete_vector("doc1")
        assert result is True
        assert len(self.store._vectors) == 0
        assert self.store._needs_rebuild is True
        
        # Test deleting non-existent vector
        result = self.store.delete_vector("nonexistent")
        assert result is False

    def test_clear_all_vectors(self):
        """
        Test clearing all vectors
        全ベクトルクリアテスト
        """
        # Add multiple vectors
        self.store.add_vectors(self.sample_vectors)
        assert len(self.store._vectors) == 3
        
        # Clear all vectors
        self.store.clear()
        assert len(self.store._vectors) == 0
        assert self.store._needs_rebuild is True
        assert self.store._vector_matrix is None
        assert self.store._document_ids == []

    def test_count_vectors(self):
        """
        Test vector counting
        ベクトル数カウントテスト
        """
        # Initially empty
        assert self.store.count_vectors() == 0
        
        # Add some vectors
        self.store.add_vectors(self.sample_vectors)
        assert self.store.count_vectors() == 3
        
        # Count with metadata filter
        science_count = self.store.count_vectors(filters={"category": "science"})
        assert science_count == 2
        
        # Count with non-matching filter
        none_count = self.store.count_vectors(filters={"category": "nonexistent"})
        assert none_count == 0

    def test_get_all_vectors(self):
        """
        Test getting all vectors
        全ベクトル取得テスト
        """
        # Initially empty
        all_vectors = self.store.get_all_vectors()
        assert len(all_vectors) == 0
        
        # Add vectors and retrieve
        self.store.add_vectors(self.sample_vectors)
        all_vectors = self.store.get_all_vectors()
        
        assert len(all_vectors) == 3
        for vector in self.sample_vectors:
            assert vector in all_vectors


class TestInMemoryVectorStoreSearch:
    """
    Test search functionality for InMemoryVectorStore
    InMemoryVectorStoreの検索機能のテスト
    """

    def setup_method(self):
        """
        Set up test environment with pre-populated vectors
        ベクトルが事前設定されたテスト環境をセットアップ
        """
        self.store = InMemoryVectorStore(similarity_metric="cosine")
        
        # Create vectors with known relationships
        self.test_vectors = [
            VectorEntry(
                document_id="similar1",
                content="Machine learning algorithms",
                embedding=[1.0, 0.0, 0.0, 0.0],  # Base vector
                metadata={"topic": "AI", "difficulty": "beginner", "score": 85}
            ),
            VectorEntry(
                document_id="similar2", 
                content="Deep learning networks",
                embedding=[0.9, 0.1, 0.0, 0.0],  # Very similar to similar1
                metadata={"topic": "AI", "difficulty": "advanced", "score": 92}
            ),
            VectorEntry(
                document_id="different1",
                content="Cooking recipes",
                embedding=[0.0, 0.0, 1.0, 0.0],  # Orthogonal to AI vectors
                metadata={"topic": "cooking", "difficulty": "easy", "score": 78}
            ),
            VectorEntry(
                document_id="different2",
                content="Sports analysis",
                embedding=[0.0, 0.0, 0.0, 1.0],  # Orthogonal to all others
                metadata={"topic": "sports", "difficulty": "intermediate", "score": 88}
            )
        ]
        
        # Add all test vectors
        self.store.add_vectors(self.test_vectors)

    def test_similarity_search_cosine(self):
        """
        Test cosine similarity search
        コサイン類似度検索テスト
        """
        # Search with query vector similar to similar1
        query_vector = [0.95, 0.05, 0.0, 0.0]
        
        results = self.store.search_similar(
            query_vector=np.array(query_vector),
            limit=3
        )
        
        # Verify results
        assert len(results) == 3
        
        # First result should be most similar (similar1 or similar2)
        assert results[0].document_id in ["similar1", "similar2"]
        assert results[0].score > 0.95
        
        # Results should be ordered by similarity (descending)
        similarities = [result.score for result in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_similarity_search_euclidean(self):
        """
        Test euclidean similarity search
        ユークリッド類似度検索テスト
        """
        # Create store with euclidean similarity
        euclidean_store = InMemoryVectorStore(similarity_metric="euclidean")
        euclidean_store.add_vectors(self.test_vectors)
        
        query_vector = [1.0, 0.0, 0.0, 0.0]
        
        results = euclidean_store.search_similar(
            query_vector=np.array(query_vector),
            limit=2
        )
        
        assert len(results) == 2
        # For euclidean, smaller distance means higher similarity
        assert all(result.score >= 0 for result in results)

    def test_similarity_search_dot_product(self):
        """
        Test dot product similarity search
        ドット積類似度検索テスト
        """
        # Create store with dot product similarity
        dot_store = InMemoryVectorStore(similarity_metric="dot")
        dot_store.add_vectors(self.test_vectors)
        
        query_vector = [1.0, 0.0, 0.0, 0.0]
        
        results = dot_store.search_similar(
            query_vector=np.array(query_vector),
            limit=2
        )
        
        assert len(results) == 2
        # Dot product should give highest score to [1.0, 0.0, 0.0, 0.0]
        assert results[0].document_id == "similar1"

    def test_search_with_threshold(self):
        """
        Test similarity search with threshold filtering
        閾値フィルタリングを使った類似度検索テスト
        """
        query_vector = [1.0, 0.0, 0.0, 0.0]
        
        # High threshold - should return only very similar vectors
        results_high = self.store.search_similar(
            query_vector=np.array(query_vector),
            limit=10,
            threshold=0.95
        )
        
        assert len(results_high) <= 2  # Only similar1 and similar2 should pass
        
        # Low threshold - should return more vectors
        results_low = self.store.search_similar(
            query_vector=np.array(query_vector),
            limit=10,
            threshold=0.1
        )
        
        assert len(results_low) >= len(results_high)

    def test_search_with_metadata_filters(self):
        """
        Test similarity search with metadata filtering
        メタデータフィルタリングを使った類似度検索テスト
        """
        query_vector = [1.0, 0.0, 0.0, 0.0]
        
        # Filter by topic
        ai_results = self.store.search_similar(
            query_vector=np.array(query_vector),
            limit=10,
            filters={"topic": "AI"}
        )
        
        assert len(ai_results) == 2
        for result in ai_results:
            assert result.metadata["topic"] == "AI"
        
        # Filter by difficulty
        advanced_results = self.store.search_similar(
            query_vector=np.array(query_vector),
            limit=10,
            filters={"difficulty": "advanced"}
        )
        
        assert len(advanced_results) == 1
        assert advanced_results[0].document_id == "similar2"

    def test_search_by_metadata_only(self):
        """
        Test search by metadata without similarity
        類似度なしでのメタデータ検索テスト
        """
        # Search by exact match
        ai_vectors = self.store.search_by_metadata({"topic": "AI"})
        assert len(ai_vectors) == 2
        
        # Search by non-existent value
        none_vectors = self.store.search_by_metadata({"topic": "nonexistent"})
        assert len(none_vectors) == 0
        
        # Search by multiple filters
        complex_vectors = self.store.search_by_metadata({
            "topic": "AI",
            "difficulty": "advanced"
        })
        assert len(complex_vectors) == 1
        assert complex_vectors[0].document_id == "similar2"

    def test_search_empty_store(self):
        """
        Test search on empty store
        空のストアでの検索テスト
        """
        empty_store = InMemoryVectorStore(similarity_metric="cosine")
        
        query_vector = [1.0, 0.0, 0.0, 0.0]
        
        results = empty_store.search_similar(
            query_vector=np.array(query_vector),
            limit=5
        )
        
        assert len(results) == 0

    def test_matrix_rebuild_mechanism(self):
        """
        Test matrix rebuild mechanism
        マトリックス再構築メカニズムテスト
        """
        # Initially needs rebuild
        assert self.store._needs_rebuild is True
        assert self.store._vector_matrix is None
        
        # First search should trigger rebuild
        query_vector = [1.0, 0.0, 0.0, 0.0]
        results = self.store.search_similar(np.array(query_vector), limit=2)
        
        # After search, matrix should be built
        assert self.store._needs_rebuild is False
        assert self.store._vector_matrix is not None
        assert len(self.store._document_ids) == 4
        
        # Add new vector - should mark for rebuild
        new_vector = VectorEntry(
            document_id="new_doc",
            content="New content",
            embedding=[0.5, 0.5, 0.5, 0.5],
            metadata={"topic": "new"}
        )
        self.store.add_vector(new_vector)
        
        assert self.store._needs_rebuild is True

    def test_get_similarity_matrix(self):
        """
        Test getting similarity matrix
        類似度マトリックス取得テスト
        """
        # Initially will build the matrix
        matrix = self.store.get_similarity_matrix()
        assert matrix is not None
        
        # Trigger rebuild by searching
        query_vector = [1.0, 0.0, 0.0, 0.0]
        self.store.search_similar(np.array(query_vector), limit=1)
        
        # Now matrix should exist
        matrix = self.store.get_similarity_matrix()
        assert matrix is not None
        assert matrix.shape == (4, 4)  # 4 vectors, 4 dimensions


class TestInMemoryVectorStoreMetadataFiltering:
    """
    Test advanced metadata filtering functionality
    高度なメタデータフィルタリング機能のテスト
    """

    def setup_method(self):
        """
        Set up test environment with diverse metadata
        多様なメタデータでテスト環境をセットアップ
        """
        self.store = InMemoryVectorStore(similarity_metric="cosine")
        
        # Create vectors with diverse metadata for testing operators
        self.metadata_vectors = [
            VectorEntry(
                document_id="doc1",
                content="Content 1",
                embedding=[0.1, 0.2, 0.3, 0.4],
                metadata={
                    "score": 85,
                    "category": "science",
                    "tags": ["physics", "quantum"],
                    "year": 2023,
                    "active": True
                }
            ),
            VectorEntry(
                document_id="doc2",
                content="Content 2", 
                embedding=[0.2, 0.3, 0.4, 0.5],
                metadata={
                    "score": 92,
                    "category": "technology",
                    "tags": ["AI", "machine learning"],
                    "year": 2022,
                    "active": True
                }
            ),
            VectorEntry(
                document_id="doc3",
                content="Content 3",
                embedding=[0.3, 0.4, 0.5, 0.6],
                metadata={
                    "score": 78,
                    "category": "science",
                    "tags": ["biology", "genetics"],
                    "year": 2024,
                    "active": False
                }
            ),
            VectorEntry(
                document_id="doc4",
                content="Content 4",
                embedding=[0.4, 0.5, 0.6, 0.7],
                metadata={
                    "score": 88,
                    "category": "mathematics",
                    "tags": ["algebra", "geometry"],
                    "year": 2023,
                    "active": None  # Test None values
                }
            )
        ]
        
        self.store.add_vectors(self.metadata_vectors)

    def test_equality_operator(self):
        """
        Test $eq operator for exact matching
        完全一致のための$eq演算子テスト
        """
        # Test exact string match
        results = self.store.search_by_metadata({"category": {"$eq": "science"}})
        assert len(results) == 2
        
        # Test exact number match
        results = self.store.search_by_metadata({"score": {"$eq": 85}})
        assert len(results) == 1
        assert results[0].document_id == "doc1"
        
        # Test boolean match
        results = self.store.search_by_metadata({"active": {"$eq": True}})
        assert len(results) == 2

    def test_not_equal_operator(self):
        """
        Test $ne operator for non-equality
        非等価のための$ne演算子テスト
        """
        # Test not equal to string
        results = self.store.search_by_metadata({"category": {"$ne": "science"}})
        assert len(results) == 2
        for result in results:
            assert result.metadata["category"] != "science"
        
        # Test not equal to number
        results = self.store.search_by_metadata({"score": {"$ne": 85}})
        assert len(results) == 3

    def test_in_operator(self):
        """
        Test $in operator for membership testing
        メンバーシップテストのための$in演算子テスト
        """
        # Test string membership
        results = self.store.search_by_metadata({"category": {"$in": ["science", "technology"]}})
        assert len(results) == 3
        
        # Test number membership
        results = self.store.search_by_metadata({"score": {"$in": [85, 92]}})
        assert len(results) == 2
        
        # Test empty list
        results = self.store.search_by_metadata({"category": {"$in": []}})
        assert len(results) == 0

    def test_not_in_operator(self):
        """
        Test $nin operator for negative membership testing
        否定メンバーシップテストのための$nin演算子テスト
        """
        # Test string not in membership
        results = self.store.search_by_metadata({"category": {"$nin": ["science"]}})
        assert len(results) == 2
        for result in results:
            assert result.metadata["category"] != "science"

    def test_comparison_operators(self):
        """
        Test numeric comparison operators
        数値比較演算子テスト
        """
        # Test $gt (greater than)
        results = self.store.search_by_metadata({"score": {"$gt": 85}})
        assert len(results) == 2  # 92 and 88
        
        # Test $gte (greater than or equal)
        results = self.store.search_by_metadata({"score": {"$gte": 85}})
        assert len(results) == 3  # 85, 92, and 88
        
        # Test $lt (less than)
        results = self.store.search_by_metadata({"score": {"$lt": 85}})
        assert len(results) == 1  # 78
        
        # Test $lte (less than or equal)
        results = self.store.search_by_metadata({"score": {"$lte": 85}})
        assert len(results) == 2  # 85 and 78
        
        # Test with year
        results = self.store.search_by_metadata({"year": {"$gte": 2023}})
        assert len(results) == 3  # 2023 (2 docs) and 2024 (1 doc)

    def test_contains_operator(self):
        """
        Test $contains operator for substring matching
        部分文字列マッチングのための$contains演算子テスト
        """
        # Test substring in category
        results = self.store.search_by_metadata({"category": {"$contains": "sci"}})
        assert len(results) == 2  # "science" contains "sci"
        
        # Test case sensitivity
        results = self.store.search_by_metadata({"category": {"$contains": "SCI"}})
        assert len(results) == 0  # Case sensitive
        
        # Test contains in non-string field (should not match)
        results = self.store.search_by_metadata({"score": {"$contains": "8"}})
        assert len(results) == 0

    def test_multiple_filter_conditions(self):
        """
        Test multiple filter conditions (AND logic)
        複数フィルタ条件のテスト（AND論理）
        """
        # Multiple exact matches
        results = self.store.search_by_metadata({
            "category": "science",
            "active": True
        })
        assert len(results) == 1
        assert results[0].document_id == "doc1"
        
        # Mix of operators
        results = self.store.search_by_metadata({
            "score": {"$gte": 80},
            "category": {"$in": ["science", "technology"]},
            "year": {"$gte": 2022}
        })
        assert len(results) == 2  # doc1 and doc2

    def test_none_value_handling(self):
        """
        Test handling of None values in metadata
        メタデータでのNone値の処理テスト
        """
        # Test exact match with None
        results = self.store.search_by_metadata({"active": None})
        assert len(results) == 1
        assert results[0].document_id == "doc4"
        
        # Test not equal with None
        results = self.store.search_by_metadata({"active": {"$ne": None}})
        assert len(results) == 3

    def test_nonexistent_field_filtering(self):
        """
        Test filtering on non-existent metadata fields
        存在しないメタデータフィールドでのフィルタリングテスト
        """
        # Filter by non-existent field
        results = self.store.search_by_metadata({"nonexistent_field": "value"})
        assert len(results) == 0
        
        # Filter with operator on non-existent field
        results = self.store.search_by_metadata({"nonexistent_field": {"$gt": 10}})
        assert len(results) == 0

    def test_invalid_operator_handling(self):
        """
        Test handling of invalid operators
        無効な演算子の処理テスト
        """
        # Invalid operator should return False (no matches)
        results = self.store.search_by_metadata({"score": {"$invalid_op": 85}})
        assert len(results) == 0


class TestInMemoryVectorStoreStatistics:
    """
    Test statistics and monitoring functionality
    統計とモニタリング機能のテスト
    """

    def setup_method(self):
        """
        Set up test environment with sample data
        サンプルデータでテスト環境をセットアップ
        """
        self.store = InMemoryVectorStore(similarity_metric="cosine")
        
        # Add sample vectors for statistics testing
        self.sample_vectors = [
            VectorEntry(
                document_id=f"doc{i}",
                content=f"Content {i}",
                embedding=[i * 0.1, (i+1) * 0.1, (i+2) * 0.1, (i+3) * 0.1],
                metadata={"category": "test", "index": i}
            )
            for i in range(5)
        ]
        
        self.store.add_vectors(self.sample_vectors)

    def test_get_stats_basic(self):
        """
        Test basic statistics retrieval
        基本統計取得テスト
        """
        stats = self.store.get_stats()
        
        # Verify basic statistics structure (matching actual implementation)
        assert stats.total_vectors == 5
        assert stats.vector_dimension == 4
        assert stats.storage_size_bytes > 0
        assert stats.index_type == "exact_memory"

    def test_get_stats_with_metadata_breakdown(self):
        """
        Test statistics with metadata breakdown
        メタデータ内訳付きの統計テスト
        """
        stats = self.store.get_stats()
        
        # Get stats returns VectorStoreStats object
        assert isinstance(stats, VectorStoreStats)
        assert stats.total_vectors == 5

    def test_get_stats_empty_store(self):
        """
        Test statistics on empty store
        空のストアでの統計テスト
        """
        empty_store = InMemoryVectorStore(similarity_metric="cosine")
        stats = empty_store.get_stats()
        
        assert stats.total_vectors == 0
        assert stats.vector_dimension == 0
        assert stats.storage_size_bytes == 0

    def test_get_stats_after_operations(self):
        """
        Test statistics after various operations
        各種操作後の統計テスト
        """
        # Initial stats
        initial_stats = self.store.get_stats()
        assert initial_stats.total_vectors == 5
        
        # Add more vectors
        new_vector = VectorEntry(
            document_id="new_doc",
            content="New content",
            embedding=[0.9, 0.8, 0.7, 0.6],
            metadata={"category": "new"}
        )
        self.store.add_vector(new_vector)
        
        # Updated stats
        updated_stats = self.store.get_stats()
        assert updated_stats.total_vectors == 6
        
        # Delete vector
        self.store.delete_vector("new_doc")
        
        # Final stats
        final_stats = self.store.get_stats()
        assert final_stats.total_vectors == 5


class TestInMemoryVectorStoreErrorHandling:
    """
    Test error handling and edge cases
    エラーハンドリングとエッジケースのテスト
    """

    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.store = InMemoryVectorStore(similarity_metric="cosine")

    def test_matrix_rebuild_failure_handling(self):
        """
        Test handling of matrix rebuild failures
        マトリックス再構築失敗の処理テスト
        """
        # Add vectors with inconsistent dimensions
        vectors = [
            VectorEntry(
                document_id="doc1",
                content="Content 1",
                embedding=[0.1, 0.2, 0.3],  # 3 dimensions
                metadata={}
            ),
            VectorEntry(
                document_id="doc2",
                content="Content 2",
                embedding=[0.1, 0.2, 0.3, 0.4],  # 4 dimensions
                metadata={}
            )
        ]
        
        self.store.add_vectors(vectors)
        
        # Search should handle matrix build failure gracefully
        with patch.object(self.store, '_rebuild_matrix', side_effect=Exception("Matrix build failed")):
            # Should raise StorageError due to exception
            with pytest.raises(StorageError):
                self.store.search_similar(np.array([0.1, 0.2, 0.3]), limit=5)

    def test_invalid_query_vector_dimensions(self):
        """
        Test search with invalid query vector dimensions
        無効なクエリベクトル次元での検索テスト
        """
        # Add some valid vectors
        vector = VectorEntry(
            document_id="doc1",
            content="Content",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={}
        )
        self.store.add_vector(vector)
        
        # Search with wrong dimensions - this will likely cause an error in sklearn
        with pytest.raises((ValueError, Exception)):
            self.store.search_similar(np.array([0.1, 0.2]), limit=5)  # 2 dims instead of 4

    def test_storage_error_wrapping(self):
        """
        Test that internal errors are wrapped in StorageError
        内部エラーがStorageErrorでラップされることのテスト
        """
        # Add a vector first so search_by_metadata has something to process
        entry = VectorEntry(
            document_id="test_doc",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Test content",
            metadata={"key": "value"}
        )
        self.store.add_vector(entry)
        
        # Mock an internal method to raise an exception during processing
        with patch.object(self.store, '_matches_filters', side_effect=Exception("Internal error")):
            with pytest.raises(StorageError, match="Failed to search by metadata"):
                self.store.search_by_metadata({"key": "value"})

    def test_concurrent_access_safety(self):
        """
        Test thread safety of basic operations
        基本操作のスレッドセーフティテスト
        """
        import threading
        import time
        
        errors = []
        
        def add_vectors_worker(start_idx):
            try:
                vectors = [
                    VectorEntry(
                        document_id=f"thread_{start_idx}_{i}",
                        content=f"Content {start_idx}_{i}",
                        embedding=[start_idx * 0.1, i * 0.1, 0.3, 0.4],
                        metadata={"thread": start_idx}
                    )
                    for i in range(10)
                ]
                self.store.add_vectors(vectors)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads adding vectors
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_vectors_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check for errors
        if errors:
            pytest.fail(f"Concurrent access errors: {errors}")
        
        # Verify all vectors were added
        stats = self.store.get_stats()
        assert stats.total_vectors == 30  # 3 threads * 10 vectors each

    def test_memory_cleanup_on_clear(self):
        """
        Test memory cleanup when clearing store
        ストアクリア時のメモリクリーンアップテスト
        """
        # Add many vectors
        large_vectors = [
            VectorEntry(
                document_id=f"large_doc_{i}",
                content=f"Large content {i}",
                embedding=[i * 0.01] * 100,  # Large embeddings
                metadata={"index": i}
            )
            for i in range(100)
        ]
        
        self.store.add_vectors(large_vectors)
        
        # Check memory usage
        stats_before = self.store.get_stats()
        assert stats_before.total_vectors == 100
        assert stats_before.storage_size_bytes > 0
        
        # Clear store
        self.store.clear()
        
        # Check memory cleanup
        stats_after = self.store.get_stats()
        assert stats_after.total_vectors == 0
        assert stats_after.storage_size_bytes < stats_before.storage_size_bytes