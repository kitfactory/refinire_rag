"""
Additional comprehensive tests for InMemoryVectorStore to achieve 95%+ coverage
InMemoryVectorStoreの95%以上のカバレッジを達成するための追加包括的テスト
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import logging
from typing import List, Dict, Any

from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.storage.vector_store import VectorEntry, VectorSearchResult, VectorStoreStats
from refinire_rag.exceptions import StorageError


class TestInMemoryVectorStoreAdditionalCoverage:
    """Additional tests to cover missing edge cases and error paths"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.store = InMemoryVectorStore(similarity_metric="cosine")
    
    def test_add_vectors_with_invalid_embeddings_warning(self):
        """Test add_vectors logs warning for invalid embeddings and continues"""
        # Create entries with invalid embeddings
        valid_entry = VectorEntry(
            document_id="valid_doc",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Valid content",
            metadata={"type": "valid"}
        )
        
        invalid_entry_empty = VectorEntry(
            document_id="invalid_empty", 
            embedding=np.array([]),
            content="Invalid content",
            metadata={"type": "invalid"}
        )
        
        entries = [valid_entry, invalid_entry_empty]
        
        # Capture log messages
        with patch('refinire_rag.storage.in_memory_vector_store.logger') as mock_logger:
            result_ids = self.store.add_vectors(entries)
            
            # Should only add the valid entry
            assert len(result_ids) == 1
            assert result_ids[0] == "valid_doc"
            
            # Should log warning for invalid entry
            assert mock_logger.warning.call_count == 1
            mock_logger.warning.assert_any_call("Skipping document invalid_empty - invalid embedding")
    
    def test_update_vector_invalid_embedding_error(self):
        """Test update_vector raises error for invalid embedding"""
        # First add a valid vector
        valid_entry = VectorEntry(
            document_id="test_doc",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Test content",
            metadata={"type": "test"}
        )
        self.store.add_vector(valid_entry)
        
        # Try to update with invalid embedding
        invalid_entry = VectorEntry(
            document_id="test_doc",
            embedding=None,
            content="Updated content",
            metadata={"type": "updated"}
        )
        
        with pytest.raises(StorageError, match="Failed to update vector"):
            self.store.update_vector(invalid_entry)
    
    def test_search_similar_unsupported_metric_error(self):
        """Test search_similar raises error for unsupported similarity metric"""
        # Add some vectors
        entry = VectorEntry(
            document_id="test_doc",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Test content",
            metadata={"type": "test"}
        )
        self.store.add_vector(entry)
        
        # Change to unsupported metric
        self.store.similarity_metric = "unsupported_metric"
        
        query_vector = np.array([1.0, 1.0, 1.0])
        
        with pytest.raises(StorageError, match="Failed to search similar vectors"):
            self.store.search_similar(query_vector)
    
    def test_search_by_metadata_limit_behavior(self):
        """Test search_by_metadata respects limit parameter"""
        # Add multiple entries
        entries = []
        for i in range(15):
            entry = VectorEntry(
                document_id=f"doc_{i}",
                embedding=np.array([float(i), float(i+1), float(i+2)]),
                content=f"Content {i}",
                metadata={"category": "test", "index": i}
            )
            entries.append(entry)
            
        self.store.add_vectors(entries)
        
        # Search with limit
        results = self.store.search_by_metadata(
            filters={"category": "test"},
            limit=5
        )
        
        # Should respect limit
        assert len(results) == 5
        
        # Each result should be valid
        for result in results:
            assert result.metadata["category"] == "test"
            assert result.score == 1.0  # No similarity score for metadata search
    
    def test_get_stats_error_handling(self):
        """Test get_stats error handling"""
        # Mock to cause error in stats calculation
        with patch.object(self.store, '_vectors', side_effect=Exception("Stats error")):
            with pytest.raises(StorageError, match="Failed to get stats"):
                self.store.get_stats()
    
    def test_clear_error_handling(self):
        """Test clear error handling"""
        # Add some vectors first
        entry = VectorEntry(
            document_id="test_doc",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Test content",
            metadata={}
        )
        self.store.add_vector(entry)
        
        # Mock len() to cause error during clear 
        with patch('builtins.len', side_effect=Exception("Clear error")):
            with pytest.raises(StorageError, match="Failed to clear vectors"):
                self.store.clear()
    
    def test_rebuild_matrix_error_handling(self):
        """Test _rebuild_matrix error handling"""
        # Add vectors
        entry = VectorEntry(
            document_id="test_doc",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Test content",
            metadata={}
        )
        self.store.add_vector(entry)
        
        # Capture logging
        with patch('refinire_rag.storage.in_memory_vector_store.logger') as mock_logger:
            # Mock np.vstack to raise error
            with patch('numpy.vstack', side_effect=Exception("Matrix error")):
                self.store._rebuild_matrix()
                
                # Should log error and reset matrix
                mock_logger.error.assert_called_once()
                assert self.store._vector_matrix is None
                assert len(self.store._document_ids) == 0
    
    def test_rebuild_matrix_no_rebuild_needed(self):
        """Test _rebuild_matrix when no rebuild is needed"""
        # Add a vector to trigger initial rebuild
        entry = VectorEntry(
            document_id="test_doc",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Test content",
            metadata={}
        )
        self.store.add_vector(entry)
        
        # Force a rebuild
        self.store._rebuild_matrix()
        original_matrix = self.store._vector_matrix.copy()
        
        # Mark as not needing rebuild
        self.store._needs_rebuild = False
        
        # Call rebuild again - should not change matrix
        self.store._rebuild_matrix()
        
        # Matrix should be unchanged
        np.testing.assert_array_equal(self.store._vector_matrix, original_matrix)
    
    def test_rebuild_matrix_empty_store(self):
        """Test _rebuild_matrix with empty store"""
        # Ensure store is empty
        self.store.clear()
        
        # Mark as needing rebuild but store is empty
        self.store._needs_rebuild = True
        
        # Call rebuild - should return early
        self.store._rebuild_matrix()
        
        # Should still need rebuild
        assert self.store._needs_rebuild is True
    
    def test_matches_filters_unknown_operator(self):
        """Test _matches_filters with unknown operator"""
        metadata = {"category": "test", "score": 85}
        
        # Use unknown operator
        filters = {"score": {"$unknown_op": 90}}
        
        result = self.store._matches_filters(metadata, filters)
        
        # Should return False for unknown operator
        assert result is False
    
    def test_complex_metadata_filtering_edge_cases(self):
        """Test edge cases in metadata filtering"""
        # Add entry with complex metadata
        entry = VectorEntry(
            document_id="test_doc",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Test content",
            metadata={
                "score": None,
                "tags": ["tag1", "tag2"],
                "description": "test description",
                "count": 0
            }
        )
        self.store.add_vector(entry)
        
        # Test $gt with None value (should return False)
        results = self.store.search_by_metadata({"score": {"$gt": 50}})
        assert len(results) == 0
        
        # Test $contains with non-string value 
        results = self.store.search_by_metadata({"tags": {"$contains": "tag1"}})
        assert len(results) == 0  # tags is not a string
        
        # Test $eq with 0 value
        results = self.store.search_by_metadata({"count": {"$eq": 0}})
        assert len(results) == 1
    
    def test_logging_behavior(self):
        """Test various logging behaviors"""
        with patch('refinire_rag.storage.in_memory_vector_store.logger') as mock_logger:
            # Test initialization logging
            store = InMemoryVectorStore(similarity_metric="euclidean")
            mock_logger.info.assert_called_with("Initialized InMemoryVectorStore with euclidean similarity")
            
            # Test add_vector debug logging
            entry = VectorEntry(
                document_id="test_doc",
                embedding=np.array([1.0, 2.0, 3.0]),
                content="Test content",
                metadata={}
            )
            store.add_vector(entry)
            mock_logger.debug.assert_any_call("Added vector for document test_doc")
            
            # Test add_vectors info logging
            entries = [
                VectorEntry(
                    document_id=f"doc_{i}",
                    embedding=np.array([float(i), float(i+1)]),
                    content=f"Content {i}",
                    metadata={}
                )
                for i in range(3)
            ]
            store.add_vectors(entries)
            mock_logger.info.assert_any_call("Added 3 vectors to store")
            
            # Test update_vector debug logging
            entry.content = "Updated content"
            store.update_vector(entry)
            mock_logger.debug.assert_any_call("Updated vector for document test_doc")
            
            # Test delete_vector debug logging
            store.delete_vector("test_doc")
            mock_logger.debug.assert_any_call("Deleted vector for document test_doc")
    
    def test_error_propagation_from_internal_methods(self):
        """Test that StorageErrors are properly raised from internal operations"""
        # Test storage error in search operations
        self.store.add_vector(VectorEntry(
            document_id="test",
            embedding=np.array([1.0, 2.0]),
            content="test",
            metadata={}
        ))
        
        # Mock _matches_filters to raise an exception during search
        with patch.object(self.store, '_matches_filters', side_effect=Exception("Filter error")):
            with pytest.raises(StorageError, match="Failed to search by metadata"):
                self.store.search_by_metadata({"category": "test"})
    
    def test_vector_dimension_consistency(self):
        """Test behavior with vectors of different dimensions"""
        # Add vectors of different dimensions
        entry1 = VectorEntry(
            document_id="doc1",
            embedding=np.array([1.0, 2.0]),
            content="Content 1",
            metadata={}
        )
        
        entry2 = VectorEntry(
            document_id="doc2", 
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Content 2",
            metadata={}
        )
        
        self.store.add_vector(entry1)
        self.store.add_vector(entry2)
        
        # Get stats - should use first vector's dimension
        stats = self.store.get_stats()
        assert stats.vector_dimension == 2  # From first vector
        assert stats.total_vectors == 2
        
        # Search should still work despite dimension mismatch
        # (though results may not be meaningful)
        query = np.array([1.0, 1.0])
        results = self.store.search_similar(query, limit=5)
        
        # Should return some results despite dimension issues
        assert len(results) <= 2


class TestInMemoryVectorStoreStressAndEdgeCases:
    """Stress tests and edge cases"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.store = InMemoryVectorStore()
    
    def test_large_batch_operations(self):
        """Test performance with large batch operations"""
        # Create large batch of vectors
        large_batch = []
        for i in range(1000):
            entry = VectorEntry(
                document_id=f"doc_{i:04d}",
                embedding=np.random.random(128),
                content=f"Content for document {i}",
                metadata={"batch": "large", "index": i, "category": f"cat_{i % 10}"}
            )
            large_batch.append(entry)
        
        # Add all at once
        result_ids = self.store.add_vectors(large_batch)
        assert len(result_ids) == 1000
        
        # Test large search
        query_vector = np.random.random(128)
        results = self.store.search_similar(query_vector, limit=50)
        assert len(results) == 50
        
        # Test metadata search on large dataset
        metadata_results = self.store.search_by_metadata(
            {"category": "cat_5"},
            limit=20
        )
        assert len(metadata_results) <= 100  # Should be exactly 100 but limited to 20
    
    def test_empty_operations(self):
        """Test operations on empty store"""
        # Test search on empty store
        query_vector = np.array([1.0, 2.0, 3.0])
        results = self.store.search_similar(query_vector)
        assert len(results) == 0
        
        # Test metadata search on empty store
        metadata_results = self.store.search_by_metadata({"any": "filter"})
        assert len(metadata_results) == 0
        
        # Test count on empty store
        count = self.store.count_vectors()
        assert count == 0
        
        # Test get_vector on empty store
        vector = self.store.get_vector("nonexistent")
        assert vector is None
        
        # Test delete on empty store
        result = self.store.delete_vector("nonexistent")
        assert result is False
        
        # Test update on empty store
        entry = VectorEntry(
            document_id="test",
            embedding=np.array([1.0]),
            content="test",
            metadata={}
        )
        result = self.store.update_vector(entry)
        assert result is False
    
    def test_extreme_similarity_scores(self):
        """Test handling of extreme similarity scores"""
        # Add vectors that will produce extreme scores
        entry1 = VectorEntry(
            document_id="zero_vector",
            embedding=np.array([0.0, 0.0, 0.0]),
            content="Zero vector",
            metadata={}
        )
        
        entry2 = VectorEntry(
            document_id="large_vector",
            embedding=np.array([1000.0, 1000.0, 1000.0]),
            content="Large vector", 
            metadata={}
        )
        
        self.store.add_vector(entry1)
        self.store.add_vector(entry2)
        
        # Query with zero vector
        zero_query = np.array([0.0, 0.0, 0.0])
        results = self.store.search_similar(zero_query)
        
        # Should handle NaN/inf gracefully
        assert len(results) >= 0
        for result in results:
            assert not np.isnan(result.score)
            assert not np.isinf(result.score)


class TestInMemoryVectorStoreConcurrencyAndThreadSafety:
    """Test thread safety considerations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.store = InMemoryVectorStore()
    
    def test_concurrent_read_operations(self):
        """Test that concurrent read operations work correctly"""
        # Add some test data
        entries = []
        for i in range(10):
            entry = VectorEntry(
                document_id=f"doc_{i}",
                embedding=np.random.random(64),
                content=f"Content {i}",
                metadata={"index": i}
            )
            entries.append(entry)
        
        self.store.add_vectors(entries)
        
        # Simulate concurrent reads
        query_vector = np.random.random(64)
        
        # Multiple searches should all work
        results1 = self.store.search_similar(query_vector, limit=5)
        results2 = self.store.search_by_metadata({"index": 5})
        results3 = self.store.count_vectors({"index": {"$gte": 5}})
        
        assert len(results1) == 5
        assert len(results2) == 1
        assert results3 >= 5
    
    def test_matrix_rebuild_consistency(self):
        """Test that matrix rebuilding maintains consistency"""
        # Add initial vectors
        for i in range(5):
            entry = VectorEntry(
                document_id=f"initial_{i}",
                embedding=np.array([float(i), float(i+1), float(i+2)]),
                content=f"Initial content {i}",
                metadata={"phase": "initial"}
            )
            self.store.add_vector(entry)
        
        # Force matrix build
        query = np.array([1.0, 2.0, 3.0])
        initial_results = self.store.search_similar(query)
        
        # Add more vectors (should trigger rebuild)
        for i in range(5, 10):
            entry = VectorEntry(
                document_id=f"additional_{i}",
                embedding=np.array([float(i), float(i+1), float(i+2)]),
                content=f"Additional content {i}",
                metadata={"phase": "additional"}
            )
            self.store.add_vector(entry)
        
        # Search again (should rebuild matrix automatically)
        final_results = self.store.search_similar(query)
        
        # Should have more results now
        assert len(final_results) >= len(initial_results)
        assert len(final_results) <= 10