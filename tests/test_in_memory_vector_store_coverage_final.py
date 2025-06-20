"""
Final coverage tests for InMemoryVectorStore to reach 99%+ coverage
InMemoryVectorStoreの99%以上のカバレッジを達成するための最終テスト
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import logging

from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.storage.vector_store import VectorEntry, VectorSearchResult, VectorStoreStats
from refinire_rag.exceptions import StorageError


class TestInMemoryVectorStoreFinalCoverage:
    """Final tests to achieve 99%+ coverage for remaining lines"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.store = InMemoryVectorStore(similarity_metric="cosine")
    
    def test_add_vectors_empty_list_handling(self):
        """Test add_vectors with empty list - covers line 74-78"""
        # Test empty list
        result_ids = self.store.add_vectors([])
        assert len(result_ids) == 0
        
        # Test list with only invalid entries
        invalid_entry = VectorEntry(
            document_id="invalid_doc",
            embedding=np.array([]),
            content="Invalid content",
            metadata={}
        )
        
        with patch('refinire_rag.storage.in_memory_vector_store.logger') as mock_logger:
            result_ids = self.store.add_vectors([invalid_entry])
            
            # Should return empty list and not trigger info log
            assert len(result_ids) == 0
            # Should not call info log since added_ids is empty
            mock_logger.info.assert_not_called()
    
    def test_add_vectors_info_logging_coverage(self):
        """Test add_vectors info logging when vectors are actually added - covers line 76-77"""
        entries = [
            VectorEntry(
                document_id="doc1",
                embedding=np.array([1.0, 2.0]),
                content="Content 1",
                metadata={}
            ),
            VectorEntry(
                document_id="doc2", 
                embedding=np.array([3.0, 4.0]),
                content="Content 2",
                metadata={}
            )
        ]
        
        with patch('refinire_rag.storage.in_memory_vector_store.logger') as mock_logger:
            result_ids = self.store.add_vectors(entries)
            
            # Should add both vectors
            assert len(result_ids) == 2
            
            # Should call info log since added_ids is not empty
            mock_logger.info.assert_called_once_with("Added 2 vectors to store")
    
    def test_update_vector_invalid_embedding_exception_coverage(self):
        """Test update_vector with invalid embedding - covers line 97"""
        # First add a valid vector
        valid_entry = VectorEntry(
            document_id="test_doc",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Test content",
            metadata={}
        )
        self.store.add_vector(valid_entry)
        
        # Try to update with invalid (empty) embedding
        invalid_entry = VectorEntry(
            document_id="test_doc",
            embedding=np.array([]),
            content="Updated content",
            metadata={}
        )
        
        with pytest.raises(StorageError, match="Failed to update vector"):
            self.store.update_vector(invalid_entry)
    
    def test_delete_vector_exception_coverage(self):
        """Test delete_vector exception handling - covers line 120-121"""
        # Add a vector first
        entry = VectorEntry(
            document_id="test_doc",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Test content",
            metadata={}
        )
        self.store.add_vector(entry)
        
        # Mock the dictionary deletion to raise an exception
        with patch.object(self.store._vectors, '__delitem__', side_effect=Exception("Delete error")):
            with pytest.raises(StorageError, match="Failed to delete vector"):
                self.store.delete_vector("test_doc")
    
    def test_count_vectors_exception_coverage(self):
        """Test count_vectors exception handling - covers line 230-231"""
        # Add some vectors
        entry = VectorEntry(
            document_id="test_doc",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Test content",
            metadata={"category": "test"}
        )
        self.store.add_vector(entry)
        
        # Mock _matches_filters to raise an exception
        with patch.object(self.store, '_matches_filters', side_effect=Exception("Count error")):
            with pytest.raises(StorageError, match="Failed to count vectors"):
                self.store.count_vectors({"category": "test"})
    
    def test_get_stats_exception_coverage(self):
        """Test get_stats exception handling - covers line 259-260"""
        # Add some vectors to make stats calculation meaningful
        entry = VectorEntry(
            document_id="test_doc",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Test content",
            metadata={}
        )
        self.store.add_vector(entry)
        
        # Mock iter() to cause error during stats calculation
        with patch('builtins.iter', side_effect=Exception("Stats error")):
            with pytest.raises(StorageError, match="Failed to get stats"):
                self.store.get_stats()
    
    def test_clear_exception_coverage(self):
        """Test clear exception handling - covers line 275-276"""
        # Add some vectors first
        entry = VectorEntry(
            document_id="test_doc",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Test content",
            metadata={}
        )
        self.store.add_vector(entry)
        
        # Mock len() to cause error in the first line of clear method
        original_len = len
        def mock_len(obj):
            if obj is self.store._vectors:
                raise Exception("Clear error")
            return original_len(obj)
        
        with patch('builtins.len', side_effect=mock_len):
            with pytest.raises(StorageError, match="Failed to clear vectors"):
                self.store.clear()
    
    def test_none_embedding_handling(self):
        """Test handling of None embeddings in various methods"""
        # Test add_vector with None embedding
        none_entry = VectorEntry(
            document_id="none_doc",
            embedding=None,
            content="None embedding content",
            metadata={}
        )
        
        with pytest.raises(StorageError, match="Failed to add vector"):
            self.store.add_vector(none_entry)
    
    def test_edge_case_logging_scenarios(self):
        """Test specific logging scenarios for coverage"""
        # Test debug logging in various operations
        with patch('refinire_rag.storage.in_memory_vector_store.logger') as mock_logger:
            # Add vector - should trigger debug log
            entry = VectorEntry(
                document_id="debug_test",
                embedding=np.array([1.0, 2.0]),
                content="Debug test",
                metadata={}
            )
            self.store.add_vector(entry)
            mock_logger.debug.assert_any_call("Added vector for document debug_test")
            
            # Update vector - should trigger debug log  
            entry.content = "Updated content"
            self.store.update_vector(entry)
            mock_logger.debug.assert_any_call("Updated vector for document debug_test")
            
            # Delete vector - should trigger debug log
            self.store.delete_vector("debug_test")
            mock_logger.debug.assert_any_call("Deleted vector for document debug_test")
    
    def test_search_similar_debug_logging(self):
        """Test search_similar debug logging coverage"""
        # Add vectors
        entries = [
            VectorEntry(
                document_id=f"doc_{i}",
                embedding=np.array([float(i), float(i+1)]),
                content=f"Content {i}",
                metadata={}
            )
            for i in range(3)
        ]
        self.store.add_vectors(entries)
        
        # Search and verify debug logging
        with patch('refinire_rag.storage.in_memory_vector_store.logger') as mock_logger:
            query_vector = np.array([1.0, 2.0])
            results = self.store.search_similar(query_vector, limit=2)
            
            # Should log debug message about found results
            mock_logger.debug.assert_any_call(f"Found {len(results)} similar vectors for query")
    
    def test_search_by_metadata_debug_logging(self):
        """Test search_by_metadata debug logging coverage"""
        # Add vectors with metadata
        entries = [
            VectorEntry(
                document_id=f"meta_doc_{i}",
                embedding=np.array([float(i), float(i+1)]),
                content=f"Meta content {i}",
                metadata={"category": "test", "index": i}
            )
            for i in range(3)
        ]
        self.store.add_vectors(entries)
        
        # Search by metadata and verify debug logging
        with patch('refinire_rag.storage.in_memory_vector_store.logger') as mock_logger:
            results = self.store.search_by_metadata({"category": "test"}, limit=2)
            
            # Should log debug message about found results
            mock_logger.debug.assert_any_call(f"Found {len(results)} vectors matching metadata filters")
    
    def test_rebuild_matrix_debug_logging(self):
        """Test _rebuild_matrix debug logging coverage"""
        # Add vectors
        entry = VectorEntry(
            document_id="matrix_test",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Matrix test",
            metadata={}
        )
        self.store.add_vector(entry)
        
        # Force rebuild and verify debug logging
        with patch('refinire_rag.storage.in_memory_vector_store.logger') as mock_logger:
            self.store._rebuild_matrix()
            
            # Should log debug message about matrix shape
            expected_call = f"Rebuilt vector matrix: {self.store._vector_matrix.shape}"
            mock_logger.debug.assert_any_call(expected_call)


class TestInMemoryVectorStoreCompleteEdgeCases:
    """Complete edge case coverage for remaining scenarios"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.store = InMemoryVectorStore()
    
    def test_invalid_operations_on_empty_store(self):
        """Test operations on completely empty store"""
        # Ensure store is empty
        assert len(self.store._vectors) == 0
        
        # Test get_stats on empty store
        stats = self.store.get_stats()
        assert stats.total_vectors == 0
        assert stats.vector_dimension == 0
        assert stats.storage_size_bytes == 0
        
        # Test clear on empty store
        result = self.store.clear()
        assert result is True
        
        # Test rebuild matrix on empty store
        self.store._needs_rebuild = True
        self.store._rebuild_matrix()
        assert self.store._vector_matrix is None
        assert len(self.store._document_ids) == 0
    
    def test_similarity_metrics_edge_cases(self):
        """Test all similarity metrics with edge case vectors"""
        metrics = ["cosine", "euclidean", "dot"]
        
        for metric in metrics:
            store = InMemoryVectorStore(similarity_metric=metric)
            
            # Add test vectors
            entries = [
                VectorEntry(
                    document_id=f"{metric}_doc_1",
                    embedding=np.array([1.0, 0.0]),
                    content="Test 1",
                    metadata={}
                ),
                VectorEntry(
                    document_id=f"{metric}_doc_2", 
                    embedding=np.array([0.0, 1.0]),
                    content="Test 2",
                    metadata={}
                )
            ]
            store.add_vectors(entries)
            
            # Query
            query_vector = np.array([1.0, 1.0])
            results = store.search_similar(query_vector, limit=2)
            
            # Should get results for all metrics
            assert len(results) == 2
            
            # Scores should be valid floats
            for result in results:
                assert isinstance(result.score, float)
                assert not np.isnan(result.score)
                assert not np.isinf(result.score)