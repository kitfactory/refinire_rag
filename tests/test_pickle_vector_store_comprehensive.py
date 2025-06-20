"""
Comprehensive tests for PickleVectorStore to achieve maximum coverage
PickleVectorStoreの最大カバレッジを達成するための包括的テスト
"""

import pytest
import tempfile
import pickle
import shutil
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import List, Dict, Any

from refinire_rag.storage.pickle_vector_store import PickleVectorStore
from refinire_rag.storage.vector_store import VectorEntry, VectorSearchResult, VectorStoreStats
from refinire_rag.exceptions import StorageError


class TestPickleVectorStoreInitialization:
    """Test PickleVectorStore initialization and setup"""
    
    def test_initialization_with_default_parameters(self):
        """Test initialization with default parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = str(Path(temp_dir) / "test_vectors.pkl")
            store = PickleVectorStore(file_path=file_path)
            
            assert store.file_path == Path(file_path)
            assert store.similarity_metric == "cosine"
            assert store.auto_save is True
            assert store.save_interval == 10
            assert store._operations_since_save == 0
            
            store.close()
    
    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = str(Path(temp_dir) / "custom_vectors.pkl")
            config = {
                "file_path": file_path,
                "similarity_metric": "euclidean",
                "auto_save": False,
                "save_interval": 5
            }
            
            store = PickleVectorStore(
                file_path=file_path,
                similarity_metric="euclidean",
                auto_save=False,
                save_interval=5,
                config=config
            )
            
            assert store.file_path == Path(file_path)
            assert store.similarity_metric == "euclidean"
            assert store.auto_save is False
            assert store.save_interval == 5
            
            store.close()
    
    def test_initialization_creates_parent_directories(self):
        """Test that initialization creates parent directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "directories" / "vectors.pkl"
            store = PickleVectorStore(file_path=str(nested_path))
            
            assert nested_path.parent.exists()
            assert store.file_path == nested_path
            
            store.close()
    
    def test_initialization_loads_existing_file(self):
        """Test that initialization loads existing pickle file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "existing_vectors.pkl"
            
            # Create a pickle file with test data
            test_data = {
                'vectors': {
                    'doc1': VectorEntry(
                        document_id='doc1',
                        embedding=np.array([1.0, 2.0, 3.0]),
                        content='Test content',
                        metadata={'type': 'test'}
                    )
                },
                'similarity_metric': 'cosine',
                'metadata': {
                    'total_vectors': 1,
                    'vector_dimension': 3,
                    'save_timestamp': np.datetime64('now')
                }
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(test_data, f)
            
            # Initialize store - should load existing data
            store = PickleVectorStore(file_path=str(file_path))
            
            assert len(store._vectors) == 1
            assert 'doc1' in store._vectors
            assert np.array_equal(store._vectors['doc1'].embedding, [1.0, 2.0, 3.0])
            
            store.close()
    
    def test_initialization_with_corrupted_file(self):
        """Test initialization with corrupted pickle file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "corrupted_vectors.pkl"
            
            # Create a corrupted file
            with open(file_path, 'w') as f:
                f.write("This is not a valid pickle file")
            
            # Should handle gracefully and start with empty store
            store = PickleVectorStore(file_path=str(file_path))
            
            assert len(store._vectors) == 0
            
            store.close()
    
    def test_initialization_with_similarity_metric_mismatch(self):
        """Test initialization when saved metric differs from requested"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "metric_mismatch.pkl"
            
            # Create pickle file with cosine metric
            test_data = {
                'vectors': {},
                'similarity_metric': 'cosine',
                'metadata': {}
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(test_data, f)
            
            # Initialize with different metric
            with patch('refinire_rag.storage.pickle_vector_store.logger') as mock_logger:
                store = PickleVectorStore(
                    file_path=str(file_path),
                    similarity_metric="euclidean"
                )
                
                # Should log warning about mismatch
                mock_logger.warning.assert_called()
                
                store.close()


class TestPickleVectorStoreVectorOperations:
    """Test vector CRUD operations with persistence"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.file_path = Path(self.temp_dir) / "test_vectors.pkl"
        self.store = PickleVectorStore(
            file_path=str(self.file_path),
            auto_save=False  # Manual save for testing
        )
        
        self.test_entry = VectorEntry(
            document_id="test_doc_001",
            embedding=np.array([1.0, 2.0, 3.0, 4.0]),
            content="This is test content for vector operations",
            metadata={"type": "test", "category": "unit_test"}
        )
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_vector_with_manual_save(self):
        """Test adding vector and manually saving"""
        # Add vector
        doc_id = self.store.add_vector(self.test_entry)
        assert doc_id == "test_doc_001"
        
        # Save to disk
        success = self.store.save_to_disk()
        assert success is True
        assert self.file_path.exists()
        
        # Verify file contains data
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
        
        assert 'vectors' in data
        assert 'test_doc_001' in data['vectors']
        assert data['metadata']['total_vectors'] == 1
    
    def test_add_vector_with_auto_save(self):
        """Test adding vector with auto-save enabled"""
        auto_store = PickleVectorStore(
            file_path=str(self.file_path.with_name("auto_save.pkl")),
            auto_save=True,
            save_interval=1
        )
        
        try:
            # Add vector - should trigger auto-save
            auto_store.add_vector(self.test_entry)
            
            # File should exist
            assert auto_store.file_path.exists()
            
        finally:
            auto_store.close()
    
    def test_add_multiple_vectors(self):
        """Test adding multiple vectors at once"""
        entries = [
            VectorEntry(
                document_id=f"doc_{i:03d}",
                embedding=np.array([i, i+1, i+2]),
                content=f"Content {i}",
                metadata={"index": i}
            )
            for i in range(5)
        ]
        
        doc_ids = self.store.add_vectors(entries)
        assert len(doc_ids) == 5
        assert all(doc_id.startswith("doc_") for doc_id in doc_ids)
        
        # Verify all vectors are stored
        assert len(self.store._vectors) == 5
    
    def test_update_vector_with_persistence(self):
        """Test updating vector and saving changes"""
        # Add initial vector
        self.store.add_vector(self.test_entry)
        
        # Update vector
        updated_entry = VectorEntry(
            document_id="test_doc_001",
            embedding=np.array([5.0, 6.0, 7.0, 8.0]),
            content="Updated content",
            metadata={"type": "updated", "version": 2}
        )
        
        success = self.store.update_vector(updated_entry)
        assert success is True
        
        # Save and verify
        self.store.save_to_disk()
        
        # Reload to verify persistence
        new_store = PickleVectorStore(file_path=str(self.file_path))
        try:
            retrieved = new_store.get_vector("test_doc_001")
            assert retrieved is not None
            assert retrieved.content == "Updated content"
            assert retrieved.metadata["version"] == 2
            assert np.array_equal(retrieved.embedding, [5.0, 6.0, 7.0, 8.0])
        finally:
            new_store.close()
    
    def test_delete_vector_with_persistence(self):
        """Test deleting vector and saving changes"""
        # Add vector
        self.store.add_vector(self.test_entry)
        assert len(self.store._vectors) == 1
        
        # Delete vector
        success = self.store.delete_vector("test_doc_001")
        assert success is True
        assert len(self.store._vectors) == 0
        
        # Save and verify
        self.store.save_to_disk()
        
        # Reload to verify deletion persisted
        new_store = PickleVectorStore(file_path=str(self.file_path))
        try:
            assert len(new_store._vectors) == 0
        finally:
            new_store.close()
    
    def test_clear_vectors_with_persistence(self):
        """Test clearing all vectors and saving"""
        # Add multiple vectors
        entries = [
            VectorEntry(
                document_id=f"doc_{i}",
                embedding=np.array([i, i+1]),
                content=f"Content {i}",
                metadata={}
            )
            for i in range(3)
        ]
        self.store.add_vectors(entries)
        assert len(self.store._vectors) == 3
        
        # Clear all
        success = self.store.clear()
        assert success is True
        assert len(self.store._vectors) == 0
        
        # Verify file was updated (clear() calls save_to_disk automatically)
        assert self.file_path.exists()
        
        # Reload to verify clearing persisted
        new_store = PickleVectorStore(file_path=str(self.file_path))
        try:
            assert len(new_store._vectors) == 0
        finally:
            new_store.close()


class TestPickleVectorStorePersistence:
    """Test persistence-specific functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.file_path = Path(self.temp_dir) / "persistence_test.pkl"
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_to_disk_success(self):
        """Test successful save to disk"""
        store = PickleVectorStore(file_path=str(self.file_path), auto_save=False)
        
        try:
            # Add some data
            entry = VectorEntry(
                document_id="test_doc",
                embedding=np.array([1.0, 2.0]),
                content="Test content",
                metadata={"test": True}
            )
            store.add_vector(entry)
            
            # Save to disk
            success = store.save_to_disk()
            assert success is True
            assert self.file_path.exists()
            
            # Verify file structure
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
            
            assert 'vectors' in data
            assert 'similarity_metric' in data
            assert 'metadata' in data
            assert data['metadata']['total_vectors'] == 1
            assert 'save_timestamp' in data['metadata']
            
        finally:
            store.close()
    
    def test_save_to_disk_failure(self):
        """Test save to disk failure handling"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_vectors.pkl"
            store = PickleVectorStore(file_path=str(file_path), auto_save=False)
            
            try:
                # Mock the pickle.dump to raise an exception
                with patch('pickle.dump', side_effect=Exception("Mock save error")):
                    success = store.save_to_disk()
                    assert success is False
                    
            finally:
                store.close()
    
    def test_load_from_disk_explicit(self):
        """Test explicit load from disk"""
        # Create store and add data
        store1 = PickleVectorStore(file_path=str(self.file_path), auto_save=False)
        
        try:
            entry = VectorEntry(
                document_id="load_test",
                embedding=np.array([3.0, 4.0, 5.0]),
                content="Load test content",
                metadata={"loaded": True}
            )
            store1.add_vector(entry)
            store1.save_to_disk()
        finally:
            store1.close()
        
        # Create new store and explicitly load
        store2 = PickleVectorStore(file_path=str(self.file_path), auto_save=False)
        
        try:
            # Should have loaded data automatically, but test explicit load
            success = store2.load_from_disk()
            assert success is True
            
            retrieved = store2.get_vector("load_test")
            assert retrieved is not None
            assert retrieved.content == "Load test content"
            assert retrieved.metadata["loaded"] is True
            
        finally:
            store2.close()
    
    def test_load_from_nonexistent_file(self):
        """Test loading when file doesn't exist"""
        nonexistent_path = str(self.file_path.with_name("nonexistent.pkl"))
        store = PickleVectorStore(file_path=nonexistent_path, auto_save=False)
        
        try:
            # Should succeed (no error) but store should be empty
            assert len(store._vectors) == 0
            
            # Explicit load should also succeed
            success = store.load_from_disk()
            assert success is True
            
        finally:
            store.close()
    
    def test_atomic_save_operation(self):
        """Test that save operation is atomic (uses temporary file)"""
        store = PickleVectorStore(file_path=str(self.file_path), auto_save=False)
        
        try:
            # Add data
            entry = VectorEntry(
                document_id="atomic_test",
                embedding=np.array([1.0]),
                content="Atomic test",
                metadata={}
            )
            store.add_vector(entry)
            
            # Mock the rename operation to verify atomic behavior
            with patch('pathlib.Path.replace') as mock_replace:
                store.save_to_disk()
                
                # Verify that replace (atomic rename) was called
                mock_replace.assert_called_once()
                
        finally:
            store.close()
    
    def test_operations_since_save_counter(self):
        """Test that operations counter is properly maintained"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "counter_test.pkl"
            store = PickleVectorStore(
                file_path=str(file_path),
                auto_save=False,
                save_interval=3
            )
            
            try:
                assert store._operations_since_save == 0
                
                # Add vector
                entry = VectorEntry(
                    document_id="counter_test",
                    embedding=np.array([1.0]),
                    content="Counter test",
                    metadata={}
                )
                store.add_vector(entry)
                assert store._operations_since_save == 1
                
                # Update vector
                store.update_vector(entry)
                assert store._operations_since_save == 2
                
                # Save should reset counter
                store.save_to_disk()
                assert store._operations_since_save == 0
                
            finally:
                store.close()


class TestPickleVectorStoreAutoSave:
    """Test auto-save functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.file_path = Path(self.temp_dir) / "autosave_test.pkl"
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_auto_save_disabled(self):
        """Test that auto-save doesn't trigger when disabled"""
        store = PickleVectorStore(
            file_path=str(self.file_path),
            auto_save=False,
            save_interval=1
        )
        
        try:
            # Add vector - should not auto-save
            entry = VectorEntry(
                document_id="no_autosave",
                embedding=np.array([1.0]),
                content="No auto-save",
                metadata={}
            )
            store.add_vector(entry)
            
            # File should not exist yet
            assert not self.file_path.exists()
            
        finally:
            store.close()
    
    def test_auto_save_triggered_by_interval(self):
        """Test that auto-save triggers at specified interval"""
        store = PickleVectorStore(
            file_path=str(self.file_path),
            auto_save=True,
            save_interval=2
        )
        
        try:
            # First operation - should not trigger save
            entry1 = VectorEntry(
                document_id="autosave_1",
                embedding=np.array([1.0]),
                content="Auto-save test 1",
                metadata={}
            )
            store.add_vector(entry1)
            assert store._operations_since_save == 1
            # File might not exist yet (depending on implementation)
            
            # Second operation - should trigger save
            entry2 = VectorEntry(
                document_id="autosave_2",
                embedding=np.array([2.0]),
                content="Auto-save test 2",
                metadata={}
            )
            store.add_vector(entry2)
            
            # File should now exist and counter should be reset
            assert self.file_path.exists()
            assert store._operations_since_save == 0
            
        finally:
            store.close()
    
    def test_auto_save_with_update_operations(self):
        """Test auto-save with update operations"""
        store = PickleVectorStore(
            file_path=str(self.file_path),
            auto_save=True,
            save_interval=2
        )
        
        try:
            # Add vector
            entry = VectorEntry(
                document_id="update_test",
                embedding=np.array([1.0, 2.0]),
                content="Original content",
                metadata={"version": 1}
            )
            store.add_vector(entry)
            assert store._operations_since_save == 1
            
            # Update vector - should trigger auto-save
            updated_entry = VectorEntry(
                document_id="update_test",
                embedding=np.array([3.0, 4.0]),
                content="Updated content",
                metadata={"version": 2}
            )
            store.update_vector(updated_entry)
            
            # Should have auto-saved
            assert self.file_path.exists()
            assert store._operations_since_save == 0
            
        finally:
            store.close()
    
    def test_auto_save_with_delete_operations(self):
        """Test auto-save with delete operations"""
        store = PickleVectorStore(
            file_path=str(self.file_path),
            auto_save=True,
            save_interval=2
        )
        
        try:
            # Add vector
            entry = VectorEntry(
                document_id="delete_test",
                embedding=np.array([1.0]),
                content="To be deleted",
                metadata={}
            )
            store.add_vector(entry)
            assert store._operations_since_save == 1
            
            # Delete vector - should trigger auto-save
            store.delete_vector("delete_test")
            
            # Should have auto-saved
            assert self.file_path.exists()
            assert store._operations_since_save == 0
            
        finally:
            store.close()
    
    def test_auto_save_with_failed_operations(self):
        """Test that failed operations don't trigger auto-save"""
        store = PickleVectorStore(
            file_path=str(self.file_path),
            auto_save=True,
            save_interval=1
        )
        
        try:
            # Try to update non-existent vector (should fail)
            entry = VectorEntry(
                document_id="nonexistent",
                embedding=np.array([1.0]),
                content="Non-existent",
                metadata={}
            )
            result = store.update_vector(entry)
            assert result is False
            
            # Counter should not increment for failed operations
            assert store._operations_since_save == 0
            
            # Try to delete non-existent vector (should fail)
            result = store.delete_vector("nonexistent")
            assert result is False
            assert store._operations_since_save == 0
            
        finally:
            store.close()


class TestPickleVectorStoreBackupRestore:
    """Test backup and restore functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.file_path = Path(self.temp_dir) / "main_store.pkl"
        self.backup_path = Path(self.temp_dir) / "backup_store.pkl"
        
        self.store = PickleVectorStore(file_path=str(self.file_path), auto_save=False)
        
        # Add test data
        self.test_entries = [
            VectorEntry(
                document_id=f"backup_test_{i}",
                embedding=np.array([i, i+1, i+2]),
                content=f"Backup test content {i}",
                metadata={"index": i, "group": "backup_test"}
            )
            for i in range(3)
        ]
        
        for entry in self.test_entries:
            self.store.add_vector(entry)
        
        self.store.save_to_disk()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_backup_to_file_success(self):
        """Test successful backup creation"""
        success = self.store.backup_to_file(str(self.backup_path))
        
        assert success is True
        assert self.backup_path.exists()
        
        # Verify backup contains same data as original
        with open(self.backup_path, 'rb') as f:
            backup_data = pickle.load(f)
        
        assert len(backup_data['vectors']) == 3
        assert 'backup_test_0' in backup_data['vectors']
        assert 'backup_test_1' in backup_data['vectors']
        assert 'backup_test_2' in backup_data['vectors']
    
    def test_backup_to_file_creates_directories(self):
        """Test that backup creates necessary directories"""
        nested_backup = Path(self.temp_dir) / "nested" / "backup" / "store.pkl"
        
        success = self.store.backup_to_file(str(nested_backup))
        
        assert success is True
        assert nested_backup.exists()
        assert nested_backup.parent.exists()
    
    def test_backup_to_file_failure(self):
        """Test backup failure handling"""
        # Use invalid path
        invalid_backup = "/invalid/path/backup.pkl"
        
        success = self.store.backup_to_file(invalid_backup)
        
        assert success is False
    
    def test_restore_from_backup_success(self):
        """Test successful restore from backup"""
        # Create backup
        self.store.backup_to_file(str(self.backup_path))
        
        # Modify original store
        new_entry = VectorEntry(
            document_id="new_entry",
            embedding=np.array([99.0]),
            content="New entry",
            metadata={}
        )
        self.store.add_vector(new_entry)
        self.store.save_to_disk()
        assert len(self.store._vectors) == 4
        
        # Restore from backup
        success = self.store.restore_from_backup(str(self.backup_path))
        
        assert success is True
        assert len(self.store._vectors) == 3  # Should be back to original
        assert "new_entry" not in self.store._vectors
        assert "backup_test_0" in self.store._vectors
    
    def test_restore_from_nonexistent_backup(self):
        """Test restore from non-existent backup file"""
        nonexistent_backup = str(self.backup_path.with_name("nonexistent.pkl"))
        
        success = self.store.restore_from_backup(nonexistent_backup)
        
        assert success is False
        # Original data should remain intact
        assert len(self.store._vectors) == 3
    
    def test_restore_from_backup_failure(self):
        """Test restore failure handling"""
        # Create invalid backup file
        with open(self.backup_path, 'w') as f:
            f.write("Invalid backup content")
        
        success = self.store.restore_from_backup(str(self.backup_path))
        
        assert success is False


class TestPickleVectorStoreOptimization:
    """Test storage optimization and maintenance"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.file_path = Path(self.temp_dir) / "optimization_test.pkl"
        self.store = PickleVectorStore(file_path=str(self.file_path), auto_save=False)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_optimize_storage_success(self):
        """Test successful storage optimization"""
        # Add some data
        entry = VectorEntry(
            document_id="optimize_test",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Optimization test",
            metadata={"optimized": False}
        )
        self.store.add_vector(entry)
        
        # Optimize storage
        success = self.store.optimize_storage()
        
        assert success is True
        assert self.file_path.exists()
        
        # Verify data is still intact
        retrieved = self.store.get_vector("optimize_test")
        assert retrieved is not None
        assert retrieved.content == "Optimization test"
    
    def test_optimize_storage_failure(self):
        """Test optimize storage failure handling"""
        # Mock save_to_disk to fail
        with patch.object(self.store, 'save_to_disk', return_value=False):
            success = self.store.optimize_storage()
            assert success is False


class TestPickleVectorStoreFileInfo:
    """Test file information and status methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.file_path = Path(self.temp_dir) / "file_info_test.pkl"
        self.store = PickleVectorStore(
            file_path=str(self.file_path),
            auto_save=True,
            save_interval=5
        )
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_file_info_existing_file(self):
        """Test getting file info for existing file"""
        # Add data and save
        entry = VectorEntry(
            document_id="file_info_test",
            embedding=np.array([1.0, 2.0]),
            content="File info test",
            metadata={}
        )
        self.store.add_vector(entry)
        self.store.save_to_disk()
        
        file_info = self.store.get_file_info()
        
        assert file_info["exists"] is True
        assert file_info["path"] == str(self.file_path)
        assert file_info["size_bytes"] > 0
        assert "modified_time" in file_info
        assert file_info["operations_since_save"] == 0
        assert file_info["auto_save"] is True
        assert file_info["save_interval"] == 5
    
    def test_get_file_info_nonexistent_file(self):
        """Test getting file info for non-existent file"""
        # Create store without saving
        nonexistent_path = str(self.file_path.with_name("nonexistent.pkl"))
        temp_store = PickleVectorStore(file_path=nonexistent_path, auto_save=False)
        
        try:
            file_info = temp_store.get_file_info()
            
            assert file_info["exists"] is False
            assert file_info["path"] == nonexistent_path
            
        finally:
            temp_store.close()
    
    def test_get_file_info_with_operations_count(self):
        """Test file info shows correct operations count"""
        # Perform some operations without saving
        entry = VectorEntry(
            document_id="ops_count_test",
            embedding=np.array([1.0]),
            content="Operations count test",
            metadata={}
        )
        
        # Temporarily disable auto-save
        self.store.auto_save = False
        
        self.store.add_vector(entry)
        self.store.update_vector(entry)
        
        file_info = self.store.get_file_info()
        
        assert file_info["operations_since_save"] == 2
    
    def test_get_file_info_error_handling(self):
        """Test file info error handling"""
        # Mock file stat to raise exception
        with patch.object(Path, 'stat', side_effect=OSError("Stat failed")):
            file_info = self.store.get_file_info()
            
            assert "error" in file_info
            assert "Stat failed" in file_info["error"]


class TestPickleVectorStoreStats:
    """Test statistics functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.file_path = Path(self.temp_dir) / "stats_test.pkl"
        self.store = PickleVectorStore(file_path=str(self.file_path), auto_save=False)
        
        # Add test data
        self.test_entries = [
            VectorEntry(
                document_id=f"stats_test_{i}",
                embedding=np.array([i, i+1, i+2, i+3]),  # 4-dimensional vectors
                content=f"Stats test content {i}",
                metadata={"index": i}
            )
            for i in range(5)
        ]
        
        for entry in self.test_entries:
            self.store.add_vector(entry)
        
        self.store.save_to_disk()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_stats_with_file_size(self):
        """Test that statistics include file size information"""
        stats = self.store.get_stats()
        
        assert isinstance(stats, VectorStoreStats)
        assert stats.total_vectors == 5
        assert stats.vector_dimension == 4
        assert stats.index_type == "exact_pickle"
        assert stats.storage_size_bytes > 0  # Should have file size
        assert stats.similarity_metric == "cosine"
    
    def test_get_stats_no_file(self):
        """Test statistics when file doesn't exist"""
        # Create new store without saving
        temp_path = str(self.file_path.with_name("no_file.pkl"))
        temp_store = PickleVectorStore(file_path=temp_path, auto_save=False)
        
        try:
            stats = temp_store.get_stats()
            
            assert stats.total_vectors == 0
            assert stats.storage_size_bytes == 0  # No file, so no size
            assert stats.index_type == "exact_pickle"
            
        finally:
            temp_store.close()


class TestPickleVectorStoreCloseAndCleanup:
    """Test close and cleanup functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.file_path = Path(self.temp_dir) / "cleanup_test.pkl"
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_close_with_auto_save_pending(self):
        """Test close with pending auto-save operations"""
        store = PickleVectorStore(
            file_path=str(self.file_path),
            auto_save=True,
            save_interval=10  # High interval to prevent auto-save
        )
        
        # Add data without triggering auto-save
        entry = VectorEntry(
            document_id="close_test",
            embedding=np.array([1.0, 2.0]),
            content="Close test",
            metadata={}
        )
        store.add_vector(entry)
        
        assert store._operations_since_save > 0
        assert not self.file_path.exists()  # Should not exist yet
        
        # Close should trigger save
        store.close()
        
        # File should now exist
        assert self.file_path.exists()
    
    def test_close_without_auto_save(self):
        """Test close without auto-save enabled"""
        store = PickleVectorStore(
            file_path=str(self.file_path),
            auto_save=False
        )
        
        # Add data
        entry = VectorEntry(
            document_id="no_auto_save_test",
            embedding=np.array([1.0]),
            content="No auto-save test",
            metadata={}
        )
        store.add_vector(entry)
        
        # Close should not save
        store.close()
        
        # File should not exist
        assert not self.file_path.exists()
    
    def test_destructor_cleanup(self):
        """Test that destructor properly cleans up"""
        store = PickleVectorStore(
            file_path=str(self.file_path),
            auto_save=True,
            save_interval=1
        )
        
        # Add data
        entry = VectorEntry(
            document_id="destructor_test",
            embedding=np.array([1.0]),
            content="Destructor test",
            metadata={}
        )
        store.add_vector(entry)
        
        # Force destructor call
        del store
        
        # Should not raise any exceptions and file should exist
        assert self.file_path.exists()
    
    def test_multiple_close_calls(self):
        """Test that multiple close calls don't cause issues"""
        store = PickleVectorStore(file_path=str(self.file_path), auto_save=False)
        
        # Multiple close calls should be safe
        store.close()
        store.close()
        store.close()
        
        # Should not raise any exceptions


class TestPickleVectorStoreErrorHandling:
    """Test error handling and edge cases"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.file_path = Path(self.temp_dir) / "error_test.pkl"
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_to_disk_permission_error(self):
        """Test save failure due to permission issues"""
        store = PickleVectorStore(file_path=str(self.file_path), auto_save=False)
        
        try:
            # Add data
            entry = VectorEntry(
                document_id="permission_test",
                embedding=np.array([1.0]),
                content="Permission test",
                metadata={}
            )
            store.add_vector(entry)
            
            # Mock file operations to raise permission error
            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                success = store.save_to_disk()
                assert success is False
                
        finally:
            store.close()
    
    def test_load_from_disk_with_pickle_error(self):
        """Test load failure due to pickle corruption"""
        # Create corrupted pickle file
        with open(self.file_path, 'wb') as f:
            f.write(b"corrupted pickle data")
        
        # Should handle gracefully
        store = PickleVectorStore(file_path=str(self.file_path), auto_save=False)
        
        try:
            # Should start with empty store
            assert len(store._vectors) == 0
            
        finally:
            store.close()
    
    def test_operations_with_disk_full(self):
        """Test operations when disk is full"""
        store = PickleVectorStore(
            file_path=str(self.file_path),
            auto_save=True,
            save_interval=1
        )
        
        try:
            # Mock save_to_disk to simulate disk full
            with patch.object(store, 'save_to_disk', return_value=False):
                entry = VectorEntry(
                    document_id="disk_full_test",
                    embedding=np.array([1.0]),
                    content="Disk full test",
                    metadata={}
                )
                
                # Add should succeed even if save fails
                doc_id = store.add_vector(entry)
                assert doc_id == "disk_full_test"
                
                # Data should still be in memory
                retrieved = store.get_vector("disk_full_test")
                assert retrieved is not None
                
        finally:
            store.close()
    
    def test_concurrent_access_protection(self):
        """Test behavior with potential concurrent access"""
        store = PickleVectorStore(file_path=str(self.file_path), auto_save=False)
        
        try:
            # Add data
            entry = VectorEntry(
                document_id="concurrent_test",
                embedding=np.array([1.0, 2.0]),
                content="Concurrent test",
                metadata={}
            )
            store.add_vector(entry)
            store.save_to_disk()
            
            # Simulate file being modified by another process
            # (This is a basic test - real concurrent access would be more complex)
            with open(self.file_path, 'ab') as f:
                f.write(b"extra data from another process")
            
            # Try to load - should handle gracefully
            success = store.load_from_disk()
            # Might succeed or fail depending on corruption level
            # Main thing is it shouldn't crash the application
            
        finally:
            store.close()


class TestPickleVectorStoreInheritance:
    """Test inheritance from InMemoryVectorStore"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.file_path = Path(self.temp_dir) / "inheritance_test.pkl"
        self.store = PickleVectorStore(file_path=str(self.file_path), auto_save=False)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_inherited_search_functionality(self):
        """Test that search functionality is inherited correctly"""
        # Add test vectors
        entries = [
            VectorEntry(
                document_id=f"search_test_{i}",
                embedding=np.array([i, i*2, i*3]),
                content=f"Search test content {i}",
                metadata={"category": "test", "value": i}
            )
            for i in range(3)
        ]
        
        for entry in entries:
            self.store.add_vector(entry)
        
        # Test search functionality (inherited from InMemoryVectorStore)
        query_vector = np.array([1.0, 2.0, 3.0])
        results = self.store.search_similar(query_vector, limit=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, VectorSearchResult) for r in results)
        
        # Test with metadata filters
        filtered_results = self.store.search_by_metadata({"category": "test"})
        assert len(filtered_results) == 3
    
    def test_inherited_vector_dimension_calculation(self):
        """Test that vector dimension calculation is inherited"""
        # Add vector
        entry = VectorEntry(
            document_id="dimension_test",
            embedding=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            content="Dimension test",
            metadata={}
        )
        self.store.add_vector(entry)
        
        # Test inherited dimension calculation
        dimension = self.store.get_vector_dimension()
        assert dimension == 5
    
    def test_persistence_preserves_inheritance_behavior(self):
        """Test that saved/loaded data maintains inherited behavior"""
        # Add vector and save
        entry = VectorEntry(
            document_id="persistence_inheritance_test",
            embedding=np.array([1.0, 2.0, 3.0]),
            content="Persistence inheritance test",
            metadata={"type": "inheritance_test"}
        )
        self.store.add_vector(entry)
        self.store.save_to_disk()
        
        # Create new store and load
        new_store = PickleVectorStore(file_path=str(self.file_path), auto_save=False)
        
        try:
            # Test that inherited methods work correctly
            retrieved = new_store.get_vector("persistence_inheritance_test")
            assert retrieved is not None
            
            # Test search functionality
            query_vector = np.array([1.0, 2.0, 3.0])
            results = new_store.search_similar(query_vector, limit=1)
            assert len(results) == 1
            assert results[0].document_id == "persistence_inheritance_test"
            
            # Test metadata search
            metadata_results = new_store.search_by_metadata({"type": "inheritance_test"})
            assert len(metadata_results) == 1
            
        finally:
            new_store.close()