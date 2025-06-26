"""
Comprehensive tests for SQLiteDocumentStore
SQLiteDocumentStoreの包括的テスト
"""

import pytest
import tempfile
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.models.document import Document
from refinire_rag.exceptions import StorageError


class TestSQLiteDocumentStoreInitialization:
    """Test SQLiteDocumentStore initialization"""
    
    def test_init_default_path(self):
        """Test initialization with default path"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            store = SQLiteDocumentStore(str(db_path))
            
            assert store.db_path == db_path
            assert store.conn is not None
            assert store.db_path.exists()
            store.close()
    
    def test_init_creates_parent_directory(self):
        """Test that parent directories are created"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "subdir" / "test.db"
            store = SQLiteDocumentStore(str(db_path))
            
            assert db_path.parent.exists()
            assert db_path.exists()
            store.close()
    
    def test_init_in_memory_database(self):
        """Test initialization with in-memory database"""
        store = SQLiteDocumentStore(":memory:")
        
        assert store.conn is not None
        assert store.json_enabled is not None
        store.close()
    
    def test_json_extension_detection(self):
        """Test JSON1 extension detection"""
        store = SQLiteDocumentStore(":memory:")
        
        # JSON extension should be available in most modern SQLite
        assert isinstance(store.json_enabled, bool)
        store.close()
    
    def test_schema_initialization(self):
        """Test that database schema is properly initialized"""
        store = SQLiteDocumentStore(":memory:")
        
        # Check if main table exists
        cursor = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        )
        assert cursor.fetchone() is not None
        
        # Check if FTS table exists (should be created lazily)
        cursor = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'"
        )
        fts_exists = cursor.fetchone() is not None
        
        store.close()


class TestSQLiteDocumentStoreBasicOperations:
    """Test basic CRUD operations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.store = SQLiteDocumentStore(":memory:")
        
        # Sample documents
        self.doc1 = Document(
            id="doc1",
            content="This is test document 1",
            metadata={"type": "test", "category": "A"}
        )
        self.doc2 = Document(
            id="doc2", 
            content="This is test document 2",
            metadata={"type": "test", "category": "B"}
        )
    
    def teardown_method(self):
        """Clean up"""
        self.store.close()
    
    def test_store_document(self):
        """Test storing a document"""
        doc_id = self.store.store_document(self.doc1)
        
        assert doc_id == "doc1"
        
        # Verify document was stored
        cursor = self.store.conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row["id"] == "doc1"
        assert row["content"] == "This is test document 1"
    
    def test_store_document_with_json_metadata(self):
        """Test storing document with complex metadata"""
        complex_doc = Document(
            id="complex",
            content="Complex document",
            metadata={
                "type": "complex",
                "tags": ["tag1", "tag2"],
                "nested": {"key": "value"},
                "number": 42
            }
        )
        
        doc_id = self.store.store_document(complex_doc)
        assert doc_id == "complex"
        
        # Verify metadata is properly stored as JSON
        cursor = self.store.conn.execute("SELECT metadata FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        stored_metadata = json.loads(row["metadata"])
        
        assert stored_metadata["type"] == "complex"
        assert stored_metadata["tags"] == ["tag1", "tag2"]
        assert stored_metadata["nested"]["key"] == "value"
        assert stored_metadata["number"] == 42
    
    def test_get_document(self):
        """Test retrieving a document"""
        self.store.store_document(self.doc1)
        
        retrieved_doc = self.store.get_document("doc1")
        
        assert retrieved_doc is not None
        assert retrieved_doc.id == "doc1"
        assert retrieved_doc.content == "This is test document 1"
        assert retrieved_doc.metadata["type"] == "test"
    
    def test_get_nonexistent_document(self):
        """Test retrieving a document that doesn't exist"""
        retrieved_doc = self.store.get_document("nonexistent")
        assert retrieved_doc is None
    
    def test_update_document(self):
        """Test updating an existing document"""
        self.store.store_document(self.doc1)
        
        # Update the document
        updated_doc = Document(
            id="doc1",
            content="Updated content",
            metadata={"type": "updated", "new_field": "value"}
        )
        
        doc_id = self.store.store_document(updated_doc)
        assert doc_id == "doc1"
        
        # Verify update
        retrieved_doc = self.store.get_document("doc1")
        assert retrieved_doc.content == "Updated content"
        assert retrieved_doc.metadata["type"] == "updated"
        assert retrieved_doc.metadata["new_field"] == "value"
    
    def test_delete_document(self):
        """Test deleting a document"""
        self.store.store_document(self.doc1)
        
        # Verify document exists
        assert self.store.get_document("doc1") is not None
        
        # Delete document
        success = self.store.delete_document("doc1")
        assert success is True
        
        # Verify document is gone
        assert self.store.get_document("doc1") is None
    
    def test_delete_nonexistent_document(self):
        """Test deleting a document that doesn't exist"""
        success = self.store.delete_document("nonexistent")
        assert success is False
    
    def test_document_exists(self):
        """Test checking if document exists"""
        assert self.store.document_exists("doc1") is False
        
        self.store.store_document(self.doc1)
        assert self.store.document_exists("doc1") is True
        
        self.store.delete_document("doc1")
        assert self.store.document_exists("doc1") is False


class TestSQLiteDocumentStoreListOperations:
    """Test listing and filtering operations"""
    
    def setup_method(self):
        """Set up test fixtures with multiple documents"""
        self.store = SQLiteDocumentStore(":memory:")
        
        # Create test documents with different timestamps
        self.docs = [
            Document(id="doc1", content="Content 1", metadata={"type": "A", "priority": "high"}),
            Document(id="doc2", content="Content 2", metadata={"type": "B", "priority": "low"}),
            Document(id="doc3", content="Content 3", metadata={"type": "A", "priority": "medium"}),
            Document(id="doc4", content="Content 4", metadata={"type": "C", "priority": "high"}),
            Document(id="doc5", content="Content 5", metadata={"type": "B", "priority": "high"})
        ]
        
        for doc in self.docs:
            self.store.store_document(doc)
    
    def teardown_method(self):
        """Clean up"""
        self.store.close()
    
    def test_list_documents_all(self):
        """Test listing all documents"""
        docs = self.store.list_documents()
        
        assert len(docs) == 5
        doc_ids = [doc.id for doc in docs]
        assert "doc1" in doc_ids
        assert "doc5" in doc_ids
    
    def test_list_documents_with_limit(self):
        """Test listing documents with limit"""
        docs = self.store.list_documents(limit=3)
        
        assert len(docs) == 3
    
    def test_list_documents_with_offset(self):
        """Test listing documents with offset"""
        docs = self.store.list_documents(limit=2, offset=2)
        
        assert len(docs) == 2
    
    def test_list_documents_with_sorting(self):
        """Test listing documents with sorting"""
        docs = self.store.list_documents(sort_by="id", sort_order="asc")
        
        doc_ids = [doc.id for doc in docs]
        assert doc_ids == ["doc1", "doc2", "doc3", "doc4", "doc5"]
        
        docs = self.store.list_documents(sort_by="id", sort_order="desc")
        doc_ids = [doc.id for doc in docs]
        assert doc_ids == ["doc5", "doc4", "doc3", "doc2", "doc1"]
    
    def test_count_documents(self):
        """Test counting documents"""
        count = self.store.count_documents()
        assert count == 5
    
    def test_clear_all_documents(self):
        """Test clearing all documents"""
        assert self.store.count_documents() == 5
        
        self.store.clear_all()
        
        assert self.store.count_documents() == 0
        assert len(self.store.list_documents()) == 0


class TestSQLiteDocumentStoreSearchOperations:
    """Test search functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.store = SQLiteDocumentStore(":memory:")
        
        # Create documents for searching
        self.search_docs = [
            Document(id="search1", content="Python programming tutorial", 
                    metadata={"language": "python", "difficulty": "beginner"}),
            Document(id="search2", content="Advanced Python techniques", 
                    metadata={"language": "python", "difficulty": "advanced"}),
            Document(id="search3", content="JavaScript fundamentals", 
                    metadata={"language": "javascript", "difficulty": "beginner"}),
            Document(id="search4", content="Machine learning with Python", 
                    metadata={"language": "python", "topic": "ml"})
        ]
        
        for doc in self.search_docs:
            self.store.store_document(doc)
    
    def teardown_method(self):
        """Clean up"""
        self.store.close()
    
    def test_search_by_metadata_single_filter(self):
        """Test searching by single metadata filter"""
        # Search for Python documents
        results = self.store.search_by_metadata({"language": "python"})
        
        assert len(results) == 3
        doc_ids = [result.document.id for result in results]
        assert "search1" in doc_ids
        assert "search2" in doc_ids
        assert "search4" in doc_ids
    
    def test_search_by_metadata_multiple_filters(self):
        """Test searching by multiple metadata filters"""
        # Search for beginner Python documents
        results = self.store.search_by_metadata({
            "language": "python",
            "difficulty": "beginner"
        })
        
        assert len(results) == 1
        assert results[0].document.id == "search1"
    
    def test_search_by_metadata_no_results(self):
        """Test searching with no matching results"""
        results = self.store.search_by_metadata({"language": "nonexistent"})
        assert len(results) == 0
    
    def test_search_by_metadata_with_limit(self):
        """Test searching with limit"""
        results = self.store.search_by_metadata({"language": "python"}, limit=2)
        assert len(results) == 2
    
    def test_search_by_content_basic(self):
        """Test basic content search"""
        # Initialize FTS first
        self.store._init_fts()
        
        # Search for Python content
        results = self.store.search_by_content("Python")
        
        # Should find documents containing "Python"
        assert len(results) >= 2
        
        # Check that results contain search scores
        for result in results:
            assert hasattr(result, 'score')
            assert result.score is not None


class TestSQLiteDocumentStoreErrorHandling:
    """Test error handling scenarios"""
    
    def test_store_document_database_error(self):
        """Test error handling when database operation fails"""
        store = SQLiteDocumentStore(":memory:")
        
        # Close the connection to cause database error
        store.conn.close()
        
        with pytest.raises(StorageError, match="Failed to store document"):
            store.store_document(Document(id="test", content="test", metadata={}))
    
    def test_get_document_database_error(self):
        """Test error handling when retrieving document fails"""
        store = SQLiteDocumentStore(":memory:")
        
        # Close the connection to cause database error
        store.conn.close()
        
        with pytest.raises(StorageError, match="Failed to get document"):
            store.get_document("test")
    
    def test_search_metadata_database_error(self):
        """Test error handling when metadata search fails"""
        store = SQLiteDocumentStore(":memory:")
        
        # Close the connection to cause database error
        store.conn.close()
        
        with pytest.raises(StorageError, match="Failed to search by metadata"):
            store.search_by_metadata({"key": "value"})
    
    def test_search_content_database_error(self):
        """Test error handling when content search fails"""
        store = SQLiteDocumentStore(":memory:")
        store._init_fts()
        
        # Close the connection to cause database error
        store.conn.close()
        
        with pytest.raises(StorageError, match="Failed to search by content"):
            store.search_by_content("test")


class TestSQLiteDocumentStoreFTSOperations:
    """Test Full-Text Search specific operations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.store = SQLiteDocumentStore(":memory:")
        
        # Documents for FTS testing
        self.fts_docs = [
            Document(id="fts1", content="Machine learning algorithms and neural networks", 
                    metadata={"topic": "AI"}),
            Document(id="fts2", content="Deep learning with TensorFlow and PyTorch", 
                    metadata={"topic": "AI"}),
            Document(id="fts3", content="Web development using React and Node.js", 
                    metadata={"topic": "Web"}),
        ]
        
        for doc in self.fts_docs:
            self.store.store_document(doc)
    
    def teardown_method(self):
        """Clean up"""
        self.store.close()
    
    def test_fts_initialization(self):
        """Test FTS table initialization"""
        # FTS should not be initialized yet
        assert self.store.fts_initialized is False
        
        # Initialize FTS
        self.store._init_fts()
        
        # Check if initialized
        assert self.store.fts_initialized is True
        
        # Verify FTS table exists
        cursor = self.store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'"
        )
        assert cursor.fetchone() is not None
    
    def test_fts_search_functionality(self):
        """Test FTS search with actual queries"""
        self.store._init_fts()
        
        # Search for machine learning
        results = self.store.search_by_content("machine learning")
        
        assert len(results) >= 1
        # Should find the document about machine learning
        found_ml_doc = any(result.document.id == "fts1" for result in results)
        assert found_ml_doc
    
    def test_fts_search_ranking(self):
        """Test that FTS search returns ranked results"""
        self.store._init_fts()
        
        results = self.store.search_by_content("learning")
        
        if len(results) > 1:
            # Results should have scores
            for result in results:
                assert hasattr(result, 'score')
                assert result.score is not None


class TestSQLiteDocumentStoreUtilityMethods:
    """Test utility methods"""
    
    def test_get_storage_stats(self):
        """Test getting storage statistics"""
        store = SQLiteDocumentStore(":memory:")
        
        # Add some documents
        for i in range(3):
            doc = Document(id=f"doc{i}", content=f"Content {i}", metadata={"index": i})
            store.store_document(doc)
        
        stats = store.get_storage_stats()
        
        assert stats.total_documents == 3
        assert stats.storage_size_bytes > 0
        assert stats.oldest_document is not None
        assert stats.newest_document is not None
        
        store.close()
    
    def test_close_connection(self):
        """Test closing database connection"""
        store = SQLiteDocumentStore(":memory:")
        
        # Connection should be active
        assert store.conn is not None
        
        # Close connection
        store.close()
        
        # Connection should still be set but operations should fail
        with pytest.raises(Exception):
            store.conn.execute("SELECT 1")
    
    def test_context_manager_usage(self):
        """Test using store as context manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "context_test.db"
            
            # Use as context manager
            with SQLiteDocumentStore(str(db_path)) as store:
                doc = Document(id="ctx_test", content="Context test", metadata={})
                store.store_document(doc)
                
                # Verify document was stored
                retrieved = store.get_document("ctx_test")
                assert retrieved is not None
            
            # Connection should be closed after context


class TestSQLiteDocumentStoreEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_content_document(self):
        """Test storing document with empty content"""
        store = SQLiteDocumentStore(":memory:")
        
        empty_doc = Document(id="empty", content="", metadata={"type": "empty"})
        doc_id = store.store_document(empty_doc)
        
        assert doc_id == "empty"
        retrieved = store.get_document("empty")
        assert retrieved.content == ""
        
        store.close()
    
    def test_large_content_document(self):
        """Test storing document with large content"""
        store = SQLiteDocumentStore(":memory:")
        
        # Create document with large content (1MB)
        large_content = "A" * (1024 * 1024)
        large_doc = Document(id="large", content=large_content, metadata={"size": "1MB"})
        
        doc_id = store.store_document(large_doc)
        assert doc_id == "large"
        
        retrieved = store.get_document("large")
        assert len(retrieved.content) == 1024 * 1024
        
        store.close()
    
    def test_special_characters_in_content(self):
        """Test documents with special characters"""
        store = SQLiteDocumentStore(":memory:")
        
        special_content = "Special chars: 中文 🚀 'quotes' \"double\" \n\t\r"
        special_doc = Document(
            id="special", 
            content=special_content, 
            metadata={"type": "unicode"}
        )
        
        doc_id = store.store_document(special_doc)
        retrieved = store.get_document("special")
        
        assert retrieved.content == special_content
        
        store.close()
    
    def test_null_and_none_handling(self):
        """Test handling of null/None values"""
        store = SQLiteDocumentStore(":memory:")
        
        # Document with None in metadata
        doc_with_none = Document(
            id="none_test", 
            content="Test content", 
            metadata={"value": None, "empty_string": ""}
        )
        
        doc_id = store.store_document(doc_with_none)
        retrieved = store.get_document("none_test")
        
        assert retrieved.metadata["value"] is None
        assert retrieved.metadata["empty_string"] == ""
        
        store.close()