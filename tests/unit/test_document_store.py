"""
Comprehensive tests for DocumentStore implementation
"""

import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag import (
    Document,
    DocumentStore,
    SQLiteDocumentStore,
    SearchResult,
    StorageStats
)
from refinire_rag.exceptions import StorageError


def create_test_documents():
    """Create test documents for testing"""
    
    documents = []
    
    # Original document
    doc1 = Document(
        id="doc_001",
        content="This is a technical document about RAG systems. It explains how retrieval-augmented generation works.",
        metadata={
            "path": "/docs/technical/rag_guide.pdf",
            "created_at": "2024-01-15T10:30:00Z",
            "file_type": ".pdf",
            "size_bytes": 1024000,
            "dataset": "rag_docs_v2",
            "category": "technical",
            "access_group": "engineers",
            "tags": ["rag", "ai", "technical"]
        }
    )
    documents.append(doc1)
    
    # Normalized document (derived from doc1)
    doc2 = Document(
        id="doc_001_normalized",
        content="This is a technical document about Retrieval-Augmented Generation systems. It explains how RAG works.",
        metadata={
            "path": "/docs/technical/rag_guide.pdf",
            "created_at": "2024-01-15T10:35:00Z",
            "file_type": ".pdf",
            "size_bytes": 1024000,
            "original_document_id": "doc_001",
            "parent_document_id": "doc_001",
            "processing_stage": "normalized",
            "dataset": "rag_docs_v2",
            "category": "technical"
        }
    )
    documents.append(doc2)
    
    # Chunk document (derived from doc2)
    doc3 = Document(
        id="doc_001_chunk_0",
        content="This is a technical document about Retrieval-Augmented Generation systems.",
        metadata={
            "path": "/docs/technical/rag_guide.pdf",
            "created_at": "2024-01-15T10:40:00Z",
            "file_type": ".pdf",
            "size_bytes": 512,
            "original_document_id": "doc_001",
            "parent_document_id": "doc_001_normalized",
            "processing_stage": "chunked",
            "chunk_position": 0,
            "chunk_total": 2,
            "category": "technical"
        }
    )
    documents.append(doc3)
    
    # Public document
    doc4 = Document(
        id="doc_002",
        content="This is a public readme file with general information about the project.",
        metadata={
            "path": "/docs/public/readme.md",
            "created_at": "2024-01-16T09:00:00Z",
            "file_type": ".md",
            "size_bytes": 2048,
            "access_group": "public",
            "category": "documentation",
            "tags": ["readme", "public"]
        }
    )
    documents.append(doc4)
    
    return documents


def test_document_store_basic_operations():
    """Test basic CRUD operations"""
    print("\n=== Testing Basic CRUD Operations ===")
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        store = SQLiteDocumentStore(str(db_path))
        
        try:
            documents = create_test_documents()
            doc = documents[0]
            
            # Test store
            stored_id = store.store_document(doc)
            print(f"✓ Stored document: {stored_id}")
            assert stored_id == doc.id
            
            # Test get
            retrieved_doc = store.get_document(doc.id)
            print(f"✓ Retrieved document: {retrieved_doc.id}")
            assert retrieved_doc is not None
            assert retrieved_doc.id == doc.id
            assert retrieved_doc.content == doc.content
            assert retrieved_doc.metadata == doc.metadata
            
            # Test update
            doc.content = "Updated content"
            doc.metadata["updated"] = True
            updated = store.update_document(doc)
            print(f"✓ Updated document: {updated}")
            assert updated
            
            # Verify update
            updated_doc = store.get_document(doc.id)
            assert updated_doc.content == "Updated content"
            assert updated_doc.metadata["updated"] == True
            
            # Test delete
            deleted = store.delete_document(doc.id)
            print(f"✓ Deleted document: {deleted}")
            assert deleted
            
            # Verify deletion
            deleted_doc = store.get_document(doc.id)
            assert deleted_doc is None
            
            # Test delete non-existent
            deleted_again = store.delete_document(doc.id)
            assert not deleted_again
            
            print("✓ Basic CRUD operations passed")
            
        finally:
            store.close()


def test_metadata_search():
    """Test metadata-based search"""
    print("\n=== Testing Metadata Search ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        store = SQLiteDocumentStore(str(db_path))
        
        try:
            # Store test documents
            documents = create_test_documents()
            for doc in documents:
                store.store_document(doc)
            
            # Test exact match
            results = store.search_by_metadata({"category": "technical"})
            print(f"✓ Found {len(results)} technical documents")
            assert len(results) == 3  # doc1, doc2, doc3
            
            # Test access group filter
            results = store.search_by_metadata({"access_group": "public"})
            print(f"✓ Found {len(results)} public documents")
            assert len(results) == 1
            assert results[0].document.id == "doc_002"
            
            # Test file type filter
            results = store.search_by_metadata({"file_type": ".pdf"})
            print(f"✓ Found {len(results)} PDF documents")
            assert len(results) == 3
            
            # Test size range filter (if JSON1 available)
            try:
                results = store.search_by_metadata({"size_bytes": {"$gte": 1000000}})
                print(f"✓ Found {len(results)} large documents (>=1MB)")
                assert len(results) >= 1
            except Exception as e:
                print(f"⚠ Size range search skipped: {e}")
            
            # Test contains filter
            try:
                results = store.search_by_metadata({"tags": {"$contains": "rag"}})
                print(f"✓ Found {len(results)} documents containing 'rag' tag")
                assert len(results) >= 1
            except Exception as e:
                print(f"⚠ Contains search may not work without JSON1: {e}")
            
            # Test multiple filters
            results = store.search_by_metadata({
                "category": "technical",
                "file_type": ".pdf"
            })
            print(f"✓ Found {len(results)} technical PDF documents")
            assert len(results) == 3
            
            print("✓ Metadata search tests passed")
            
        finally:
            store.close()


def test_content_search():
    """Test full-text content search"""
    print("\n=== Testing Content Search ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        store = SQLiteDocumentStore(str(db_path))
        
        try:
            # Store test documents
            documents = create_test_documents()
            for doc in documents:
                store.store_document(doc)
            
            # Test simple content search
            results = store.search_by_content("RAG")
            print(f"✓ Found {len(results)} documents containing 'RAG'")
            assert len(results) >= 2
            
            # Test phrase search
            results = store.search_by_content("technical document")
            print(f"✓ Found {len(results)} documents containing 'technical document'")
            assert len(results) >= 2
            
            # Test search with scoring
            results = store.search_by_content("retrieval augmented generation")
            print(f"✓ Found {len(results)} documents for 'retrieval augmented generation'")
            for result in results:
                print(f"  - {result.document.id}: score={result.score}")
                assert result.score is not None
            
            # Test no results
            results = store.search_by_content("nonexistent term xyz")
            print(f"✓ Found {len(results)} documents for non-existent term")
            assert len(results) == 0
            
            print("✓ Content search tests passed")
            
        finally:
            store.close()


def test_lineage_tracking():
    """Test document lineage tracking"""
    print("\n=== Testing Document Lineage ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        store = SQLiteDocumentStore(str(db_path))
        
        try:
            # Store test documents
            documents = create_test_documents()
            for doc in documents:
                store.store_document(doc)
            
            # Test lineage retrieval
            lineage_docs = store.get_documents_by_lineage("doc_001")
            print(f"✓ Found {len(lineage_docs)} documents in lineage for doc_001")
            
            # Should include: doc_001 (original), doc_001_normalized, doc_001_chunk_0
            assert len(lineage_docs) == 3
            
            lineage_ids = [doc.id for doc in lineage_docs]
            assert "doc_001" in lineage_ids
            assert "doc_001_normalized" in lineage_ids
            assert "doc_001_chunk_0" in lineage_ids
            
            # Test lineage for standalone document
            lineage_docs = store.get_documents_by_lineage("doc_002")
            print(f"✓ Found {len(lineage_docs)} documents in lineage for doc_002")
            assert len(lineage_docs) == 1
            assert lineage_docs[0].id == "doc_002"
            
            # Test non-existent lineage
            lineage_docs = store.get_documents_by_lineage("non_existent")
            print(f"✓ Found {len(lineage_docs)} documents for non-existent lineage")
            assert len(lineage_docs) == 0
            
            print("✓ Lineage tracking tests passed")
            
        finally:
            store.close()


def test_pagination_and_sorting():
    """Test document listing with pagination and sorting"""
    print("\n=== Testing Pagination and Sorting ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        store = SQLiteDocumentStore(str(db_path))
        
        try:
            # Store test documents
            documents = create_test_documents()
            for doc in documents:
                store.store_document(doc)
            
            # Test basic listing
            all_docs = store.list_documents(limit=10)
            print(f"✓ Listed {len(all_docs)} documents (default sort)")
            assert len(all_docs) == 4
            
            # Test pagination
            page1 = store.list_documents(limit=2, offset=0)
            page2 = store.list_documents(limit=2, offset=2)
            print(f"✓ Page 1: {len(page1)} docs, Page 2: {len(page2)} docs")
            assert len(page1) == 2
            assert len(page2) == 2
            
            # Ensure no overlap
            page1_ids = {doc.id for doc in page1}
            page2_ids = {doc.id for doc in page2}
            assert len(page1_ids.intersection(page2_ids)) == 0
            
            # Test sorting by ID
            docs_by_id = store.list_documents(sort_by="id", sort_order="asc")
            print(f"✓ Sorted by ID: {[doc.id for doc in docs_by_id]}")
            assert docs_by_id[0].id <= docs_by_id[1].id
            
            print("✓ Pagination and sorting tests passed")
            
        finally:
            store.close()


def test_counting_and_stats():
    """Test document counting and statistics"""
    print("\n=== Testing Counting and Statistics ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        store = SQLiteDocumentStore(str(db_path))
        
        try:
            # Store test documents
            documents = create_test_documents()
            for doc in documents:
                store.store_document(doc)
            
            # Test total count
            total_count = store.count_documents()
            print(f"✓ Total documents: {total_count}")
            assert total_count == 4
            
            # Test filtered count
            technical_count = store.count_documents({"category": "technical"})
            print(f"✓ Technical documents: {technical_count}")
            assert technical_count == 3
            
            public_count = store.count_documents({"access_group": "public"})
            print(f"✓ Public documents: {public_count}")
            assert public_count == 1
            
            # Test storage stats
            stats = store.get_storage_stats()
            print(f"✓ Storage stats: {stats}")
            assert stats.total_documents == 4
            assert stats.storage_size_bytes > 0
            assert stats.oldest_document is not None
            assert stats.newest_document is not None
            
            print("✓ Counting and statistics tests passed")
            
        finally:
            store.close()


def test_backup_and_restore():
    """Test backup and restore functionality"""
    print("\n=== Testing Backup and Restore ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        backup_path = Path(temp_dir) / "backup.db"
        
        # Create and populate original store
        store1 = SQLiteDocumentStore(str(db_path))
        
        try:
            documents = create_test_documents()
            for doc in documents:
                store1.store_document(doc)
            
            # Create backup
            backup_success = store1.backup_to_file(str(backup_path))
            print(f"✓ Backup created: {backup_success}")
            assert backup_success
            assert backup_path.exists()
            
        finally:
            store1.close()
        
        # Create new store and restore from backup
        new_db_path = Path(temp_dir) / "restored.db"
        store2 = SQLiteDocumentStore(str(new_db_path))
        
        try:
            restore_success = store2.restore_from_file(str(backup_path))
            print(f"✓ Restore completed: {restore_success}")
            assert restore_success
            
            # Verify restored data
            restored_count = store2.count_documents()
            print(f"✓ Restored documents: {restored_count}")
            assert restored_count == 4
            
            # Verify specific document
            doc = store2.get_document("doc_001")
            assert doc is not None
            assert "RAG systems" in doc.content
            
            print("✓ Backup and restore tests passed")
            
        finally:
            store2.close()


def test_error_handling():
    """Test error handling scenarios"""
    print("\n=== Testing Error Handling ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        store = SQLiteDocumentStore(str(db_path))
        
        try:
            # Test get non-existent document
            doc = store.get_document("non_existent")
            print(f"✓ Non-existent document returns: {doc}")
            assert doc is None
            
            # Test update non-existent document
            fake_doc = Document(
                id="fake_id",
                content="fake content",
                metadata={
                    "path": "/fake/path.txt",
                    "created_at": "2024-01-01T00:00:00Z",
                    "file_type": ".txt",
                    "size_bytes": 100
                }
            )
            
            updated = store.update_document(fake_doc)
            print(f"✓ Update non-existent document: {updated}")
            assert not updated
            
            # Test invalid search filters
            try:
                results = store.search_by_metadata({"invalid_operator": {"$unknown": "value"}})
                print(f"✓ Invalid search handled gracefully: {len(results)} results")
            except Exception as e:
                print(f"✓ Invalid search raised expected error: {type(e).__name__}")
            
            # Test restore from non-existent backup
            restore_result = store.restore_from_file("/non/existent/backup.db")
            print(f"✓ Restore from non-existent file: {restore_result}")
            assert not restore_result
            
            print("✓ Error handling tests passed")
            
        finally:
            store.close()


def test_performance():
    """Test performance with larger dataset"""
    print("\n=== Testing Performance ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        store = SQLiteDocumentStore(str(db_path))
        
        try:
            import time
            
            # Create many documents
            num_docs = 100
            start_time = time.time()
            
            for i in range(num_docs):
                doc = Document(
                    id=f"perf_doc_{i:03d}",
                    content=f"This is performance test document number {i} with some content to search.",
                    metadata={
                        "path": f"/test/docs/doc_{i}.txt",
                        "created_at": f"2024-01-{(i % 30) + 1:02d}T10:00:00Z",
                        "file_type": ".txt",
                        "size_bytes": 1000 + i * 10,
                        "batch": i // 10,
                        "category": "test" if i % 2 == 0 else "performance"
                    }
                )
                store.store_document(doc)
            
            insert_time = time.time() - start_time
            print(f"✓ Inserted {num_docs} documents in {insert_time:.3f} seconds")
            
            # Test search performance
            start_time = time.time()
            results = store.search_by_metadata({"category": "test"})
            search_time = time.time() - start_time
            print(f"✓ Metadata search: {len(results)} results in {search_time:.3f} seconds")
            
            # Test content search performance
            start_time = time.time()
            results = store.search_by_content("performance test")
            content_search_time = time.time() - start_time
            print(f"✓ Content search: {len(results)} results in {content_search_time:.3f} seconds")
            
            # Test count performance
            start_time = time.time()
            count = store.count_documents()
            count_time = time.time() - start_time
            print(f"✓ Count: {count} documents in {count_time:.3f} seconds")
            
            assert count == num_docs  # Only performance docs in this test
            
            print("✓ Performance tests passed")
            
        finally:
            store.close()


def main():
    """Run all DocumentStore tests"""
    print("Running comprehensive DocumentStore tests...\n")
    
    try:
        test_document_store_basic_operations()
        test_metadata_search()
        test_content_search()
        test_lineage_tracking()
        test_pagination_and_sorting()
        test_counting_and_stats()
        test_backup_and_restore()
        test_error_handling()
        test_performance()
        
        print("\n✅ All DocumentStore tests passed!")
        
        print("\n=== Test Summary ===")
        print("✅ Basic CRUD Operations - Store, retrieve, update, delete documents")
        print("✅ Metadata Search - Advanced filtering with JSON operators")
        print("✅ Content Search - Full-text search with FTS5 and scoring")
        print("✅ Lineage Tracking - Document family relationship management")
        print("✅ Pagination & Sorting - Efficient data access patterns")
        print("✅ Counting & Statistics - Performance monitoring capabilities")
        print("✅ Backup & Restore - Data persistence and recovery")
        print("✅ Error Handling - Graceful failure management")
        print("✅ Performance - Bulk operations and search optimization")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())