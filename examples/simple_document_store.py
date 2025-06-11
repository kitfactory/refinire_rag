"""
Simple DocumentStore demonstration
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag import (
    Document,
    SQLiteDocumentStore
)


def main():
    """Simple DocumentStore demonstration"""
    print("Simple DocumentStore Demonstration")
    print("=" * 40)
    
    # Use a simple local database file
    db_path = "./test_documents.db"
    doc_store = SQLiteDocumentStore(db_path)
    
    try:
        # Create a sample document
        doc = Document(
            id="simple_test_001",
            content="This is a simple test document for demonstrating DocumentStore functionality.",
            metadata={
                "path": "/test/simple.txt",
                "created_at": "2024-01-15T10:00:00Z",
                "file_type": ".txt",
                "size_bytes": 500,
                "category": "test",
                "importance": "low"
            }
        )
        
        print(f"1. Storing document: {doc.id}")
        stored_id = doc_store.store_document(doc)
        print(f"   ✓ Stored: {stored_id}")
        
        print(f"\n2. Retrieving document: {doc.id}")
        retrieved_doc = doc_store.get_document(doc.id)
        if retrieved_doc:
            print(f"   ✓ Retrieved: {retrieved_doc.id}")
            print(f"   Content preview: {retrieved_doc.content[:50]}...")
        else:
            print("   ❌ Document not found")
        
        print(f"\n3. Searching by metadata:")
        results = doc_store.search_by_metadata({"category": "test"})
        print(f"   ✓ Found {len(results)} test documents")
        
        print(f"\n4. Content search:")
        results = doc_store.search_by_content("simple test")
        print(f"   ✓ Found {len(results)} documents containing 'simple test'")
        
        print(f"\n5. Storage statistics:")
        stats = doc_store.get_storage_stats()
        print(f"   Total documents: {stats.total_documents}")
        print(f"   Storage size: {stats.storage_size_bytes} bytes")
        
        print(f"\n6. Cleanup - deleting test document")
        deleted = doc_store.delete_document(doc.id)
        print(f"   ✓ Deleted: {deleted}")
        
        print(f"\n✅ Simple DocumentStore demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        doc_store.close()
        # Clean up database file
        try:
            Path(db_path).unlink()
            print(f"Cleaned up database file: {db_path}")
        except:
            pass
    
    return 0


if __name__ == "__main__":
    exit(main())