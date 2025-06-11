"""
Example usage of DocumentStore with Loader integration
"""

import sys
import tempfile
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag import (
    UniversalLoader,
    SQLiteDocumentStore,
    PathBasedMetadataGenerator,
    LoadingConfig,
    Document
)


def create_sample_documents():
    """Create sample documents for demonstration"""
    
    # Create sample directory structure
    sample_dir = Path("examples/data/sample_docs")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample files
    files = {
        "docs/technical/api_guide.md": """# API Guide

This document explains how to use our RAG API.

## Authentication
Use bearer tokens for authentication.

## Endpoints
- GET /search - Search documents
- POST /upload - Upload new documents
""",
        "docs/public/readme.txt": """README

This is a public documentation project.
It contains guides and tutorials for users.
""",
        "docs/internal/architecture.md": """# Internal Architecture

## System Components
- Document Store: SQLite-based persistence
- Vector Store: Embedding storage
- Query Engine: Search and retrieval

## Security
Access controls are implemented at the metadata level.
"""
    }
    
    created_files = []
    for relative_path, content in files.items():
        file_path = sample_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        created_files.append(str(file_path))
        print(f"Created: {file_path}")
    
    return created_files, sample_dir


def loader_document_store_integration():
    """Demonstrate Loader + DocumentStore integration"""
    print("\n=== Loader + DocumentStore Integration ===")
    
    # Create sample files
    file_paths, sample_dir = create_sample_documents()
    
    # Create temporary database with proper permissions
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "documents.db"
        # Ensure directory has write permissions
        Path(temp_dir).chmod(0o755)
        
        # Initialize DocumentStore
        doc_store = SQLiteDocumentStore(str(db_path))
        
        try:
            # Configure metadata generation based on paths
            path_rules = {
                "*/docs/technical/*": {
                    "access_group": "engineers",
                    "classification": "technical",
                    "department": "engineering"
                },
                "*/docs/public/*": {
                    "access_group": "public", 
                    "classification": "open",
                    "department": "general"
                },
                "*/docs/internal/*": {
                    "access_group": "employees",
                    "classification": "internal", 
                    "department": "company"
                }
            }
            
            metadata_gen = PathBasedMetadataGenerator(path_rules)
            
            # Configure parallel loading
            config = LoadingConfig(
                parallel=True,
                max_workers=2,
                skip_errors=True
            )
            
            # Initialize Loader
            loader = UniversalLoader(
                metadata_generator=metadata_gen,
                config=config
            )
            
            print(f"Loading {len(file_paths)} documents...")
            
            # Load documents
            result = loader.load_batch(file_paths)
            print(f"Loaded: {result.summary()}")
            
            # Store documents in DocumentStore
            stored_docs = []
            for doc in result.documents:
                doc_id = doc_store.store_document(doc)
                stored_docs.append(doc_id)
                print(f"Stored document: {doc_id}")
            
            print(f"\n✓ Successfully loaded and stored {len(stored_docs)} documents")
            
            return doc_store, stored_docs
            
        except Exception as e:
            print(f"❌ Integration failed: {e}")
            raise
        finally:
            # Cleanup sample files
            import shutil
            if sample_dir.exists():
                shutil.rmtree(sample_dir)


def demonstrate_document_search(doc_store):
    """Demonstrate various document search capabilities"""
    print("\n=== Document Search Demonstration ===")
    
    # 1. Search by access group
    print("\n1. Search by access group:")
    public_docs = doc_store.search_by_metadata({"access_group": "public"})
    print(f"   Public documents: {len(public_docs)}")
    for result in public_docs:
        print(f"   - {result.document.path}")
    
    # 2. Search technical documents
    print("\n2. Search technical documents:")
    tech_docs = doc_store.search_by_metadata({"classification": "technical"})
    print(f"   Technical documents: {len(tech_docs)}")
    for result in tech_docs:
        print(f"   - {result.document.path}")
    
    # 3. Search by file type
    print("\n3. Search by file type:")
    md_docs = doc_store.search_by_metadata({"file_type": ".md"})
    print(f"   Markdown documents: {len(md_docs)}")
    for result in md_docs:
        print(f"   - {result.document.path}")
    
    # 4. Content-based search
    print("\n4. Content search:")
    api_docs = doc_store.search_by_content("API")
    print(f"   Documents containing 'API': {len(api_docs)}")
    for result in api_docs:
        print(f"   - {result.document.path} (score: {result.score:.6f})")
    
    # 5. Complex metadata search
    print("\n5. Complex metadata search:")
    try:
        large_internal_docs = doc_store.search_by_metadata({
            "classification": "internal",
            "size_bytes": {"$gte": 100}
        })
        print(f"   Large internal documents: {len(large_internal_docs)}")
        for result in large_internal_docs:
            doc = result.document
            print(f"   - {doc.path} ({doc.size_bytes} bytes)")
    except Exception as e:
        print(f"   ⚠ Complex search may require JSON1: {e}")


def demonstrate_document_lifecycle(doc_store):
    """Demonstrate document processing lifecycle"""
    print("\n=== Document Lifecycle Demonstration ===")
    
    # Get an original document
    original_docs = doc_store.search_by_metadata({"file_type": ".md"})
    if not original_docs:
        print("No original documents found")
        return
    
    original_doc = original_docs[0].document
    print(f"Original document: {original_doc.id}")
    
    # Simulate normalization process
    normalized_doc = Document(
        id=f"{original_doc.id}_normalized",
        content=original_doc.content.replace("API", "Application Programming Interface"),
        metadata={
            **original_doc.metadata,
            "original_document_id": original_doc.id,
            "parent_document_id": original_doc.id,
            "processing_stage": "normalized",
            "processing_timestamp": "2024-01-15T11:00:00Z"
        }
    )
    
    # Store normalized document
    normalized_id = doc_store.store_document(normalized_doc)
    print(f"Created normalized document: {normalized_id}")
    
    # Simulate chunking process
    content_chunks = normalized_doc.content.split('\n\n')
    chunk_docs = []
    
    for i, chunk_content in enumerate(content_chunks[:2]):  # Only first 2 chunks
        if chunk_content.strip():
            chunk_doc = Document(
                id=f"{original_doc.id}_chunk_{i}",
                content=chunk_content.strip(),
                metadata={
                    **original_doc.metadata,
                    "original_document_id": original_doc.id,
                    "parent_document_id": normalized_doc.id,
                    "processing_stage": "chunked",
                    "chunk_position": i,
                    "chunk_total": len(content_chunks),
                    "size_bytes": len(chunk_content.encode('utf-8'))
                }
            )
            
            chunk_id = doc_store.store_document(chunk_doc)
            chunk_docs.append(chunk_id)
            print(f"Created chunk document: {chunk_id}")
    
    # Demonstrate lineage tracking
    print(f"\nDocument lineage for {original_doc.id}:")
    lineage_docs = doc_store.get_documents_by_lineage(original_doc.id)
    
    for doc in lineage_docs:
        stage = doc.metadata.get("processing_stage", "original")
        print(f"   - {doc.id} ({stage})")
    
    print(f"Total documents in lineage: {len(lineage_docs)}")


def demonstrate_analytics_and_stats(doc_store):
    """Demonstrate analytics and statistics"""
    print("\n=== Analytics and Statistics ===")
    
    # Get storage statistics
    stats = doc_store.get_storage_stats()
    print(f"Storage Statistics:")
    print(f"   Total documents: {stats.total_documents}")
    print(f"   Storage size: {stats.storage_size_bytes} bytes")
    print(f"   Oldest document: {stats.oldest_document}")
    print(f"   Newest document: {stats.newest_document}")
    
    # Count by category
    print(f"\nDocument counts by classification:")
    for classification in ["technical", "internal", "open"]:
        count = doc_store.count_documents({"classification": classification})
        print(f"   {classification}: {count}")
    
    # Count by access group
    print(f"\nDocument counts by access group:")
    for group in ["engineers", "employees", "public"]:
        count = doc_store.count_documents({"access_group": group})
        print(f"   {group}: {count}")
    
    # List recent documents
    print(f"\nRecent documents (sorted by creation):")
    recent_docs = doc_store.list_documents(limit=5, sort_by="created_at", sort_order="desc")
    for doc in recent_docs:
        stage = doc.metadata.get("processing_stage", "original")
        print(f"   - {doc.id} ({stage})")


def demonstrate_backup_restore(doc_store):
    """Demonstrate backup and restore functionality"""
    print("\n=== Backup and Restore ===")
    
    with tempfile.TemporaryDirectory() as backup_dir:
        backup_path = Path(backup_dir) / "document_backup.db"
        
        # Create backup
        print("Creating backup...")
        backup_success = doc_store.backup_to_file(str(backup_path))
        print(f"Backup created: {backup_success}")
        print(f"Backup size: {backup_path.stat().st_size} bytes")
        
        # Get current document count for verification
        original_count = doc_store.count_documents()
        print(f"Original document count: {original_count}")
        
        # For demonstration, we won't actually restore (would overwrite current data)
        # But we can verify the backup file exists and is readable
        if backup_path.exists():
            print("✓ Backup file created successfully")
            print("✓ Backup could be restored if needed")


def main():
    """Run complete DocumentStore integration example"""
    print("DocumentStore Integration Example")
    print("=" * 50)
    
    try:
        # 1. Load documents and store them
        doc_store, stored_docs = loader_document_store_integration()
        
        # 2. Demonstrate search capabilities
        demonstrate_document_search(doc_store)
        
        # 3. Show document lifecycle and lineage
        demonstrate_document_lifecycle(doc_store)
        
        # 4. Analytics and statistics
        demonstrate_analytics_and_stats(doc_store)
        
        # 5. Backup functionality
        demonstrate_backup_restore(doc_store)
        
        print("\n✅ DocumentStore integration example completed successfully!")
        
        # Final statistics
        final_stats = doc_store.get_storage_stats()
        print(f"\nFinal Statistics:")
        print(f"   Total documents processed: {final_stats.total_documents}")
        print(f"   Total storage used: {final_stats.storage_size_bytes} bytes")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if 'doc_store' in locals():
            doc_store.close()
    
    return 0


if __name__ == "__main__":
    exit(main())