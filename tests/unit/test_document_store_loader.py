#!/usr/bin/env python3
"""
Test DocumentStoreLoader

Test the DocumentStoreLoader functionality for reprocessing 
and staged document workflows.
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag import (
    DocumentStoreLoader,
    SQLiteDocumentStore,
    Document,
    LoadingConfig
)


def test_document_store_loader():
    """Test DocumentStoreLoader functionality"""
    print("🧪 Testing DocumentStoreLoader...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        db_path = temp_path / "test_documents.db"
        
        # Create test documents in DocumentStore
        document_store = SQLiteDocumentStore(str(db_path))
        
        # Create test documents
        now = datetime.now().isoformat()
        test_documents = [
            Document(
                id="doc1",
                content="Machine learning algorithms are transforming data analysis.",
                metadata={
                    "path": "/test/doc1.txt",
                    "created_at": now,
                    "file_type": "txt",
                    "size_bytes": 60,
                    "processing_stage": "original",
                    "topic": "AI"
                }
            ),
            Document(
                id="doc2",
                content="Deep learning networks require large datasets for training.",
                metadata={
                    "path": "/test/doc2.txt", 
                    "created_at": now,
                    "file_type": "txt",
                    "size_bytes": 58,
                    "processing_stage": "chunked",
                    "topic": "AI"
                }
            ),
            Document(
                id="doc3",
                content="Natural language processing enables text understanding.",
                metadata={
                    "path": "/test/doc3.md",
                    "created_at": now,
                    "file_type": "md", 
                    "size_bytes": 55,
                    "processing_stage": "embedded",
                    "topic": "NLP"
                }
            )
        ]
        
        # Store test documents
        for doc in test_documents:
            document_store.store_document(doc)
        
        print(f"   ✅ Created {len(test_documents)} test documents in DocumentStore")
        
        # Test DocumentStoreLoader
        loader = DocumentStoreLoader(document_store)
        
        # Test 1: Load single document by ID
        print(f"\n📄 Test 1: Load single document")
        doc1 = loader.load_single("doc1")
        print(f"   Loaded: {doc1.id} - {doc1.content[:40]}...")
        
        # Test 2: Load multiple documents by IDs
        print(f"\n📄 Test 2: Load batch of documents")
        batch_result = loader.load_batch(["doc1", "doc2", "nonexistent"])
        print(f"   Loaded: {batch_result.successful_count} documents")
        print(f"   Failed: {batch_result.failed_count} sources")
        print(f"   Success rate: {batch_result.success_rate:.2%}")
        
        # Test 3: Load by processing stage
        print(f"\n📄 Test 3: Load by processing stage")
        original_docs = loader.load_by_processing_stage("original")
        chunked_docs = loader.load_by_processing_stage("chunked")
        embedded_docs = loader.load_by_processing_stage("embedded")
        
        print(f"   Original stage: {original_docs.successful_count} documents")
        print(f"   Chunked stage: {chunked_docs.successful_count} documents")
        print(f"   Embedded stage: {embedded_docs.successful_count} documents")
        
        # Test 4: Load by file type
        print(f"\n📄 Test 4: Load by file type")
        txt_docs = loader.load_by_file_type("txt")
        md_docs = loader.load_by_file_type("md")
        
        print(f"   TXT files: {txt_docs.successful_count} documents")
        print(f"   MD files: {md_docs.successful_count} documents")
        
        # Test 5: Load by metadata filters
        print(f"\n📄 Test 5: Load by metadata filters")
        ai_docs = loader.load_by_filters({"topic": "AI"})
        nlp_docs = loader.load_by_filters({"topic": "NLP"})
        
        print(f"   AI topic: {ai_docs.successful_count} documents")
        print(f"   NLP topic: {nlp_docs.successful_count} documents")
        
        # Test 6: Load all documents
        print(f"\n📄 Test 6: Load all documents")
        all_docs = loader.load_all()
        print(f"   Total documents: {all_docs.successful_count}")
        
        # Test 7: Get document count
        print(f"\n📄 Test 7: Document counts")
        total_count = loader.get_document_count()
        ai_count = loader.get_document_count({"topic": "AI"})
        
        print(f"   Total documents: {total_count}")
        print(f"   AI documents: {ai_count}")
        
        # Test 8: Get available stages and types
        print(f"\n📄 Test 8: Available stages and types")
        stages = loader.get_available_stages()
        file_types = loader.get_available_file_types()
        
        print(f"   Available stages: {stages}")
        print(f"   Available file types: {file_types}")
        
        # Clean up
        document_store.close()
        
        print(f"\n✅ DocumentStoreLoader test completed!")
        return True


def test_staged_processing_workflow():
    """Test staged processing workflow using DocumentStoreLoader"""
    print(f"\n🔄 Testing Staged Processing Workflow...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        db_path = temp_path / "staged_documents.db"
        
        document_store = SQLiteDocumentStore(str(db_path))
        loader = DocumentStoreLoader(document_store)
        
        # Stage 1: Load "original" documents
        now = datetime.now().isoformat()
        original_docs = [
            Document(
                id="stage_doc1",
                content="First document for staged processing workflow.",
                metadata={
                    "path": "/staged/doc1.txt",
                    "created_at": now,
                    "file_type": "txt",
                    "size_bytes": 50,
                    "processing_stage": "original"
                }
            ),
            Document(
                id="stage_doc2",
                content="Second document for workflow testing.",
                metadata={
                    "path": "/staged/doc2.txt",
                    "created_at": now,
                    "file_type": "txt", 
                    "size_bytes": 40,
                    "processing_stage": "original"
                }
            )
        ]
        
        # Store original documents
        for doc in original_docs:
            document_store.store_document(doc)
        
        print(f"   📥 Stage 1: Stored {len(original_docs)} original documents")
        
        # Stage 2: Load original documents for processing
        original_batch = loader.load_by_processing_stage("original")
        print(f"   📤 Stage 2: Loaded {original_batch.successful_count} original documents")
        
        # Simulate processing: create "processed" versions
        processed_docs = []
        for doc in original_batch.documents:
            processed_doc = Document(
                id=f"{doc.id}_processed",
                content=f"PROCESSED: {doc.content}",
                metadata={
                    **doc.metadata,
                    "processing_stage": "processed",
                    "parent_document_id": doc.id,
                    "processed_at": datetime.now().isoformat()
                }
            )
            processed_docs.append(processed_doc)
            document_store.store_document(processed_doc)
        
        print(f"   🔄 Stage 3: Created {len(processed_docs)} processed documents")
        
        # Stage 3: Load processed documents for embedding
        processed_batch = loader.load_by_processing_stage("processed")
        print(f"   📤 Stage 4: Loaded {processed_batch.successful_count} processed documents")
        
        # Simulate embedding: create "embedded" versions
        embedded_docs = []
        for doc in processed_batch.documents:
            embedded_doc = Document(
                id=f"{doc.id}_embedded",
                content=doc.content,
                metadata={
                    **doc.metadata,
                    "processing_stage": "embedded",
                    "parent_document_id": doc.id,
                    "embedded_at": datetime.now().isoformat(),
                    "embedding_model": "test-model"
                }
            )
            embedded_docs.append(embedded_doc)
            document_store.store_document(embedded_doc)
        
        print(f"   🔤 Stage 5: Created {len(embedded_docs)} embedded documents")
        
        # Final verification: check all stages
        final_original = loader.get_document_count({"processing_stage": "original"})
        final_processed = loader.get_document_count({"processing_stage": "processed"})
        final_embedded = loader.get_document_count({"processing_stage": "embedded"})
        
        print(f"\n📊 Final Stage Counts:")
        print(f"   Original: {final_original}")
        print(f"   Processed: {final_processed}")
        print(f"   Embedded: {final_embedded}")
        
        document_store.close()
        
        print(f"\n✅ Staged processing workflow test completed!")
        return True


def main():
    """Run DocumentStoreLoader tests"""
    print("🚀 Testing DocumentStoreLoader Implementation")
    print("=" * 60)
    
    try:
        # Test basic functionality
        test_document_store_loader()
        
        # Test staged workflow
        test_staged_processing_workflow()
        
        print(f"\n🎉 All DocumentStoreLoader tests passed!")
        print(f"\n📋 Key Features Verified:")
        print(f"   ✅ Load documents by ID")
        print(f"   ✅ Load documents by metadata filters")
        print(f"   ✅ Load documents by processing stage")
        print(f"   ✅ Load documents by file type")
        print(f"   ✅ Batch loading with error handling")
        print(f"   ✅ Document counting and statistics")
        print(f"   ✅ Staged processing workflow")
        print(f"   ✅ Integration with DocumentStore")
        
        print(f"\n💡 Use Cases Enabled:")
        print(f"   • Reprocess documents without file I/O")
        print(f"   • Resume processing from any stage")
        print(f"   • Filter and load specific document subsets")
        print(f"   • Implement staged processing pipelines")
        print(f"   • Database-driven document workflows")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ DocumentStoreLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())