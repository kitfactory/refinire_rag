"""
Test script for DocumentProcessor and DocumentPipeline
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag import (
    Document,
    DocumentProcessor,
    DocumentPipeline,
    DocumentProcessorConfig,
    SQLiteDocumentStore
)


# Example document processors for testing
class TestNormalizer(DocumentProcessor):
    """Test normalizer that converts text to uppercase"""
    
    def process(self, document: Document, config=None) -> list[Document]:
        """Normalize document by converting content to uppercase"""
        
        normalized_content = document.content.upper()
        
        # Create normalized document with lineage metadata
        normalized_doc = Document(
            id=f"{document.id}_normalized",
            content=normalized_content,
            metadata={
                **document.metadata,
                "original_document_id": document.id,
                "parent_document_id": document.id,
                "processing_stage": "normalized",
                "normalization_type": "uppercase",
                "processed_at": datetime.now().isoformat()
            }
        )
        
        return [normalized_doc]


class TestChunker(DocumentProcessor):
    """Test chunker that splits document into sentences"""
    
    def process(self, document: Document, config=None) -> list[Document]:
        """Chunk document by splitting on periods"""
        
        # Simple sentence splitting on periods
        sentences = [s.strip() for s in document.content.split('.') if s.strip()]
        
        chunk_docs = []
        for i, sentence in enumerate(sentences):
            chunk_doc = Document(
                id=f"{document.id}_chunk_{i}",
                content=sentence,
                metadata={
                    **document.metadata,
                    "original_document_id": document.metadata.get("original_document_id", document.id),
                    "parent_document_id": document.id,
                    "processing_stage": "chunked",
                    "chunk_position": i,
                    "chunk_total": len(sentences),
                    "chunk_type": "sentence",
                    "processed_at": datetime.now().isoformat()
                }
            )
            chunk_docs.append(chunk_doc)
        
        return chunk_docs


def test_document_processor():
    """Test basic DocumentProcessor functionality"""
    print("=== Testing DocumentProcessor ===")
    
    # Create test document
    test_doc = Document(
        id="test_001",
        content="This is a test document. It has multiple sentences. We will process it.",
        metadata={
            "path": "/test/doc.txt",
            "created_at": "2024-01-15T10:00:00Z",
            "file_type": ".txt",
            "size_bytes": 100,
            "category": "test"
        }
    )
    
    # Test normalizer
    normalizer = TestNormalizer()
    normalized_docs = normalizer.process_with_stats(test_doc)
    
    print(f"Original: {test_doc.content}")
    print(f"Normalized: {normalized_docs[0].content}")
    print(f"Normalizer stats: {normalizer.get_processing_stats()}")
    
    # Test chunker
    chunker = TestChunker() 
    chunk_docs = chunker.process_with_stats(normalized_docs[0])
    
    print(f"Created {len(chunk_docs)} chunks:")
    for chunk in chunk_docs:
        print(f"  - {chunk.id}: '{chunk.content}'")
    
    print(f"Chunker stats: {chunker.get_processing_stats()}")
    
    return normalized_docs[0], chunk_docs


def test_document_pipeline():
    """Test DocumentPipeline functionality"""
    print("\n=== Testing DocumentPipeline ===")
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_pipeline.db"
        doc_store = SQLiteDocumentStore(str(db_path))
        
        try:
            # Create pipeline with normalizer and chunker
            processors = [TestNormalizer(), TestChunker()]
            pipeline = DocumentPipeline(
                processors=processors,
                document_store=doc_store,
                store_intermediate_results=True
            )
            
            # Create test document
            test_doc = Document(
                id="pipeline_test_001",
                content="This is a pipeline test. It should normalize text. Then chunk it into pieces.",
                metadata={
                    "path": "/test/pipeline_doc.txt",
                    "created_at": "2024-01-15T11:00:00Z",
                    "file_type": ".txt",
                    "size_bytes": 150,
                    "category": "pipeline_test"
                }
            )
            
            print(f"Processing document: {test_doc.id}")
            print(f"Original content: '{test_doc.content}'")
            
            # Process through pipeline
            all_results = pipeline.process_document(test_doc)
            
            print(f"\nPipeline produced {len(all_results)} total documents:")
            for doc in all_results:
                stage = doc.metadata.get("processing_stage", "original")
                print(f"  - {doc.id} ({stage}): '{doc.content[:50]}{'...' if len(doc.content) > 50 else ''}'")
            
            # Check what's stored in the database
            print(f"\nDocuments in DocumentStore:")
            stored_docs = doc_store.list_documents()
            for doc in stored_docs:
                stage = doc.metadata.get("processing_stage", "original")
                print(f"  - {doc.id} ({stage})")
            
            # Test lineage tracking
            original_id = "pipeline_test_001"
            lineage_docs = doc_store.get_documents_by_lineage(original_id)
            print(f"\nDocuments in lineage for {original_id}:")
            for doc in lineage_docs:
                stage = doc.metadata.get("processing_stage", "original")
                print(f"  - {doc.id} ({stage})")
            
            # Get pipeline statistics
            pipeline_stats = pipeline.get_pipeline_stats()
            print(f"\nPipeline Statistics:")
            print(f"  Documents processed: {pipeline_stats['documents_processed']}")
            print(f"  Total time: {pipeline_stats['total_pipeline_time']:.3f}s")
            print(f"  Average time: {pipeline_stats['average_pipeline_time']:.3f}s")
            print(f"  Errors: {pipeline_stats['errors']}")
            
            for processor_name, stats in pipeline_stats['processor_stats'].items():
                print(f"  {processor_name}: {stats['documents_processed']} docs, {stats['total_time']:.3f}s")
            
            return all_results, pipeline_stats
            
        finally:
            doc_store.close()


def test_multiple_documents():
    """Test processing multiple documents through pipeline"""
    print("\n=== Testing Multiple Documents ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_multi.db"
        doc_store = SQLiteDocumentStore(str(db_path))
        
        try:
            # Create pipeline
            processors = [TestNormalizer(), TestChunker()]
            pipeline = DocumentPipeline(
                processors=processors,
                document_store=doc_store
            )
            
            # Create multiple test documents
            test_docs = [
                Document(
                    id="multi_001",
                    content="First document content. Has two sentences.",
                    metadata={
                        "path": "/test/doc1.txt",
                        "created_at": "2024-01-15T12:00:00Z",
                        "file_type": ".txt",
                        "size_bytes": 50
                    }
                ),
                Document(
                    id="multi_002", 
                    content="Second document is here. It also has content. Multiple parts to process.",
                    metadata={
                        "path": "/test/doc2.txt",
                        "created_at": "2024-01-15T12:01:00Z",
                        "file_type": ".txt",
                        "size_bytes": 75
                    }
                ),
                Document(
                    id="multi_003",
                    content="Third and final document. Short but sweet.",
                    metadata={
                        "path": "/test/doc3.txt",
                        "created_at": "2024-01-15T12:02:00Z",
                        "file_type": ".txt",
                        "size_bytes": 40
                    }
                )
            ]
            
            print(f"Processing {len(test_docs)} documents through pipeline")
            
            # Process all documents
            all_results = pipeline.process_documents(test_docs)
            
            print(f"Pipeline produced {len(all_results)} total documents from {len(test_docs)} input documents")
            
            # Group by original document
            by_original = {}
            for doc in all_results:
                original_id = doc.metadata.get("original_document_id", doc.id)
                if original_id not in by_original:
                    by_original[original_id] = []
                by_original[original_id].append(doc)
            
            for original_id, docs in by_original.items():
                print(f"\nDocuments derived from {original_id}:")
                for doc in docs:
                    stage = doc.metadata.get("processing_stage", "original")
                    print(f"  - {doc.id} ({stage})")
            
            # Final statistics
            final_stats = pipeline.get_pipeline_stats()
            print(f"\nFinal Pipeline Statistics:")
            print(f"  Input documents: {len(test_docs)}")
            print(f"  Output documents: {len(all_results)}")
            print(f"  Total processing time: {final_stats['total_pipeline_time']:.3f}s")
            print(f"  Documents in store: {doc_store.count_documents()}")
            
            return all_results
            
        finally:
            doc_store.close()


def main():
    """Run all DocumentProcessor tests"""
    print("DocumentProcessor and DocumentPipeline Test Suite")
    print("=" * 60)
    
    try:
        # Test basic processors
        normalized_doc, chunk_docs = test_document_processor()
        
        # Test pipeline
        pipeline_results, pipeline_stats = test_document_pipeline()
        
        # Test multiple documents
        multi_results = test_multiple_documents()
        
        print("\n" + "=" * 60)
        print("✅ All DocumentProcessor tests completed successfully!")
        
        print(f"\nTest Summary:")
        print(f"  - Basic processor test: ✅")
        print(f"  - Pipeline test: ✅")
        print(f"  - Multiple documents test: ✅")
        print(f"  - Total documents processed: {len(multi_results)}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())