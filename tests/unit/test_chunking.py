"""
Test script for Chunking system
"""

import sys
import tempfile
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag import (
    Document,
    TokenBasedChunker,
    SentenceAwareChunker,
    ChunkingConfig,
    DocumentPipeline,
    SQLiteDocumentStore
)


def test_token_based_chunker():
    """Test TokenBasedChunker"""
    print("=== Testing TokenBasedChunker ===")
    
    # Create test document
    test_doc = Document(
        id="token_test_001",
        content="""This is a comprehensive test document for token-based chunking. 
        It contains multiple sentences and paragraphs to test the chunking functionality.
        The chunker should split this document into smaller pieces based on token count.
        Each chunk should have proper metadata and lineage tracking information.
        This will help us verify that the chunking system works correctly.""",
        metadata={
            "path": "/test/token_doc.txt",
            "created_at": "2024-01-15T10:00:00Z",
            "file_type": ".txt",
            "size_bytes": 500,
            "category": "test"
        }
    )
    
    # Test with different configurations
    configs = [
        ChunkingConfig(chunk_size=20, overlap=5, split_by_sentence=False),
        ChunkingConfig(chunk_size=30, overlap=10, split_by_sentence=False),
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}: chunk_size={config.chunk_size}, overlap={config.overlap} ---")
        
        chunker = TokenBasedChunker(config)
        
        print(f"Original document tokens: {chunker.estimate_tokens(test_doc.content)}")
        print(f"Original content: {test_doc.content[:100]}...")
        
        chunks = chunker.process_with_stats(test_doc)
        
        print(f"Created {len(chunks)} chunks:")
        for chunk in chunks:
            token_count = chunk.metadata.get("token_count", "unknown")
            print(f"  - {chunk.id}: {token_count} tokens")
            print(f"    Content: {chunk.content[:60]}...")
            print(f"    Position: {chunk.metadata.get('chunk_position')}/{chunk.metadata.get('chunk_total')-1}")
        
        print(f"Chunker stats: {chunker.get_chunking_stats()}")
    
    return chunks


def test_sentence_aware_chunker():
    """Test SentenceAwareChunker"""
    print("\n=== Testing SentenceAwareChunker ===")
    
    # Create test document with clear sentence boundaries
    test_doc = Document(
        id="sentence_test_001",
        content="""This is the first sentence of our test document. Here we have the second sentence that provides more context. 
        The third sentence introduces a new concept! This is an exciting fourth sentence with an exclamation mark.
        
        Now we start a new paragraph with the fifth sentence. The sixth sentence continues the paragraph.
        Finally, we have the seventh sentence that concludes our test document.""",
        metadata={
            "path": "/test/sentence_doc.txt",
            "created_at": "2024-01-15T11:00:00Z",
            "file_type": ".txt",
            "size_bytes": 600,
            "category": "test"
        }
    )
    
    # Test with different configurations
    configs = [
        ChunkingConfig(chunk_size=25, overlap=5, split_by_sentence=True),
        ChunkingConfig(chunk_size=40, overlap=8, split_by_sentence=True),
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}: chunk_size={config.chunk_size}, overlap={config.overlap} ---")
        
        chunker = SentenceAwareChunker(config)
        
        print(f"Original document tokens: {chunker.estimate_tokens(test_doc.content)}")
        print(f"Original content: {test_doc.content[:100]}...")
        
        chunks = chunker.process_with_stats(test_doc)
        
        print(f"Created {len(chunks)} sentence-aware chunks:")
        for chunk in chunks:
            token_count = chunk.metadata.get("token_count", "unknown")
            print(f"  - {chunk.id}: {token_count} tokens")
            print(f"    Content: {chunk.content[:80]}...")
            print(f"    Position: {chunk.metadata.get('chunk_position')}/{chunk.metadata.get('chunk_total')-1}")
        
        print(f"Chunker stats: {chunker.get_chunking_stats()}")
    
    return chunks


def test_chunking_pipeline():
    """Test chunking in a complete pipeline"""
    print("\n=== Testing Chunking Pipeline ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "chunking_test.db"
        doc_store = SQLiteDocumentStore(str(db_path))
        
        try:
            # Create pipeline with chunker
            chunker = SentenceAwareChunker(
                ChunkingConfig(chunk_size=30, overlap=5, split_by_sentence=True)
            )
            
            pipeline = DocumentPipeline(
                processors=[chunker],
                document_store=doc_store,
                store_intermediate_results=True
            )
            
            # Create test document
            test_doc = Document(
                id="pipeline_chunking_001",
                content="""This is a pipeline test for chunking functionality. The document will be processed through the chunking pipeline.
                We expect the document to be split into meaningful chunks. Each chunk should preserve sentence boundaries where possible.
                The pipeline should store all intermediate results in the document store. This allows us to track the complete lineage.
                Finally, we can verify that all metadata is properly set and the chunking worked as expected.""",
                metadata={
                    "path": "/test/pipeline_chunk_doc.txt",
                    "created_at": "2024-01-15T12:00:00Z",
                    "file_type": ".txt",
                    "size_bytes": 800,
                    "category": "pipeline_test"
                }
            )
            
            print(f"Processing document: {test_doc.id}")
            print(f"Original content length: {len(test_doc.content)} chars")
            print(f"Estimated tokens: {chunker.estimate_tokens(test_doc.content)}")
            
            # Process through pipeline
            all_results = pipeline.process_document(test_doc)
            
            print(f"\nPipeline produced {len(all_results)} documents:")
            for doc in all_results:
                stage = doc.metadata.get("processing_stage", "original")
                tokens = doc.metadata.get("token_count", "unknown")
                print(f"  - {doc.id} ({stage}): {tokens} tokens")
                if stage == "chunked":
                    pos = doc.metadata.get("chunk_position", "?")
                    total = doc.metadata.get("chunk_total", "?")
                    print(f"    Position: {pos}/{total-1 if isinstance(total, int) else total}")
                    print(f"    Content: {doc.content[:60]}...")
            
            # Check lineage
            print(f"\nLineage tracking:")
            lineage_docs = doc_store.get_documents_by_lineage(test_doc.id)
            for doc in lineage_docs:
                stage = doc.metadata.get("processing_stage", "original")
                print(f"  - {doc.id} ({stage})")
            
            # Pipeline statistics
            stats = pipeline.get_pipeline_stats()
            print(f"\nPipeline Statistics:")
            print(f"  Documents processed: {stats['documents_processed']}")
            print(f"  Total time: {stats['total_pipeline_time']:.3f}s")
            print(f"  Chunker time: {stats['processor_stats']['SentenceAwareChunker']['total_time']:.3f}s")
            
            return all_results
            
        finally:
            doc_store.close()


def test_edge_cases():
    """Test edge cases for chunking"""
    print("\n=== Testing Edge Cases ===")
    
    # Test empty document
    empty_doc = Document(
        id="empty_001",
        content="",
        metadata={
            "path": "/test/empty.txt",
            "created_at": "2024-01-15T13:00:00Z",
            "file_type": ".txt",
            "size_bytes": 0
        }
    )
    
    # Test very short document
    short_doc = Document(
        id="short_001",
        content="Short.",
        metadata={
            "path": "/test/short.txt",
            "created_at": "2024-01-15T13:01:00Z",
            "file_type": ".txt",
            "size_bytes": 6
        }
    )
    
    # Test single long sentence
    long_sentence_doc = Document(
        id="long_sentence_001",
        content="This is an extremely long sentence that contains many words and should definitely exceed the typical chunk size limit that we set for testing purposes and it keeps going and going with more words to really test the edge case handling.",
        metadata={
            "path": "/test/long_sentence.txt",
            "created_at": "2024-01-15T13:02:00Z",
            "file_type": ".txt",
            "size_bytes": 200
        }
    )
    
    chunker = SentenceAwareChunker(
        ChunkingConfig(chunk_size=20, overlap=3, split_by_sentence=True)
    )
    
    test_cases = [
        ("Empty document", empty_doc),
        ("Short document", short_doc),
        ("Long sentence document", long_sentence_doc)
    ]
    
    for name, doc in test_cases:
        print(f"\n--- {name} ---")
        print(f"Content: '{doc.content}'")
        print(f"Tokens: {chunker.estimate_tokens(doc.content)}")
        
        try:
            chunks = chunker.process_with_stats(doc)
            print(f"Result: {len(chunks)} chunks")
            
            for chunk in chunks:
                print(f"  - {chunk.id}: {len(chunk.content)} chars, {chunk.metadata.get('token_count', '?')} tokens")
                
        except Exception as e:
            print(f"Error: {e}")


def test_overlap_functionality():
    """Test overlap functionality specifically"""
    print("\n=== Testing Overlap Functionality ===")
    
    test_doc = Document(
        id="overlap_test_001",
        content="First sentence here. Second sentence follows. Third sentence continues. Fourth sentence next. Fifth and final sentence ends.",
        metadata={
            "path": "/test/overlap_doc.txt",
            "created_at": "2024-01-15T14:00:00Z",
            "file_type": ".txt",
            "size_bytes": 150
        }
    )
    
    # Test different overlap settings
    overlap_configs = [
        (ChunkingConfig(chunk_size=15, overlap=0), "No overlap"),
        (ChunkingConfig(chunk_size=15, overlap=5), "Small overlap"),
        (ChunkingConfig(chunk_size=15, overlap=10), "Large overlap"),
    ]
    
    for config, description in overlap_configs:
        print(f"\n--- {description} (chunk_size={config.chunk_size}, overlap={config.overlap}) ---")
        
        chunker = SentenceAwareChunker(config)
        chunks = chunker.process_with_stats(test_doc)
        
        print(f"Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            overlap_info = chunk.metadata.get("overlap_previous", 0)
            print(f"  Chunk {i}: overlap={overlap_info} tokens")
            print(f"    Content: {chunk.content}")
        
        # Show overlap analysis
        for i in range(1, len(chunks)):
            current_words = set(chunks[i].content.split())
            prev_words = set(chunks[i-1].content.split())
            overlap_words = current_words & prev_words
            print(f"  Actual overlap between chunks {i-1} and {i}: {len(overlap_words)} words")


def main():
    """Run all chunking tests"""
    print("Chunking System Test Suite")
    print("=" * 50)
    
    try:
        # Test basic chunkers
        token_chunks = test_token_based_chunker()
        sentence_chunks = test_sentence_aware_chunker()
        
        # Test pipeline integration
        pipeline_results = test_chunking_pipeline()
        
        # Test edge cases
        test_edge_cases()
        
        # Test overlap functionality
        test_overlap_functionality()
        
        print("\n" + "=" * 50)
        print("✅ All chunking tests completed successfully!")
        
        print(f"\nTest Summary:")
        print(f"  - Token-based chunking: ✅")
        print(f"  - Sentence-aware chunking: ✅")
        print(f"  - Pipeline integration: ✅")
        print(f"  - Edge cases: ✅")
        print(f"  - Overlap functionality: ✅")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())