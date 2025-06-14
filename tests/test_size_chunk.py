"""
Tests for SizeChunkProcessor
SizeChunkProcessorのテスト
"""

import pytest
from refinire_rag.processor import SizeChunkProcessor
from refinire_rag.models.document import Document

def test_small_document():
    """Test processing of document smaller than chunk size
    チャンクサイズより小さい文書の処理テスト"""
    
    # Create processor with 100 byte chunks
    processor = SizeChunkProcessor(chunk_size=100)
    
    # Create small document
    doc = Document(
        id="small_doc",
        content="This is a small document",
        metadata={"source": "test"}
    )
    
    # Process document
    results = list(processor.process([doc]))
    
    # Should return single document unchanged
    assert len(results) == 1
    assert results[0].content == doc.content
    assert results[0].metadata == doc.metadata

def test_large_document():
    """Test processing of document larger than chunk size
    チャンクサイズより大きい文書の処理テスト"""
    
    # Create processor with 10 byte chunks
    processor = SizeChunkProcessor(chunk_size=10)
    
    # Create large document
    doc = Document(
        id="large_doc",
        content="This is a larger document that needs to be chunked",
        metadata={"source": "test"}
    )
    
    # Process document
    results = list(processor.process([doc]))
    
    # Should return multiple chunks
    assert len(results) > 1
    
    # Check first chunk
    assert len(results[0].content) <= 10
    assert results[0].metadata["chunk_index"] == 0
    assert results[0].metadata["original_document_id"] == doc.id
    
    # Check last chunk
    assert results[-1].metadata["chunk_index"] == len(results) - 1
    
    # Verify content reconstruction
    reconstructed = "".join(chunk.content for chunk in results)
    assert reconstructed == doc.content

def test_overlapping_chunks():
    """Test processing with overlapping chunks
    オーバーラップ付きチャンクの処理テスト"""
    
    # Create processor with 10 byte chunks and 2 byte overlap
    processor = SizeChunkProcessor(chunk_size=10, overlap_size=2)
    
    # Create document
    doc = Document(
        id="overlap_doc",
        content="This is a document for testing overlapping chunks",
        metadata={"source": "test"}
    )
    
    # Process document
    results = list(processor.process([doc]))
    
    # Should have overlapping content
    for i in range(len(results) - 1):
        current_chunk = results[i].content
        next_chunk = results[i + 1].content
        overlap = processor.config['overlap_size']
        if len(current_chunk) >= overlap and len(next_chunk) >= overlap and overlap > 0:
            assert current_chunk[-overlap:] == next_chunk[:overlap]

def test_multiple_documents():
    """Test processing of multiple documents
    複数文書の処理テスト"""
    
    # Create processor
    processor = SizeChunkProcessor(chunk_size=10)
    
    # Create multiple documents
    docs = [
        Document(id="doc1", content="First document", metadata={"source": "test1"}),
        Document(id="doc2", content="Second document that is longer", metadata={"source": "test2"}),
        Document(id="doc3", content="Third document", metadata={"source": "test3"})
    ]
    
    # Process documents
    results = list(processor.process(docs))
    
    # Should process all documents
    assert len(results) >= len(docs)
    
    # Verify each document's content is preserved
    current_doc = 0
    current_content = ""
    current_doc_id = docs[0].id
    for chunk in results:
        if chunk.metadata["original_document_id"] != current_doc_id:
            assert current_content == docs[current_doc].content
            current_doc += 1
            current_doc_id = chunk.metadata["original_document_id"]
            current_content = chunk.content
        else:
            current_content += chunk.content

    # Check last document
    assert current_content == docs[-1].content 