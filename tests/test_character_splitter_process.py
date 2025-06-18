"""
Comprehensive tests for CharacterTextSplitter process method
CharacterTextSplitterのprocessメソッドの包括的テスト

This module tests the actual text splitting functionality of CharacterTextSplitter.
このモジュールは、CharacterTextSplitterの実際のテキスト分割機能をテストします。
"""

import pytest
from refinire_rag.splitter.character_splitter import CharacterTextSplitter
from refinire_rag.models.document import Document


class TestCharacterTextSplitterProcess:
    """
    Test CharacterTextSplitter process method functionality
    CharacterTextSplitterのprocessメソッド機能のテスト
    """

    def test_process_short_text_no_splitting(self):
        """
        Test processing text shorter than chunk size (no splitting needed)
        チャンクサイズより短いテキストの処理テスト（分割不要）
        """
        splitter = CharacterTextSplitter(chunk_size=100, overlap_size=0)
        
        doc = Document(
            id="test_doc_1",
            content="This is a short text that should not be split.",
            metadata={"source": "test"}
        )
        
        chunks = list(splitter.process([doc]))
        
        # Should have only one chunk
        assert len(chunks) == 1
        assert chunks[0].content == doc.content
        assert chunks[0].metadata['origin_id'] == doc.id
        assert chunks[0].metadata['chunk_index'] == 0
        assert chunks[0].metadata['total_chunks'] == 1
        assert chunks[0].metadata['chunk_start'] == 0
        assert chunks[0].metadata['chunk_end'] == len(doc.content)

    def test_process_long_text_with_splitting(self):
        """
        Test processing text longer than chunk size (splitting required)
        チャンクサイズより長いテキストの処理テスト（分割必要）
        """
        splitter = CharacterTextSplitter(chunk_size=20, overlap_size=0)
        
        # Create a text that will be split into multiple chunks
        content = "This is a very long text that should be split into multiple chunks for testing purposes."
        doc = Document(id="test_doc_2", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Check that all chunks are created correctly
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata['origin_id'] == doc.id
            assert chunk.metadata['chunk_index'] == i
            assert chunk.metadata['total_chunks'] == total_chunks
            assert len(chunk.content) <= 20
            
        # Check that all content is preserved
        reconstructed = ''.join(chunk.content for chunk in chunks)
        assert reconstructed == content

    def test_process_with_overlap(self):
        """
        Test processing with overlap between chunks
        チャンク間のオーバーラップありでの処理テスト
        """
        splitter = CharacterTextSplitter(chunk_size=10, overlap_size=3)
        
        content = "0123456789abcdefghijklmnopqrstuvwxyz"
        doc = Document(id="test_doc_3", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should have multiple chunks with overlap
        assert len(chunks) > 1
        
        # Check overlap: each chunk (except first) should start with overlap from previous
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            # Check that there's overlap content
            expected_overlap_start = chunks[i-1].metadata['chunk_end'] - 3
            actual_start = curr_chunk.metadata['chunk_start']
            assert actual_start == expected_overlap_start

    def test_process_empty_content(self):
        """
        Test processing document with empty content
        空のコンテンツを持つドキュメントの処理テスト
        """
        splitter = CharacterTextSplitter(chunk_size=100, overlap_size=0)
        
        doc = Document(id="empty_doc", content="", metadata={"source": "test"})
        chunks = list(splitter.process([doc]))
        
        # Should produce no chunks for empty content
        assert len(chunks) == 0

    def test_process_none_content(self):
        """
        Test processing document with None content
        Noneコンテンツを持つドキュメントの処理テスト
        """
        splitter = CharacterTextSplitter(chunk_size=100, overlap_size=0)
        
        doc = Document(id="none_doc", content=None, metadata={"source": "test"})
        chunks = list(splitter.process([doc]))
        
        # Should produce no chunks for None content
        assert len(chunks) == 0

    def test_process_multiple_documents(self):
        """
        Test processing multiple documents at once
        複数のドキュメントを同時に処理するテスト
        """
        splitter = CharacterTextSplitter(chunk_size=15, overlap_size=2)
        
        docs = [
            Document(id="doc1", content="First document content", metadata={"source": "test1"}),
            Document(id="doc2", content="Second document with longer content", metadata={"source": "test2"}),
            Document(id="doc3", content="Third", metadata={"source": "test3"})
        ]
        
        chunks = list(splitter.process(docs))
        
        # Should have chunks from all documents
        doc1_chunks = [c for c in chunks if c.metadata['origin_id'] == 'doc1']
        doc2_chunks = [c for c in chunks if c.metadata['origin_id'] == 'doc2']
        doc3_chunks = [c for c in chunks if c.metadata['origin_id'] == 'doc3']
        
        assert len(doc1_chunks) >= 1
        assert len(doc2_chunks) >= 1
        assert len(doc3_chunks) == 1  # Short content, single chunk
        
        # Check that original metadata is preserved
        for chunk in chunks:
            original_doc = next(d for d in docs if d.id == chunk.metadata['origin_id'])
            assert chunk.metadata['source'] == original_doc.metadata['source']

    def test_process_zero_chunk_size(self):
        """
        Test processing with zero chunk size
        チャンクサイズ0での処理テスト
        """
        splitter = CharacterTextSplitter(chunk_size=0, overlap_size=0)
        
        doc = Document(id="test_doc", content="Some content", metadata={"source": "test"})
        chunks = list(splitter.process([doc]))
        
        # Should produce no chunks for zero chunk size
        assert len(chunks) == 0

    def test_process_overlap_larger_than_chunk(self):
        """
        Test processing when overlap size is larger than chunk size
        オーバーラップサイズがチャンクサイズより大きい場合の処理テスト
        """
        splitter = CharacterTextSplitter(chunk_size=5, overlap_size=10)
        
        content = "This is a test content for overlap testing"
        doc = Document(id="test_doc", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should still produce chunks, with overlap automatically adjusted
        assert len(chunks) > 1
        
        # All chunks should be valid
        for chunk in chunks:
            assert len(chunk.content) <= 5
            assert chunk.metadata['origin_id'] == doc.id

    def test_process_chunk_boundaries(self):
        """
        Test that chunk boundaries are correctly calculated
        チャンクの境界が正しく計算されることをテスト
        """
        splitter = CharacterTextSplitter(chunk_size=10, overlap_size=2)
        
        content = "0123456789abcdefghij"  # 20 characters
        doc = Document(id="boundary_test", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should have 3 chunks: [0:10], [8:18], [16:20]
        assert len(chunks) == 3
        
        # Check first chunk
        assert chunks[0].content == "0123456789"
        assert chunks[0].metadata['chunk_start'] == 0
        assert chunks[0].metadata['chunk_end'] == 10
        
        # Check second chunk
        assert chunks[1].content == "89abcdefgh"
        assert chunks[1].metadata['chunk_start'] == 8
        assert chunks[1].metadata['chunk_end'] == 18
        
        # Check third chunk (remainder)
        assert chunks[2].content == "ghij"
        assert chunks[2].metadata['chunk_start'] == 16
        assert chunks[2].metadata['chunk_end'] == 20

    def test_process_metadata_preservation(self):
        """
        Test that original document metadata is preserved in chunks
        元のドキュメントのメタデータがチャンクに保持されることをテスト
        """
        splitter = CharacterTextSplitter(chunk_size=10, overlap_size=0)
        
        original_metadata = {
            "source": "test_source",
            "author": "test_author",
            "timestamp": "2024-01-01",
            "category": "test_category"
        }
        
        content = "This is a long content that will be split into multiple chunks"
        doc = Document(id="metadata_test", content=content, metadata=original_metadata)
        
        chunks = list(splitter.process([doc]))
        
        # Check that all chunks preserve original metadata
        for chunk in chunks:
            # Original metadata should be preserved
            for key, value in original_metadata.items():
                assert chunk.metadata[key] == value
            
            # Added metadata should be present
            assert 'origin_id' in chunk.metadata
            assert 'chunk_index' in chunk.metadata
            assert 'chunk_start' in chunk.metadata
            assert 'chunk_end' in chunk.metadata
            assert 'total_chunks' in chunk.metadata

    def test_process_edge_case_single_character_chunks(self):
        """
        Test processing with single character chunk size
        文字サイズ1のチャンクでの処理テスト
        """
        splitter = CharacterTextSplitter(chunk_size=1, overlap_size=0)
        
        content = "abc"
        doc = Document(id="single_char_test", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should have 3 chunks, each with one character
        assert len(chunks) == 3
        assert chunks[0].content == "a"
        assert chunks[1].content == "b"
        assert chunks[2].content == "c"
        
        # Check metadata for each chunk
        for i, chunk in enumerate(chunks):
            assert chunk.metadata['chunk_index'] == i
            assert chunk.metadata['total_chunks'] == 3
            assert chunk.metadata['chunk_start'] == i
            assert chunk.metadata['chunk_end'] == i + 1

    def test_process_unicode_content(self):
        """
        Test processing with Unicode content
        Unicodeコンテンツでの処理テスト
        """
        splitter = CharacterTextSplitter(chunk_size=5, overlap_size=1)
        
        content = "こんにちは世界！Hello🌍"
        doc = Document(id="unicode_test", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should handle Unicode characters correctly
        assert len(chunks) > 1
        
        # Reconstruct content to verify it's preserved
        reconstructed = ""
        for i, chunk in enumerate(chunks):
            if i == 0:
                reconstructed += chunk.content
            else:
                # Remove overlap for reconstruction
                reconstructed += chunk.content[1:]
        
        assert reconstructed == content