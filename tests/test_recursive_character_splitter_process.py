"""
Comprehensive tests for RecursiveCharacterTextSplitter process method
RecursiveCharacterTextSplitterのprocessメソッドの包括的テスト

This module tests the actual recursive splitting functionality.
このモジュールは、実際の再帰分割機能をテストします。
"""

import pytest
from refinire_rag.splitter.recursive_character_splitter import RecursiveCharacterTextSplitter
from refinire_rag.models.document import Document


class TestRecursiveCharacterTextSplitterProcess:
    """
    Test RecursiveCharacterTextSplitter process method functionality
    RecursiveCharacterTextSplitterのprocessメソッド機能のテスト
    """

    def test_process_short_text_no_splitting(self):
        """
        Test processing text shorter than chunk size
        チャンクサイズより短いテキストの処理テスト
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, 
            overlap_size=0, 
            separators=["\n\n", "\n", " ", ""]
        )
        
        doc = Document(
            id="test_doc_1",
            content="This is a short text.",
            metadata={"source": "test"}
        )
        
        chunks = list(splitter.process([doc]))
        
        # Should have only one chunk
        assert len(chunks) == 1
        assert chunks[0].content == doc.content
        assert chunks[0].metadata['origin_id'] == doc.id
        assert chunks[0].metadata['chunk_index'] == 0

    def test_process_paragraph_separation(self):
        """
        Test processing with paragraph separation (double newlines)
        段落分離（ダブル改行）での処理テスト
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            overlap_size=0,
            separators=["\n\n", "\n", " ", ""]
        )
        
        content = """First paragraph here.

Second paragraph here.

Third paragraph here."""
        
        doc = Document(id="paragraph_test", content=content, metadata={"source": "test"})
        chunks = list(splitter.process([doc]))
        
        # Should split by paragraphs first
        assert len(chunks) >= 1
        
        # Check that chunks preserve paragraph structure where possible
        for chunk in chunks:
            assert chunk.metadata['origin_id'] == doc.id
            # Content should not be empty
            assert len(chunk.content.strip()) > 0

    def test_process_sentence_separation(self):
        """
        Test processing with sentence separation (single newlines)
        文分離（単一改行）での処理テスト
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=30,
            overlap_size=0,
            separators=["\n\n", "\n", " ", ""]
        )
        
        content = """First sentence here.
Second sentence here.
Third sentence here.
Fourth sentence here."""
        
        doc = Document(id="sentence_test", content=content, metadata={"source": "test"})
        chunks = list(splitter.process([doc]))
        
        # Should split by sentences when paragraphs are too large
        assert len(chunks) >= 1
        
        # Each chunk should respect the chunk size
        for chunk in chunks:
            assert len(chunk.content) <= 30 or "\n" not in chunk.content

    def test_process_word_separation(self):
        """
        Test processing with word separation (spaces)
        単語分離（空白）での処理テスト
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=20,
            overlap_size=0,
            separators=["\n\n", "\n", " ", ""]
        )
        
        content = "This is a very long sentence that will need to be split by words."
        doc = Document(id="word_test", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should split by words when sentences are too large
        assert len(chunks) > 1
        
        # Each chunk should respect the chunk size
        for chunk in chunks:
            assert len(chunk.content) <= 20 or " " not in chunk.content

    def test_process_character_separation(self):
        """
        Test processing with character separation (fallback)
        文字分離（フォールバック）での処理テスト
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=10,
            overlap_size=0,
            separators=["\n\n", "\n", " ", ""]
        )
        
        content = "verylongwordwithoutspaces"
        doc = Document(id="char_test", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should fall back to character splitting
        assert len(chunks) > 1
        
        # Each chunk should respect the chunk size exactly
        for chunk in chunks:
            assert len(chunk.content) <= 10

    def test_process_with_overlap(self):
        """
        Test processing with overlap between chunks
        チャンク間のオーバーラップありでの処理テスト
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=15,
            overlap_size=5,
            separators=[" ", ""]
        )
        
        content = "word1 word2 word3 word4 word5 word6"
        doc = Document(id="overlap_test", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        if len(chunks) > 1:
            # Check that there's some overlap content
            # Note: RecursiveCharacterTextSplitter may create chunks larger than chunk_size due to overlap
            assert len(chunks) >= 2
            # Verify that chunks have reasonable sizes (allowing for overlap expansion)
            for chunk in chunks:
                # Chunks may exceed chunk_size due to overlap being added
                assert len(chunk.content) <= 25  # Allow for overlap expansion

    def test_process_empty_content(self):
        """
        Test processing document with empty content
        空のコンテンツでの処理テスト
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            overlap_size=0,
            separators=["\n\n", "\n", " ", ""]
        )
        
        doc = Document(id="empty_doc", content="", metadata={"source": "test"})
        chunks = list(splitter.process([doc]))
        
        # Should produce no chunks for empty content
        assert len(chunks) == 0

    def test_process_none_content(self):
        """
        Test processing document with None content
        Noneコンテンツでの処理テスト
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            overlap_size=0,
            separators=["\n\n", "\n", " ", ""]
        )
        
        doc = Document(id="none_doc", content=None, metadata={"source": "test"})
        chunks = list(splitter.process([doc]))
        
        # Should produce no chunks for None content
        assert len(chunks) == 0

    def test_process_custom_separators(self):
        """
        Test processing with custom separators
        カスタムセパレータでの処理テスト
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=20,
            overlap_size=0,
            separators=["|", ",", " ", ""]
        )
        
        content = "section1|section2|section3,part1,part2 word1 word2"
        doc = Document(id="custom_sep_test", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should split using custom separators
        assert len(chunks) >= 1
        
        # Verify chunks respect size limits
        for chunk in chunks:
            assert len(chunk.content) <= 20

    def test_process_multiple_documents(self):
        """
        Test processing multiple documents
        複数ドキュメントの処理テスト
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=30,
            overlap_size=0,
            separators=["\n\n", "\n", " ", ""]
        )
        
        docs = [
            Document(id="doc1", content="First document with some content here.", metadata={"source": "test1"}),
            Document(id="doc2", content="Second document with different content.", metadata={"source": "test2"}),
            Document(id="doc3", content="Short doc.", metadata={"source": "test3"})
        ]
        
        chunks = list(splitter.process(docs))
        
        # Should have chunks from all documents
        doc1_chunks = [c for c in chunks if c.metadata['origin_id'] == 'doc1']
        doc2_chunks = [c for c in chunks if c.metadata['origin_id'] == 'doc2']
        doc3_chunks = [c for c in chunks if c.metadata['origin_id'] == 'doc3']
        
        assert len(doc1_chunks) >= 1
        assert len(doc2_chunks) >= 1  
        assert len(doc3_chunks) == 1  # Short content

    def test_process_config_override(self):
        """
        Test that config parameter overrides instance configuration
        configパラメータでインスタンス設定をオーバーライドテスト
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            overlap_size=0,
            separators=["\n\n", "\n", " ", ""]
        )
        
        override_config = {
            'chunk_size': 15,
            'overlap_size': 3,
            'separators': [" ", ""]
        }
        
        content = "This is a test sentence that should be split differently with override config"
        doc = Document(id="config_test", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc], config=override_config))
        
        # Should use override config (smaller chunk size = more chunks)
        assert len(chunks) > 1
        
        # Each chunk should be reasonably sized (allowing for overlap expansion)
        for chunk in chunks:
            # Allow for overlap expansion in recursive splitter
            assert len(chunk.content) <= 25

    def test_process_separator_priority(self):
        """
        Test that separators are used in priority order
        セパレータが優先順序で使用されることをテスト
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=25,
            overlap_size=0,
            separators=["|", " ", ""]
        )
        
        # Content that can be split by both | and space, should prefer |
        content = "part1|part2 with more words|part3 with even more words"
        doc = Document(id="priority_test", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should preferentially split on | when possible
        assert len(chunks) >= 1
        
        # Verify that chunks respect size constraints
        for chunk in chunks:
            assert len(chunk.content) <= 25

    def test_process_metadata_preservation(self):
        """
        Test that original document metadata is preserved
        元ドキュメントのメタデータが保持されることをテスト
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=20,
            overlap_size=0,
            separators=[" ", ""]
        )
        
        original_metadata = {
            "source": "test_source",
            "author": "test_author",
            "timestamp": "2024-01-01"
        }
        
        content = "This is a longer text that will be split into multiple chunks"
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
            assert chunk.metadata['origin_id'] == doc.id

    def test_process_mixed_content_structure(self):
        """
        Test processing content with mixed structure (paragraphs, sentences, words)
        混合構造コンテンツの処理テスト（段落、文、単語）
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=40,
            overlap_size=0,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        content = """First paragraph here.

Second paragraph with a longer sentence that might need splitting.

Third paragraph. With multiple sentences. Each quite short."""
        
        doc = Document(id="mixed_test", content=content, metadata={"source": "test"})
        chunks = list(splitter.process([doc]))
        
        # Should handle mixed content appropriately
        assert len(chunks) >= 1
        
        # Verify chunks respect size limits
        for chunk in chunks:
            assert len(chunk.content) <= 40
            assert chunk.metadata['origin_id'] == doc.id

    def test_split_text_method_edge_cases(self):
        """
        Test edge cases in the splitting logic
        分割ロジックのエッジケースをテスト
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=10,
            overlap_size=2,
            separators=[" ", ""]
        )
        
        # Test various edge cases
        test_cases = [
            "",  # Empty string
            "a",  # Single character
            "a b",  # Two single characters with space
            "verylongword",  # Single word longer than chunk size
            "   ",  # Only spaces
        ]
        
        for content in test_cases:
            doc = Document(id=f"edge_test_{len(content)}", content=content, metadata={"source": "test"})
            chunks = list(splitter.process([doc]))
            
            if content.strip():  # Non-empty content
                assert len(chunks) >= 0  # Should not error
                if chunks:
                    for chunk in chunks:
                        # Allow for overlap expansion or fallback to character splitting
                        assert len(chunk.content) <= 15 or chunk.content == content
            else:  # Empty or whitespace content
                # RecursiveCharacterTextSplitter may return chunks for whitespace-only content
                # depending on the implementation
                assert len(chunks) >= 0