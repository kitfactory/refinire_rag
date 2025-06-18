"""
Comprehensive tests for TokenTextSplitter process method
TokenTextSplitterのprocessメソッドの包括的テスト

This module tests the actual token splitting functionality of TokenTextSplitter.
このモジュールは、TokenTextSplitterの実際のトークン分割機能をテストします。
"""

import pytest
from refinire_rag.splitter.token_splitter import TokenTextSplitter
from refinire_rag.models.document import Document


class TestTokenTextSplitterProcess:
    """
    Test TokenTextSplitter process method functionality
    TokenTextSplitterのprocessメソッド機能のテスト
    """

    def test_process_short_text_no_splitting(self):
        """
        Test processing text with fewer tokens than chunk size
        チャンクサイズより少ないトークンのテキスト処理テスト
        """
        splitter = TokenTextSplitter(chunk_size=10, overlap_size=0, separator=" ")
        
        doc = Document(
            id="test_doc_1",
            content="This is a short text",
            metadata={"source": "test"}
        )
        
        chunks = list(splitter.process([doc]))
        
        # Should have only one chunk since token count < chunk_size
        assert len(chunks) == 1
        assert chunks[0].content == doc.content
        assert chunks[0].metadata['origin_id'] == doc.id
        assert chunks[0].metadata['chunk_index'] == 0

    def test_process_long_text_with_splitting(self):
        """
        Test processing text with more tokens than chunk size
        チャンクサイズより多いトークンのテキスト処理テスト
        """
        splitter = TokenTextSplitter(chunk_size=5, overlap_size=0, separator=" ")
        
        content = "This is a very long text that contains many words and should be split into multiple chunks"
        doc = Document(id="test_doc_2", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Check that each chunk has at most 5 tokens
        for chunk in chunks:
            tokens = chunk.content.split(" ")
            assert len(tokens) <= 5
            assert chunk.metadata['origin_id'] == doc.id

    def test_process_with_overlap(self):
        """
        Test processing with token overlap between chunks
        チャンク間のトークンオーバーラップありでの処理テスト
        """
        splitter = TokenTextSplitter(chunk_size=4, overlap_size=2, separator=" ")
        
        # 8 tokens: "one two three four five six seven eight"
        content = "one two three four five six seven eight"
        doc = Document(id="test_doc_3", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should have multiple chunks with overlap
        assert len(chunks) > 1
        
        # Check first chunk: "one two three four"
        first_tokens = chunks[0].content.split(" ")
        assert len(first_tokens) == 4
        assert first_tokens == ["one", "two", "three", "four"]
        
        # Check second chunk: "three four five six" (2 token overlap)
        second_tokens = chunks[1].content.split(" ")
        assert len(second_tokens) == 4
        assert second_tokens == ["three", "four", "five", "six"]

    def test_process_custom_separator(self):
        """
        Test processing with custom token separator
        カスタムトークンセパレータでの処理テスト
        """
        splitter = TokenTextSplitter(chunk_size=3, overlap_size=0, separator="|")
        
        content = "apple|banana|cherry|date|elderberry|fig|grape"
        doc = Document(id="custom_sep_test", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should split on "|" separator
        assert len(chunks) > 1
        
        # Check first chunk
        first_tokens = chunks[0].content.split("|")
        assert len(first_tokens) == 3
        assert first_tokens == ["apple", "banana", "cherry"]

    def test_process_empty_content(self):
        """
        Test processing document with empty content
        空のコンテンツを持つドキュメントの処理テスト
        """
        splitter = TokenTextSplitter(chunk_size=10, overlap_size=0, separator=" ")
        
        doc = Document(id="empty_doc", content="", metadata={"source": "test"})
        chunks = list(splitter.process([doc]))
        
        # Should produce no chunks for empty content
        assert len(chunks) == 0

    def test_process_none_content(self):
        """
        Test processing document with None content
        Noneコンテンツを持つドキュメントの処理テスト
        """
        splitter = TokenTextSplitter(chunk_size=10, overlap_size=0, separator=" ")
        
        doc = Document(id="none_doc", content=None, metadata={"source": "test"})
        chunks = list(splitter.process([doc]))
        
        # Should produce no chunks for None content
        assert len(chunks) == 0

    def test_process_single_token(self):
        """
        Test processing content with single token
        単一トークンのコンテンツ処理テスト
        """
        splitter = TokenTextSplitter(chunk_size=5, overlap_size=0, separator=" ")
        
        doc = Document(id="single_token", content="hello", metadata={"source": "test"})
        chunks = list(splitter.process([doc]))
        
        # Should have one chunk
        assert len(chunks) == 1
        assert chunks[0].content == "hello"

    def test_process_multiple_documents(self):
        """
        Test processing multiple documents simultaneously
        複数のドキュメントを同時に処理するテスト
        """
        splitter = TokenTextSplitter(chunk_size=3, overlap_size=1, separator=" ")
        
        docs = [
            Document(id="doc1", content="first document content here", metadata={"source": "test1"}),
            Document(id="doc2", content="second document with different content", metadata={"source": "test2"}),
            Document(id="doc3", content="short", metadata={"source": "test3"})
        ]
        
        chunks = list(splitter.process(docs))
        
        # Should have chunks from all documents
        doc1_chunks = [c for c in chunks if c.metadata['origin_id'] == 'doc1']
        doc2_chunks = [c for c in chunks if c.metadata['origin_id'] == 'doc2']
        doc3_chunks = [c for c in chunks if c.metadata['origin_id'] == 'doc3']
        
        assert len(doc1_chunks) >= 1
        assert len(doc2_chunks) >= 1
        assert len(doc3_chunks) == 1  # Short content

    def test_process_whitespace_handling(self):
        """
        Test processing with various whitespace scenarios
        様々な空白文字シナリオでの処理テスト
        """
        splitter = TokenTextSplitter(chunk_size=3, overlap_size=0, separator=" ")
        
        # Test with multiple spaces
        content = "word1  word2   word3    word4"
        doc = Document(id="whitespace_test", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should handle multiple spaces correctly
        assert len(chunks) >= 1
        
        # Reconstruct and check that content is preserved
        all_tokens = []
        for chunk in chunks:
            tokens = chunk.content.split(" ")
            all_tokens.extend(tokens)
        
        original_tokens = content.split(" ")
        # Filter out empty tokens that might result from multiple spaces
        all_tokens = [t for t in all_tokens if t]
        original_tokens = [t for t in original_tokens if t]
        
        assert len(all_tokens) >= len(original_tokens)

    def test_process_overlap_larger_than_chunk(self):
        """
        Test processing when overlap is larger than chunk size
        オーバーラップがチャンクサイズより大きい場合の処理テスト
        """
        splitter = TokenTextSplitter(chunk_size=3, overlap_size=5, separator=" ")
        
        content = "one two three four five six seven eight nine ten"
        doc = Document(id="large_overlap_test", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc]))
        
        # Should still process without errors
        assert len(chunks) >= 1
        
        # Each chunk should have at most 3 tokens
        for chunk in chunks:
            tokens = chunk.content.split(" ")
            assert len(tokens) <= 3

    def test_split_text_method_directly(self):
        """
        Test the _split_text method directly
        _split_textメソッドを直接テストする
        """
        splitter = TokenTextSplitter(chunk_size=3, overlap_size=1, separator=" ")
        
        text = "apple banana cherry date elderberry"
        chunks = splitter._split_text(text, chunk_size=3, overlap_size=1, separator=" ")
        
        # Should return list of strings
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)
        
        # Check specific chunks
        assert chunks[0] == "apple banana cherry"
        assert chunks[1] == "cherry date elderberry"

    def test_split_text_no_tokens(self):
        """
        Test _split_text with text that produces no tokens
        トークンが生成されないテキストでの_split_textテスト
        """
        splitter = TokenTextSplitter(chunk_size=3, overlap_size=1, separator=" ")
        
        # Empty string should return empty list
        assert splitter._split_text("", 3, 1, " ") == []
        
        # String with only separator should return empty list or handle gracefully
        result = splitter._split_text("   ", 3, 1, " ")
        # The method might return an empty list or a list with empty strings
        # Both are acceptable behaviors
        assert isinstance(result, list)

    def test_process_metadata_preservation(self):
        """
        Test that original document metadata is preserved
        元のドキュメントメタデータが保持されることをテスト
        """
        splitter = TokenTextSplitter(chunk_size=3, overlap_size=0, separator=" ")
        
        original_metadata = {
            "source": "test_source",
            "author": "test_author", 
            "category": "test_category"
        }
        
        content = "This is a longer text that will be split into multiple token chunks"
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

    def test_process_config_override(self):
        """
        Test that config parameter can override instance configuration
        configパラメータでインスタンス設定をオーバーライドできることをテスト
        """
        splitter = TokenTextSplitter(chunk_size=10, overlap_size=0, separator=" ")
        
        # Override configuration
        override_config = {
            'chunk_size': 2,
            'overlap_size': 1,
            'separator': ' '
        }
        
        content = "one two three four five"
        doc = Document(id="config_test", content=content, metadata={"source": "test"})
        
        chunks = list(splitter.process([doc], config=override_config))
        
        # Should use override config (chunk_size=2)
        assert len(chunks) > 2  # More chunks due to smaller chunk size
        
        # Each chunk should have at most 2 tokens
        for chunk in chunks:
            tokens = chunk.content.split(" ")
            assert len(tokens) <= 2

    def test_process_different_separators(self):
        """
        Test processing with different token separators
        異なるトークンセパレータでの処理テスト
        """
        test_cases = [
            (",", "apple,banana,cherry,date", 2),
            ("|", "one|two|three|four|five", 3),
            ("\t", "tab\tseparated\tvalues\there", 2),
            ("-", "dash-separated-content-here", 2)
        ]
        
        for separator, content, chunk_size in test_cases:
            splitter = TokenTextSplitter(chunk_size=chunk_size, overlap_size=0, separator=separator)
            doc = Document(id=f"sep_test_{separator}", content=content, metadata={"source": "test"})
            
            chunks = list(splitter.process([doc]))
            
            # Should produce chunks
            assert len(chunks) >= 1
            
            # Each chunk should use the correct separator
            for chunk in chunks:
                if separator in chunk.content:
                    tokens = chunk.content.split(separator)
                    assert len(tokens) <= chunk_size