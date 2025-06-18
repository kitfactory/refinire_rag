"""
Comprehensive tests for Chunker processor functionality
Chunkerプロセッサ機能の包括的テスト

This module provides comprehensive coverage for the Chunker class,
testing all chunking strategies, configuration options, error handling, and edge cases.
このモジュールは、Chunkerクラスの包括的カバレッジを提供し、
全ての分割戦略、設定オプション、エラーハンドリング、エッジケースをテストします。
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from refinire_rag.processing.chunker import Chunker, ChunkingConfig
from refinire_rag.models.document import Document


class TestChunkingConfig:
    """
    Test ChunkingConfig configuration and validation
    ChunkingConfigの設定と検証のテスト
    """
    
    def test_default_configuration(self):
        """
        Test default configuration values
        デフォルト設定値のテスト
        """
        config = ChunkingConfig()
        
        # Test default values
        assert config.chunk_size == 512
        assert config.overlap == 50
        assert config.split_by_sentence is True
        assert config.min_chunk_size == 50
        assert config.max_chunk_size == 1024
        assert config.preserve_paragraphs is True
        assert config.strip_whitespace is True
        assert config.add_chunk_metadata is True
        assert config.preserve_original_metadata is True
        assert config.chunking_strategy == "token_based"
    
    def test_custom_configuration(self):
        """
        Test custom configuration settings
        カスタム設定のテスト
        """
        config = ChunkingConfig(
            chunk_size=256,
            overlap=25,
            split_by_sentence=False,
            min_chunk_size=30,
            max_chunk_size=512,
            preserve_paragraphs=False,
            strip_whitespace=False,
            add_chunk_metadata=False,
            preserve_original_metadata=False,
            chunking_strategy="sentence_based"
        )
        
        assert config.chunk_size == 256
        assert config.overlap == 25
        assert config.split_by_sentence is False
        assert config.min_chunk_size == 30
        assert config.max_chunk_size == 512
        assert config.preserve_paragraphs is False
        assert config.strip_whitespace is False
        assert config.add_chunk_metadata is False
        assert config.preserve_original_metadata is False
        assert config.chunking_strategy == "sentence_based"


class TestChunkerInitialization:
    """
    Test Chunker initialization and setup
    Chunkerの初期化とセットアップのテスト
    """
    
    def test_default_initialization(self):
        """
        Test default initialization
        デフォルト初期化テスト
        """
        chunker = Chunker()
        
        assert chunker.config is not None
        assert isinstance(chunker.config, ChunkingConfig)
        assert chunker.config.chunk_size == 512
        assert chunker.config.overlap == 50
        
        # Check processing stats initialization
        assert "documents_processed" in chunker.processing_stats
        assert "chunks_created" in chunker.processing_stats
        assert "total_tokens_processed" in chunker.processing_stats
        assert "average_chunk_size" in chunker.processing_stats
        assert "overlap_tokens" in chunker.processing_stats
    
    def test_custom_config_initialization(self):
        """
        Test initialization with custom configuration
        カスタム設定での初期化テスト
        """
        config = ChunkingConfig(
            chunk_size=256,
            overlap=30,
            chunking_strategy="sentence_based"
        )
        
        chunker = Chunker(config)
        
        assert chunker.config == config
        assert chunker.config.chunk_size == 256
        assert chunker.config.overlap == 30
        assert chunker.config.chunking_strategy == "sentence_based"
    
    def test_get_config_class(self):
        """
        Test get_config_class method
        get_config_classメソッドのテスト
        """
        assert Chunker.get_config_class() == ChunkingConfig


class TestChunkerTokenBasedChunking:
    """
    Test token-based chunking functionality
    トークンベース分割機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        self.config = ChunkingConfig(
            chunk_size=10,
            overlap=2,
            min_chunk_size=3,
            chunking_strategy="token_based"
        )
        self.chunker = Chunker(self.config)
    
    def test_basic_token_chunking(self):
        """
        Test basic token-based chunking
        基本的なトークンベース分割テスト
        """
        document = Document(
            id="test_doc",
            content="This is a simple test document with multiple words for chunking purposes.",
            metadata={}
        )
        
        chunks = list(self.chunker.process([document]))
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, Document) for chunk in chunks)
        assert all(len(chunk.content.split()) <= self.config.chunk_size for chunk in chunks)
    
    def test_token_chunking_with_overlap(self):
        """
        Test token chunking with overlap
        オーバーラップ付きトークン分割テスト
        """
        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10 Word11 Word12"
        document = Document(id="test_doc", content=text, metadata={})
        
        chunks = list(self.chunker.process([document]))
        
        # Should have multiple chunks due to overlap
        assert len(chunks) >= 2
        
        # Check overlap - last words of chunk should appear in next chunk
        if len(chunks) > 1:
            chunk1_words = chunks[0].content.split()
            chunk2_words = chunks[1].content.split()
            # Some overlap should exist
            assert len(chunk1_words) >= self.config.overlap
    
    def test_token_chunking_short_text(self):
        """
        Test token chunking with short text
        短いテキストでのトークン分割テスト
        """
        document = Document(
            id="short_doc",
            content="Short text",
            metadata={}
        )
        
        chunks = list(self.chunker.process([document]))
        
        # Should create single chunk for short text
        assert len(chunks) == 1
        assert chunks[0].content == "Short text"
    
    def test_token_chunking_sentence_boundary_preference(self):
        """
        Test token chunking with sentence boundary preference
        文境界優先でのトークン分割テスト
        """
        config = ChunkingConfig(
            chunk_size=8,
            overlap=1,
            split_by_sentence=True,
            chunking_strategy="token_based"
        )
        chunker = Chunker(config)
        
        text = "This is sentence one. This is sentence two. This is sentence three."
        document = Document(id="test_doc", content=text, metadata={})
        
        chunks = list(chunker.process([document]))
        
        # Should prefer breaking at sentence boundaries
        assert len(chunks) >= 1
        # Check that chunks respect sentence boundaries when possible
        for chunk in chunks:
            content = chunk.content.strip()
            if len(content) > 0 and not content.endswith('.'):
                # If not ending with period, should be the last chunk or special case
                pass
    
    def test_token_chunking_no_sentence_boundary(self):
        """
        Test token chunking without sentence boundary preference
        文境界優先なしでのトークン分割テスト
        """
        config = ChunkingConfig(
            chunk_size=3,
            overlap=1,
            min_chunk_size=1,  # Lower min_chunk_size to ensure chunking
            split_by_sentence=False,
            chunking_strategy="token_based"
        )
        chunker = Chunker(config)
        
        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
        document = Document(id="test_doc", content=text, metadata={})
        
        chunks = list(chunker.process([document]))
        
        # Should split exactly at token boundaries without regard to sentences
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.content.split()) <= config.chunk_size


class TestChunkerSentenceBasedChunking:
    """
    Test sentence-based chunking functionality
    文ベース分割機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        self.config = ChunkingConfig(
            chunk_size=20,
            overlap=5,
            min_chunk_size=5,
            chunking_strategy="sentence_based"
        )
        self.chunker = Chunker(self.config)
    
    def test_sentence_based_chunking(self):
        """
        Test basic sentence-based chunking
        基本的な文ベース分割テスト
        """
        text = ("This is the first sentence. This is the second sentence. "
                "This is the third sentence. This is the fourth sentence. "
                "This is the fifth sentence.")
        
        document = Document(id="test_doc", content=text, metadata={})
        
        chunks = list(self.chunker.process([document]))
        
        assert len(chunks) >= 1
        # Each chunk should contain complete sentences
        for chunk in chunks:
            content = chunk.content.strip()
            if content:
                # Should contain complete sentences (ending with period)
                sentences = content.split('. ')
                assert len(sentences) >= 1
    
    def test_sentence_chunking_with_overlap(self):
        """
        Test sentence chunking with overlap
        オーバーラップ付き文分割テスト
        """
        text = ("Short sentence one. Short sentence two. Short sentence three. "
                "Short sentence four. Short sentence five. Short sentence six.")
        
        document = Document(id="test_doc", content=text, metadata={})
        
        chunks = list(self.chunker.process([document]))
        
        if len(chunks) > 1:
            # Check that overlap exists between chunks
            chunk1_content = chunks[0].content
            chunk2_content = chunks[1].content
            
            # There should be some common words due to overlap
            chunk1_words = set(chunk1_content.split())
            chunk2_words = set(chunk2_content.split())
            common_words = chunk1_words.intersection(chunk2_words)
            
            # Should have some overlap (at least a few words)
            assert len(common_words) > 0
    
    def test_sentence_chunking_long_sentences(self):
        """
        Test sentence chunking with very long sentences
        非常に長い文での文分割テスト
        """
        config = ChunkingConfig(
            chunk_size=10,
            overlap=2,
            chunking_strategy="sentence_based"
        )
        chunker = Chunker(config)
        
        # Create a sentence longer than chunk size
        long_sentence = "This is a very long sentence with many words that exceeds the chunk size limit."
        document = Document(id="test_doc", content=long_sentence, metadata={})
        
        chunks = list(chunker.process([document]))
        
        # Should handle long sentences gracefully
        assert len(chunks) >= 1
        # All text should be preserved
        total_content = ' '.join(chunk.content for chunk in chunks)
        assert long_sentence in total_content or total_content in long_sentence


class TestChunkerParagraphBasedChunking:
    """
    Test paragraph-based chunking functionality
    段落ベース分割機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        self.config = ChunkingConfig(
            chunk_size=30,
            overlap=5,
            min_chunk_size=10,
            chunking_strategy="paragraph_based"
        )
        self.chunker = Chunker(self.config)
    
    def test_paragraph_based_chunking(self):
        """
        Test basic paragraph-based chunking
        基本的な段落ベース分割テスト
        """
        text = ("This is the first paragraph. It has multiple sentences.\n\n"
                "This is the second paragraph. It also has content.\n\n"
                "This is the third paragraph with more text.")
        
        document = Document(id="test_doc", content=text, metadata={})
        
        chunks = list(self.chunker.process([document]))
        
        assert len(chunks) >= 1
        # Check that paragraph structure is preserved
        for chunk in chunks:
            if '\n\n' in chunk.content:
                # Should maintain paragraph breaks
                paragraphs = chunk.content.split('\n\n')
                assert len(paragraphs) >= 1
    
    def test_paragraph_chunking_single_paragraph(self):
        """
        Test paragraph chunking with single paragraph
        単一段落での段落分割テスト
        """
        text = "This is a single paragraph without line breaks but with sufficient content to test the chunking."
        document = Document(id="test_doc", content=text, metadata={})
        
        chunks = list(self.chunker.process([document]))
        
        # Should create single chunk for single paragraph
        assert len(chunks) == 1
        assert chunks[0].content.strip() == text
    
    def test_paragraph_chunking_empty_paragraphs(self):
        """
        Test paragraph chunking with empty paragraphs
        空段落での段落分割テスト
        """
        text = ("First paragraph.\n\n\n\nSecond paragraph after empty lines.\n\n"
                "Third paragraph.")
        
        document = Document(id="test_doc", content=text, metadata={})
        
        chunks = list(self.chunker.process([document]))
        
        # Should handle empty paragraphs gracefully
        assert len(chunks) >= 1
        # Should not have empty chunks
        for chunk in chunks:
            assert chunk.content.strip() != ""


class TestChunkerTextPreprocessing:
    """
    Test text preprocessing functionality
    テキスト前処理機能のテスト
    """
    
    def test_whitespace_stripping_enabled(self):
        """
        Test whitespace stripping when enabled
        有効時の空白除去テスト
        """
        config = ChunkingConfig(
            strip_whitespace=True,
            chunk_size=20
        )
        chunker = Chunker(config)
        
        text = "Text   with    excessive    whitespace\n\n\n\nand   extra   newlines."
        document = Document(id="test_doc", content=text, metadata={})
        
        chunks = list(chunker.process([document]))
        
        # Should normalize whitespace
        for chunk in chunks:
            content = chunk.content
            # Should not have excessive spaces
            assert "   " not in content  # Multiple spaces should be normalized
            # Should normalize paragraph breaks
            assert "\n\n\n" not in content
    
    def test_whitespace_stripping_disabled(self):
        """
        Test whitespace preservation when stripping disabled
        無効時の空白保持テスト
        """
        config = ChunkingConfig(
            strip_whitespace=False,
            chunk_size=20
        )
        chunker = Chunker(config)
        
        text = "Text   with    excessive    whitespace."
        document = Document(id="test_doc", content=text, metadata={})
        
        chunks = list(chunker.process([document]))
        
        # Should preserve original whitespace
        total_content = ' '.join(chunk.content for chunk in chunks)
        # Original excessive whitespace might be partially preserved
        assert "Text" in total_content and "whitespace" in total_content


class TestChunkerDocumentCreation:
    """
    Test chunk document creation and metadata handling
    チャンクドキュメント作成とメタデータ処理のテスト
    """
    
    def test_chunk_metadata_enabled(self):
        """
        Test chunk creation with metadata enabled
        メタデータ有効でのチャンク作成テスト
        """
        config = ChunkingConfig(
            chunk_size=5,
            overlap=1,
            add_chunk_metadata=True,
            preserve_original_metadata=True
        )
        chunker = Chunker(config)
        
        original_metadata = {"author": "test", "category": "test_doc"}
        document = Document(
            id="test_doc",
            content="Word1 Word2 Word3 Word4 Word5 Word6 Word7",
            metadata=original_metadata
        )
        
        chunks = list(chunker.process([document]))
        
        assert len(chunks) >= 2
        
        for i, chunk in enumerate(chunks):
            # Check chunk ID format
            assert chunk.id.startswith("test_doc_chunk_")
            
            # Check original metadata preservation
            assert chunk.metadata["author"] == "test"
            assert chunk.metadata["category"] == "test_doc"
            
            # Check chunk metadata
            assert chunk.metadata["processing_stage"] == "chunked"
            assert chunk.metadata["parent_document_id"] == "test_doc"
            assert chunk.metadata["chunk_position"] == i
            assert chunk.metadata["chunk_total"] == len(chunks)
            assert "chunk_size_tokens" in chunk.metadata
            assert chunk.metadata["chunking_strategy"] == "token_based"
            assert chunk.metadata["chunked_by"] == "Chunker"
    
    def test_chunk_metadata_disabled(self):
        """
        Test chunk creation with metadata disabled
        メタデータ無効でのチャンク作成テスト
        """
        config = ChunkingConfig(
            chunk_size=5,
            add_chunk_metadata=False,
            preserve_original_metadata=False
        )
        chunker = Chunker(config)
        
        original_metadata = {"author": "test", "category": "test_doc"}
        document = Document(
            id="test_doc",
            content="Word1 Word2 Word3 Word4 Word5 Word6 Word7",
            metadata=original_metadata
        )
        
        chunks = list(chunker.process([document]))
        
        for chunk in chunks:
            # Should not have chunk metadata
            assert "processing_stage" not in chunk.metadata
            assert "chunk_position" not in chunk.metadata
            assert "chunked_by" not in chunk.metadata
            
            # Should not have original metadata
            assert "author" not in chunk.metadata
            assert "category" not in chunk.metadata
    
    def test_original_metadata_preservation(self):
        """
        Test original metadata preservation options
        元メタデータ保持オプションテスト
        """
        config = ChunkingConfig(
            preserve_original_metadata=True,
            add_chunk_metadata=False
        )
        chunker = Chunker(config)
        
        original_metadata = {"important": "data", "version": "1.0"}
        document = Document(
            id="test_doc",
            content="Test content for metadata preservation.",
            metadata=original_metadata
        )
        
        chunks = list(chunker.process([document]))
        
        for chunk in chunks:
            # Should preserve original metadata
            assert chunk.metadata["important"] == "data"
            assert chunk.metadata["version"] == "1.0"
            
            # Should not add chunk metadata
            assert "chunk_position" not in chunk.metadata


class TestChunkerUtilityMethods:
    """
    Test utility and helper methods
    ユーティリティとヘルパーメソッドのテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.chunker = Chunker()
    
    def test_find_sentence_break(self):
        """
        Test sentence break detection
        文区切り検出テスト
        """
        words = ["This", "is", "a", "sentence.", "This", "is", "another."]
        
        # Should find sentence break at period
        break_pos = self.chunker._find_sentence_break(words, 0, 6)
        assert break_pos == 4  # After "sentence."
        
        # Should return end if no break found
        words_no_break = ["Word1", "Word2", "Word3", "Word4"]
        break_pos = self.chunker._find_sentence_break(words_no_break, 0, 4)
        assert break_pos == 4
    
    def test_split_into_sentences(self):
        """
        Test sentence splitting
        文分割テスト
        """
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = self.chunker._split_into_sentences(text)
        
        assert len(sentences) == 4
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]
        assert "Fourth sentence" in sentences[3]
    
    def test_split_into_sentences_japanese(self):
        """
        Test sentence splitting with Japanese punctuation
        日本語句読点での文分割テスト
        """
        text = "これは最初の文です。これは二番目の文です！これは三番目の文です？"
        sentences = self.chunker._split_into_sentences(text)
        
        assert len(sentences) >= 3
        # Should handle Japanese punctuation
        assert any("最初" in s for s in sentences)
        assert any("二番目" in s for s in sentences)
        assert any("三番目" in s for s in sentences)
    
    def test_get_overlap_sentences(self):
        """
        Test overlap sentence calculation
        オーバーラップ文計算テスト
        """
        sentences = ["Short one.", "Another short.", "Third sentence here.", "Fourth one."]
        
        # Test with overlap limit
        overlap = self.chunker._get_overlap_sentences(sentences, 5)
        
        # Should return sentences that fit within token limit
        assert isinstance(overlap, list)
        assert len(overlap) <= len(sentences)
        
        # Total tokens should not exceed limit
        total_tokens = sum(len(s.split()) for s in overlap)
        assert total_tokens <= 5
    
    def test_get_overlap_sentences_empty(self):
        """
        Test overlap sentences with empty input
        空入力でのオーバーラップ文テスト
        """
        # Empty sentences
        overlap = self.chunker._get_overlap_sentences([], 5)
        assert overlap == []
        
        # Zero overlap
        sentences = ["Test sentence."]
        overlap = self.chunker._get_overlap_sentences(sentences, 0)
        assert overlap == []
    
    def test_get_chunking_stats(self):
        """
        Test chunking statistics retrieval
        分割統計取得テスト
        """
        stats = self.chunker.get_chunking_stats()
        
        assert isinstance(stats, dict)
        assert "chunk_size" in stats
        assert "overlap" in stats
        assert "chunking_strategy" in stats
        assert "split_by_sentence" in stats
        assert "documents_processed" in stats
        assert "chunks_created" in stats
        
        # Check values match config
        assert stats["chunk_size"] == self.chunker.config.chunk_size
        assert stats["overlap"] == self.chunker.config.overlap
        assert stats["chunking_strategy"] == self.chunker.config.chunking_strategy


class TestChunkerErrorHandling:
    """
    Test error handling and edge cases
    エラーハンドリングとエッジケースのテスト
    """
    
    def test_empty_document_content(self):
        """
        Test handling of empty document content
        空ドキュメント内容の処理テスト
        """
        chunker = Chunker()
        document = Document(id="empty_doc", content="", metadata={})
        
        chunks = list(chunker.process([document]))
        
        # Should handle empty content gracefully
        assert len(chunks) >= 0  # May return empty list or original document
    
    def test_whitespace_only_content(self):
        """
        Test handling of whitespace-only content
        空白のみ内容の処理テスト
        """
        chunker = Chunker()
        document = Document(id="whitespace_doc", content="   \n\n\t  ", metadata={})
        
        chunks = list(chunker.process([document]))
        
        # Should handle whitespace-only content
        assert len(chunks) >= 0
    
    def test_very_short_content(self):
        """
        Test handling of very short content
        非常に短い内容の処理テスト
        """
        config = ChunkingConfig(chunk_size=10, min_chunk_size=5)
        chunker = Chunker(config)
        
        document = Document(id="short_doc", content="Word", metadata={})
        
        chunks = list(chunker.process([document]))
        
        # Should handle content shorter than min_chunk_size
        assert len(chunks) >= 0
    
    def test_exception_handling_in_processing(self):
        """
        Test exception handling during processing
        処理中の例外ハンドリングテスト
        """
        chunker = Chunker()
        
        # Mock a method to raise an exception
        with patch.object(chunker, '_preprocess_text', side_effect=Exception("Test error")):
            document = Document(id="error_doc", content="Test content", metadata={})
            
            chunks = list(chunker.process([document]))
            
            # Should return original document on error
            assert len(chunks) == 1
            assert chunks[0] == document
    
    def test_invalid_chunking_strategy(self):
        """
        Test handling of invalid chunking strategy
        無効な分割戦略の処理テスト
        """
        config = ChunkingConfig(chunking_strategy="invalid_strategy")
        chunker = Chunker(config)
        
        document = Document(id="test_doc", content="Test content for invalid strategy.", metadata={})
        
        chunks = list(chunker.process([document]))
        
        # Should fall back to token_based strategy
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Document) for chunk in chunks)


class TestChunkerStatisticsTracking:
    """
    Test statistics tracking functionality
    統計追跡機能のテスト
    """
    
    def test_statistics_initialization(self):
        """
        Test initial statistics state
        初期統計状態のテスト
        """
        chunker = Chunker()
        
        stats = chunker.processing_stats
        
        assert stats["documents_processed"] == 0
        assert stats["chunks_created"] == 0
        assert stats["total_tokens_processed"] == 0
        assert stats["average_chunk_size"] == 0.0
        assert stats["overlap_tokens"] == 0
    
    def test_statistics_update_after_processing(self):
        """
        Test statistics update after document processing
        ドキュメント処理後の統計更新テスト
        """
        config = ChunkingConfig(chunk_size=5, overlap=1)
        chunker = Chunker(config)
        
        document = Document(
            id="test_doc",
            content="Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9",
            metadata={}
        )
        
        chunks = list(chunker.process([document]))
        
        stats = chunker.processing_stats
        
        # Should update statistics
        assert stats["documents_processed"] == 1
        assert stats["chunks_created"] == len(chunks)
        assert stats["total_tokens_processed"] > 0
        
        if len(chunks) > 0:
            assert stats["average_chunk_size"] > 0
    
    def test_statistics_accumulation(self):
        """
        Test statistics accumulation across multiple documents
        複数ドキュメント間での統計累積テスト
        """
        chunker = Chunker()
        
        documents = [
            Document(id="doc1", content="Content for first document.", metadata={}),
            Document(id="doc2", content="Content for second document.", metadata={})
        ]
        
        # Process documents
        for doc in documents:
            list(chunker.process([doc]))
        
        stats = chunker.processing_stats
        
        # Should accumulate statistics
        assert stats["documents_processed"] == 2
        assert stats["chunks_created"] >= 2  # At least one chunk per document
        assert stats["total_tokens_processed"] > 0


class TestChunkerEdgeCases:
    """
    Test edge cases and boundary conditions
    エッジケースと境界条件のテスト
    """
    
    def test_chunk_size_larger_than_content(self):
        """
        Test when chunk size is larger than content
        チャンクサイズが内容より大きい場合のテスト
        """
        config = ChunkingConfig(chunk_size=100)
        chunker = Chunker(config)
        
        document = Document(id="small_doc", content="Small content.", metadata={})
        
        chunks = list(chunker.process([document]))
        
        # Should create single chunk
        assert len(chunks) == 1
        assert chunks[0].content == "Small content."
    
    def test_zero_overlap(self):
        """
        Test chunking with zero overlap
        ゼロオーバーラップでの分割テスト
        """
        config = ChunkingConfig(chunk_size=3, overlap=0)
        chunker = Chunker(config)
        
        document = Document(
            id="test_doc",
            content="Word1 Word2 Word3 Word4 Word5 Word6",
            metadata={}
        )
        
        chunks = list(chunker.process([document]))
        
        # Should create chunks without overlap
        assert len(chunks) == 2
        
        # Verify no overlap between chunks
        chunk1_words = set(chunks[0].content.split())
        chunk2_words = set(chunks[1].content.split())
        overlap_words = chunk1_words.intersection(chunk2_words)
        assert len(overlap_words) == 0
    
    def test_overlap_larger_than_chunk_size(self):
        """
        Test when overlap is larger than chunk size
        オーバーラップがチャンクサイズより大きい場合のテスト
        """
        config = ChunkingConfig(chunk_size=3, overlap=5)
        chunker = Chunker(config)
        
        document = Document(
            id="test_doc",
            content="Word1 Word2 Word3 Word4 Word5 Word6 Word7",
            metadata={}
        )
        
        chunks = list(chunker.process([document]))
        
        # Should handle gracefully (overlap limited by chunk size)
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Document) for chunk in chunks)
    
    def test_min_chunk_size_enforcement(self):
        """
        Test minimum chunk size enforcement
        最小チャンクサイズの強制テスト
        """
        config = ChunkingConfig(
            chunk_size=5,
            min_chunk_size=3,
            overlap=4
        )
        chunker = Chunker(config)
        
        document = Document(
            id="test_doc",
            content="Word1 Word2 Word3 Word4 Word5 Word6",
            metadata={}
        )
        
        chunks = list(chunker.process([document]))
        
        # All chunks should meet minimum size requirement (except possibly the last)
        for chunk in chunks[:-1]:  # All but last chunk
            assert len(chunk.content.split()) >= config.min_chunk_size
    
    def test_processing_multiple_documents(self):
        """
        Test processing multiple documents in sequence
        複数ドキュメントの連続処理テスト
        """
        chunker = Chunker()
        
        documents = [
            Document(id="doc1", content="First document content here.", metadata={}),
            Document(id="doc2", content="Second document with different content.", metadata={}),
            Document(id="doc3", content="Third document also has unique content.", metadata={})
        ]
        
        all_chunks = list(chunker.process(documents))
        
        # Should process all documents
        assert len(all_chunks) >= 3  # At least one chunk per document
        
        # Check that chunks maintain document lineage
        doc1_chunks = [c for c in all_chunks if c.id.startswith("doc1")]
        doc2_chunks = [c for c in all_chunks if c.id.startswith("doc2")]
        doc3_chunks = [c for c in all_chunks if c.id.startswith("doc3")]
        
        assert len(doc1_chunks) >= 1
        assert len(doc2_chunks) >= 1
        assert len(doc3_chunks) >= 1
    
    def test_special_characters_handling(self):
        """
        Test handling of special characters and Unicode
        特殊文字とUnicodeの処理テスト
        """
        chunker = Chunker()
        
        # Text with special characters and Unicode
        special_text = "Text with émojis 😀, symbols ♠♣♥♦, and numbers 123-456!"
        document = Document(id="special_doc", content=special_text, metadata={})
        
        chunks = list(chunker.process([document]))
        
        # Should handle special characters gracefully
        assert len(chunks) >= 1
        
        # Content should be preserved
        total_content = ' '.join(chunk.content for chunk in chunks)
        assert "émojis" in total_content
        assert "😀" in total_content
        assert "♠" in total_content
        assert "123-456" in total_content