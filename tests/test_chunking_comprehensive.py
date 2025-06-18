"""
Comprehensive tests for Chunking functionality
Chunking機能の包括的テスト

This module provides comprehensive coverage for the Chunker base class and TokenBasedChunker,
testing all configuration options, chunking strategies, error handling, and edge cases.
このモジュールは、ChunkerベースクラスとTokenBasedChunkerの包括的カバレッジを提供し、
全ての設定オプション、チャンキング戦略、エラーハンドリング、エッジケースをテストします。
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List
from datetime import datetime

from refinire_rag.chunking.chunker import Chunker, ChunkingConfig
from refinire_rag.chunking.token_chunker import TokenBasedChunker
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
        assert config.preserve_formatting is False
        assert config.min_chunk_size == 10
        assert config.max_chunk_size == 1024
    
    def test_custom_configuration(self):
        """
        Test custom configuration settings
        カスタム設定のテスト
        """
        config = ChunkingConfig(
            chunk_size=256,
            overlap=25,
            split_by_sentence=False,
            preserve_formatting=True,
            min_chunk_size=5,
            max_chunk_size=2048
        )
        
        assert config.chunk_size == 256
        assert config.overlap == 25
        assert config.split_by_sentence is False
        assert config.preserve_formatting is True
        assert config.min_chunk_size == 5
        assert config.max_chunk_size == 2048
    
    def test_config_to_dict(self):
        """
        Test configuration serialization to dict
        辞書への設定シリアライゼーションテスト
        """
        config = ChunkingConfig(
            chunk_size=128,
            overlap=20,
            split_by_sentence=True
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["chunk_size"] == 128
        assert config_dict["overlap"] == 20
        assert config_dict["split_by_sentence"] is True
        # Should include all fields
        assert "min_chunk_size" in config_dict
        assert "max_chunk_size" in config_dict
        assert "preserve_formatting" in config_dict


class ConcreteChunker(Chunker):
    """
    Concrete implementation of Chunker for testing
    テスト用のChunkerの具象実装
    """
    
    def chunk(self, document: Document, config: ChunkingConfig) -> List[Document]:
        """Test implementation of chunk method"""
        content = document.content
        chunk_size = config.chunk_size
        
        # Simple character-based chunking for testing
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i + chunk_size]
            chunk_doc = Document(
                id=self._generate_chunk_id(document.id, len(chunks)),
                content=chunk_content,
                metadata=document.metadata.copy()
            )
            chunks.append(chunk_doc)
        
        return chunks
    
    def estimate_tokens(self, text: str) -> int:
        """Test implementation of estimate_tokens"""
        # Simple estimation: split by whitespace
        return len(text.split()) if text.strip() else 0


class TestChunkerBase:
    """
    Test Chunker base class functionality
    Chunkerベースクラス機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        self.config = ChunkingConfig(
            chunk_size=100,
            overlap=20,
            min_chunk_size=5,
            max_chunk_size=500
        )
        
        self.chunker = ConcreteChunker(config=self.config)
        
        self.test_document = Document(
            id="test_doc",
            content="This is a test document with multiple sentences. It should be chunked properly.",
            metadata={"type": "test", "author": "tester"}
        )
    
    def test_initialization_with_config(self):
        """
        Test chunker initialization with custom config
        カスタム設定でのチャンカー初期化テスト
        """
        config = ChunkingConfig(chunk_size=256, overlap=30)
        chunker = ConcreteChunker(config=config)
        
        assert chunker.config == config
        assert chunker.config.chunk_size == 256
        assert chunker.config.overlap == 30
    
    def test_initialization_without_config(self):
        """
        Test chunker initialization without config (uses defaults)
        設定なしでのチャンカー初期化テスト（デフォルト使用）
        """
        chunker = ConcreteChunker()
        
        assert isinstance(chunker.config, ChunkingConfig)
        assert chunker.config.chunk_size == 512  # Default value
    
    def test_get_config_class(self):
        """
        Test get_config_class method
        get_config_classメソッドテスト
        """
        config_class = ConcreteChunker.get_config_class()
        
        assert config_class == ChunkingConfig
    
    def test_process_document_basic(self):
        """
        Test basic document processing
        基本的な文書処理テスト
        """
        chunks = self.chunker.process(self.test_document)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Check that all chunks are Document objects
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert chunk.id.startswith(self.test_document.id)
            assert chunk.content
    
    def test_process_adds_metadata(self):
        """
        Test that process adds proper chunking metadata
        processが適切なチャンキングメタデータを追加することのテスト
        """
        chunks = self.chunker.process(self.test_document)
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata
            
            # Check required metadata fields
            assert "original_document_id" in metadata
            assert "parent_document_id" in metadata
            assert "processing_stage" in metadata
            assert "chunk_position" in metadata
            assert "chunk_total" in metadata
            assert "chunking_method" in metadata
            assert "chunk_config" in metadata
            assert "chunked_at" in metadata
            assert "token_count" in metadata
            
            # Check values
            assert metadata["original_document_id"] == self.test_document.id
            assert metadata["parent_document_id"] == self.test_document.id
            assert metadata["processing_stage"] == "chunked"
            assert metadata["chunk_position"] == i
            assert metadata["chunk_total"] == len(chunks)
            assert metadata["chunking_method"] == "ConcreteChunker"
            assert isinstance(metadata["chunk_config"], dict)
            assert metadata["token_count"] >= 0
    
    def test_process_with_config_override(self):
        """
        Test process with configuration override
        設定上書きでのprocessテスト
        """
        override_config = ChunkingConfig(chunk_size=50, overlap=10)
        
        chunks = self.chunker.process(self.test_document, config=override_config)
        
        # Check that override config was used in metadata
        for chunk in chunks:
            config_dict = chunk.metadata["chunk_config"]
            assert config_dict["chunk_size"] == 50
            assert config_dict["overlap"] == 10
    
    def test_generate_chunk_id(self):
        """
        Test chunk ID generation
        チャンクID生成テスト
        """
        chunk_id = self.chunker._generate_chunk_id("parent_doc", 5)
        
        assert chunk_id == "parent_doc_chunk_5"
    
    def test_validate_chunk_size_valid(self):
        """
        Test chunk size validation with valid chunk
        有効なチャンクでのチャンクサイズ検証テスト
        """
        valid_chunk = "This is a valid chunk with appropriate length."
        
        is_valid = self.chunker._validate_chunk_size(valid_chunk, self.config)
        
        assert is_valid is True
    
    def test_validate_chunk_size_too_small(self):
        """
        Test chunk size validation with too small chunk
        小さすぎるチャンクでのチャンクサイズ検証テスト
        """
        small_chunk = "Too small"
        
        with patch.object(self.chunker, 'estimate_tokens', return_value=2):
            is_valid = self.chunker._validate_chunk_size(small_chunk, self.config)
            
            assert is_valid is False
    
    def test_validate_chunk_size_too_large(self):
        """
        Test chunk size validation with too large chunk
        大きすぎるチャンクでのチャンクサイズ検証テスト
        """
        large_chunk = "This is a very large chunk " * 100
        
        with patch.object(self.chunker, 'estimate_tokens', return_value=1000):
            is_valid = self.chunker._validate_chunk_size(large_chunk, self.config)
            
            assert is_valid is False
    
    def test_get_chunking_stats(self):
        """
        Test chunking statistics retrieval
        チャンキング統計取得テスト
        """
        stats = self.chunker.get_chunking_stats()
        
        assert isinstance(stats, dict)
        assert "chunker_class" in stats
        assert "config" in stats
        assert "chunk_size_target" in stats
        assert "overlap_size" in stats
        
        assert stats["chunker_class"] == "ConcreteChunker"
        assert stats["chunk_size_target"] == self.config.chunk_size
        assert stats["overlap_size"] == self.config.overlap
    
    def test_process_preserves_original_metadata(self):
        """
        Test that processing preserves original document metadata
        処理が元の文書メタデータを保持することのテスト
        """
        chunks = self.chunker.process(self.test_document)
        
        for chunk in chunks:
            # Original metadata should be preserved
            assert chunk.metadata["type"] == "test"
            assert chunk.metadata["author"] == "tester"
    
    def test_process_with_overlap_metadata(self):
        """
        Test overlap metadata addition
        重複メタデータ追加テスト
        """
        config_with_overlap = ChunkingConfig(chunk_size=20, overlap=5)
        
        chunks = self.chunker.process(self.test_document, config=config_with_overlap)
        
        # Second chunk and beyond should have overlap metadata
        if len(chunks) > 1:
            for i, chunk in enumerate(chunks[1:], 1):
                assert "overlap_previous" in chunk.metadata
                assert chunk.metadata["overlap_previous"] == 5


class TestTokenBasedChunker:
    """
    Test TokenBasedChunker functionality
    TokenBasedChunker機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        self.config = ChunkingConfig(
            chunk_size=10,  # 10 tokens per chunk
            overlap=2,
            min_chunk_size=2,
            max_chunk_size=50
        )
        
        self.chunker = TokenBasedChunker(config=self.config)
        
        self.test_document = Document(
            id="token_test_doc",
            content="This is a comprehensive test document with many words that should be chunked into multiple token-based chunks for testing purposes.",
            metadata={"type": "token_test"}
        )
    
    def test_initialization(self):
        """
        Test TokenBasedChunker initialization
        TokenBasedChunker初期化テスト
        """
        chunker = TokenBasedChunker()
        
        assert isinstance(chunker.config, ChunkingConfig)
        assert hasattr(chunker, 'token_pattern')
        assert chunker.token_pattern is not None
    
    def test_tokenize_basic(self):
        """
        Test basic tokenization
        基本的なトークン化テスト
        """
        text = "Hello world, this is a test."
        tokens = self.chunker._tokenize(text)
        
        expected_tokens = ["Hello", "world", "this", "is", "a", "test"]
        assert tokens == expected_tokens
    
    def test_estimate_tokens(self):
        """
        Test token estimation
        トークン推定テスト
        """
        text = "This has exactly five tokens."
        token_count = self.chunker.estimate_tokens(text)
        
        assert token_count == 5
    
    def test_estimate_tokens_empty(self):
        """
        Test token estimation with empty text
        空テキストでのトークン推定テスト
        """
        assert self.chunker.estimate_tokens("") == 0
        assert self.chunker.estimate_tokens("   ") == 0
    
    def test_chunk_document_basic(self):
        """
        Test basic document chunking
        基本的な文書チャンキングテスト
        """
        chunks = self.chunker.chunk(self.test_document, self.config)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1  # Should create multiple chunks
        
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert chunk.content.strip()
            assert chunk.id.startswith("token_test_doc_chunk_")
    
    def test_chunk_small_document(self):
        """
        Test chunking of small document (single chunk)
        小さい文書のチャンキングテスト（単一チャンク）
        """
        small_doc = Document(
            id="small_doc",
            content="Small document with few words.",
            metadata={}
        )
        
        chunks = self.chunker.chunk(small_doc, self.config)
        
        assert len(chunks) == 1
        assert chunks[0].content == small_doc.content
        assert chunks[0].id == "small_doc_chunk_0"
    
    def test_chunk_empty_document(self):
        """
        Test chunking of empty document
        空文書のチャンキングテスト
        """
        empty_doc = Document(
            id="empty_doc",
            content="",
            metadata={}
        )
        
        chunks = self.chunker.chunk(empty_doc, self.config)
        
        assert chunks == []
    
    def test_chunk_whitespace_document(self):
        """
        Test chunking of whitespace-only document
        空白のみの文書のチャンキングテスト
        """
        whitespace_doc = Document(
            id="whitespace_doc",
            content="   \n\t  ",
            metadata={}
        )
        
        chunks = self.chunker.chunk(whitespace_doc, self.config)
        
        assert chunks == []
    
    def test_create_overlapping_chunks(self):
        """
        Test overlapping chunk creation
        重複チャンク作成テスト
        """
        tokens = ["word1", "word2", "word3", "word4", "word5", "word6", "word7", "word8"]
        config = ChunkingConfig(chunk_size=4, overlap=1)
        
        chunks = self.chunker._create_overlapping_chunks(tokens, config)
        
        assert len(chunks) == 2
        assert chunks[0] == ["word1", "word2", "word3", "word4"]
        assert chunks[1] == ["word4", "word5", "word6", "word7"]  # Overlap with word4
    
    def test_create_overlapping_chunks_no_overlap(self):
        """
        Test chunk creation without overlap
        重複なしのチャンク作成テスト
        """
        tokens = ["word1", "word2", "word3", "word4", "word5", "word6"]
        config = ChunkingConfig(chunk_size=3, overlap=0)
        
        chunks = self.chunker._create_overlapping_chunks(tokens, config)
        
        assert len(chunks) == 2
        assert chunks[0] == ["word1", "word2", "word3"]
        assert chunks[1] == ["word4", "word5", "word6"]
    
    def test_create_overlapping_chunks_large_overlap(self):
        """
        Test chunk creation with large overlap
        大きな重複でのチャンク作成テスト
        """
        tokens = ["word1", "word2", "word3", "word4", "word5"]
        config = ChunkingConfig(chunk_size=3, overlap=2)
        
        chunks = self.chunker._create_overlapping_chunks(tokens, config)
        
        # Should still make progress despite large overlap
        assert len(chunks) >= 2
        assert chunks[0] == ["word1", "word2", "word3"]
    
    def test_reconstruct_text_basic(self):
        """
        Test basic text reconstruction from tokens
        トークンからの基本的なテキスト再構築テスト
        """
        tokens = ["Hello", "world", "test"]
        original_text = "Hello world test"
        
        reconstructed = self.chunker._reconstruct_text(tokens, original_text)
        
        assert "Hello" in reconstructed
        assert "world" in reconstructed
        assert "test" in reconstructed
    
    def test_reconstruct_text_empty(self):
        """
        Test text reconstruction with empty tokens
        空トークンでのテキスト再構築テスト
        """
        reconstructed = self.chunker._reconstruct_text([], "original text")
        
        assert reconstructed == ""
    
    def test_preserve_sentence_boundaries(self):
        """
        Test sentence boundary preservation
        文境界保持テスト
        """
        reconstructed = "This is a test sentence"
        original = "This is a test sentence. Another sentence."
        
        result = self.chunker._preserve_sentence_boundaries(reconstructed, original)
        
        # Should return the text as-is (no automatic punctuation)
        assert result == reconstructed
    
    def test_calculate_char_positions(self):
        """
        Test character position calculation
        文字位置計算テスト
        """
        original_text = "This is the original text with the chunk somewhere inside."
        chunk_text = "chunk somewhere"
        position = 1
        
        start_char, end_char = self.chunker._calculate_char_positions(original_text, chunk_text, position)
        
        assert isinstance(start_char, int)
        assert isinstance(end_char, int)
        assert start_char >= 0
        assert end_char > start_char
        assert end_char <= len(original_text)
    
    def test_calculate_char_positions_not_found(self):
        """
        Test character position calculation when chunk not found
        チャンクが見つからない場合の文字位置計算テスト
        """
        original_text = "This is the original text."
        chunk_text = "nonexistent chunk content"
        position = 2
        
        start_char, end_char = self.chunker._calculate_char_positions(original_text, chunk_text, position)
        
        # Should provide fallback positions
        assert isinstance(start_char, int)
        assert isinstance(end_char, int)
        assert start_char >= 0
        assert end_char <= len(original_text)
    
    def test_chunk_with_validation_skip(self):
        """
        Test chunking with chunk validation that skips invalid chunks
        無効なチャンクをスキップするチャンク検証付きチャンキングテスト
        """
        # Create a scenario where some chunks might be invalid
        with patch.object(self.chunker, '_validate_chunk_size') as mock_validate:
            # Make every other chunk invalid
            mock_validate.side_effect = lambda chunk_text, config: len(chunk_text) % 2 == 0
            
            chunks = self.chunker.chunk(self.test_document, self.config)
            
            # Should have fewer chunks due to validation filtering
            assert isinstance(chunks, list)
            # At least some chunks should pass validation
            assert len(chunks) > 0


class TestTokenBasedChunkerEdgeCases:
    """
    Test edge cases and boundary conditions for TokenBasedChunker
    TokenBasedChunkerのエッジケースと境界条件のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.chunker = TokenBasedChunker()
    
    def test_chunk_with_special_characters(self):
        """
        Test chunking with special characters
        特殊文字でのチャンキングテスト
        """
        doc = Document(
            id="special_doc",
            content="Hello! @world #test $money %percent ^power &and *star (parentheses) [brackets] {braces}",
            metadata={}
        )
        
        config = ChunkingConfig(chunk_size=5, overlap=1)
        chunks = self.chunker.chunk(doc, config)
        
        assert len(chunks) > 0
        # Should handle special characters gracefully
        for chunk in chunks:
            assert chunk.content.strip()
    
    def test_chunk_with_unicode(self):
        """
        Test chunking with Unicode characters
        Unicode文字でのチャンキングテスト
        """
        doc = Document(
            id="unicode_doc",
            content="こんにちは 世界 これは 日本語 テスト です emoji 🌟 café naïve résumé",
            metadata={}
        )
        
        config = ChunkingConfig(chunk_size=4, overlap=1)
        chunks = self.chunker.chunk(doc, config)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content.strip()
    
    def test_chunk_with_numbers(self):
        """
        Test chunking with numbers
        数字でのチャンキングテスト
        """
        doc = Document(
            id="numbers_doc",
            content="The year 2023 has 365 days and 12 months with 52 weeks total.",
            metadata={}
        )
        
        config = ChunkingConfig(chunk_size=6, overlap=2)
        chunks = self.chunker.chunk(doc, config)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content.strip()
    
    def test_chunk_very_long_document(self):
        """
        Test chunking with very long document
        非常に長い文書でのチャンキングテスト
        """
        long_content = "This is a test sentence. " * 200  # 1000 words
        doc = Document(
            id="long_doc",
            content=long_content,
            metadata={}
        )
        
        config = ChunkingConfig(chunk_size=50, overlap=10)
        chunks = self.chunker.chunk(doc, config)
        
        assert len(chunks) > 5  # Should create many chunks
        for chunk in chunks:
            assert chunk.content.strip()
            # Check token count is reasonable
            token_count = self.chunker.estimate_tokens(chunk.content)
            assert token_count <= config.max_chunk_size
    
    def test_chunk_single_word_document(self):
        """
        Test chunking with single word document
        単語一つの文書でのチャンキングテスト
        """
        doc = Document(
            id="single_word_doc",
            content="Word",
            metadata={}
        )
        
        config = ChunkingConfig(chunk_size=10, overlap=2)
        chunks = self.chunker.chunk(doc, config)
        
        assert len(chunks) == 1
        assert chunks[0].content == "Word"
    
    def test_tokenize_edge_cases(self):
        """
        Test tokenization edge cases
        トークン化エッジケースのテスト
        """
        # Test various punctuation and spacing scenarios
        test_cases = [
            ("", []),
            ("word", ["word"]),
            ("word1 word2", ["word1", "word2"]),
            ("hello,world", ["hello", "world"]),
            ("test...dots", ["test", "dots"]),
            ("123 abc", ["123", "abc"]),
            ("under_score", ["under_score"]),
        ]
        
        for text, expected in test_cases:
            tokens = self.chunker._tokenize(text)
            assert tokens == expected, f"Failed for text: '{text}'"


class TestChunkingIntegration:
    """
    Test integration scenarios and workflows
    統合シナリオとワークフローのテスト
    """
    
    def test_chunking_workflow_end_to_end(self):
        """
        Test complete chunking workflow from document to chunks
        文書からチャンクまでの完全なチャンキングワークフローテスト
        """
        # Create a realistic document
        doc = Document(
            id="workflow_doc",
            content="Machine learning is a subset of artificial intelligence. It focuses on algorithms that can learn from data. Deep learning is a subset of machine learning that uses neural networks. These networks can process complex patterns in data.",
            metadata={
                "title": "ML Overview",
                "author": "AI Researcher",
                "date": "2024-01-01"
            }
        )
        
        # Configure chunker
        config = ChunkingConfig(
            chunk_size=15,  # 15 tokens
            overlap=3,
            min_chunk_size=5,
            max_chunk_size=30
        )
        
        chunker = TokenBasedChunker(config=config)
        
        # Process document
        chunks = chunker.process(doc)
        
        # Verify results
        assert len(chunks) > 1
        
        for i, chunk in enumerate(chunks):
            # Verify document structure
            assert isinstance(chunk, Document)
            assert chunk.id == f"workflow_doc_chunk_{i}"
            assert chunk.content.strip()
            
            # Verify metadata preservation and enhancement
            assert chunk.metadata["title"] == "ML Overview"
            assert chunk.metadata["author"] == "AI Researcher"
            assert chunk.metadata["date"] == "2024-01-01"
            
            # Verify chunking metadata
            assert chunk.metadata["original_document_id"] == "workflow_doc"
            assert chunk.metadata["parent_document_id"] == "workflow_doc"
            assert chunk.metadata["processing_stage"] == "chunked"
            assert chunk.metadata["chunk_position"] == i
            assert chunk.metadata["chunk_total"] == len(chunks)
            assert chunk.metadata["chunking_method"] == "TokenBasedChunker"
            
            # Verify token count
            token_count = chunk.metadata["token_count"]
            assert token_count >= config.min_chunk_size
            assert token_count <= config.max_chunk_size
    
    def test_chunking_preserves_document_lineage(self):
        """
        Test that chunking preserves document lineage through metadata
        チャンキングがメタデータを通じて文書系譜を保持することのテスト
        """
        # Document that already has processing history
        doc = Document(
            id="processed_doc_v2",
            content="This document has already been processed once.",
            metadata={
                "original_document_id": "raw_doc",
                "processing_history": ["loaded", "normalized"],
                "version": 2
            }
        )
        
        chunker = TokenBasedChunker()
        chunks = chunker.process(doc)
        
        for chunk in chunks:
            # Should preserve original lineage
            assert chunk.metadata["original_document_id"] == "raw_doc"
            assert chunk.metadata["parent_document_id"] == "processed_doc_v2"
            
            # Should preserve processing history
            assert chunk.metadata["processing_history"] == ["loaded", "normalized"]
            assert chunk.metadata["version"] == 2
    
    def test_multiple_chunkers_different_configs(self):
        """
        Test using multiple chunkers with different configurations
        異なる設定での複数チャンカー使用テスト
        """
        doc = Document(
            id="multi_chunk_doc",
            content="This is a document that will be chunked with different strategies to compare results.",
            metadata={}
        )
        
        # Fine-grained chunking
        fine_config = ChunkingConfig(chunk_size=5, overlap=1)
        fine_chunker = TokenBasedChunker(config=fine_config)
        fine_chunks = fine_chunker.process(doc)
        
        # Coarse-grained chunking
        coarse_config = ChunkingConfig(chunk_size=15, overlap=3)
        coarse_chunker = TokenBasedChunker(config=coarse_config)
        coarse_chunks = coarse_chunker.process(doc)
        
        # Fine chunking should produce more chunks
        assert len(fine_chunks) > len(coarse_chunks)
        
        # Both should cover the same content
        fine_total_tokens = sum(chunk.metadata["token_count"] for chunk in fine_chunks)
        coarse_total_tokens = sum(chunk.metadata["token_count"] for chunk in coarse_chunks)
        
        # Total tokens should be similar (accounting for overlap)
        assert abs(fine_total_tokens - coarse_total_tokens) < 10