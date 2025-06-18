"""
Comprehensive tests for RecursiveChunker functionality
RecursiveChunker機能の包括的テスト

This module provides comprehensive coverage for the RecursiveChunker class,
testing all configuration options, recursive splitting logic, environment variables,
metadata handling, and edge cases.
このモジュールは、RecursiveChunkerクラスの包括的カバレッジを提供し、
全ての設定オプション、再帰分割ロジック、環境変数、メタデータ処理、エッジケースをテストします。
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional

from refinire_rag.processing.recursive_chunker import RecursiveChunker, RecursiveChunkerConfig
from refinire_rag.models.document import Document


class TestRecursiveChunkerConfig:
    """
    Test RecursiveChunkerConfig configuration and validation
    RecursiveChunkerConfigの設定と検証のテスト
    """
    
    def test_default_configuration(self):
        """
        Test default configuration values
        デフォルト設定値のテスト
        """
        config = RecursiveChunkerConfig()
        
        # Test default values
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.separators == ["\n\n", "\n", " ", ""]
        assert config.keep_separator is True
        assert config.is_separator_regex is False
        assert config.strip_whitespace is True
        assert config.add_chunk_metadata is True
        assert config.preserve_original_metadata is True
    
    def test_custom_configuration(self):
        """
        Test custom configuration settings
        カスタム設定のテスト
        """
        # Create config without post_init to avoid environment variable interference
        config = RecursiveChunkerConfig()
        config.chunk_size = 500
        config.chunk_overlap = 100
        config.separators = ["\n", ".", " "]
        config.keep_separator = False
        config.is_separator_regex = True
        config.strip_whitespace = False
        config.add_chunk_metadata = False
        config.preserve_original_metadata = False
        
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.separators == ["\n", ".", " "]
        assert config.keep_separator is False
        assert config.is_separator_regex is True
        assert config.strip_whitespace is False
        assert config.add_chunk_metadata is False
        assert config.preserve_original_metadata is False
    
    @patch.dict('os.environ', {
        'REFINIRE_RAG_CHUNK_SIZE': '800',
        'REFINIRE_RAG_CHUNK_OVERLAP': '150',
        'REFINIRE_RAG_SEPARATORS': '\\n\\n,\\n,.,!,?',
        'REFINIRE_RAG_KEEP_SEPARATOR': 'false',
        'REFINIRE_RAG_IS_SEPARATOR_REGEX': 'true'
    })
    def test_environment_variable_configuration(self):
        """
        Test configuration from environment variables
        環境変数からの設定テスト
        """
        config = RecursiveChunkerConfig()
        
        assert config.chunk_size == 800
        assert config.chunk_overlap == 150
        assert config.separators == ['\\n\\n', '\\n', '.', '!', '?']
        assert config.keep_separator is False
        assert config.is_separator_regex is True
    
    @patch.dict('os.environ', {
        'REFINIRE_RAG_SEPARATORS': ' , , \\n , '
    })
    def test_environment_separators_with_empty_values(self):
        """
        Test environment separator parsing with empty values
        空値を含む環境変数セパレータの解析テスト
        """
        config = RecursiveChunkerConfig()
        
        # Should filter out empty separators after stripping
        assert config.separators == ['\\n']
    
    @patch.dict('os.environ', {
        'REFINIRE_RAG_SEPARATORS': '   ,   ,   '
    })
    def test_environment_separators_all_empty(self):
        """
        Test environment separators when all are empty
        全て空の環境変数セパレータテスト
        """
        config = RecursiveChunkerConfig()
        
        # Should fall back to defaults
        assert config.separators == ["\n\n", "\n", " ", ""]
    
    @patch.dict('os.environ', {
        'REFINIRE_RAG_CHUNK_SIZE': 'invalid',
        'REFINIRE_RAG_CHUNK_OVERLAP': '150'
    })
    def test_environment_variable_invalid_values(self):
        """
        Test environment variables with invalid values
        無効な値を持つ環境変数のテスト
        """
        # Should raise ValueError for invalid int
        with pytest.raises(ValueError):
            RecursiveChunkerConfig()
    
    def test_config_to_dict(self):
        """
        Test configuration serialization to dictionary
        辞書への設定シリアライゼーションテスト
        """
        config = RecursiveChunkerConfig(chunk_size=500, chunk_overlap=50)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["chunk_size"] == 500
        assert config_dict["chunk_overlap"] == 50
        assert "separators" in config_dict
        assert "keep_separator" in config_dict


class TestRecursiveChunkerInitialization:
    """
    Test RecursiveChunker initialization scenarios
    RecursiveChunker初期化シナリオのテスト
    """
    
    def test_default_initialization(self):
        """
        Test default initialization
        デフォルト初期化テスト
        """
        chunker = RecursiveChunker()
        
        assert chunker.config is not None
        assert isinstance(chunker.config, RecursiveChunkerConfig)
        assert chunker.config.chunk_size == 1000
        assert chunker.config.chunk_overlap == 200
        assert chunker.processing_stats["documents_processed"] == 0
        assert chunker.processing_stats["chunks_created"] == 0
        assert chunker.processing_stats["total_chars_processed"] == 0
        assert chunker.processing_stats["separator_usage"] == {}
    
    def test_custom_config_initialization(self):
        """
        Test initialization with custom configuration
        カスタム設定での初期化テスト
        """
        config = RecursiveChunkerConfig(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n", ".", " "]
        )
        
        chunker = RecursiveChunker(config)
        
        assert chunker.config == config
        assert chunker.config.chunk_size == 500
        assert chunker.config.chunk_overlap == 100
        assert chunker.config.separators == ["\n", ".", " "]
    
    def test_get_config_class(self):
        """
        Test get_config_class method
        get_config_classメソッドテスト
        """
        config_class = RecursiveChunker.get_config_class()
        assert config_class == RecursiveChunkerConfig
    
    @patch.dict('os.environ', {
        'REFINIRE_RAG_CHUNK_SIZE': '750',
        'REFINIRE_RAG_CHUNK_OVERLAP': '125'
    })
    def test_from_env_class_method(self):
        """
        Test from_env class method
        from_envクラスメソッドテスト
        """
        chunker = RecursiveChunker.from_env()
        
        assert isinstance(chunker, RecursiveChunker)
        assert chunker.config.chunk_size == 750
        assert chunker.config.chunk_overlap == 125
    
    def test_initialization_with_none_config(self):
        """
        Test initialization with None config
        None設定での初期化テスト
        """
        chunker = RecursiveChunker(None)
        
        assert chunker.config is not None
        assert isinstance(chunker.config, RecursiveChunkerConfig)
        assert chunker.config.chunk_size == 1000  # Default value


class TestRecursiveChunkerBasicProcessing:
    """
    Test RecursiveChunker basic document processing
    RecursiveChunkerの基本的な文書処理テスト
    """
    
    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        self.config = RecursiveChunkerConfig(
            chunk_size=100,
            chunk_overlap=20,
            separators=["\n\n", "\n", " ", ""]
        )
        self.chunker = RecursiveChunker(self.config)
        
        self.test_document = Document(
            id="test_doc",
            content="This is a test document.\n\nIt has multiple paragraphs.\nAnd some sentences.\n\nFor testing purposes.",
            metadata={"type": "test", "author": "tester"}
        )
    
    def test_single_document_processing(self):
        """
        Test processing a single document
        単一文書処理テスト
        """
        chunks = list(self.chunker._process_single_document(self.test_document, self.config))
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Check chunk properties
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert chunk.id.startswith("test_doc_recursive_chunk_")
            assert len(chunk.content) <= self.config.chunk_size
            assert chunk.content.strip()  # Should not be empty
    
    def test_process_iterator_interface(self):
        """
        Test process method iterator interface
        processメソッドのイテレータインターフェースのテスト
        """
        documents = [self.test_document]
        
        # Test that process returns an iterator
        chunks = list(self.chunker.process(documents))
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
    
    def test_multiple_documents_processing(self):
        """
        Test processing multiple documents
        複数文書処理テスト
        """
        doc1 = Document(id="doc1", content="First document content.", metadata={})
        doc2 = Document(id="doc2", content="Second document content.", metadata={})
        
        documents = [doc1, doc2]
        chunks = list(self.chunker.process(documents))
        
        # Should have chunks from both documents
        doc1_chunks = [c for c in chunks if c.id.startswith("doc1")]
        doc2_chunks = [c for c in chunks if c.id.startswith("doc2")]
        
        assert len(doc1_chunks) > 0
        assert len(doc2_chunks) > 0
    
    def test_process_with_config_override(self):
        """
        Test processing with configuration override
        設定オーバーライドでの処理テスト
        """
        override_config = RecursiveChunkerConfig(
            chunk_size=50,
            chunk_overlap=10,
            separators=[" ", ""]
        )
        
        chunks = list(self.chunker._process_single_document(self.test_document, override_config))
        
        # Should use override config
        for chunk in chunks:
            assert len(chunk.content) <= 50  # Override chunk size
            if "chunk_overlap" in chunk.metadata:
                assert chunk.metadata["chunk_overlap"] == 10
    
    def test_short_document_processing(self):
        """
        Test processing document shorter than chunk size
        チャンクサイズより短い文書の処理テスト
        """
        short_doc = Document(
            id="short_doc",
            content="Short content.",
            metadata={}
        )
        
        chunks = list(self.chunker._process_single_document(short_doc, self.config))
        
        assert len(chunks) == 1
        assert chunks[0].content == "Short content."
        assert chunks[0].id == "short_doc_recursive_chunk_000"
    
    def test_empty_document_processing(self):
        """
        Test processing empty document
        空文書の処理テスト
        """
        empty_doc = Document(id="empty_doc", content="", metadata={})
        
        chunks = list(self.chunker._process_single_document(empty_doc, self.config))
        
        assert len(chunks) == 0
    
    def test_whitespace_only_document_processing(self):
        """
        Test processing whitespace-only document
        空白のみ文書の処理テスト
        """
        whitespace_doc = Document(id="whitespace_doc", content="   \n\t  ", metadata={})
        
        chunks = list(self.chunker._process_single_document(whitespace_doc, self.config))
        
        assert len(chunks) == 0
    
    def test_processing_statistics_update(self):
        """
        Test that processing statistics are updated correctly
        処理統計が正しく更新されることのテスト
        """
        initial_docs = self.chunker.processing_stats["documents_processed"]
        initial_chunks = self.chunker.processing_stats["chunks_created"]
        
        chunks = list(self.chunker._process_single_document(self.test_document, self.config))
        
        assert self.chunker.processing_stats["documents_processed"] == initial_docs + 1
        assert self.chunker.processing_stats["chunks_created"] == initial_chunks + len(chunks)
        assert self.chunker.processing_stats["total_chars_processed"] > 0
        assert self.chunker.processing_stats["average_chunk_size"] > 0


class TestRecursiveChunkerTextSplitting:
    """
    Test RecursiveChunker text splitting algorithms
    RecursiveChunkerのテキスト分割アルゴリズムテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.config = RecursiveChunkerConfig(
            chunk_size=50,
            chunk_overlap=10,
            separators=["\n\n", "\n", " ", ""]
        )
        self.chunker = RecursiveChunker(self.config)
    
    def test_recursive_text_splitting_paragraphs(self):
        """
        Test recursive splitting with paragraph separators
        段落セパレータでの再帰分割テスト
        """
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        
        chunks = self.chunker._split_text_recursively(text, self.config)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= self.config.chunk_size for chunk in chunks)
        assert any("First paragraph" in chunk for chunk in chunks)
        assert any("Second paragraph" in chunk for chunk in chunks)
    
    def test_recursive_text_splitting_sentences(self):
        """
        Test recursive splitting with sentence separators
        文セパレータでの再帰分割テスト
        """
        text = "First sentence.\nSecond sentence.\nThird sentence."
        
        chunks = self.chunker._split_text_recursively(text, self.config)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= self.config.chunk_size for chunk in chunks)
    
    def test_recursive_text_splitting_words(self):
        """
        Test recursive splitting with word separators
        単語セパレータでの再帰分割テスト
        """
        text = "word " * 20  # 20 words, should exceed chunk size
        
        chunks = self.chunker._split_text_recursively(text, self.config)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= self.config.chunk_size for chunk in chunks)
    
    def test_character_level_splitting(self):
        """
        Test character-level splitting when no separators work
        セパレータが効かない場合の文字レベル分割テスト
        """
        text = "a" * 100  # 100 characters without separators
        
        chunks = self.chunker._split_by_character(text, self.config)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= self.config.chunk_size for chunk in chunks)
    
    def test_character_splitting_with_word_boundaries(self):
        """
        Test character splitting that respects word boundaries
        単語境界を尊重する文字分割テスト
        """
        text = "verylongwordthatexceedschunksize and some more words"
        
        chunks = self.chunker._split_by_character(text, self.config)
        
        assert len(chunks) > 1
        # Should try to break at word boundaries when possible
        for chunk in chunks:
            assert len(chunk) <= self.config.chunk_size
    
    def test_split_text_with_regex_separators(self):
        """
        Test text splitting with regex separators
        正規表現セパレータでのテキスト分割テスト
        """
        config = RecursiveChunkerConfig(
            chunk_size=30,
            chunk_overlap=5,
            separators=[r'\d+\.', r'[.!?]', ' '],
            is_separator_regex=True
        )
        chunker = RecursiveChunker(config)
        
        text = "1. First item. 2. Second item. 3. Third item."
        
        chunks = chunker._split_text_recursively(text, config)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= config.chunk_size for chunk in chunks)
    
    def test_split_text_without_separators(self):
        """
        Test splitting when separators list is empty
        セパレータリストが空の場合の分割テスト
        """
        config = RecursiveChunkerConfig(
            chunk_size=20,
            chunk_overlap=5,
            separators=[]
        )
        chunker = RecursiveChunker(config)
        
        text = "Long text without any separators to test fallback"
        
        chunks = chunker._split_text_recursively(text, config)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= config.chunk_size for chunk in chunks)
    
    def test_merge_splits_with_separator_keeping(self):
        """
        Test merging splits while keeping separators
        セパレータを保持しながらの分割統合テスト
        """
        splits = ["First part", "Second part", "Third part"]
        separator = "\n"
        
        config = RecursiveChunkerConfig(
            chunk_size=50,
            chunk_overlap=10,
            keep_separator=True
        )
        
        chunks = self.chunker._merge_splits(splits, separator, config)
        
        assert len(chunks) > 0
        # Should contain separators
        assert any(separator in chunk for chunk in chunks if len(chunk) > 1)
    
    def test_merge_splits_without_separator_keeping(self):
        """
        Test merging splits without keeping separators
        セパレータを保持しない分割統合テスト
        """
        splits = ["First", "Second", "Third"]
        separator = "\n"
        
        config = RecursiveChunkerConfig(
            chunk_size=50,
            chunk_overlap=10,
            keep_separator=False
        )
        
        chunks = self.chunker._merge_splits(splits, separator, config)
        
        assert len(chunks) > 0
        # Should not contain separators (unless naturally present)
        # This is harder to test definitively without knowing exact content
    
    def test_apply_overlap(self):
        """
        Test overlap application between chunks
        チャンク間の重複適用テスト
        """
        existing_chunks = ["This is the first chunk content"]
        new_split = "This is new content"
        separator = " "
        
        config = RecursiveChunkerConfig(chunk_overlap=10, keep_separator=True)
        
        result = self.chunker._apply_overlap(existing_chunks, new_split, separator, config)
        
        # Should contain overlap from previous chunk
        assert "content" in result  # Overlap from end of first chunk
        assert "This is new content" in result  # New split content
    
    def test_apply_overlap_no_existing_chunks(self):
        """
        Test overlap application with no existing chunks
        既存チャンクなしでの重複適用テスト
        """
        existing_chunks = []
        new_split = "First content"
        separator = " "
        
        config = RecursiveChunkerConfig(chunk_overlap=10)
        
        result = self.chunker._apply_overlap(existing_chunks, new_split, separator, config)
        
        assert result == "First content"
    
    def test_apply_overlap_zero_overlap(self):
        """
        Test overlap application with zero overlap
        ゼロ重複での重複適用テスト
        """
        existing_chunks = ["Previous chunk"]
        new_split = "New content"
        separator = " "
        
        config = RecursiveChunkerConfig(chunk_overlap=0)
        
        result = self.chunker._apply_overlap(existing_chunks, new_split, separator, config)
        
        assert result == "New content"


class TestRecursiveChunkerMetadata:
    """
    Test RecursiveChunker metadata handling
    RecursiveChunkerのメタデータ処理テスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.config = RecursiveChunkerConfig(
            chunk_size=50,
            chunk_overlap=10,
            add_chunk_metadata=True,
            preserve_original_metadata=True
        )
        self.chunker = RecursiveChunker(self.config)
        
        self.test_document = Document(
            id="test_doc",
            content="This is a test document with enough content to create multiple chunks.",
            metadata={
                "title": "Test Document",
                "author": "Test Author",
                "category": "Testing",
                "original_document_id": "root_doc"
            }
        )
    
    def test_chunk_metadata_creation(self):
        """
        Test chunk metadata creation
        チャンクメタデータ作成テスト
        """
        chunks = list(self.chunker._process_single_document(self.test_document, self.config))
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata
            
            # Check required metadata fields
            assert metadata["processing_stage"] == "recursive_chunked"
            assert metadata["parent_document_id"] == "test_doc"
            assert metadata["chunk_position"] == i
            assert metadata["chunk_total"] == len(chunks)
            assert metadata["chunking_method"] == "recursive"
            assert metadata["chunk_overlap"] == self.config.chunk_overlap
            assert metadata["chunked_by"] == "RecursiveChunker"
            assert "chunk_size_chars" in metadata
            assert "separators_used" in metadata
    
    def test_preserve_original_metadata(self):
        """
        Test preservation of original document metadata
        元文書メタデータの保持テスト
        """
        chunks = list(self.chunker._process_single_document(self.test_document, self.config))
        
        for chunk in chunks:
            # Original metadata should be preserved
            assert chunk.metadata["title"] == "Test Document"
            assert chunk.metadata["author"] == "Test Author"
            assert chunk.metadata["category"] == "Testing"
            assert chunk.metadata["original_document_id"] == "root_doc"
    
    def test_metadata_inheritance_hierarchy(self):
        """
        Test metadata inheritance hierarchy
        メタデータ継承階層テスト
        """
        chunks = list(self.chunker._process_single_document(self.test_document, self.config))
        
        for chunk in chunks:
            # Should preserve original_document_id from parent
            assert chunk.metadata["original_document_id"] == "root_doc"
            # Parent should be immediate parent
            assert chunk.metadata["parent_document_id"] == "test_doc"
    
    def test_no_chunk_metadata_configuration(self):
        """
        Test configuration with chunk metadata disabled
        チャンクメタデータ無効設定のテスト
        """
        config = RecursiveChunkerConfig(
            chunk_size=50,
            add_chunk_metadata=False,
            preserve_original_metadata=True
        )
        
        chunks = list(self.chunker._process_single_document(self.test_document, config))
        
        for chunk in chunks:
            # Should have original metadata
            assert chunk.metadata["title"] == "Test Document"
            
            # Should not have chunking metadata
            assert "processing_stage" not in chunk.metadata
            assert "chunk_position" not in chunk.metadata
            assert "chunking_method" not in chunk.metadata
    
    def test_no_original_metadata_preservation(self):
        """
        Test configuration without original metadata preservation
        元メタデータ保持なし設定のテスト
        """
        config = RecursiveChunkerConfig(
            chunk_size=50,
            add_chunk_metadata=True,
            preserve_original_metadata=False
        )
        
        chunks = list(self.chunker._process_single_document(self.test_document, config))
        
        for chunk in chunks:
            # Should have chunking metadata
            assert chunk.metadata["processing_stage"] == "recursive_chunked"
            
            # Should not have original metadata
            assert "title" not in chunk.metadata
            assert "author" not in chunk.metadata
            assert "category" not in chunk.metadata
    
    def test_chunk_id_generation(self):
        """
        Test chunk ID generation
        チャンクID生成テスト
        """
        chunks = list(self.chunker._process_single_document(self.test_document, self.config))
        
        for i, chunk in enumerate(chunks):
            expected_id = f"test_doc_recursive_chunk_{i:03d}"
            assert chunk.id == expected_id
    
    def test_separator_usage_tracking(self):
        """
        Test separator usage tracking in metadata
        メタデータでのセパレータ使用追跡テスト
        """
        text_doc = Document(
            id="text_doc",
            content="First paragraph.\n\nSecond paragraph.\nWith multiple sentences.",
            metadata={}
        )
        
        chunks = list(self.chunker._process_single_document(text_doc, self.config))
        
        # Check that separator usage is tracked
        assert len(self.chunker.processing_stats["separator_usage"]) > 0
        
        for chunk in chunks:
            if "separators_used" in chunk.metadata:
                assert isinstance(chunk.metadata["separators_used"], list)


class TestRecursiveChunkerErrorHandling:
    """
    Test RecursiveChunker error handling and edge cases
    RecursiveChunkerのエラーハンドリングとエッジケーステスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.config = RecursiveChunkerConfig(chunk_size=50, chunk_overlap=10)
        self.chunker = RecursiveChunker(self.config)
    
    def test_processing_error_handling(self):
        """
        Test error handling during processing
        処理中のエラーハンドリングテスト
        """
        test_doc = Document(id="error_doc", content="Test content", metadata={})
        
        # Mock an error in split_text_recursively
        with patch.object(self.chunker, '_split_text_recursively', side_effect=Exception("Split error")):
            chunks = list(self.chunker._process_single_document(test_doc, self.config))
            
            # Should return original document on error
            assert len(chunks) == 1
            assert chunks[0] == test_doc
    
    def test_very_long_single_word(self):
        """
        Test handling of very long single word
        非常に長い単語の処理テスト
        """
        long_word = "a" * 200  # Much longer than chunk size
        doc = Document(id="long_word_doc", content=long_word, metadata={})
        
        chunks = list(self.chunker._process_single_document(doc, self.config))
        
        assert len(chunks) > 1
        assert all(len(chunk.content) <= self.config.chunk_size for chunk in chunks)
    
    def test_special_characters_handling(self):
        """
        Test handling of special characters
        特殊文字の処理テスト
        """
        special_text = "Special chars: 你好 こんにちは ñ ü é 🌟 💡 🚀"
        doc = Document(id="special_doc", content=special_text, metadata={})
        
        chunks = list(self.chunker._process_single_document(doc, self.config))
        
        assert len(chunks) > 0
        # Should handle unicode characters properly
        for chunk in chunks:
            assert isinstance(chunk.content, str)
            assert chunk.content.strip()
    
    def test_whitespace_handling_configuration(self):
        """
        Test whitespace handling configuration
        空白処理設定のテスト
        """
        text_with_whitespace = "  Text with   extra   whitespace   "
        doc = Document(id="whitespace_doc", content=text_with_whitespace, metadata={})
        
        # Test with whitespace stripping enabled
        config_strip = RecursiveChunkerConfig(chunk_size=20, strip_whitespace=True)
        chunks_strip = list(self.chunker._process_single_document(doc, config_strip))
        
        # Test with whitespace stripping disabled
        config_no_strip = RecursiveChunkerConfig(chunk_size=20, strip_whitespace=False)
        chunks_no_strip = list(self.chunker._process_single_document(doc, config_no_strip))
        
        # Results should be different
        assert chunks_strip[0].content != chunks_no_strip[0].content
        assert chunks_strip[0].content.strip() == chunks_strip[0].content
    
    def test_empty_splits_handling(self):
        """
        Test handling of empty splits
        空分割の処理テスト
        """
        splits = ["", "content", "", "more content", ""]
        separator = "\n"
        
        config = RecursiveChunkerConfig(chunk_size=50)
        
        chunks = self.chunker._merge_splits(splits, separator, config)
        
        # Should filter out empty chunks
        assert all(chunk.strip() for chunk in chunks)
    
    def test_chunk_size_validation(self):
        """
        Test behavior with very small chunk sizes
        非常に小さなチャンクサイズでの動作テスト
        """
        config = RecursiveChunkerConfig(chunk_size=5, chunk_overlap=2)
        chunker = RecursiveChunker(config)
        
        doc = Document(id="small_chunk_doc", content="This is a test document.", metadata={})
        
        chunks = list(chunker._process_single_document(doc, config))
        
        assert len(chunks) > 1
        assert all(len(chunk.content) <= config.chunk_size for chunk in chunks)
    
    def test_overlap_larger_than_chunk_size(self):
        """
        Test behavior when overlap is larger than chunk size
        重複がチャンクサイズより大きい場合の動作テスト
        """
        config = RecursiveChunkerConfig(chunk_size=10, chunk_overlap=15)
        chunker = RecursiveChunker(config)
        
        doc = Document(id="large_overlap_doc", content="Short text content here.", metadata={})
        
        # Should handle gracefully without infinite loops
        chunks = list(chunker._process_single_document(doc, config))
        
        assert len(chunks) > 0
        assert all(len(chunk.content) <= config.chunk_size for chunk in chunks)


class TestRecursiveChunkerAdvancedFeatures:
    """
    Test RecursiveChunker advanced features and configurations
    RecursiveChunkerの高度機能と設定のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.chunker = RecursiveChunker()
    
    def test_custom_separator_hierarchy(self):
        """
        Test custom separator hierarchy
        カスタムセパレータ階層テスト
        """
        config = RecursiveChunkerConfig(
            chunk_size=30,
            chunk_overlap=5,
            separators=["###", "##", "#", "\n", " ", ""]
        )
        
        text = "# Header 1\nContent 1\n## Header 2\nContent 2\n### Header 3\nContent 3"
        
        chunker = RecursiveChunker(config)
        chunks = chunker._split_text_recursively(text, config)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= config.chunk_size for chunk in chunks)
    
    def test_regex_separator_patterns(self):
        """
        Test regex separator patterns
        正規表現セパレータパターンテスト
        """
        config = RecursiveChunkerConfig(
            chunk_size=40,
            chunk_overlap=5,
            separators=[r'\n\s*\n', r'[.!?]+\s*', r'[,;]\s*', ' '],
            is_separator_regex=True
        )
        
        text = "First sentence. Second sentence!\n\nNew paragraph, with comma; and semicolon."
        
        chunker = RecursiveChunker(config)
        chunks = chunker._split_text_recursively(text, config)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= config.chunk_size for chunk in chunks)
    
    def test_processing_statistics_comprehensive(self):
        """
        Test comprehensive processing statistics
        包括的処理統計テスト
        """
        config = RecursiveChunkerConfig(chunk_size=30, chunk_overlap=5)
        chunker = RecursiveChunker(config)
        
        documents = [
            Document(id="doc1", content="First document content with multiple sentences.", metadata={}),
            Document(id="doc2", content="Second document content with different structure.", metadata={}),
            Document(id="doc3", content="Third document with various separators.\n\nNew paragraph here.", metadata={})
        ]
        
        all_chunks = []
        for doc in documents:
            chunks = list(chunker._process_single_document(doc, config))
            all_chunks.extend(chunks)
        
        stats = chunker.get_chunking_stats()
        
        assert stats["documents_processed"] == 3
        assert stats["chunks_created"] == len(all_chunks)
        assert stats["total_chars_processed"] > 0
        assert stats["average_chunk_size"] > 0
        assert stats["chunk_size"] == 30
        assert stats["chunk_overlap"] == 5
        assert "separator_usage" in stats
    
    def test_separator_usage_statistics(self):
        """
        Test separator usage statistics tracking
        セパレータ使用統計追跡テスト
        """
        config = RecursiveChunkerConfig(
            chunk_size=25,
            separators=["\n\n", "\n", " ", ""]
        )
        chunker = RecursiveChunker(config)
        
        text = "Para 1.\n\nPara 2.\nLine 2.\n\nPara 3."
        doc = Document(id="stats_doc", content=text, metadata={})
        
        chunks = list(chunker._process_single_document(doc, config))
        
        usage_stats = chunker.processing_stats["separator_usage"]
        
        # Should have recorded separator usage
        assert len(usage_stats) > 0
        assert all(isinstance(count, int) and count > 0 for count in usage_stats.values())
    
    def test_chunk_content_validation(self):
        """
        Test chunk content validation and filtering
        チャンクコンテンツ検証とフィルタリングテスト
        """
        text = "Content.\n\n\n\n  \n   \n\nMore content."
        doc = Document(id="validation_doc", content=text, metadata={})
        
        config = RecursiveChunkerConfig(chunk_size=20, strip_whitespace=True)
        chunker = RecursiveChunker(config)
        
        chunks = list(chunker._process_single_document(doc, config))
        
        # Should filter out empty/whitespace-only chunks
        assert all(chunk.content.strip() for chunk in chunks)
        assert len(chunks) > 0
    
    def test_chunking_stats_retrieval(self):
        """
        Test chunking statistics retrieval method
        チャンキング統計取得メソッドテスト
        """
        config = RecursiveChunkerConfig(chunk_size=40, chunk_overlap=8)
        chunker = RecursiveChunker(config)
        
        stats = chunker.get_chunking_stats()
        
        # Should include both processing stats and chunking-specific stats
        assert "documents_processed" in stats
        assert "chunks_created" in stats
        assert "chunk_size" in stats
        assert "chunk_overlap" in stats
        assert "separators" in stats
        assert "keep_separator" in stats
        assert "separator_usage" in stats
    
    def test_different_document_types_processing(self):
        """
        Test processing different types of document content
        異なるタイプの文書コンテンツ処理テスト
        """
        documents = [
            # Code-like content
            Document(id="code_doc", content="def function():\n    return True\n\nclass MyClass:\n    pass", metadata={}),
            # Structured text
            Document(id="structured_doc", content="Title\n====\n\nSection 1\n---------\nContent", metadata={}),
            # Plain text
            Document(id="plain_doc", content="This is plain text with sentences. Multiple sentences here.", metadata={}),
            # Mixed content
            Document(id="mixed_doc", content="Text\n\n```code\nblock\n```\n\nMore text.", metadata={})
        ]
        
        config = RecursiveChunkerConfig(chunk_size=30, chunk_overlap=5)
        chunker = RecursiveChunker(config)
        
        for doc in documents:
            chunks = list(chunker._process_single_document(doc, config))
            
            assert len(chunks) > 0
            assert all(isinstance(chunk, Document) for chunk in chunks)
            assert all(len(chunk.content) <= config.chunk_size for chunk in chunks)
    
    def test_edge_case_separator_handling(self):
        """
        Test edge cases in separator handling
        セパレータ処理のエッジケーステスト
        """
        # Test with separator that doesn't exist in text
        config = RecursiveChunkerConfig(
            chunk_size=20,
            separators=["|||", "###", "\n", " ", ""]
        )
        chunker = RecursiveChunker(config)
        
        text = "Simple text without special separators."
        doc = Document(id="edge_doc", content=text, metadata={})
        
        chunks = list(chunker._process_single_document(doc, config))
        
        assert len(chunks) > 0
        assert all(len(chunk.content) <= config.chunk_size for chunk in chunks)
    
    def test_performance_with_large_text(self):
        """
        Test performance characteristics with large text
        大きなテキストでのパフォーマンス特性テスト
        """
        # Generate large text content
        large_text = "This is a sentence with multiple words. " * 100  # ~4000 characters
        doc = Document(id="large_doc", content=large_text, metadata={})
        
        config = RecursiveChunkerConfig(chunk_size=200, chunk_overlap=50)
        chunker = RecursiveChunker(config)
        
        chunks = list(chunker._process_single_document(doc, config))
        
        # Should handle large text efficiently
        assert len(chunks) > 1
        assert all(len(chunk.content) <= config.chunk_size for chunk in chunks)
        
        # Should have reasonable statistics
        stats = chunker.processing_stats
        assert stats["total_chars_processed"] == len(large_text)
        assert stats["chunks_created"] == len(chunks)
        assert stats["average_chunk_size"] > 0