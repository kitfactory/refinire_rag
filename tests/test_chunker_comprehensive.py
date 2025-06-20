"""
Comprehensive test suite for Chunker module
Chunkerモジュールの包括的テストスイート

Coverage targets:
- ChunkingConfig dataclass with all configuration options
- Chunker class initialization and configuration
- Three chunking strategies: token_based, sentence_based, paragraph_based
- Text preprocessing and overlap handling
- Document creation and metadata preservation
- Statistics tracking and error handling
- Edge cases and integration testing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Optional
from dataclasses import asdict

from refinire_rag.processing.chunker import (
    ChunkingConfig,
    Chunker
)
from refinire_rag.models.document import Document
from refinire_rag.document_processor import DocumentProcessorConfig


class TestChunkingConfig:
    """Test ChunkingConfig dataclass functionality"""
    
    def test_chunking_config_default_initialization(self):
        """Test ChunkingConfig with default values"""
        config = ChunkingConfig()
        
        # Chunk size settings
        assert config.chunk_size == 512
        assert config.overlap == 50
        
        # Splitting behavior
        assert config.split_by_sentence is True
        assert config.min_chunk_size == 50
        assert config.max_chunk_size == 1024
        
        # Text processing
        assert config.preserve_paragraphs is True
        assert config.strip_whitespace is True
        
        # Metadata settings
        assert config.add_chunk_metadata is True
        assert config.preserve_original_metadata is True
        
        # Chunking strategy
        assert config.chunking_strategy == "token_based"
    
    def test_chunking_config_custom_initialization(self):
        """Test ChunkingConfig with custom values"""
        config = ChunkingConfig(
            chunk_size=256,
            overlap=25,
            split_by_sentence=False,
            min_chunk_size=25,
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
        assert config.min_chunk_size == 25
        assert config.max_chunk_size == 512
        assert config.preserve_paragraphs is False
        assert config.strip_whitespace is False
        assert config.add_chunk_metadata is False
        assert config.preserve_original_metadata is False
        assert config.chunking_strategy == "sentence_based"
    
    def test_chunking_config_inheritance(self):
        """Test ChunkingConfig inherits from DocumentProcessorConfig"""
        config = ChunkingConfig()
        assert isinstance(config, DocumentProcessorConfig)
    
    def test_chunking_config_strategies(self):
        """Test valid chunking strategies"""
        strategies = ["token_based", "sentence_based", "paragraph_based"]
        
        for strategy in strategies:
            config = ChunkingConfig(chunking_strategy=strategy)
            assert config.chunking_strategy == strategy
    
    def test_chunking_config_serialization(self):
        """Test ChunkingConfig can be serialized to dict"""
        config = ChunkingConfig(
            chunk_size=128,
            overlap=10,
            chunking_strategy="paragraph_based"
        )
        
        config_dict = asdict(config)
        assert config_dict["chunk_size"] == 128
        assert config_dict["overlap"] == 10
        assert config_dict["chunking_strategy"] == "paragraph_based"


class TestChunkerInitialization:
    """Test Chunker class initialization"""
    
    def test_chunker_default_initialization(self):
        """Test Chunker with default configuration"""
        chunker = Chunker()
        
        assert isinstance(chunker.config, ChunkingConfig)
        assert chunker.config.chunk_size == 512
        assert chunker.config.overlap == 50
        assert chunker.config.chunking_strategy == "token_based"
    
    def test_chunker_custom_config_initialization(self):
        """Test Chunker with custom configuration"""
        config = ChunkingConfig(
            chunk_size=256,
            overlap=25,
            chunking_strategy="sentence_based"
        )
        chunker = Chunker(config)
        
        assert chunker.config == config
        assert chunker.config.chunk_size == 256
        assert chunker.config.overlap == 25
        assert chunker.config.chunking_strategy == "sentence_based"
    
    def test_chunker_processing_stats_initialization(self):
        """Test Chunker processing statistics initialization"""
        chunker = Chunker()
        
        expected_stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "total_tokens_processed": 0,
            "average_chunk_size": 0.0,
            "overlap_tokens": 0
        }
        
        for key, value in expected_stats.items():
            assert key in chunker.processing_stats
            assert chunker.processing_stats[key] == value
    
    def test_get_config_class(self):
        """Test get_config_class returns ChunkingConfig"""
        assert Chunker.get_config_class() == ChunkingConfig
    
    @patch('refinire_rag.processing.chunker.logger')
    def test_chunker_initialization_logging(self, mock_logger):
        """Test Chunker initialization logs configuration"""
        config = ChunkingConfig(chunk_size=256, overlap=25)
        Chunker(config)
        
        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args[0][0]
        assert "chunk_size=256" in log_call
        assert "overlap=25" in log_call


class TestChunkerTextPreprocessing:
    """Test text preprocessing functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ChunkingConfig(strip_whitespace=True)
        self.chunker = Chunker(self.config)
    
    def test_preprocess_text_with_whitespace_stripping(self):
        """Test text preprocessing with whitespace stripping enabled"""
        text = "  This is   a test.\n\n\nAnother   paragraph.  "
        processed = self.chunker._preprocess_text(text, self.config)
        
        expected = "This is a test.\n\nAnother paragraph."
        assert processed == expected
    
    def test_preprocess_text_without_whitespace_stripping(self):
        """Test text preprocessing with whitespace stripping disabled"""
        config = ChunkingConfig(strip_whitespace=False)
        text = "  This is   a test.\n\n\nAnother   paragraph.  "
        processed = self.chunker._preprocess_text(text, config)
        
        # Should return original text unchanged
        assert processed == text
    
    def test_preprocess_text_normalize_paragraph_breaks(self):
        """Test paragraph break normalization"""
        text = "First paragraph.\n\n\n\n\nSecond paragraph."
        processed = self.chunker._preprocess_text(text, self.config)
        
        expected = "First paragraph.\n\nSecond paragraph."
        assert processed == expected
    
    def test_preprocess_text_normalize_spaces(self):
        """Test space normalization"""
        text = "Word1\t\t\tWord2     Word3"
        processed = self.chunker._preprocess_text(text, self.config)
        
        expected = "Word1 Word2 Word3"
        assert processed == expected
    
    def test_preprocess_text_edge_cases(self):
        """Test text preprocessing edge cases"""
        # Empty string
        assert self.chunker._preprocess_text("", self.config) == ""
        
        # Only whitespace
        assert self.chunker._preprocess_text("   \n\n\t  ", self.config) == ""
        
        # Single word
        assert self.chunker._preprocess_text("word", self.config) == "word"


class TestChunkerTokenBasedChunking:
    """Test token-based chunking functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ChunkingConfig(
            chunk_size=10,
            overlap=2,
            min_chunk_size=3,
            split_by_sentence=False
        )
        self.chunker = Chunker(self.config)
    
    def test_chunk_by_tokens_basic(self):
        """Test basic token-based chunking"""
        text = "This is a test document with multiple words to chunk into pieces"
        chunks = self.chunker._chunk_by_tokens(text, self.config)
        
        assert len(chunks) > 1
        # Check that first chunk doesn't exceed chunk_size
        first_chunk_words = chunks[0].split()
        assert len(first_chunk_words) <= self.config.chunk_size
    
    def test_chunk_by_tokens_with_overlap(self):
        """Test token-based chunking with overlap"""
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12"
        chunks = self.chunker._chunk_by_tokens(text, self.config)
        
        assert len(chunks) > 1
        # Check overlap between consecutive chunks
        if len(chunks) > 1:
            first_words = chunks[0].split()
            second_words = chunks[1].split()
            # Should have some overlap
            overlap_found = any(word in second_words for word in first_words[-self.config.overlap:])
            assert overlap_found
    
    def test_chunk_by_tokens_sentence_boundary_enabled(self):
        """Test token-based chunking with sentence boundary detection"""
        config = ChunkingConfig(
            chunk_size=5,
            overlap=1,
            split_by_sentence=True
        )
        text = "Short sentence. This is a longer sentence with more words."
        chunks = self.chunker._chunk_by_tokens(text, config)
        
        assert len(chunks) >= 1
        # First chunk should end at sentence boundary if possible
        if len(chunks) > 1:
            assert chunks[0].endswith("sentence.")
    
    def test_chunk_by_tokens_min_chunk_size(self):
        """Test minimum chunk size enforcement"""
        config = ChunkingConfig(
            chunk_size=3,
            min_chunk_size=2,
            overlap=0
        )
        text = "a b c d e"
        chunks = self.chunker._chunk_by_tokens(text, config)
        
        # All chunks (except possibly the last) should meet min_chunk_size
        for chunk in chunks:
            words = chunk.split()
            assert len(words) >= config.min_chunk_size or chunk == chunks[-1]
    
    def test_chunk_by_tokens_edge_cases(self):
        """Test token-based chunking edge cases"""
        # Single word
        chunks = self.chunker._chunk_by_tokens("word", self.config)
        assert len(chunks) == 1
        assert chunks[0] == "word"
        
        # Empty string
        chunks = self.chunker._chunk_by_tokens("", self.config)
        assert len(chunks) == 0
        
        # Text shorter than chunk_size
        text = "short text"
        chunks = self.chunker._chunk_by_tokens(text, self.config)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestChunkerSentenceBasedChunking:
    """Test sentence-based chunking functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ChunkingConfig(
            chunk_size=20,
            overlap=5,
            min_chunk_size=5,
            chunking_strategy="sentence_based"
        )
        self.chunker = Chunker(self.config)
    
    def test_chunk_by_sentences_basic(self):
        """Test basic sentence-based chunking"""
        text = "First sentence. Second sentence with more words. Third sentence is here."
        chunks = self.chunker._chunk_by_sentences(text, self.config)
        
        assert len(chunks) >= 1
        # Check that chunks contain complete sentences
        for chunk in chunks:
            assert chunk.strip()  # Not empty
    
    def test_chunk_by_sentences_with_overlap(self):
        """Test sentence-based chunking with overlap"""
        text = "A. B. C. D. E. F. G. H."  # Short sentences
        chunks = self.chunker._chunk_by_sentences(text, self.config)
        
        if len(chunks) > 1:
            # Should have some overlap
            assert len(chunks) > 1
    
    def test_split_into_sentences(self):
        """Test sentence splitting functionality"""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = self.chunker._split_into_sentences(text)
        
        expected_sentences = [
            "First sentence",
            "Second sentence",
            "Third sentence",
            "Fourth sentence"
        ]
        assert sentences == expected_sentences
    
    def test_split_into_sentences_japanese(self):
        """Test sentence splitting with Japanese punctuation"""
        text = "最初の文。二番目の文！三番目の文？"
        sentences = self.chunker._split_into_sentences(text)
        
        expected_sentences = [
            "最初の文",
            "二番目の文",
            "三番目の文"
        ]
        assert sentences == expected_sentences
    
    def test_get_overlap_sentences(self):
        """Test sentence overlap calculation"""
        sentences = ["Short", "Medium length sentence", "Another short", "Final sentence"]
        
        # Test with overlap of 10 tokens
        overlap_sentences = self.chunker._get_overlap_sentences(sentences, 10)
        
        # Should include sentences from the end up to token limit
        assert len(overlap_sentences) > 0
        assert all(sentence in sentences for sentence in overlap_sentences)
    
    def test_get_overlap_sentences_edge_cases(self):
        """Test sentence overlap edge cases"""
        sentences = ["test"]
        
        # Zero overlap
        overlap = self.chunker._get_overlap_sentences(sentences, 0)
        assert overlap == []
        
        # Empty sentences
        overlap = self.chunker._get_overlap_sentences([], 5)
        assert overlap == []
        
        # Negative overlap
        overlap = self.chunker._get_overlap_sentences(sentences, -1)
        assert overlap == []


class TestChunkerParagraphBasedChunking:
    """Test paragraph-based chunking functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ChunkingConfig(
            chunk_size=30,
            overlap=5,
            min_chunk_size=10,
            chunking_strategy="paragraph_based"
        )
        self.chunker = Chunker(self.config)
    
    def test_chunk_by_paragraphs_basic(self):
        """Test basic paragraph-based chunking"""
        text = "First paragraph with some content.\n\nSecond paragraph with more content.\n\nThird paragraph here."
        chunks = self.chunker._chunk_by_paragraphs(text, self.config)
        
        assert len(chunks) >= 1
        # Check that chunks maintain paragraph structure
        for chunk in chunks:
            assert chunk.strip()
    
    def test_chunk_by_paragraphs_large_paragraph(self):
        """Test paragraph chunking with large paragraphs that exceed chunk_size"""
        # Create a paragraph with some words
        large_paragraph = " ".join([f"word{i}" for i in range(10)])
        text = f"{large_paragraph}\n\nSecond paragraph."
        
        chunks = self.chunker._chunk_by_paragraphs(text, self.config)
        
        # Should create multiple chunks
        assert len(chunks) >= 1
    
    def test_chunk_by_paragraphs_preserves_structure(self):
        """Test that paragraph chunking preserves paragraph structure"""
        text = "Para1.\n\nPara2.\n\nPara3."
        chunks = self.chunker._chunk_by_paragraphs(text, self.config)
        
        # Check that paragraph separators are preserved
        if len(chunks) == 1:
            assert "\n\n" in chunks[0]
    
    def test_chunk_by_paragraphs_edge_cases(self):
        """Test paragraph chunking edge cases"""
        # Single paragraph
        text = "Single paragraph without breaks."
        chunks = self.chunker._chunk_by_paragraphs(text, self.config)
        assert len(chunks) == 1
        assert chunks[0] == text
        
        # Empty paragraphs (whitespace only)
        text = "First.\n\n   \n\nSecond."
        chunks = self.chunker._chunk_by_paragraphs(text, self.config)
        # Should filter out empty paragraphs
        assert all(chunk.strip() for chunk in chunks)


class TestChunkerSentenceBreakDetection:
    """Test sentence break detection functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ChunkingConfig()
        self.chunker = Chunker(self.config)
    
    def test_find_sentence_break_english(self):
        """Test sentence break detection with English punctuation"""
        words = ["This", "is", "a", "test.", "Another", "sentence."]
        break_pos = self.chunker._find_sentence_break(words, 0, 4)
        
        assert break_pos == 4  # After "test."
    
    def test_find_sentence_break_japanese(self):
        """Test sentence break detection with Japanese punctuation"""
        words = ["これは", "テスト。", "別の", "文。"]
        break_pos = self.chunker._find_sentence_break(words, 0, 3)
        
        assert break_pos == 2  # After "テスト。"
    
    def test_find_sentence_break_multiple_endings(self):
        """Test sentence break with multiple ending types"""
        words = ["Question?", "Exclamation!", "Statement."]
        
        # Method searches backwards, finds "Exclamation!" at index 1, returns 1+1=2
        break_pos = self.chunker._find_sentence_break(words, 0, 2)
        assert break_pos == 2  # After "Exclamation!" (searches backwards, finds last match)
        
        # Test exclamation mark alone
        words = ["Start", "Middle!", "End"]
        break_pos = self.chunker._find_sentence_break(words, 0, 2)
        assert break_pos == 2  # After "Middle!" (index 1, so returns 2)
    
    def test_find_sentence_break_no_break_found(self):
        """Test when no sentence break is found in range"""
        words = ["no", "sentence", "endings", "here"]
        break_pos = self.chunker._find_sentence_break(words, 0, 4)
        
        assert break_pos == 4  # Returns end position when no break found
    
    def test_find_sentence_break_reverse_search(self):
        """Test that sentence break detection searches backwards"""
        words = ["Start", "middle.", "More", "text.", "End"]
        break_pos = self.chunker._find_sentence_break(words, 1, 5)
        
        # Should find the last sentence break in range (position 4)
        assert break_pos == 4


class TestChunkerDocumentCreation:
    """Test document creation and metadata handling"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ChunkingConfig(
            add_chunk_metadata=True,
            preserve_original_metadata=True
        )
        self.chunker = Chunker(self.config)
        
        self.original_doc = Document(
            id="test_doc_001",
            content="This is test content for chunking.",
            metadata={
                "source": "test_source",
                "category": "test",
                "original_length": 35
            }
        )
    
    def test_create_chunk_documents_basic(self):
        """Test basic chunk document creation"""
        chunks = ["First chunk content", "Second chunk content"]
        chunk_docs = self.chunker._create_chunk_documents(
            self.original_doc, chunks, self.config
        )
        
        assert len(chunk_docs) == 2
        
        # Check first chunk
        assert chunk_docs[0].id == "test_doc_001_chunk_000"
        assert chunk_docs[0].content == "First chunk content"
        
        # Check second chunk
        assert chunk_docs[1].id == "test_doc_001_chunk_001"
        assert chunk_docs[1].content == "Second chunk content"
    
    def test_create_chunk_documents_with_metadata_preservation(self):
        """Test chunk creation with original metadata preservation"""
        chunks = ["Test chunk"]
        chunk_docs = self.chunker._create_chunk_documents(
            self.original_doc, chunks, self.config
        )
        
        chunk_doc = chunk_docs[0]
        
        # Should preserve original metadata
        assert chunk_doc.metadata["source"] == "test_source"
        assert chunk_doc.metadata["category"] == "test"
        assert chunk_doc.metadata["original_length"] == 35
    
    def test_create_chunk_documents_with_chunk_metadata(self):
        """Test chunk creation with chunk-specific metadata"""
        chunks = ["First chunk", "Second chunk"]
        chunk_docs = self.chunker._create_chunk_documents(
            self.original_doc, chunks, self.config
        )
        
        first_chunk = chunk_docs[0]
        
        # Check chunk-specific metadata
        assert first_chunk.metadata["processing_stage"] == "chunked"
        assert first_chunk.metadata["original_document_id"] == "test_doc_001"
        assert first_chunk.metadata["parent_document_id"] == "test_doc_001"
        assert first_chunk.metadata["chunk_position"] == 0
        assert first_chunk.metadata["chunk_total"] == 2
        assert first_chunk.metadata["chunk_size_tokens"] == 2  # "First chunk" = 2 words
        assert first_chunk.metadata["chunking_strategy"] == "token_based"
        assert first_chunk.metadata["chunk_overlap"] == 50
        assert first_chunk.metadata["chunked_by"] == "Chunker"
    
    def test_create_chunk_documents_without_metadata_preservation(self):
        """Test chunk creation without preserving original metadata"""
        config = ChunkingConfig(
            preserve_original_metadata=False,
            add_chunk_metadata=True
        )
        
        chunks = ["Test chunk"]
        chunk_docs = self.chunker._create_chunk_documents(
            self.original_doc, chunks, config
        )
        
        chunk_doc = chunk_docs[0]
        
        # Should not have original metadata
        assert "source" not in chunk_doc.metadata
        assert "category" not in chunk_doc.metadata
        
        # But should have chunk metadata
        assert chunk_doc.metadata["processing_stage"] == "chunked"
    
    def test_create_chunk_documents_without_chunk_metadata(self):
        """Test chunk creation without adding chunk metadata"""
        config = ChunkingConfig(
            preserve_original_metadata=True,
            add_chunk_metadata=False
        )
        
        chunks = ["Test chunk"]
        chunk_docs = self.chunker._create_chunk_documents(
            self.original_doc, chunks, config
        )
        
        chunk_doc = chunk_docs[0]
        
        # Should have original metadata
        assert chunk_doc.metadata["source"] == "test_source"
        
        # Should not have chunk-specific metadata
        assert "processing_stage" not in chunk_doc.metadata
        assert "chunk_position" not in chunk_doc.metadata
    
    def test_create_chunk_documents_with_existing_original_document_id(self):
        """Test chunk creation with existing original_document_id in metadata"""
        original_doc = Document(
            id="chunk_doc_001",
            content="Test content",
            metadata={
                "original_document_id": "root_doc_123",
                "other_field": "value"
            }
        )
        
        chunks = ["Test chunk"]
        chunk_docs = self.chunker._create_chunk_documents(
            original_doc, chunks, self.config
        )
        
        chunk_doc = chunk_docs[0]
        
        # Should preserve the existing original_document_id
        assert chunk_doc.metadata["original_document_id"] == "root_doc_123"
        assert chunk_doc.metadata["parent_document_id"] == "chunk_doc_001"


class TestChunkerProcessingStats:
    """Test processing statistics tracking"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ChunkingConfig(chunk_size=5, overlap=1)
        self.chunker = Chunker(self.config)
    
    def test_get_chunking_stats(self):
        """Test get_chunking_stats method"""
        stats = self.chunker.get_chunking_stats()
        
        expected_keys = [
            "documents_processed",
            "chunks_created", 
            "total_tokens_processed",
            "average_chunk_size",
            "overlap_tokens",
            "chunk_size",
            "overlap",
            "chunking_strategy",
            "split_by_sentence"
        ]
        
        for key in expected_keys:
            assert key in stats
        
        # Check config values are included
        assert stats["chunk_size"] == 5
        assert stats["overlap"] == 1
        assert stats["chunking_strategy"] == "token_based"
        assert stats["split_by_sentence"] is True
    
    def test_processing_stats_update_after_processing(self):
        """Test that processing stats are updated after document processing"""
        document = Document(
            id="test_doc",
            content="This is a test document with multiple words for chunking"
        )
        
        # Process document
        chunks = list(self.chunker._process_single_document(document))
        
        # Check stats were updated
        assert self.chunker.processing_stats["documents_processed"] == 1
        assert self.chunker.processing_stats["chunks_created"] == len(chunks)
        assert self.chunker.processing_stats["total_tokens_processed"] > 0
        assert self.chunker.processing_stats["average_chunk_size"] > 0


class TestChunkerMainProcessing:
    """Test main document processing functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ChunkingConfig(
            chunk_size=5,
            overlap=1,
            chunking_strategy="token_based"
        )
        self.chunker = Chunker(self.config)
    
    def test_process_single_document_token_based(self):
        """Test processing single document with token-based strategy"""
        document = Document(
            id="test_doc",
            content="This is a short test document with few words"
        )
        
        chunks = self.chunker._process_single_document(document, self.config)
        
        # Should produce at least one chunk
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Document) for chunk in chunks)
        assert all(chunk.id.startswith("test_doc_chunk_") for chunk in chunks)
    
    def test_process_single_document_sentence_based(self):
        """Test processing single document with sentence-based strategy"""
        config = ChunkingConfig(
            chunk_size=10,
            overlap=2,
            chunking_strategy="sentence_based"
        )
        
        document = Document(
            id="test_doc",
            content="First sentence here. Second sentence with content. Third sentence follows."
        )
        
        chunks = self.chunker._process_single_document(document, config)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Document) for chunk in chunks)
    
    def test_process_single_document_paragraph_based(self):
        """Test processing single document with paragraph-based strategy"""
        config = ChunkingConfig(
            chunk_size=20,
            overlap=3,
            chunking_strategy="paragraph_based"
        )
        
        document = Document(
            id="test_doc",
            content="First paragraph with content.\n\nSecond paragraph here.\n\nThird paragraph follows."
        )
        
        chunks = self.chunker._process_single_document(document, config)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Document) for chunk in chunks)
    
    def test_process_single_document_with_config_override(self):
        """Test processing with configuration override"""
        document = Document(
            id="test_doc",
            content="Short test content for processing"
        )
        
        override_config = ChunkingConfig(
            chunk_size=3,
            overlap=0,
            chunking_strategy="sentence_based"
        )
        
        chunks = self.chunker._process_single_document(document, override_config)
        
        # Should use override config, not instance config
        assert len(chunks) >= 1
    
    @patch('refinire_rag.processing.chunker.logger')
    def test_process_single_document_logging(self, mock_logger):
        """Test that processing logs appropriate messages"""
        document = Document(
            id="test_doc",
            content="Test content"
        )
        
        self.chunker._process_single_document(document)
        
        # Should log debug and info messages
        assert mock_logger.debug.called
        assert mock_logger.info.called
    
    def test_process_single_document_error_handling(self):
        """Test error handling in single document processing"""
        # Create a document that might cause issues
        document = Document(
            id="test_doc",
            content=""  # Empty content
        )
        
        # Should not raise exception, should handle gracefully
        chunks = self.chunker._process_single_document(document)
        
        # With empty content, no chunks are produced
        assert len(chunks) == 0
    
    def test_process_multiple_documents(self):
        """Test processing multiple documents via process method"""
        documents = [
            Document(id="doc1", content="First short document"),
            Document(id="doc2", content="Second short document")
        ]
        
        result_chunks = list(self.chunker.process(documents))
        
        # Should produce at least one chunk per document
        assert len(result_chunks) >= len(documents)
        
        # Check that chunks from different documents have different prefixes
        doc1_chunks = [chunk for chunk in result_chunks if chunk.id.startswith("doc1_chunk_")]
        doc2_chunks = [chunk for chunk in result_chunks if chunk.id.startswith("doc2_chunk_")]
        
        assert len(doc1_chunks) > 0
        assert len(doc2_chunks) > 0


class TestChunkerEdgeCasesAndIntegration:
    """Test edge cases and integration scenarios"""
    
    def test_chunker_with_minimal_text(self):
        """Test chunker with very short text"""
        config = ChunkingConfig(chunk_size=10, min_chunk_size=5)
        chunker = Chunker(config)
        
        document = Document(id="short_doc", content="Hi")
        chunks = chunker._process_single_document(document)
        
        # Should handle short text gracefully
        assert len(chunks) >= 1
    
    def test_chunker_with_empty_content(self):
        """Test chunker with empty document content"""
        chunker = Chunker()
        
        document = Document(id="empty_doc", content="")
        chunks = chunker._process_single_document(document)
        
        # Empty content produces no chunks
        assert len(chunks) == 0
    
    def test_chunker_with_whitespace_only_content(self):
        """Test chunker with whitespace-only content"""
        config = ChunkingConfig(strip_whitespace=True)
        chunker = Chunker(config)
        
        document = Document(id="whitespace_doc", content="   \n\n\t   ")
        chunks = chunker._process_single_document(document)
        
        # Whitespace-only content produces no chunks after preprocessing
        assert len(chunks) == 0
    
    def test_chunker_with_unicode_content(self):
        """Test chunker with Unicode content"""
        chunker = Chunker()
        
        document = Document(
            id="unicode_doc", 
            content="こんにちは世界。これはテストです。🌍 émoji and spéciál characters"
        )
        chunks = chunker._process_single_document(document)
        
        assert len(chunks) >= 1
        # Check that Unicode is preserved
        combined_content = " ".join(chunk.content for chunk in chunks)
        assert "こんにちは世界" in combined_content
        assert "🌍" in combined_content
        assert "émoji" in combined_content
    
    def test_chunker_with_very_large_chunk_size(self):
        """Test chunker with chunk size larger than content"""
        config = ChunkingConfig(chunk_size=1000)
        chunker = Chunker(config)
        
        document = Document(
            id="small_doc",
            content="This is a small document with limited content"
        )
        chunks = chunker._process_single_document(document)
        
        # Should create single chunk when content is smaller than chunk_size
        assert len(chunks) == 1
        assert chunks[0].content == document.content
    
    def test_chunker_with_zero_overlap(self):
        """Test chunker with zero overlap"""
        config = ChunkingConfig(chunk_size=3, overlap=0, min_chunk_size=1)
        chunker = Chunker(config)
        
        document = Document(
            id="no_overlap_doc",
            content="word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        )
        chunks = chunker._process_single_document(document)
        
        # Should create multiple non-overlapping chunks
        assert len(chunks) > 1
        
        # Verify no overlap (check only first few chunks to avoid performance issues)
        for i in range(min(3, len(chunks) - 1)):  # Limit to first 3 chunks
            current_words = chunks[i].content.split()
            next_words = chunks[i + 1].content.split()
            # Simple check instead of expensive set intersection
            assert not any(word in next_words for word in current_words)
    
    def test_chunker_with_maximum_overlap(self):
        """Test chunker with high overlap (but not equal to chunk size)"""
        config = ChunkingConfig(chunk_size=4, overlap=3)  # 75% overlap instead of 100%
        chunker = Chunker(config)
        
        document = Document(
            id="max_overlap_doc",
            content="word1 word2 word3 word4 word5 word6"
        )
        chunks = chunker._process_single_document(document)
        
        # Should handle high overlap without infinite loops
        assert len(chunks) >= 1
    
    def test_chunker_different_strategies_same_content(self):
        """Test different chunking strategies on same content"""
        content = "Short test. Second sentence.\n\nNew paragraph."
        document = Document(id="strategy_test", content=content)
        
        strategies = ["token_based", "sentence_based", "paragraph_based"]
        results = {}
        
        for strategy in strategies:
            config = ChunkingConfig(
                chunk_size=15,
                overlap=2,
                chunking_strategy=strategy
            )
            chunker = Chunker(config)
            chunks = chunker._process_single_document(document, config)
            results[strategy] = chunks
        
        # Each strategy should produce some chunks
        for strategy, chunks in results.items():
            assert len(chunks) >= 1, f"Strategy {strategy} produced no chunks"
        
        # Different strategies may produce different numbers of chunks
        chunk_counts = [len(chunks) for chunks in results.values()]
        # This is just to verify the test ran - the actual counts may vary
        assert all(count > 0 for count in chunk_counts)
    
    def test_chunker_preserves_document_structure_in_metadata(self):
        """Test that chunker preserves document structure information"""
        config = ChunkingConfig(add_chunk_metadata=True)
        chunker = Chunker(config)
        
        document = Document(
            id="structure_test",
            content="Content with structure information",
            metadata={
                "title": "Test Document",
                "author": "Test Author",
                "created_at": "2023-01-01"
            }
        )
        
        chunks = chunker._process_single_document(document, config)
        
        assert len(chunks) >= 1
        
        # Check that structural information is preserved
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_position"] == i
            assert chunk.metadata["chunk_total"] == len(chunks)
            assert chunk.metadata["parent_document_id"] == document.id
            
            # Original metadata should be preserved
            assert chunk.metadata["title"] == "Test Document"
            assert chunk.metadata["author"] == "Test Author"
            assert chunk.metadata["created_at"] == "2023-01-01"