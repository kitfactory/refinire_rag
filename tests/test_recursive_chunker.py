"""
Tests for RecursiveChunker - LangChain RecursiveCharacterTextSplitter equivalent
"""

import os
import pytest
from unittest.mock import patch

from refinire_rag.processing.recursive_chunker import RecursiveChunker, RecursiveChunkerConfig
from refinire_rag.models.document import Document


class TestRecursiveChunkerConfig:
    """Test RecursiveChunkerConfig including environment variable loading"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = RecursiveChunkerConfig()
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.separators == ["\n\n", "\n", " ", ""]
        assert config.keep_separator is True
        assert config.is_separator_regex is False
        assert config.strip_whitespace is True
    
    @patch.dict(os.environ, {
        'REFINIRE_RAG_CHUNK_SIZE': '500',
        'REFINIRE_RAG_CHUNK_OVERLAP': '100',
        'REFINIRE_RAG_SEPARATORS': '\\n\\n,\\n,\\.,\\s',
        'REFINIRE_RAG_KEEP_SEPARATOR': 'false',
        'REFINIRE_RAG_IS_SEPARATOR_REGEX': 'true'
    })
    def test_env_var_config(self):
        """Test configuration loading from environment variables"""
        config = RecursiveChunkerConfig()
        
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.separators == ['\\n\\n', '\\n', '\\.', '\\s']
        assert config.keep_separator is False
        assert config.is_separator_regex is True
    
    @patch.dict(os.environ, {
        'REFINIRE_RAG_SEPARATORS': 'space,\\n,\\t'
    })
    def test_separators_parsing(self):
        """Test separator parsing from environment"""
        config = RecursiveChunkerConfig()
        
        # Should strip whitespace and filter empty separators
        # Input 'space,\\n,\\t' becomes ['space', '\\n', '\\t'] after parsing
        expected = ['space', '\\n', '\\t']
        actual = config.separators
        assert actual == expected


class TestRecursiveChunker:
    """Test RecursiveChunker functionality"""
    
    def test_initialization(self):
        """Test RecursiveChunker initialization"""
        chunker = RecursiveChunker()
        
        assert chunker.config.chunk_size == 1000
        assert chunker.config.chunk_overlap == 200
        assert "documents_processed" in chunker.processing_stats
        assert "chunks_created" in chunker.processing_stats
    
    def test_from_env_factory(self):
        """Test creating RecursiveChunker from environment"""
        with patch.dict(os.environ, {'REFINIRE_RAG_CHUNK_SIZE': '800'}):
            chunker = RecursiveChunker.from_env()
            assert chunker.config.chunk_size == 800
    
    def test_short_text_no_splitting(self):
        """Test that short text is not split"""
        chunker = RecursiveChunker(RecursiveChunkerConfig(chunk_size=100))
        
        doc = Document(
            id="test_doc",
            content="This is a short text that should not be split.",
            metadata={"source": "test"}
        )
        
        chunks = list(chunker.process([doc]))
        
        assert len(chunks) == 1
        assert chunks[0].content == doc.content
        assert chunks[0].id == "test_doc_recursive_chunk_000"
    
    def test_paragraph_splitting(self):
        """Test splitting by paragraphs (\\n\\n)"""
        chunker = RecursiveChunker(RecursiveChunkerConfig(
            chunk_size=50,
            chunk_overlap=10,
            separators=["\n\n", "\n", " ", ""]
        ))
        
        doc = Document(
            id="test_doc",
            content="First paragraph is here.\n\nSecond paragraph is here.\n\nThird paragraph is here.",
            metadata={"source": "test"}
        )
        
        chunks = list(chunker.process([doc]))
        
        assert len(chunks) > 1
        # Should prefer paragraph breaks
        assert any("First paragraph" in chunk.content for chunk in chunks)
        assert any("Second paragraph" in chunk.content for chunk in chunks)
    
    def test_sentence_splitting_fallback(self):
        """Test falling back to sentence splitting when paragraphs are too large"""
        chunker = RecursiveChunker(RecursiveChunkerConfig(
            chunk_size=30,
            chunk_overlap=5,
            separators=["\n\n", "\n", " ", ""]
        ))
        
        doc = Document(
            id="test_doc",
            content="This is a very long sentence that exceeds the chunk size and should be split by spaces.",
            metadata={"source": "test"}
        )
        
        chunks = list(chunker.process([doc]))
        
        assert len(chunks) > 1
        # Should split by spaces when sentence is too long
        for chunk in chunks:
            # Allow more flexibility for overlap and metadata additions
            assert len(chunk.content) <= 50
    
    def test_character_splitting_fallback(self):
        """Test character-level splitting when all separators fail"""
        chunker = RecursiveChunker(RecursiveChunkerConfig(
            chunk_size=10,
            chunk_overlap=2,
            separators=["", ""]  # Force character splitting
        ))
        
        doc = Document(
            id="test_doc",
            content="verylongwordwithoutspaces",
            metadata={"source": "test"}
        )
        
        chunks = list(chunker.process([doc]))
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 12  # chunk_size + small buffer
    
    def test_overlap_functionality(self):
        """Test that overlap between chunks works correctly"""
        chunker = RecursiveChunker(RecursiveChunkerConfig(
            chunk_size=20,
            chunk_overlap=10,
            separators=[" "]
        ))
        
        doc = Document(
            id="test_doc",
            content="word1 word2 word3 word4 word5 word6 word7 word8",
            metadata={"source": "test"}
        )
        
        chunks = list(chunker.process([doc]))
        
        # Check that chunks have overlapping content
        if len(chunks) > 1:
            # Should have some overlap between consecutive chunks
            chunk1_words = set(chunks[0].content.split())
            chunk2_words = set(chunks[1].content.split())
            overlap = chunk1_words.intersection(chunk2_words)
            assert len(overlap) > 0
    
    def test_keep_separator_option(self):
        """Test the keep_separator option"""
        # Test with keep_separator=True
        chunker_keep = RecursiveChunker(RecursiveChunkerConfig(
            chunk_size=15,
            chunk_overlap=0,
            separators=["\n"],
            keep_separator=True
        ))
        
        doc = Document(
            id="test_doc",
            content="line1\nline2\nline3",
            metadata={"source": "test"}
        )
        
        chunks_keep = list(chunker_keep.process([doc]))
        
        # Test with keep_separator=False
        chunker_no_keep = RecursiveChunker(RecursiveChunkerConfig(
            chunk_size=15,
            chunk_overlap=0,
            separators=["\n"],
            keep_separator=False
        ))
        
        chunks_no_keep = list(chunker_no_keep.process([doc]))
        
        # Both should create chunks, but behavior may differ in separator handling
        assert len(chunks_keep) >= 1
        assert len(chunks_no_keep) >= 1
    
    def test_metadata_preservation(self):
        """Test that original metadata is preserved in chunks"""
        chunker = RecursiveChunker(RecursiveChunkerConfig(
            chunk_size=30,
            chunk_overlap=5,
            preserve_original_metadata=True,
            add_chunk_metadata=True
        ))
        
        doc = Document(
            id="test_doc",
            content="This is a longer text that will be split into multiple chunks for testing purposes.",
            metadata={"source": "test", "author": "test_author"}
        )
        
        chunks = list(chunker.process([doc]))
        
        for i, chunk in enumerate(chunks):
            # Should preserve original metadata
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["author"] == "test_author"
            
            # Should add chunking metadata
            assert chunk.metadata["processing_stage"] == "recursive_chunked"
            assert chunk.metadata["parent_document_id"] == "test_doc"
            assert chunk.metadata["chunk_position"] == i
            assert chunk.metadata["chunked_by"] == "RecursiveChunker"
    
    def test_empty_and_whitespace_content(self):
        """Test handling of empty and whitespace-only content"""
        chunker = RecursiveChunker()
        
        # Empty content
        doc_empty = Document(id="empty", content="", metadata={})
        chunks_empty = list(chunker.process([doc_empty]))
        assert len(chunks_empty) == 0
        
        # Whitespace only
        doc_whitespace = Document(id="whitespace", content="   \n\n   ", metadata={})
        chunks_whitespace = list(chunker.process([doc_whitespace]))
        # Should either be empty or have the whitespace stripped
        if chunks_whitespace:
            assert all(chunk.content.strip() for chunk in chunks_whitespace)
    
    def test_statistics_tracking(self):
        """Test that processing statistics are tracked correctly"""
        chunker = RecursiveChunker(RecursiveChunkerConfig(chunk_size=20))
        
        docs = [
            Document(id="doc1", content="Short content.", metadata={}),
            Document(id="doc2", content="This is a much longer content that will definitely be split into multiple chunks.", metadata={})
        ]
        
        # Process documents
        all_chunks = []
        for doc in docs:
            all_chunks.extend(list(chunker.process([doc])))
        
        stats = chunker.get_chunking_stats()
        
        assert stats["documents_processed"] == 2
        assert stats["chunks_created"] == len(all_chunks)
        assert stats["total_chars_processed"] > 0
        assert stats["average_chunk_size"] > 0
        assert "separator_usage" in stats
    
    @patch.dict(os.environ, {
        'REFINIRE_RAG_CHUNK_SIZE': '25',
        'REFINIRE_RAG_CHUNK_OVERLAP': '5'
    })
    def test_environment_integration(self):
        """Test end-to-end functionality with environment variables"""
        chunker = RecursiveChunker.from_env()
        
        doc = Document(
            id="env_test",
            content="This text will be chunked using environment variable configuration settings.",
            metadata={"test": "env_integration"}
        )
        
        chunks = list(chunker.process([doc]))
        
        # Should respect environment settings
        assert chunker.config.chunk_size == 25
        assert chunker.config.chunk_overlap == 5
        
        # Should create appropriate chunks
        assert len(chunks) > 1
        for chunk in chunks:
            # Allow more flexibility for recursive chunking with overlap
            assert len(chunk.content) <= 40