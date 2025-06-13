"""
Sentence-aware chunking implementation
"""

import logging
import re
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime

from refinire.rag.models.document import Document
from refinire.rag.chunking.chunker import Chunker, ChunkingConfig

logger = logging.getLogger(__name__)


class SentenceAwareChunker(Chunker):
    """Sentence-aware document chunker that respects sentence boundaries
    文境界を尊重する文章認識文書チャンカー"""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize sentence-aware chunker
        文境界認識チャンカーを初期化
        
        Args:
            config: Optional chunking configuration
        """
        super().__init__(config)
        
        # Sentence boundary patterns for Japanese and English
        self.sentence_patterns = [
            # English sentence endings
            re.compile(r'[.!?]+\s+'),
            # Japanese sentence endings
            re.compile(r'[。！？]+'),
            # Additional patterns for mixed content
            re.compile(r'\n\s*\n'),  # Paragraph breaks
        ]
        
        # Token estimation pattern
        self.token_pattern = re.compile(r'\b\w+\b|[^\w\s]')
        
        logger.info(f"Initialized SentenceAwareChunker with chunk_size={self.config.chunk_size}, overlap={self.config.overlap}")
    
    def chunk(self, document: Document, config: ChunkingConfig) -> List[Document]:
        """Split document into sentence-aware chunks
        文書を文境界認識チャンクに分割
        
        Args:
            document: Input document to chunk
            config: Chunking configuration
            
        Returns:
            List of chunk documents
        """
        content = document.content
        
        if not content.strip():
            logger.warning(f"Document {document.id} has empty content, skipping chunking")
            return []
        
        # Split into sentences
        sentences = self._split_into_sentences(content)
        
        if not sentences:
            logger.warning(f"No sentences found in document {document.id}")
            return []
        
        # If all sentences fit in one chunk, return single chunk
        total_tokens = sum(self.estimate_tokens(sentence) for sentence in sentences)
        if total_tokens <= config.chunk_size:
            chunk_doc = Document(
                id=self._generate_chunk_id(document.id, 0),
                content=content,
                metadata=document.metadata.copy()
            )
            return [chunk_doc]
        
        # Group sentences into chunks respecting size limits
        chunks = self._group_sentences_into_chunks(sentences, config)
        
        # Create Document objects for chunks
        chunk_docs = []
        for i, chunk_sentences in enumerate(chunks):
            chunk_text = self._join_sentences(chunk_sentences)
            
            # Validate chunk size
            if not self._validate_chunk_size(chunk_text, config):
                logger.warning(f"Skipping invalid chunk {i} for document {document.id}")
                continue
            
            chunk_doc = Document(
                id=self._generate_chunk_id(document.id, i),
                content=chunk_text,
                metadata=document.metadata.copy()
            )
            
            chunk_docs.append(chunk_doc)
        
        logger.debug(f"Created {len(chunk_docs)} sentence-aware chunks from document {document.id}")
        
        return chunk_docs
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text
        テキストのトークン数を推定
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        if not text.strip():
            return 0
        
        tokens = self.token_pattern.findall(text)
        return len(tokens)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences
        テキストを文に分割
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        sentences = []
        current_pos = 0
        
        # Use multiple patterns to find sentence boundaries
        all_matches = []
        for pattern in self.sentence_patterns:
            for match in pattern.finditer(text):
                all_matches.append((match.start(), match.end()))
        
        # Sort matches by position
        all_matches.sort()
        
        # Split text at sentence boundaries
        for start, end in all_matches:
            if start > current_pos:
                sentence = text[current_pos:end].strip()
                if sentence:
                    sentences.append(sentence)
                current_pos = end
        
        # Add remaining text as final sentence
        if current_pos < len(text):
            final_sentence = text[current_pos:].strip()
            if final_sentence:
                sentences.append(final_sentence)
        
        # If no sentence boundaries found, split by line breaks or return whole text
        if not sentences:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            sentences = lines if lines else [text]
        
        return sentences
    
    def _group_sentences_into_chunks(self, sentences: List[str], config: ChunkingConfig) -> List[List[str]]:
        """Group sentences into chunks respecting token limits
        トークン制限を尊重して文をチャンクにグループ化
        
        Args:
            sentences: List of sentences to group
            config: Chunking configuration
            
        Returns:
            List of sentence groups (chunks)
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            # If single sentence exceeds chunk size, split it further
            if sentence_tokens > config.chunk_size:
                # If we have accumulated sentences, save them first
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                
                # Split the large sentence
                sub_chunks = self._split_large_sentence(sentence, config)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding this sentence would exceed the limit
            if current_tokens + sentence_tokens > config.chunk_size and current_chunk:
                # Save current chunk and start a new one
                chunks.append(current_chunk)
                
                # Handle overlap by keeping some sentences from the end
                if config.overlap > 0:
                    overlap_sentences = self._get_overlap_sentences(current_chunk, config)
                    current_chunk = overlap_sentences
                    current_tokens = sum(self.estimate_tokens(s) for s in overlap_sentences)
                else:
                    current_chunk = []
                    current_tokens = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_large_sentence(self, sentence: str, config: ChunkingConfig) -> List[List[str]]:
        """Split a sentence that's too large into smaller chunks
        大きすぎる文をより小さなチャンクに分割
        
        Args:
            sentence: Large sentence to split
            config: Chunking configuration
            
        Returns:
            List of sentence chunks
        """
        # Simple approach: split by commas or other punctuation
        parts = re.split(r'[,;:、；：]', sentence)
        
        if len(parts) == 1:
            # No obvious split points, fall back to word-based splitting
            words = sentence.split()
            part_size = config.chunk_size // 2  # Conservative estimate
            parts = [' '.join(words[i:i+part_size]) for i in range(0, len(words), part_size)]
        
        # Group parts into chunks
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            part_tokens = self.estimate_tokens(part)
            
            if current_tokens + part_tokens > config.chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(part)
            current_tokens += part_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _get_overlap_sentences(self, chunk: List[str], config: ChunkingConfig) -> List[str]:
        """Get sentences from end of chunk for overlap
        重複用にチャンクの終わりから文を取得
        
        Args:
            chunk: Current chunk sentences
            config: Chunking configuration
            
        Returns:
            List of sentences for overlap
        """
        overlap_sentences = []
        overlap_tokens = 0
        
        # Start from the end and work backwards
        for sentence in reversed(chunk):
            sentence_tokens = self.estimate_tokens(sentence)
            
            if overlap_tokens + sentence_tokens > config.overlap:
                break
            
            overlap_sentences.insert(0, sentence)
            overlap_tokens += sentence_tokens
        
        return overlap_sentences
    
    def _join_sentences(self, sentences: List[str]) -> str:
        """Join sentences back into coherent text
        文を一貫性のあるテキストに再結合
        
        Args:
            sentences: List of sentences to join
            
        Returns:
            Joined text
        """
        if not sentences:
            return ""
        
        # Simple joining - could be enhanced to preserve original formatting
        return ' '.join(sentence.strip() for sentence in sentences if sentence.strip())
    
    def _calculate_char_positions(self, original_text: str, chunk_text: str, position: int) -> tuple:
        """Calculate character positions of chunk in original text
        オリジナルテキスト内でのチャンクの文字位置を計算
        
        Args:
            original_text: Original document text
            chunk_text: Chunk text content
            position: Position of chunk in sequence
            
        Returns:
            Tuple of (start_char, end_char)
        """
        # Find the start of the chunk text in original
        start_char = original_text.find(chunk_text[:50])  # Use first 50 chars for matching
        
        if start_char == -1:
            # Fallback to estimated position
            estimated_chars_per_chunk = len(original_text) // max(1, position + 1)
            start_char = position * estimated_chars_per_chunk
            end_char = min(start_char + len(chunk_text), len(original_text))
        else:
            end_char = start_char + len(chunk_text)
        
        return start_char, end_char