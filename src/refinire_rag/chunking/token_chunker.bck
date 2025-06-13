"""
Token-based chunking implementation
"""

import logging
import re
from typing import List, Optional
from datetime import datetime

from ..models import Document
from .chunker import Chunker, ChunkingConfig

logger = logging.getLogger(__name__)


class TokenBasedChunker(Chunker):
    """Token-based document chunker
    トークンベースの文書チャンカー"""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize token-based chunker
        トークンベースチャンカーを初期化
        
        Args:
            config: Optional chunking configuration
        """
        super().__init__(config)
        
        # Simple tokenization pattern (can be enhanced)
        self.token_pattern = re.compile(r'\b\w+\b')
        
        logger.info(f"Initialized TokenBasedChunker with chunk_size={self.config.chunk_size}, overlap={self.config.overlap}")
    
    def chunk(self, document: Document, config: ChunkingConfig) -> List[Document]:
        """Split document into token-based chunks
        文書をトークンベースのチャンクに分割
        
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
        
        # Tokenize the content
        tokens = self._tokenize(content)
        
        if len(tokens) <= config.chunk_size:
            # Document is small enough to be a single chunk
            chunk_doc = Document(
                id=self._generate_chunk_id(document.id, 0),
                content=content,
                metadata=document.metadata.copy()
            )
            return [chunk_doc]
        
        # Split into overlapping chunks
        chunks = self._create_overlapping_chunks(tokens, config)
        
        # Convert token chunks back to text and create Document objects
        chunk_docs = []
        for i, chunk_tokens in enumerate(chunks):
            # Reconstruct text from tokens (simple approach)
            chunk_text = self._reconstruct_text(chunk_tokens, content)
            
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
        
        logger.debug(f"Created {len(chunk_docs)} token-based chunks from document {document.id}")
        
        return chunk_docs
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using simple tokenization
        簡単なトークン化を使用してトークン数を推定
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        if not text.strip():
            return 0
        
        tokens = self.token_pattern.findall(text)
        return len(tokens)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into list of tokens
        テキストをトークンのリストに分割
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        return self.token_pattern.findall(text)
    
    def _create_overlapping_chunks(self, tokens: List[str], config: ChunkingConfig) -> List[List[str]]:
        """Create overlapping chunks from tokens
        トークンから重複チャンクを作成
        
        Args:
            tokens: List of tokens to chunk
            config: Chunking configuration
            
        Returns:
            List of token chunks
        """
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + config.chunk_size, len(tokens))
            chunk = tokens[start:end]
            chunks.append(chunk)
            
            if end >= len(tokens):
                break
            
            # Move start position considering overlap
            start = end - config.overlap
            
            # Ensure we make progress even with large overlap
            if len(chunks) > 1:
                # Get the start position of the previous chunk for comparison
                prev_chunk_start_token = chunks[-2][0] if chunks[-2] else ""
                # Simple progress check - ensure we're moving forward
                if start <= end - config.chunk_size:
                    start = end - min(config.overlap, config.chunk_size // 2)
        
        return chunks
    
    def _reconstruct_text(self, tokens: List[str], original_text: str) -> str:
        """Reconstruct text from tokens, preserving spacing
        トークンからテキストを再構築し、スペースを保持
        
        Args:
            tokens: List of tokens to reconstruct
            original_text: Original text for reference
            
        Returns:
            Reconstructed text
        """
        if not tokens:
            return ""
        
        # Simple reconstruction - join with spaces
        # In a more sophisticated implementation, you would preserve
        # the original spacing and punctuation
        reconstructed = " ".join(tokens)
        
        # Try to preserve sentence boundaries
        reconstructed = self._preserve_sentence_boundaries(reconstructed, original_text)
        
        return reconstructed
    
    def _preserve_sentence_boundaries(self, reconstructed: str, original_text: str) -> str:
        """Attempt to preserve sentence boundaries in reconstructed text
        再構築されたテキストで文境界を保持しようとする
        
        Args:
            reconstructed: Reconstructed text from tokens
            original_text: Original text for reference
            
        Returns:
            Text with improved sentence boundaries
        """
        # Simple approach: add periods at the end if missing and it looks like a sentence
        if not reconstructed.endswith(('.', '!', '?', '。', '！', '？')):
            # Check if the original text near this content has sentence endings
            if any(punct in original_text for punct in ['.', '!', '?', '。', '！', '？']):
                # Don't add punctuation automatically - preserve as is
                pass
        
        return reconstructed
    
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
        # Simple approach - find first occurrence
        # In practice, you'd want more sophisticated position tracking
        start_char = original_text.find(chunk_text.strip())
        if start_char == -1:
            # Fallback to estimated position
            estimated_chars_per_chunk = len(original_text) // max(1, position + 1)
            start_char = position * estimated_chars_per_chunk
            end_char = min(start_char + len(chunk_text), len(original_text))
        else:
            end_char = start_char + len(chunk_text)
        
        return start_char, end_char