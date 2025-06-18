"""
Token-based text splitter
トークンベースのテキスト分割プロセッサー

This module provides a token-based text splitter that splits text into chunks based on token count.
このモジュールは、トークン数に基づいてテキストをチャンクに分割するトークンベースのテキスト分割プロセッサーを提供します。
"""

import os
from typing import Iterator, Iterable, Optional, Any, List
from refinire_rag.splitter.splitter import Splitter
from refinire_rag.models.document import Document
from refinire_rag.utils import generate_chunk_id

class TokenTextSplitter(Splitter):
    """
    Token-based text splitter
    トークンベースのテキスト分割プロセッサー
    """
    def __init__(
        self,
        chunk_size: int = None,
        overlap_size: int = None,
        separator: str = None
    ):
        """
        Initialize token splitter
        トークン分割プロセッサーを初期化

        Args:
            chunk_size: Maximum number of tokens per chunk. If None, reads from REFINIRE_RAG_TOKEN_CHUNK_SIZE environment variable (default: 1000)
            チャンクサイズ: 各チャンクの最大トークン数。Noneの場合、REFINIRE_RAG_TOKEN_CHUNK_SIZE環境変数から読み取り (デフォルト: 1000)
            overlap_size: Number of tokens to overlap between chunks. If None, reads from REFINIRE_RAG_TOKEN_OVERLAP environment variable (default: 0)
            オーバーラップサイズ: チャンク間のオーバーラップトークン数。Noneの場合、REFINIRE_RAG_TOKEN_OVERLAP環境変数から読み取り (デフォルト: 0)
            separator: Token separator. If None, reads from REFINIRE_RAG_TOKEN_SEPARATOR environment variable (default: " ")
            セパレータ: トークンの区切り文字。Noneの場合、REFINIRE_RAG_TOKEN_SEPARATOR環境変数から読み取り (デフォルト: " ")

        Environment variables:
        環境変数:
            REFINIRE_RAG_TOKEN_CHUNK_SIZE: Maximum number of tokens per chunk (default: 1000)
            チャンクサイズ: 各チャンクの最大トークン数 (デフォルト: 1000)
            REFINIRE_RAG_TOKEN_OVERLAP: Number of tokens to overlap between chunks (default: 0)
            オーバーラップサイズ: チャンク間のオーバーラップトークン数 (デフォルト: 0)
            REFINIRE_RAG_TOKEN_SEPARATOR: Token separator (default: " ")
            セパレータ: トークンの区切り文字 (デフォルト: " ")
        """
        # Read from environment variables if arguments are not provided
        # 引数が提供されていない場合は環境変数から読み取り
        if chunk_size is None:
            chunk_size = int(os.getenv('REFINIRE_RAG_TOKEN_CHUNK_SIZE', '1000'))
        if overlap_size is None:
            overlap_size = int(os.getenv('REFINIRE_RAG_TOKEN_OVERLAP', '0'))
        if separator is None:
            separator = os.getenv('REFINIRE_RAG_TOKEN_SEPARATOR', ' ')
            
        super().__init__({
            'chunk_size': chunk_size,
            'overlap_size': overlap_size,
            'separator': separator
        })

    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """
        Split documents into chunks based on token count
        トークン数に基づいて文書をチャンクに分割

        Args:
            documents: Input documents to process
            config: Optional configuration for splitting
        Yields:
            Split documents
        """
        chunk_config = config or self.config
        chunk_size = chunk_config.get('chunk_size', 1000)
        overlap_size = chunk_config.get('overlap_size', 0)
        separator = chunk_config.get('separator', " ")

        for doc in documents:
            content = doc.content
            if not content:
                continue
            chunks = self._split_text(content, chunk_size, overlap_size, separator)
            for idx, chunk in enumerate(chunks):
                yield Document(
                    id=generate_chunk_id(),
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_index': idx,
                        'chunk_start': None,
                        'chunk_end': None,
                        'origin_id': doc.id,
                        'original_document_id': doc.id
                    }
                )

    def _split_text(self, text: str, chunk_size: int, overlap_size: int, separator: str) -> List[str]:
        """
        Split text into chunks based on token count
        トークン数に基づいてテキストをチャンクに分割

        Args:
            text: Text to split
            chunk_size: Maximum number of tokens per chunk
            overlap_size: Number of tokens to overlap between chunks
            separator: Token separator
        Returns:
            List of text chunks
        """
        # Handle empty text
        if not text:
            return []
            
        # Split text into tokens
        tokens = text.split(separator)
        if not tokens:
            return []

        # If text is shorter than chunk size, return as is
        if len(tokens) <= chunk_size:
            return [text]

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            # Calculate end index for current chunk
            end_idx = min(start_idx + chunk_size, len(tokens))
            
            # Create chunk
            chunk = separator.join(tokens[start_idx:end_idx])
            chunks.append(chunk)

            # Move to next chunk, considering overlap
            if end_idx == len(tokens):
                break
            # Ensure start_idx always advances to prevent infinite loops
            start_idx = max(start_idx + 1, end_idx - overlap_size)

        return chunks 