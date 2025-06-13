"""
Token-based text splitter
トークンベースのテキスト分割プロセッサー

This module provides a token-based text splitter that splits text into chunks based on token count.
このモジュールは、トークン数に基づいてテキストをチャンクに分割するトークンベースのテキスト分割プロセッサーを提供します。
"""

from typing import Iterator, Iterable, Optional, Any, List
from refinire.rag.splitter.splitter import Splitter
from refinire.rag.models.document import Document

class TokenTextSplitter(Splitter):
    """
    Token-based text splitter
    トークンベースのテキスト分割プロセッサー
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap_size: int = 0,
        separator: str = " "
    ):
        """
        Initialize token splitter
        トークン分割プロセッサーを初期化

        Args:
            chunk_size: Maximum number of tokens per chunk
            チャンクサイズ: 各チャンクの最大トークン数
            overlap_size: Number of tokens to overlap between chunks
            オーバーラップサイズ: チャンク間のオーバーラップトークン数
            separator: Token separator
            セパレータ: トークンの区切り文字
        """
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
                    id=f"{doc.id}_chunk_{idx}",
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_index': idx,
                        'chunk_start': None,
                        'chunk_end': None,
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
            start_idx = end_idx - overlap_size

        return chunks 