"""
Recursive character-based text splitter
再帰的文字ベースのテキスト分割プロセッサー

This module provides a recursive character-based text splitter that splits text using multiple levels of separators (e.g., paragraph, sentence, word).
このモジュールは、複数レベルのセパレータ（例：段落、文、単語）を使ってテキストを再帰的に分割する文字ベースのテキスト分割プロセッサーを提供します。
"""

from typing import Iterator, Iterable, Optional, Any, List
from refinire.rag.splitter.splitter import Splitter
from refinire.rag.models.document import Document

class RecursiveCharacterTextSplitter(Splitter):
    """
    Recursive character-based text splitter
    再帰的文字ベースのテキスト分割プロセッサー
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap_size: int = 0,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize recursive character splitter
        再帰的文字分割プロセッサーを初期化

        Args:
            chunk_size: Maximum number of characters per chunk
            チャンクサイズ: 各チャンクの最大文字数
            overlap_size: Number of characters to overlap between chunks
            オーバーラップサイズ: チャンク間のオーバーラップ文字数
            separators: List of separators to use for recursive splitting
            セパレータリスト: 再帰的分割に使用するセパレータのリスト
        """
        if separators is None:
            separators = ["\n\n", "\n", ".", " ", ""]
        super().__init__({
            'chunk_size': chunk_size,
            'overlap_size': overlap_size,
            'separators': separators
        })

    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """
        Split documents into chunks recursively using multiple separators
        複数のセパレータを使って文書を再帰的に分割

        Args:
            documents: Input documents to process
            config: Optional configuration for splitting
        Yields:
            Split documents
        """
        chunk_config = config or self.config
        chunk_size = chunk_config.get('chunk_size', 1000)
        overlap_size = chunk_config.get('overlap_size', 0)
        separators = chunk_config.get('separators', ["\n\n", "\n", ".", " ", ""])

        for doc in documents:
            content = doc.content
            if not content:
                continue
            chunks = self._split_text(content, chunk_size, overlap_size, separators)
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

    def _split_text(self, text: str, chunk_size: int, overlap_size: int, separators: List[str]) -> List[str]:
        """
        Recursively split text using the provided separators
        指定されたセパレータを使ってテキストを再帰的に分割

        Args:
            text: Text to split
            chunk_size: Maximum number of characters per chunk
            overlap_size: Number of characters to overlap between chunks
            separators: List of separators to use
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        # Try each separator in order
        for sep in separators:
            if sep == "":
                # Last resort: split by chunk_size
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                if overlap_size > 0 and len(chunks) > 1:
                    overlapped = []
                    for i in range(len(chunks)):
                        if i == 0:
                            overlapped.append(chunks[i])
                        else:
                            prev = chunks[i-1]
                            overlap = prev[-overlap_size:] if len(prev) >= overlap_size else prev
                            overlapped.append(overlap + chunks[i])
                    return overlapped
                return chunks

            # Split by current separator
            parts = text.split(sep)
            if len(parts) == 1:
                continue

            # Try to create chunks
            chunks = []
            current_chunk = ""
            for part in parts:
                if current_chunk:
                    current_chunk += sep
                if len(current_chunk) + len(part) <= chunk_size:
                    current_chunk += part
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = part

            if current_chunk:
                chunks.append(current_chunk)

            # If chunks are too large, try next separator
            if any(len(chunk) > chunk_size for chunk in chunks):
                continue

            # Apply overlap if needed
            if overlap_size > 0 and len(chunks) > 1:
                overlapped = []
                for i in range(len(chunks)):
                    if i == 0:
                        overlapped.append(chunks[i])
                    else:
                        prev = chunks[i-1]
                        overlap = prev[-overlap_size:] if len(prev) >= overlap_size else prev
                        overlapped.append(overlap + chunks[i])
                return overlapped

            return chunks

        # If no separator worked, split by chunk_size
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        if overlap_size > 0 and len(chunks) > 1:
            overlapped = []
            for i in range(len(chunks)):
                if i == 0:
                    overlapped.append(chunks[i])
                else:
                    prev = chunks[i-1]
                    overlap = prev[-overlap_size:] if len(prev) >= overlap_size else prev
                    overlapped.append(overlap + chunks[i])
            return overlapped
        return chunks 