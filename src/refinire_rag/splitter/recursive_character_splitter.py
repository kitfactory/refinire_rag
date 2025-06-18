"""
Recursive character-based text splitter
再帰的文字ベースのテキスト分割プロセッサー

This module provides a recursive character-based text splitter that splits text using multiple levels of separators (e.g., paragraph, sentence, word).
このモジュールは、複数レベルのセパレータ（例：段落、文、単語）を使ってテキストを再帰的に分割する文字ベースのテキスト分割プロセッサーを提供します。
"""

import os
from typing import Iterator, Iterable, Optional, Any, List
from refinire_rag.splitter.splitter import Splitter
from refinire_rag.models.document import Document
from refinire_rag.utils import generate_chunk_id

class RecursiveCharacterTextSplitter(Splitter):
    """
    Recursive character-based text splitter
    再帰的文字ベースのテキスト分割プロセッサー
    """
    def __init__(
        self,
        chunk_size: int = None,
        overlap_size: int = None,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize recursive character splitter
        再帰的文字分割プロセッサーを初期化

        Args:
            chunk_size: Maximum number of characters per chunk. If None, reads from REFINIRE_RAG_RECURSIVE_CHUNK_SIZE environment variable (default: 1000)
            チャンクサイズ: 各チャンクの最大文字数。Noneの場合、REFINIRE_RAG_RECURSIVE_CHUNK_SIZE環境変数から読み取り (デフォルト: 1000)
            overlap_size: Number of characters to overlap between chunks. If None, reads from REFINIRE_RAG_RECURSIVE_OVERLAP environment variable (default: 0)
            オーバーラップサイズ: チャンク間のオーバーラップ文字数。Noneの場合、REFINIRE_RAG_RECURSIVE_OVERLAP環境変数から読み取り (デフォルト: 0)
            separators: List of separators to use for recursive splitting. If None, reads from REFINIRE_RAG_RECURSIVE_SEPARATORS environment variable (default: ["\\n\\n", "\\n", ".", " ", ""])
            セパレータリスト: 再帰的分割に使用するセパレータのリスト。Noneの場合、REFINIRE_RAG_RECURSIVE_SEPARATORS環境変数から読み取り (デフォルト: ["\\n\\n", "\\n", ".", " ", ""])

        Environment variables:
        環境変数:
            REFINIRE_RAG_RECURSIVE_CHUNK_SIZE: Maximum number of characters per chunk (default: 1000)
            チャンクサイズ: 各チャンクの最大文字数 (デフォルト: 1000)
            REFINIRE_RAG_RECURSIVE_OVERLAP: Number of characters to overlap between chunks (default: 0)
            オーバーラップサイズ: チャンク間のオーバーラップ文字数 (デフォルト: 0)
            REFINIRE_RAG_RECURSIVE_SEPARATORS: Comma-separated list of separators (default: "\\n\\n,\\n,., ,")
            セパレータリスト: カンマ区切りのセパレータリスト (デフォルト: "\\n\\n,\\n,., ,")
        """
        # Read from environment variables if arguments are not provided
        # 引数が提供されていない場合は環境変数から読み取り
        if chunk_size is None:
            chunk_size = int(os.getenv('REFINIRE_RAG_RECURSIVE_CHUNK_SIZE', '1000'))
        if overlap_size is None:
            overlap_size = int(os.getenv('REFINIRE_RAG_RECURSIVE_OVERLAP', '0'))
        if separators is None:
            separators_str = os.getenv('REFINIRE_RAG_RECURSIVE_SEPARATORS', '\\n\\n,\\n,., ,')
            
            # Parse separators from comma-separated string, handling escape sequences
            separators = []
            for sep in separators_str.split(','):
                sep = sep.strip()
                # Handle escape sequences
                sep = sep.replace('\\n', '\n').replace('\\t', '\t')
                separators.append(sep)
            
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