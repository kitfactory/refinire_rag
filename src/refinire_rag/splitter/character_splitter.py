"""
Character-based text splitter
文字ベースのテキスト分割プロセッサー

This module provides a character-based text splitter that splits text into chunks of specified size.
このモジュールは、テキストを指定されたサイズのチャンクに分割する文字ベースのテキスト分割プロセッサーを提供します。
"""

from typing import Iterator, Iterable, Optional, Any
from refinire_rag.splitter.splitter import Splitter
from refinire_rag.models.document import Document
from refinire_rag.utils import generate_chunk_id

class CharacterTextSplitter(Splitter):
    """
    Processor that splits documents into chunks based on character count
    文字数ベースで文書を分割するプロセッサー
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap_size: int = 0
    ):
        """
        Initialize character splitter
        文字分割プロセッサーを初期化

        Args:
            chunk_size: Maximum number of characters per chunk
            チャンクサイズ: 各チャンクの最大文字数
            overlap_size: Number of characters to overlap between chunks
            オーバーラップサイズ: チャンク間のオーバーラップ文字数
        """
        super().__init__({
            'chunk_size': chunk_size,
            'overlap_size': overlap_size
        })
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """
        Split documents into chunks based on character count
        文書を文字数ベースで分割

        Args:
            documents: Input documents to process
            config: Optional configuration for splitting
        Yields:
            Split documents
        """
        chunk_size = self.chunk_size
        overlap_size = self.overlap_size

        # Prevent infinite loop if overlap_size >= chunk_size
        if overlap_size >= chunk_size:
            overlap_size = chunk_size - 1 if chunk_size > 1 else 0

        for doc in documents:
            content = doc.content
            if not content:
                continue

            content_len = len(content)
            if chunk_size == 0:
                continue
            
            # Calculate total chunks first
            chunks = []
            start = 0
            chunk_index = 0
            
            while start + chunk_size < content_len:
                end = start + chunk_size
                chunk_content = content[start:end]
                chunks.append({
                    'content': chunk_content,
                    'index': chunk_index,
                    'start': start,
                    'end': end
                })
                start = end - overlap_size
                chunk_index += 1
                
            # Add remainder as last chunk
            if start < content_len:
                chunk_content = content[start:content_len]
                chunks.append({
                    'content': chunk_content,
                    'index': chunk_index,
                    'start': start,
                    'end': content_len
                })
            
            total_chunks = len(chunks)
            
            # Yield all chunks with correct total_chunks metadata
            for chunk_info in chunks:
                chunk_doc = Document(
                    id=generate_chunk_id(),
                    content=chunk_info['content'],
                    metadata={
                        **doc.metadata,
                        'origin_id': doc.id,
                        'chunk_index': chunk_info['index'],
                        'chunk_start': chunk_info['start'],
                        'chunk_end': chunk_info['end'],
                        'total_chunks': total_chunks
                    }
                )
                yield chunk_doc 