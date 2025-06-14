"""
HTML-based text splitter
HTMLベースのテキスト分割プロセッサー

This module provides an HTML-aware text splitter that splits text into chunks based on HTML structure.
このモジュールは、HTML構造に基づいてテキストをチャンクに分割する分割プロセッサーを提供します。
"""

import re
from typing import Iterator, Iterable, Optional, Any, List
from refinire_rag.splitter.splitter import Splitter
from refinire_rag.models.document import Document
from refinire_rag.utils import generate_chunk_id

class HTMLTextSplitter(Splitter):
    """
    HTML-based text splitter
    HTMLベースのテキスト分割プロセッサー
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap_size: int = 0,
        separator: str = "\n"
    ):
        """
        Initialize HTML splitter
        HTML分割プロセッサーを初期化

        Args:
            chunk_size: Maximum number of characters per chunk
            チャンクサイズ: 各チャンクの最大文字数
            overlap_size: Number of characters to overlap between chunks
            オーバーラップサイズ: チャンク間のオーバーラップ文字数
            separator: Separator for joining HTML blocks
            セパレータ: HTMLブロックを結合する区切り文字
        """
        super().__init__({
            'chunk_size': chunk_size,
            'overlap_size': overlap_size,
            'separator': separator
        })

    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """
        Split documents into chunks based on HTML structure
        HTML構造に基づいて文書をチャンクに分割

        Args:
            documents: Input documents to process
            config: Optional configuration for splitting
        Yields:
            Split documents
        """
        chunk_config = config or self.config
        chunk_size = chunk_config.get('chunk_size', 1000)
        overlap_size = chunk_config.get('overlap_size', 0)
        separator = chunk_config.get('separator', "\n")

        for doc in documents:
            content = doc.content
            if not content:
                continue
            chunks = self._split_html(content, chunk_size, overlap_size, separator)
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

    def _split_html(self, text: str, chunk_size: int, overlap_size: int, separator: str) -> List[str]:
        """
        Split HTML text into chunks based on tags
        HTMLのタグごとにテキストをチャンクに分割

        Args:
            text: HTML text to split
            chunk_size: Maximum number of characters per chunk
            overlap_size: Number of characters to overlap between chunks
            separator: Separator for joining HTML blocks
        Returns:
            List of HTML text chunks
        """
        # HTMLのブロック単位（タグごと）で分割
        blocks = re.split(r'(<[^>]+>.*?</[^>]+>|<[^>]+>)', text)
        # 空要素除去
        blocks = [b for b in blocks if b and b.strip()]

        # チャンク生成
        chunks = []
        current = []
        current_len = 0
        for block in blocks:
            block_len = len(block)
            if current_len + block_len > chunk_size and current:
                chunks.append(separator.join(current))
                # オーバーラップ処理
                if overlap_size > 0:
                    overlap_text = separator.join(current)[-overlap_size:]
                    current = [overlap_text]
                    current_len = len(overlap_text)
                else:
                    current = []
                    current_len = 0
            current.append(block)
            current_len += block_len
        if current:
            chunks.append(separator.join(current))
        return chunks 