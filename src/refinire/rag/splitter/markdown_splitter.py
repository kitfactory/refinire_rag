"""
MarkdownTextSplitter - Markdown document splitter

A DocumentProcessor that splits markdown documents into chunks while preserving
markdown structure and formatting.
"""

import logging
from typing import List, Optional, Type, Any, Iterator, Dict
import re

from refinire.rag.document import Document
from refinire.rag.splitter.splitter import Splitter

logger = logging.getLogger(__name__)

class MarkdownTextSplitter(Splitter):
    """Markdownテキストを構造を保持しながら分割するスプリッター
    
    Markdownの構造（見出し、リスト、コードブロックなど）を保持しながら、
    テキストを適切なサイズのチャンクに分割します。
    """
    
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 200):
        """
        Initialize the MarkdownTextSplitter
        MarkdownTextSplitterを初期化
        Args:
            chunk_size: Maximum size of each chunk (in characters)
            chunk_size: 各チャンクの最大サイズ（文字数）
            overlap_size: Overlap size between chunks (in characters)
            overlap_size: チャンク間のオーバーラップ（文字数）
        """
        super().__init__(chunk_size, overlap_size)
        self.processing_stats = {
            "headers_preserved": 0,
            "lists_preserved": 0,
            "code_blocks_preserved": 0,
            "tables_preserved": 0
        }
        
    def process(self, documents: List[Document]) -> Iterator[Document]:
        """
        Process documents and split them into chunks
        ドキュメントを処理し、チャンクに分割
        Args:
            documents: List of documents to process
            documents: 処理対象のドキュメントリスト
        Returns:
            Iterator of processed documents
            処理済みドキュメントのイテレータ
        """
        for doc in documents:
            if not doc.content.strip():
                yield doc
                continue
            
            chunks = self._split_text(doc.content)
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata.update({
                    "original_document_id": doc.id,
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "section_index": i // self.chunk_size,
                    "total_sections": (total_chunks + self.chunk_size - 1) // self.chunk_size
                })
                yield Document(
                    content=chunk,
                    id=f"{doc.id}_chunk_{i}",
                    metadata=metadata
                )
                
    def _split_by_headers(self, content: str) -> List[str]:
        """
        Split text by markdown headers (#~######)
        Markdownの見出しでテキストを分割
        Args:
            content: Text to split
            content: 分割対象のテキスト
        Returns:
            List of sections split by headers
            見出しで分割されたテキストのリスト
        """
        header_pattern = r'^#{1,6}\s+.+$'
        lines = content.split('\n')
        sections = []
        current_section = []
        for line in lines:
            if re.match(header_pattern, line):
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
            current_section.append(line)
        if current_section:
            sections.append('\n'.join(current_section))
        return sections
        
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks, always grouping header and its following text (including blank lines) into a single chunk.
        ヘッダーとその直後のテキスト（空行も含む）を常に1チャンクにまとめる
        Args:
            text: Text to split
            text: 分割対象のテキスト
        Returns:
            List of text chunks
            チャンク化されたテキストのリスト
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('#'):
                # 既存のチャンクがあれば追加
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                # 新しいチャンクを開始
                current_chunk.append(line)
                i += 1
                # 次のヘッダーまたはファイル末尾までをこのチャンクに追加
                while i < len(lines) and not lines[i].startswith('#'):
                    current_chunk.append(lines[i])
                    i += 1
                # チャンクを追加
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
            else:
                # ヘッダー以外の行（ファイル先頭やヘッダーなしの場合）
                current_chunk.append(line)
                i += 1
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        # 空チャンクは除外
        return [c for c in chunks if c.strip()]

    def _split_section(self, text: str) -> List[str]:
        """
        Split a section of text into chunks with overlap
        セクション内のテキストをオーバーラップ付きでチャンクに分割
        Args:
            text: Text to split
            text: 分割対象のテキスト
        Returns:
            List of text chunks
            チャンク化されたテキストのリスト
        """
        if not text.strip():
            return [text]
        
        # 文単位で分割
        sentences = []
        current_sentence = []
        for line in text.split('\n'):
            if line.strip():
                current_sentence.append(line)
            else:
                if current_sentence:
                    sentences.append('\n'.join(current_sentence))
                    current_sentence = []
        if current_sentence:
            sentences.append('\n'.join(current_sentence))
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence) + 1
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                # オーバーラップの適用
                if self.overlap_size > 0 and len(current_chunk) > 1:
                    overlap_sentences = current_chunk[-self.overlap_size:]
                    current_chunk = overlap_sentences
                    current_size = sum(len(s) + 1 for s in overlap_sentences)
                else:
                    current_chunk = []
                    current_size = 0
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap to chunks
        チャンクにオーバーラップを適用
        Args:
            chunks: List of text chunks
            chunks: チャンク化されたテキストのリスト
        Returns:
            List of overlapped chunks
            オーバーラップ適用後のチャンクリスト
        """
        if self.overlap_size <= 0 or len(chunks) <= 1:
            return chunks
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                prev = overlapped[-1]
                overlap = prev[-self.overlap_size:] if len(prev) > self.overlap_size else prev
                overlapped.append(overlap + chunk)
        return overlapped 