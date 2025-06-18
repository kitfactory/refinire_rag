"""
Code text splitter module.
コードテキスト分割モジュール

This module provides a text splitter that preserves code structure when splitting text.
このモジュールは、テキストを分割する際にコード構造を保持するテキスト分割プロセッサーを提供します。
"""

import os
from typing import List, Optional, Iterable, Iterator, Any
from refinire_rag.models.document import Document
from .splitter import Splitter
from refinire_rag.utils import generate_chunk_id


class CodeTextSplitter(Splitter):
    """
    A text splitter that preserves code structure when splitting text.
    コード構造を保持してテキストを分割するテキスト分割プロセッサー
    
    This splitter is designed to handle code files and split them while maintaining
    the integrity of code blocks, functions, and other code structures.
    このスプリッターは、コードブロック、関数、その他のコード構造の完全性を維持しながら
    コードファイルを処理し、分割するように設計されています。
    
    Attributes:
        chunk_size (int): The target size of each text chunk in characters.
        チャンクサイズ: 各テキストチャンクの目標サイズ（文字数）
        overlap_size (int): The number of characters to overlap between chunks.
        オーバーラップサイズ: チャンク間でオーバーラップする文字数
        language (str): The programming language of the code being split.
        言語: 分割されるコードのプログラミング言語
    """
    
    # デフォルトの区切り文字
    default_delimiters = [
        "\n\n",  # Double newline
        "\n",    # Single newline
        ";",     # Statement end
        "}",     # Block end
        "{",     # Block start
        ")",     # Function call end
        "(",     # Function call start
        ",",     # Parameter separator
        " ",     # Space
    ]
    # 言語ごとの区切り文字
    language_delimiters = {
        "python": ["\n\n", "\n", ":", " ", "#"],
        "javascript": ["\n\n", "\n", ";", "}", "{", ",", " "],
        "java": ["\n\n", "\n", ";", "}", "{", ",", " "],
        # 必要に応じて他の言語も追加
    }
    
    def __init__(
        self,
        chunk_size: int = None,
        overlap_size: int = None,
        language: Optional[str] = None
    ):
        """
        Initialize the CodeTextSplitter.
        CodeTextSplitterを初期化
        
        Args:
            chunk_size: The target size of each text chunk in characters. If None, reads from REFINIRE_RAG_CODE_CHUNK_SIZE environment variable (default: 1000)
            チャンクサイズ: 各テキストチャンクの目標サイズ（文字数）。Noneの場合、REFINIRE_RAG_CODE_CHUNK_SIZE環境変数から読み取り (デフォルト: 1000)
            overlap_size: The number of characters to overlap between chunks. If None, reads from REFINIRE_RAG_CODE_OVERLAP environment variable (default: 200)
            オーバーラップサイズ: チャンク間でオーバーラップする文字数。Noneの場合、REFINIRE_RAG_CODE_OVERLAP環境変数から読み取り (デフォルト: 200)
            language: The programming language of the code being split. If None, reads from REFINIRE_RAG_CODE_LANGUAGE environment variable (default: None)
            言語: 分割されるコードのプログラミング言語。Noneの場合、REFINIRE_RAG_CODE_LANGUAGE環境変数から読み取り (デフォルト: None)

        Environment variables:
        環境変数:
            REFINIRE_RAG_CODE_CHUNK_SIZE: The target size of each text chunk in characters (default: 1000)
            チャンクサイズ: 各テキストチャンクの目標サイズ（文字数）(デフォルト: 1000)
            REFINIRE_RAG_CODE_OVERLAP: The number of characters to overlap between chunks (default: 200)
            オーバーラップサイズ: チャンク間でオーバーラップする文字数 (デフォルト: 200)
            REFINIRE_RAG_CODE_LANGUAGE: The programming language of the code being split (default: None)
            言語: 分割されるコードのプログラミング言語 (デフォルト: None)
        """
        # Read from environment variables if arguments are not provided
        # 引数が提供されていない場合は環境変数から読み取り
        if chunk_size is None:
            chunk_size = int(os.getenv('REFINIRE_RAG_CODE_CHUNK_SIZE', '1000'))
        if overlap_size is None:
            overlap_size = int(os.getenv('REFINIRE_RAG_CODE_OVERLAP', '200'))
        if language is None:
            language = os.getenv('REFINIRE_RAG_CODE_LANGUAGE')
            
        super().__init__({
            'chunk_size': chunk_size,
            'overlap_size': overlap_size,
            'language': language
        })
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.language = language

    
    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """
        Process documents by splitting their content while preserving code structure.
        
        Args:
            documents: Documents to process
            config: Optional configuration for splitting
            
        Yields:
            Split documents
        """
        for document in documents:
            chunks = self._split_text(document.content)
            for i, chunk in enumerate(chunks):
                yield Document(
                    content=chunk,
                    metadata={
                        **document.metadata,
                        'chunk_index': i,
                        'origin_id': document.id,
                        'original_document_id': document.id
                    },
                    id=generate_chunk_id()
                )
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks while preserving code structure."""
        if not text:
            return []

        # 関数ごとに分割するロジック
        lines = text.split('\n')
        function_chunks = []
        current_chunk = ""
        for line in lines:
            if line.strip().startswith('def '):
                if current_chunk:
                    function_chunks.append(current_chunk)
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += '\n' + line
                else:
                    current_chunk = line
        if current_chunk:
            function_chunks.append(current_chunk)

        # オーバーラップ処理
        if self.overlap_size > 0 and len(function_chunks) > 1:
            overlapped = []
            for i, chunk in enumerate(function_chunks):
                if i == 0:
                    overlapped.append(chunk)
                else:
                    prev = function_chunks[i-1]
                    overlap = prev[-self.overlap_size:] if len(prev) > self.overlap_size else prev
                    overlapped.append(overlap + chunk)
            return overlapped
        return function_chunks 