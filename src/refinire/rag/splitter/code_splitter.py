"""
Code text splitter module.

This module provides a text splitter that preserves code structure when splitting text.
"""

from typing import List, Optional
from ..document import Document
from .splitter import Splitter


class CodeTextSplitter(Splitter):
    """
    A text splitter that preserves code structure when splitting text.
    
    This splitter is designed to handle code files and split them while maintaining
    the integrity of code blocks, functions, and other code structures.
    
    Attributes:
        chunk_size (int): The target size of each text chunk in characters.
        overlap_size (int): The number of characters to overlap between chunks.
        language (str): The programming language of the code being split.
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
        chunk_size: int = 1000,
        overlap_size: int = 200,
        language: Optional[str] = None
    ):
        """
        Initialize the CodeTextSplitter.
        
        Args:
            chunk_size: The target size of each text chunk in characters.
            overlap_size: The number of characters to overlap between chunks.
            language: The programming language of the code being split.
        """
        super().__init__(chunk_size, overlap_size)
        self.language = language
    
    def process(self, document: Document) -> List[Document]:
        """
        Process a document by splitting its content while preserving code structure.
        
        Args:
            document: The document to process.
            
        Returns:
            A list of documents containing the split content.
        """
        chunks = self._split_text(document.content)
        return [
            Document(
                content=chunk,
                metadata=document.metadata,
                id=f"{document.id}_{i}"
            )
            for i, chunk in enumerate(chunks)
        ]
    
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