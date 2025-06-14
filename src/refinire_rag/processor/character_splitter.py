"""
Character-based text splitting processor
文字ベースのテキスト分割プロセッサー
"""

import logging
from typing import Iterator, Iterable, Optional, Any, List
from refinire_rag.document_processor import DocumentProcessor
from refinire_rag.models.document import Document

logger = logging.getLogger(__name__)

class CharacterTextSplitter(DocumentProcessor):
    """Processor that splits documents based on character separators
    文字セパレータを基準に文書を分割するプロセッサー"""
    
    def __init__(
        self,
        separator: str = "\n\n",  # Default separator is double newline
        keep_separator: bool = False  # Whether to keep the separator at the end of each chunk
    ):
        """Initialize character text splitter
        文字テキスト分割プロセッサーを初期化
        
        Args:
            separator: Character(s) to split on
            セパレータ: 分割に使用する文字
            keep_separator: Whether to keep the separator at the end of each chunk
            セパレータ保持: 各チャンクの末尾にセパレータを保持するかどうか
        """
        super().__init__({
            'separator': separator,
            'keep_separator': keep_separator
        })
    
    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """Split documents based on character separators
        文字セパレータを基準に文書を分割
        
        Args:
            documents: Input documents to process
            config: Optional configuration for splitting
            
        Yields:
            Split documents
        """
        split_config = config or self.config
        separator = split_config.get('separator', "\n\n")
        keep_separator = split_config.get('keep_separator', False)
        
        for doc in documents:
            # Split the content by separator
            chunks = self._split_text(
                doc.content,
                separator,
                keep_separator
            )
            
            # Create new documents for each chunk
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    id=f"{doc.id}_chunk_{i}",
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'original_document_id': doc.id
                    }
                )
                yield chunk_doc
    
    def _split_text(
        self,
        text: str,
        separator: str,
        keep_separator: bool
    ) -> List[str]:
        """Split text into chunks based on separator only
        セパレータのみでテキストを分割
        
        Args:
            text: Text to split
            separator: Character(s) to split on
            keep_separator: Whether to keep the separator at the end of each chunk
            
        Returns:
            List of text chunks
        """
        if not separator:
            return [text]
        parts = text.split(separator)
        if keep_separator:
            # Add separator to the end of each chunk except the last
            return [p + separator for p in parts[:-1]] + [parts[-1]] if len(parts) > 1 else parts
        else:
            return parts 