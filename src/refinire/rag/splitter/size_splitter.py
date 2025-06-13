"""
Size-based document splitting processor
サイズベースの文書分割プロセッサー
"""

import logging
from typing import Iterator, Iterable, Optional, Any
from refinire.rag.document_processor import DocumentProcessor
from refinire.rag.models.document import Document

logger = logging.getLogger(__name__)

class SizeSplitter(DocumentProcessor):
    """Processor that splits documents into chunks based on size
    サイズベースで文書を分割するプロセッサー"""
    
    def __init__(
        self,
        chunk_size: int = 1024,  # 1KB default
        overlap_size: int = 0    # No overlap by default
    ):
        """Initialize size splitter
        サイズ分割プロセッサーを初期化
        
        Args:
            chunk_size: Maximum size of each chunk in bytes
            チャンクサイズ: 各チャンクの最大サイズ（バイト）
            overlap_size: Size of overlap between chunks in bytes
            オーバーラップサイズ: チャンク間のオーバーラップサイズ（バイト）
        """
        super().__init__({
            'chunk_size': chunk_size,
            'overlap_size': overlap_size
        })
    
    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """Split documents into chunks based on size
        文書をサイズベースで分割
        
        Args:
            documents: Input documents to process
            config: Optional configuration for splitting
            
        Yields:
            Split documents
        """
        chunk_config = config or self.config
        chunk_size = chunk_config.get('chunk_size', 1024)
        overlap_size = chunk_config.get('overlap_size', 0)
        
        # Prevent infinite loop if overlap_size >= chunk_size
        # overlap_sizeがchunk_size以上の場合は無限ループを防ぐ
        if overlap_size >= chunk_size:
            logger.warning(
                "overlap_size (%d) >= chunk_size (%d). Setting overlap_size = chunk_size - 1.",
                overlap_size, chunk_size
            )
            overlap_size = chunk_size - 1 if chunk_size > 1 else 0
        
        for doc in documents:
            content = doc.content
            content_size = len(content.encode('utf-8'))
            
            if content_size <= chunk_size:
                # If document is smaller than chunk size, yield as is
                yield doc
                continue
            
            # Split content into chunks
            start = 0
            chunk_index = 0
            while start < len(content):
                # Calculate chunk end position
                end = start + chunk_size
                if end > len(content):
                    end = len(content)
                
                # Extract chunk
                chunk_content = content[start:end]
                
                # Create new document for chunk
                chunk_doc = Document(
                    id=f"{doc.id}_chunk_{chunk_index}",
                    content=chunk_content,
                    metadata={
                        **doc.metadata,
                        'chunk_index': chunk_index,
                        'total_chunks': (len(content) + chunk_size - 1) // chunk_size,
                        'chunk_start': start,
                        'chunk_end': end,
                        'original_document_id': doc.id
                    }
                )
                
                yield chunk_doc
                
                # Move to next chunk, considering overlap
                next_start = end - overlap_size
                if next_start <= start:
                    next_start = start + 1
                start = next_start
                chunk_index += 1 