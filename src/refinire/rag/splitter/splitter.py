"""
Base splitter for document processing
文書分割用スプリッターの基底クラス
"""

from refinire.rag.document_processor import DocumentProcessor

class Splitter(DocumentProcessor):
    """
    Base class for all document splitters
    すべての文書分割スプリッターの基底クラス
    """
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 200):
        """
        Initialize splitter
        スプリッターを初期化

        Args:
            chunk_size: The target size of each text chunk in characters.
            overlap_size: The number of characters to overlap between chunks.
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size 

    def split(self, documents: list) -> list:
        """
        Split a list of documents into chunks using the process method.
        processメソッドを使って複数のDocumentを分割し、フラットなリストで返す

        Args:
            documents: List of Document objects to split
            分割対象のDocumentのリスト
        Returns:
            List of split Document objects
            分割後のDocumentのリスト
        """
        results = []
        for doc in documents:
            results.extend(self.process(doc))
        return results 