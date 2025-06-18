"""
Base splitter for document processing
文書分割用スプリッターの基底クラス
"""

from refinire_rag.document_processor import DocumentProcessor

class Splitter(DocumentProcessor):
    """
    Base class for all document splitters
    すべての文書分割スプリッターの基底クラス
    """
    def __init__(self, config: dict = None):
        """
        Initialize splitter
        スプリッターを初期化

        Args:
            config: Configuration dictionary with chunk_size, overlap_size, etc.
        """
        if config is None:
            config = {'chunk_size': 1000, 'overlap_size': 200}
        super().__init__(config)
        self.chunk_size = config.get('chunk_size', 1000)
        self.overlap_size = config.get('overlap_size', 200) 

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
        # Pass the documents as an iterable to process method
        results.extend(list(self.process(documents)))
        return results 