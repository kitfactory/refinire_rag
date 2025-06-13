from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .models import Document

class CorpusStore(ABC):
    """
    Abstract base class for storing documents with metadata.
    メタデータを持つ文書を保存するための抽象基底クラス
    """
    @abstractmethod
    def add_document(self, document: Document) -> None:
        """
        Add a document to the store.
        ストアに文書を追加する
        """
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Get a document by its ID.
        IDによって文書を取得する
        """
        pass

    @abstractmethod
    def list_documents(self, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        List documents with optional metadata filtering.
        オプションのメタデータフィルタリングを使用して文書をリストする
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """
        Delete a document by its ID.
        IDによって文書を削除する
        """
        pass 