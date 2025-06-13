print("vector_store_base.py start")
"""
Base vector store implementation
ベクトルストアの基底実装

This module provides the base VectorStore class that all vector store implementations
should inherit from.

このモジュールは、すべてのベクトルストア実装が継承すべき基底VectorStoreクラスを提供します。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

print("before import Document")
from refinire.rag.models.document import Document
print("after import Document")
print("before import Query")
from refinire.rag.models.query import Query
print("after import Query")


class VectorStore(ABC):
    """
    Abstract base class for vector stores
    ベクトルストアの抽象基底クラス
    
    All vector store implementations must inherit from this class and implement
    its abstract methods.
    
    すべてのベクトルストア実装はこのクラスを継承し、その抽象メソッドを実装する必要があります。
    """
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store
        ベクトルストアにドキュメントを追加
        
        Args:
            documents: List of documents to add
                      追加するドキュメントのリスト
        """
        pass
    
    @abstractmethod
    def search(self, query: Query, top_k: int = 5) -> List[Document]:
        """
        Search for similar documents
        類似ドキュメントを検索
        
        Args:
            query: Query to search with
                   検索に使用するクエリ
            top_k: Number of results to return
                   返す結果の数
                   
        Returns:
            List[Document]: List of similar documents
                           類似ドキュメントのリスト
        """
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the vector store
        ベクトルストアからドキュメントを削除
        
        Args:
            document_ids: List of document IDs to delete
                         削除するドキュメントIDのリスト
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all documents from the vector store
        ベクトルストアからすべてのドキュメントをクリア
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        ベクトルストアの統計情報を取得
        
        Returns:
            Dict[str, Any]: Statistics about the vector store
                           ベクトルストアの統計情報
        """
        pass 