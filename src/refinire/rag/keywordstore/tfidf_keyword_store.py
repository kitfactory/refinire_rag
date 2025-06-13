"""
TF-IDF based keyword store implementation
TF-IDFベースのキーワードストア実装

Concrete implementation of KeywordStore using TF-IDF algorithm for
keyword-based document retrieval.

キーワードベースの文書検索のためのTF-IDFアルゴリズムを使用した
KeywordStoreの具体的な実装。
"""

import logging
import time
from typing import List, Optional, Dict, Any

from refinire.rag.retrieval.base import KeywordStore, SearchResult
from refinire.rag.models.document import Document
from ..plugins.plugin_config import PluginConfig

logger = logging.getLogger(__name__)


class TFIDFKeywordStore(KeywordStore):
    """TF-IDF based keyword search implementation
    TF-IDFベースのキーワード検索実装
    
    Simple implementation using scikit-learn's TfidfVectorizer.
    scikit-learnのTfidfVectorizerを使用したシンプルな実装。
    """
    
    def __init__(self, config: Optional[PluginConfig] = None):
        """
        Initialize TF-IDF keyword store
        TF-IDFキーワードストアを初期化
        
        Args:
            config: KeywordStore configuration
                   KeywordStore設定
        """
        super().__init__(config or PluginConfig().for_plugin_type("keyword_store"))
        self.documents: Dict[str, Document] = {}
        self.vectorizer = None
        self.tfidf_matrix = None
        self.doc_ids = []
        self.index_built = False
        
        logger.info("Initialized TFIDFKeywordStore")
    
    def retrieve(self, 
                 query: str, 
                 limit: Optional[int] = None,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Retrieve relevant documents using TF-IDF keyword search
        TF-IDFキーワード検索を使用して関連文書を取得
        
        Args:
            query: Search query text
                  検索クエリテキスト
            limit: Maximum number of results (uses config.top_k if None)
                  結果の最大数（Noneの場合はconfig.top_kを使用）
            metadata_filter: Metadata filters for constraining search
                           検索を制約するメタデータフィルタ
                           
        Returns:
            List[SearchResult]: Search results sorted by relevance
                               関連度でソートされた検索結果
        """
        start_time = time.time()
        limit = limit or self.config.top_k
        
        try:
            logger.debug(f"TF-IDF search for query: '{query}' (limit={limit})")
            
            # Ensure index is built
            if not self.index_built:
                self._build_index()
                self.index_built = True
            
            # Perform TF-IDF search
            search_results_raw = self._search_index(query, limit * 2)  # Get more for filtering
            
            # Convert to SearchResult objects with filtering
            search_results = []
            for doc_id, score in search_results_raw:
                if doc_id not in self.documents:
                    continue
                
                doc = self.documents[doc_id]
                
                # Apply similarity threshold filtering
                if self.config.enable_filtering and score < self.config.similarity_threshold:
                    continue
                
                # Apply metadata filtering
                if self.config.enable_filtering and metadata_filter:
                    if not self._matches_metadata_filter(doc.metadata, metadata_filter):
                        continue
                
                search_result = SearchResult(
                    document_id=doc_id,
                    document=doc,
                    score=score,
                    metadata={
                        "retrieval_method": "keyword_search",
                        "algorithm": "tfidf",
                        "query_length": len(query),
                        "keyword_store": "TFIDFKeywordStore"
                    }
                )
                search_results.append(search_result)
                
                # Stop when we have enough results
                if len(search_results) >= limit:
                    break
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats["queries_processed"] += 1
            self.processing_stats["processing_time"] += processing_time
            
            logger.debug(f"TF-IDF search completed: {len(search_results)} results in {processing_time:.3f}s")
            return search_results
            
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            logger.error(f"TF-IDF search failed: {e}")
            return []
    
    def index_document(self, document: Document) -> None:
        """
        Index a single document for TF-IDF search
        TF-IDF検索用に単一文書をインデックス
        
        Args:
            document: Document to index
                     インデックスする文書
        """
        try:
            self.documents[document.id] = document
            # Mark index as needing rebuild
            self.index_built = False
            
            logger.debug(f"Added document to TF-IDF index: {document.id}")
            
        except Exception as e:
            logger.error(f"Failed to index document {document.id}: {e}")
            raise
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index multiple documents efficiently
        複数の文書を効率的にインデックス
        
        Args:
            documents: List of documents to index
                      インデックスする文書のリスト
        """
        try:
            for doc in documents:
                self.documents[doc.id] = doc
            
            # Mark index as needing rebuild
            self.index_built = False
            
            logger.info(f"Added {len(documents)} documents to TF-IDF index")
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise
    
    def remove_document(self, document_id: str) -> bool:
        """
        Remove document from TF-IDF index
        TF-IDFインデックスから文書を削除
        
        Args:
            document_id: ID of document to remove
                        削除する文書のID
                        
        Returns:
            bool: True if document was found and removed
                 文書が見つかって削除された場合True
        """
        try:
            if document_id in self.documents:
                del self.documents[document_id]
                # Mark index as needing rebuild
                self.index_built = False
                logger.debug(f"Removed document from TF-IDF index: {document_id}")
                return True
            else:
                logger.warning(f"Document not found in TF-IDF index: {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove document {document_id}: {e}")
            return False
    
    def update_document(self, document: Document) -> bool:
        """
        Update an existing document in the TF-IDF index
        TF-IDFインデックスの既存文書を更新
        
        Args:
            document: Updated document
                     更新する文書
                     
        Returns:
            bool: True if document was found and updated
                 文書が見つかって更新された場合True
        """
        try:
            if document.id in self.documents:
                self.documents[document.id] = document
                # Mark index as needing rebuild
                self.index_built = False
                logger.debug(f"Updated document in TF-IDF index: {document.id}")
                return True
            else:
                logger.warning(f"Document not found for update: {document.id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update document {document.id}: {e}")
            return False
    
    def clear_index(self) -> None:
        """
        Remove all documents from the TF-IDF index
        TF-IDFインデックスからすべての文書を削除
        """
        try:
            self.documents.clear()
            self._clear_index()
            self.index_built = False
            logger.info("Cleared TF-IDF index")
            
        except Exception as e:
            logger.error(f"Failed to clear TF-IDF index: {e}")
            raise
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the TF-IDF index
        TF-IDFインデックスの文書数を取得
        
        Returns:
            int: Number of indexed documents
                インデックスされた文書数
        """
        return len(self.documents)
    
    def _build_index(self) -> None:
        """Build TF-IDF index from documents
        文書からTF-IDFインデックスを構築"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            if not self.documents:
                return
            
            # Extract document texts
            self.doc_ids = list(self.documents.keys())
            doc_texts = [self.documents[doc_id].content for doc_id in self.doc_ids]
            
            # Build TF-IDF matrix
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(doc_texts)
            
            logger.info(f"Built TF-IDF index for {len(doc_texts)} documents")
            
        except ImportError:
            logger.error("scikit-learn required for TF-IDF implementation")
            raise
        except Exception as e:
            logger.error(f"Failed to build TF-IDF index: {e}")
            raise
    
    def _search_index(self, query: str, limit: int) -> List[tuple]:
        """Search TF-IDF index
        TF-IDFインデックスを検索
        
        Args:
            query: Search query
                  検索クエリ
            limit: Maximum results
                  最大結果数
            
        Returns:
            List of (document_id, score) tuples
            (文書ID, スコア)のタプルのリスト
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            if self.vectorizer is None or self.tfidf_matrix is None:
                return []
            
            # Vectorize query
            query_vec = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[::-1][:limit]
            
            results = []
            for idx in top_indices:
                doc_id = self.doc_ids[idx]
                score = float(similarities[idx])
                if score > 0:  # Only include non-zero similarities
                    results.append((doc_id, score))
            
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}")
            return []
    
    def _clear_index(self) -> None:
        """Clear TF-IDF index
        TF-IDFインデックスをクリア"""
        self.vectorizer = None
        self.tfidf_matrix = None
        self.doc_ids = []
    
    def _matches_metadata_filter(self, metadata: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
        """
        Check if document metadata matches the filter
        文書のメタデータがフィルタに一致するかチェック
        
        Args:
            metadata: Document metadata
                     文書のメタデータ
            metadata_filter: Filter conditions
                           フィルタ条件
            
        Returns:
            bool: True if metadata matches filter
                 メタデータがフィルタに一致する場合True
        """
        for key, value in metadata_filter.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                # OR condition: metadata value must be in the list
                if metadata[key] not in value:
                    return False
            elif isinstance(value, dict):
                # Range or complex condition (could be extended)
                if "$gte" in value and metadata[key] < value["$gte"]:
                    return False
                if "$lte" in value and metadata[key] > value["$lte"]:
                    return False
                if "$ne" in value and metadata[key] == value["$ne"]:
                    return False
            else:
                # Exact match
                if metadata[key] != value:
                    return False
        
        return True
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics with TF-IDF-specific metrics
        TF-IDF固有のメトリクスを含む処理統計を取得"""
        stats = super().get_processing_stats()
        
        # Add TF-IDF-specific stats
        stats.update({
            "retriever_type": "TFIDFKeywordStore",
            "algorithm": "tfidf",
            "similarity_threshold": self.config.similarity_threshold,
            "top_k": self.config.top_k,
            "document_count": self.get_document_count(),
            "index_built": self.index_built
        })
        
        return stats