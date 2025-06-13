"""
OpenAI Embeddings-based Vector Store implementation
OpenAI埋め込みベースのベクトルストア実装

Concrete implementation of VectorStore using OpenAI embeddings and
a configurable backend vector store.

OpenAI埋め込みと設定可能なバックエンドベクトルストアを使用した
VectorStoreの具体的な実装。
"""

import logging
import time
from typing import List, Optional, Dict, Any

from refinire.rag.retrieval.base import VectorStore, SearchResult
from refinire.rag.models.document import Document
from refinire.rag.storage.vector_store import VectorStore as BackendVectorStore, VectorEntry
from refinire.rag.embedding.base import Embedder

logger = logging.getLogger(__name__)


class OpenAIVectorStore(VectorStore):
    """OpenAI Embeddings-based vector store implementation
    OpenAI埋め込みベースのベクトルストア実装
    
    Uses OpenAI embeddings to convert documents to vectors and a backend
    vector store for storage and similarity search.
    
    OpenAI埋め込みを使用して文書をベクトルに変換し、バックエンドベクトル
    ストアを使用してストレージと類似検索を行います。
    """
    
    def __init__(self, 
                 backend_store: BackendVectorStore,
                 embedder: Embedder,
                 config: Optional[PluginConfig] = None):
        """
        Initialize OpenAI VectorStore
        OpenAI VectorStoreを初期化
        
        Args:
            backend_store: Backend vector store for storage
                          ストレージ用のバックエンドベクトルストア
            embedder: Embedder for converting text to vectors
                     テキストをベクトルに変換する埋め込み器
            config: VectorStore configuration
                   VectorStore設定
        """
        super().__init__(config or PluginConfig().for_plugin_type("vector_store"))
        self.backend_store = backend_store
        self.embedder = embedder
        
        logger.info(f"Initialized OpenAIVectorStore with {type(backend_store).__name__} and {type(embedder).__name__}")
    
    def retrieve(self, 
                 query: str, 
                 limit: Optional[int] = None,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Retrieve relevant documents using vector similarity search
        ベクトル類似度検索を使用して関連文書を取得
        
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
            logger.debug(f"Vector search for query: '{query}' (limit={limit})")
            
            # Generate query embedding
            query_result = self.embedder.embed_text(query)
            query_vector = query_result.vector
            
            # Perform similarity search with metadata filtering
            if self.config.enable_filtering and metadata_filter:
                similar_docs = self.backend_store.search_by_metadata(
                    metadata_filter=metadata_filter,
                    query_vector=query_vector,
                    limit=limit
                )
            else:
                similar_docs = self.backend_store.search_similar(
                    query_vector,
                    limit=limit
                )
            
            # Convert to SearchResult objects
            search_results = []
            for result in similar_docs:
                # Apply similarity threshold filtering
                if self.config.enable_filtering and result.score < self.config.similarity_threshold:
                    continue
                
                # Create Document from VectorSearchResult
                doc = Document(
                    id=result.document_id,
                    content=result.content or "",
                    metadata=result.metadata or {}
                )
                
                search_result = SearchResult(
                    document_id=result.document_id,
                    document=doc,
                    score=result.score,
                    metadata={
                        "retrieval_method": "vector_similarity",
                        "embedding_model": self.config.embedding_model,
                        "query_length": len(query),
                        "backend_store": type(self.backend_store).__name__
                    }
                )
                search_results.append(search_result)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats["queries_processed"] += 1
            self.processing_stats["processing_time"] += processing_time
            
            logger.debug(f"Vector search completed: {len(search_results)} results in {processing_time:.3f}s")
            return search_results
            
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            logger.error(f"Vector search failed: {e}")
            return []
    
    def index_document(self, document: Document) -> None:
        """
        Index a single document for vector search
        ベクトル検索用に単一文書をインデックス
        
        Args:
            document: Document to index
                     インデックスする文書
        """
        try:
            # Generate embedding for document content
            embedding_result = self.embedder.embed_text(document.content)
            
            # Create vector entry
            vector_entry = VectorEntry(
                id=document.id,
                vector=embedding_result.vector,
                metadata=document.metadata.copy(),
                content=document.content
            )
            
            # Store in backend
            self.backend_store.add_vector(vector_entry)
            
            logger.debug(f"Indexed document: {document.id}")
            
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
            # Generate embeddings for all documents
            contents = [doc.content for doc in documents]
            embedding_results = self.embedder.embed_texts(contents)
            
            # Create vector entries
            vector_entries = []
            for doc, embedding_result in zip(documents, embedding_results):
                vector_entry = VectorEntry(
                    id=doc.id,
                    vector=embedding_result.vector,
                    metadata=doc.metadata.copy(),
                    content=doc.content
                )
                vector_entries.append(vector_entry)
            
            # Store in backend
            self.backend_store.add_vectors(vector_entries)
            
            logger.info(f"Indexed {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise
    
    def remove_document(self, document_id: str) -> bool:
        """
        Remove document from vector index
        ベクトルインデックスから文書を削除
        
        Args:
            document_id: ID of document to remove
                        削除する文書のID
                        
        Returns:
            bool: True if document was found and removed
                 文書が見つかって削除された場合True
        """
        try:
            success = self.backend_store.delete_vector(document_id)
            if success:
                logger.debug(f"Removed document from index: {document_id}")
            else:
                logger.warning(f"Document not found in index: {document_id}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to remove document {document_id}: {e}")
            return False
    
    def update_document(self, document: Document) -> bool:
        """
        Update an existing document in the vector index
        ベクトルインデックスの既存文書を更新
        
        Args:
            document: Updated document
                     更新する文書
                     
        Returns:
            bool: True if document was found and updated
                 文書が見つかって更新された場合True
        """
        try:
            # Generate new embedding
            embedding_result = self.embedder.embed_text(document.content)
            
            # Create updated vector entry
            vector_entry = VectorEntry(
                id=document.id,
                vector=embedding_result.vector,
                metadata=document.metadata.copy(),
                content=document.content
            )
            
            # Update in backend (this may involve delete + add)
            if hasattr(self.backend_store, 'update_vector'):
                success = self.backend_store.update_vector(vector_entry)
            else:
                # Fallback: delete then add
                self.backend_store.delete_vector(document.id)
                self.backend_store.add_vector(vector_entry)
                success = True
            
            if success:
                logger.debug(f"Updated document in index: {document.id}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to update document {document.id}: {e}")
            return False
    
    def clear_index(self) -> None:
        """
        Remove all documents from the vector index
        ベクトルインデックスからすべての文書を削除
        """
        try:
            self.backend_store.clear()
            logger.info("Cleared vector index")
            
        except Exception as e:
            logger.error(f"Failed to clear vector index: {e}")
            raise
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the vector index
        ベクトルインデックスの文書数を取得
        
        Returns:
            int: Number of indexed documents
                インデックスされた文書数
        """
        try:
            if hasattr(self.backend_store, 'get_stats'):
                stats = self.backend_store.get_stats()
                return stats.total_vectors
            else:
                # Fallback: not all backends may support this
                return 0
                
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics with VectorStore-specific metrics
        VectorStore固有のメトリクスを含む処理統計を取得"""
        stats = super().get_processing_stats()
        
        # Add VectorStore-specific stats
        stats.update({
            "retriever_type": "OpenAIVectorStore",
            "backend_store_type": type(self.backend_store).__name__,
            "embedder_type": type(self.embedder).__name__,
            "embedding_model": self.config.embedding_model,
            "similarity_threshold": self.config.similarity_threshold,
            "top_k": self.config.top_k,
            "document_count": self.get_document_count()
        })
        
        # Add backend store stats if available
        if hasattr(self.backend_store, 'get_stats'):
            stats["backend_store_stats"] = self.backend_store.get_stats()
        
        return stats