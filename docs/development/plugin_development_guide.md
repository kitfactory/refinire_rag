# KeywordSearch & VectorStore Plugin Development Guide / プラグイン開発ガイド

## Overview / 概要

This guide provides comprehensive instructions for developing custom KeywordSearch and VectorStore plugins for the refinire-rag system. These plugins extend the core search capabilities with custom implementations for specific use cases.

このガイドでは、refinire-ragシステム用のカスタムKeywordSearchとVectorStoreプラグインを開発するための包括的な手順を提供します。これらのプラグインは、特定のユースケース向けのカスタム実装でコア検索機能を拡張します。

## Plugin Types / プラグインタイプ

This guide focuses on two primary plugin types:

| Plugin Type | Interface | Purpose | Integration |
|-------------|-----------|---------|-------------|
| **KeywordSearch** | `KeywordSearch` (DocumentProcessor, Indexer, Retriever) | キーワードベースの文書検索 | CorpusManager, QueryEngine |
| **VectorStore** | `VectorStore` (DocumentProcessor) | ベクトルベースの文書ストレージと類似検索 | CorpusManager, QueryEngine |

## KeywordSearch Plugin Development / KeywordSearchプラグイン開発

### Interface Requirements / インターフェース要件

All KeywordSearch plugins must implement the `KeywordSearch` interface which combines:

```python
from refinire_rag.retrieval.base import KeywordSearch, SearchResult
from refinire_rag.models.document import Document
from typing import List, Optional, Dict, Any

class CustomKeywordSearch(KeywordSearch):
    """Custom keyword search implementation"""
    
    def __init__(self, **kwargs):
        """Initialize with unified configuration support
        
        Args:
            **kwargs: Configuration parameters, merged with environment variables
                     設定パラメータ、環境変数とマージされます
        """
        super().__init__(**kwargs)
        # Your initialization code here
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary
        現在の設定を辞書として取得
        """
        return {
            'setting1': getattr(self, 'setting1', 'default_value1'),
            'setting2': getattr(self, 'setting2', 'default_value2'),
            # Add your configuration parameters here
        }
    
    # Core KeywordSearch methods
    def add_document(self, document: Document) -> None:
        """Add a document to the store"""
        pass
    
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search for documents using keyword matching"""
        pass
    
    # Indexer interface methods  
    def index_document(self, document: Document) -> None:
        """Index a single document for search"""
        pass
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index multiple documents efficiently"""
        pass
    
    def remove_document(self, document_id: str) -> bool:
        """Remove document from index"""
        pass
    
    def update_document(self, document: Document) -> bool:
        """Update an existing document in the index"""
        pass
    
    def clear_index(self) -> None:
        """Remove all documents from the index"""
        pass
    
    def get_document_count(self) -> int:
        """Get the number of documents in the index"""
        pass
    
    # Retriever interface methods
    def retrieve(self, 
                 query: str, 
                 limit: Optional[int] = None,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Retrieve relevant documents for query"""
        pass
```

### Example Implementation: ElasticsearchKeywordStore / 実装例

### 必須実装事項 / Required Implementation

#### 1. 基底クラス継承とインターフェース実装

```python
from refinire_rag.retrieval.base import KeywordSearch

class CustomKeywordSearch(KeywordSearch):
    """必須：KeywordSearchインターフェースを継承"""
    
    def __init__(self, **kwargs):
        """必須：統一設定パターンのコンストラクタ"""
        super().__init__(**kwargs)
        # 環境変数サポート付き設定処理
        import os
        self.setting = kwargs.get('setting', os.getenv('REFINIRE_RAG_PLUGIN_SETTING', 'default'))
    
    def get_config(self) -> Dict[str, Any]:
        """必須：現在の設定を辞書で返却"""
        return {'setting': self.setting}
    
    # 必須メソッド群
    def add_document(self, document: Document) -> None: ...
    def search(self, query: str, limit: int = 10) -> List[SearchResult]: ...
    def index_document(self, document: Document) -> None: ...
    def retrieve(self, query: str, limit: Optional[int] = None, 
                metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]: ...
```

#### 2. 環境変数命名規則の遵守

```python
# 必須：プラグイン固有環境変数は REFINIRE_RAG_{PLUGIN}_{SETTING} パターン
REFINIRE_RAG_ELASTICSEARCH_HOST="localhost"
REFINIRE_RAG_ELASTICSEARCH_PORT="9200"
REFINIRE_RAG_ELASTICSEARCH_INDEX="documents"
```

#### 3. エラーハンドリングとログ

```python
import logging
logger = logging.getLogger(__name__)

class CustomKeywordSearch(KeywordSearch):
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        try:
            # 検索処理
            results = self._perform_search(query, limit)
            logger.info(f"Retrieved {len(results)} results for query: {query[:50]}")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []  # 必須：エラー時は空リストを返却
```

### 簡略化実装例：ElasticsearchKeywordStore

```python
class ElasticsearchKeywordStore(KeywordSearch):
    """Elasticsearch keyword search implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import os
        # 必須：環境変数サポート付き設定
        self.host = kwargs.get('host', os.getenv('REFINIRE_RAG_ELASTICSEARCH_HOST', 'localhost'))
        self.port = int(kwargs.get('port', os.getenv('REFINIRE_RAG_ELASTICSEARCH_PORT', '9200')))
        self.index_name = kwargs.get('index_name', 
                                   os.getenv('REFINIRE_RAG_ELASTICSEARCH_INDEX', 'documents'))
        self._client = None
    
    def get_config(self) -> Dict[str, Any]:
        """必須：設定情報を辞書で返却"""
        return {
            'host': self.host,
            'port': self.port, 
            'index_name': self.index_name
        }
        
        # Initialize Elasticsearch client
        try:
            self.es_client = Elasticsearch([{
                'host': self.config["host"],
                'port': self.config["port"]
            }])
            
            # Create index if it doesn't exist
            self._ensure_index_exists()
            
            logger.info(f"Initialized ElasticsearchKeywordStore with index '{self.config['index_name']}'")
            
        except Exception as e:
            raise StorageError(f"Failed to initialize Elasticsearch client: {e}") from e
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary
        現在の設定を辞書として取得
        """
        return {
            'host': self.host,
            'port': self.port,
            'index_name': self.index_name,
            'similarity_threshold': self.similarity_threshold,
            'top_k': self.top_k
        }
    
    # 必須メソッドの簡略化実装例
    def add_document(self, document: Document) -> None:
        """必須：文書をインデックスに追加"""
        try:
            # プラグイン固有の保存処理
            self._store_document(document)
            logger.debug(f"Added document: {document.id}")
        except Exception as e:
            logger.error(f"Failed to add document {document.id}: {e}")
            raise
    
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """必須：キーワード検索の実行"""
        try:
            # プラグイン固有の検索処理
            results = self._perform_search(query, limit)
            logger.info(f"Search completed: {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []  # 必須：エラー時は空リスト返却
    
    def retrieve(self, query: str, limit: Optional[int] = None,
                metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """必須：メタデータフィルタ付き検索"""
        limit = limit or 10
        try:
            # フィルタ処理を含む検索実行
            results = self._search_with_filters(query, limit, metadata_filter)
            return results
        except Exception as e:
            logger.error(f"Filtered search failed: {e}")
            return []
                es_query = {
                    "query": query_clause,
                    "size": limit
                }
            
            # Execute filtered search
            response = self.es_client.search(
                index=self.config["index_name"],
                body=es_query
            )
            
            # Process results (similar to search method)
            search_results = []
            for hit in response["hits"]["hits"]:
                doc_id = hit["_id"]
                score = float(hit["_score"])
                source = hit["_source"]
                
                document = Document(
                    id=doc_id,
                    content=source["content"],
                    metadata=source.get("metadata", {})
                )
                
                search_result = SearchResult(
                    document_id=doc_id,
                    document=document,
                    score=score,
                    metadata={
                        "retrieval_method": "elasticsearch_filtered",
                        "has_metadata_filter": bool(metadata_filter),
                        "keyword_store": "ElasticsearchKeywordStore"
                    }
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Elasticsearch filtered search failed: {e}")
            return []
    
    def index_document(self, document: Document) -> None:
        """Index a single document (alias for add_document)"""
        self.add_document(document)
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index multiple documents using bulk API"""
        try:
            # Prepare bulk operations
            bulk_body = []
            for doc in documents:
                bulk_body.extend([
                    {
                        "index": {
                            "_index": self.config["index_name"],
                            "_id": doc.id
                        }
                    },
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "timestamp": time.time()
                    }
                ])
            
            # Execute bulk operation
            response = self.es_client.bulk(body=bulk_body)
            
            # Check for errors
            if response.get("errors"):
                error_items = [item for item in response["items"] if "error" in item.get("index", {})]
                logger.warning(f"Bulk indexing had {len(error_items)} errors")
            
            logger.info(f"Bulk indexed {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            raise StorageError(f"Failed to bulk index documents: {e}") from e
    
    def remove_document(self, document_id: str) -> bool:
        """Remove document from Elasticsearch index"""
        try:
            response = self.es_client.delete(
                index=self.config["index_name"],
                id=document_id
            )
            
            success = response.get("result") == "deleted"
            if success:
                logger.debug(f"Removed document from Elasticsearch: {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to remove document {document_id}: {e}")
            return False
    
    def update_document(self, document: Document) -> bool:
        """Update an existing document in the index"""
        try:
            # Check if document exists
            if not self.es_client.exists(index=self.config["index_name"], id=document.id):
                return False
            
            # Update the document
            self.add_document(document)
            logger.debug(f"Updated document in Elasticsearch: {document.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {document.id}: {e}")
            return False
    
    def clear_index(self) -> None:
        """Remove all documents from the index"""
        try:
            self.es_client.delete_by_query(
                index=self.config["index_name"],
                body={"query": {"match_all": {}}}
            )
            
            logger.info("Cleared all documents from Elasticsearch index")
            
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            raise StorageError(f"Failed to clear index: {e}") from e
    
    def get_document_count(self) -> int:
        """Get the number of documents in the index"""
        try:
            response = self.es_client.count(index=self.config["index_name"])
            return response["count"]
            
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist"""
        try:
            if not self.es_client.indices.exists(index=self.config["index_name"]):
                # Define index mapping
                mapping = {
                    "mappings": {
                        "properties": {
                            "content": {
                                "type": "text",
                                "analyzer": "standard"
                            },
                            "metadata": {
                                "type": "object",
                                "dynamic": True
                            },
                            "timestamp": {
                                "type": "date",
                                "format": "epoch_second"
                            }
                        }
                    }
                }
                
                self.es_client.indices.create(
                    index=self.config["index_name"],
                    body=mapping
                )
                
                logger.info(f"Created Elasticsearch index: {self.config['index_name']}")
                
        except Exception as e:
            raise StorageError(f"Failed to create index: {e}") from e
```

## VectorStore Plugin Development / VectorStoreプラグイン開発

### 必須実装事項 / Required Implementation

#### 1. VectorStore基底クラス継承

```python
from refinire_rag.storage.vector_store import VectorStore

class CustomVectorStore(VectorStore):
    """必須：VectorStoreインターフェースを継承"""
    
    def __init__(self, **kwargs):
        """必須：統一設定パターンのコンストラクタ"""
        super().__init__(**kwargs)
        import os
        # 環境変数サポート付き設定
        self.setting = kwargs.get('setting', os.getenv('REFINIRE_RAG_PLUGIN_SETTING', 'default'))
    
    def get_config(self) -> Dict[str, Any]:
        """必須：現在の設定を辞書で返却"""
        return {'setting': self.setting}
    
    # 必須メソッド群（簡略化）
    def add_vector(self, entry: VectorEntry) -> str: ...
    def search_similar(self, query_vector: np.ndarray, limit: int = 10) -> List[VectorSearchResult]: ...
    def get_stats(self) -> VectorStoreStats: ...
    def clear(self) -> bool: ...
```

#### 2. エラーハンドリングパターン

```python
def search_similar(self, query_vector: np.ndarray, limit: int = 10) -> List[VectorSearchResult]:
    """必須：類似ベクトル検索"""
    try:
        # ベクトル検索処理
        results = self._perform_vector_search(query_vector, limit)
        logger.info(f"Vector search completed: {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []  # 必須：エラー時は空リスト返却
```

### 簡略化実装例：ChromaVectorStore

```python
class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import os
        # 必須：環境変数サポート付き設定
        self.collection_name = kwargs.get('collection_name', 
                                        os.getenv('REFINIRE_RAG_CHROMA_COLLECTION', 'documents'))
        self.host = kwargs.get('host', os.getenv('REFINIRE_RAG_CHROMA_HOST', 'localhost'))
        self.port = int(kwargs.get('port', os.getenv('REFINIRE_RAG_CHROMA_PORT', '8000')))
        self._client = None
    
    def get_config(self) -> Dict[str, Any]:
        """必須：設定情報返却"""
        return {
            'collection_name': self.collection_name,
            'host': self.host,
            'port': self.port
        }
        
        # Initialize ChromaDB client
        try:
            if self.config["host"] and self.config["port"]:
                # Remote ChromaDB
                self.client = chromadb.HttpClient(
                    host=self.config["host"],
                    port=self.config["port"]
                )
            else:
                # Local persistent ChromaDB
                self.client = chromadb.PersistentClient(
                    path=self.config["persist_directory"]
                )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.config["collection_name"],
                metadata={"similarity_metric": self.config["similarity_metric"]}
            )
            
            logger.info(f"Initialized ChromaVectorStore with collection '{self.config['collection_name']}'")
            
        except Exception as e:
            raise StorageError(f"Failed to initialize ChromaDB client: {e}") from e
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary
        現在の設定を辞書として取得
        """
        return {
            'host': getattr(self, 'host', 'localhost'),
            'port': getattr(self, 'port', 8000),
            'collection_name': getattr(self, 'collection_name', 'documents'),
            'similarity_metric': getattr(self, 'similarity_metric', 'cosine')
        }
    
    def add_vector(self, entry: VectorEntry) -> str:
        """Add a vector entry to ChromaDB"""
        try:
            # Validate embedding
            if entry.embedding is None or len(entry.embedding) == 0:
                raise ValueError("Entry must have a valid embedding")
            
            # Add to ChromaDB
            self.collection.add(
                ids=[entry.document_id],
                embeddings=[entry.embedding.tolist()],
                documents=[entry.content],
                metadatas=[entry.metadata]
            )
            
            logger.debug(f"Added vector to ChromaDB: {entry.document_id}")
            return entry.document_id
            
        except Exception as e:
            raise StorageError(f"Failed to add vector for {entry.document_id}: {e}") from e
    
    def add_vectors(self, entries: List[VectorEntry]) -> List[str]:
        """Add multiple vector entries to ChromaDB"""
        try:
            if not entries:
                return []
            
            # Prepare batch data
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for entry in entries:
                if entry.embedding is None or len(entry.embedding) == 0:
                    logger.warning(f"Skipping document {entry.document_id} - invalid embedding")
                    continue
                
                ids.append(entry.document_id)
                embeddings.append(entry.embedding.tolist())
                documents.append(entry.content)
                metadatas.append(entry.metadata)
            
            if ids:
                # Batch add to ChromaDB
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                
                logger.info(f"Added {len(ids)} vectors to ChromaDB")
            
            return ids
            
        except Exception as e:
            raise StorageError(f"Failed to add vectors: {e}") from e
    
    def get_vector(self, document_id: str) -> Optional[VectorEntry]:
        """Retrieve vector entry by document ID from ChromaDB"""
        try:
            result = self.collection.get(
                ids=[document_id],
                include=["embeddings", "documents", "metadatas"]
            )
            
            if not result["ids"]:
                return None
            
            # Convert ChromaDB result to VectorEntry
            embedding = np.array(result["embeddings"][0])
            content = result["documents"][0]
            metadata = result["metadatas"][0] or {}
            
            return VectorEntry(
                document_id=document_id,
                content=content,
                embedding=embedding,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to get vector for {document_id}: {e}")
            return None
    
    def update_vector(self, entry: VectorEntry) -> bool:
        """Update an existing vector entry in ChromaDB"""
        try:
            # Check if entry exists
            existing = self.get_vector(entry.document_id)
            if not existing:
                return False
            
            # ChromaDB doesn't have direct update, so we use upsert
            self.collection.upsert(
                ids=[entry.document_id],
                embeddings=[entry.embedding.tolist()],
                documents=[entry.content],
                metadatas=[entry.metadata]
            )
            
            logger.debug(f"Updated vector in ChromaDB: {entry.document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update vector for {entry.document_id}: {e}")
            return False
    
    def delete_vector(self, document_id: str) -> bool:
        """Delete vector entry by document ID from ChromaDB"""
        try:
            # Check if entry exists
            existing = self.get_vector(document_id)
            if not existing:
                return False
            
            # Delete from ChromaDB
            self.collection.delete(ids=[document_id])
            
            logger.debug(f"Deleted vector from ChromaDB: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vector for {document_id}: {e}")
            return False
    
    def search_similar(
        self, 
        query_vector: np.ndarray, 
        limit: int = 10,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors using ChromaDB"""
        try:
            # Prepare where clause for metadata filtering
            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=limit,
                where=where_clause,
                include=["embeddings", "documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            if results["ids"] and results["ids"][0]:  # ChromaDB returns nested lists
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    
                    # Convert distance to similarity score (assuming cosine distance)
                    score = 1.0 - distance
                    
                    # Apply threshold filter
                    if threshold is not None and score < threshold:
                        continue
                    
                    # Create VectorSearchResult
                    result = VectorSearchResult(
                        document_id=doc_id,
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i] or {},
                        score=score,
                        embedding=np.array(results["embeddings"][0][i])
                    )
                    search_results.append(result)
            
            logger.debug(f"Found {len(search_results)} similar vectors in ChromaDB")
            return search_results
            
        except Exception as e:
            raise StorageError(f"Failed to search similar vectors: {e}") from e
    
    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 100
    ) -> List[VectorSearchResult]:
        """Search vectors by metadata filters in ChromaDB"""
        try:
            where_clause = self._build_where_clause(filters)
            
            # Get all matching documents
            results = self.collection.get(
                where=where_clause,
                limit=limit,
                include=["embeddings", "documents", "metadatas"]
            )
            
            # Process results
            search_results = []
            for i, doc_id in enumerate(results["ids"]):
                result = VectorSearchResult(
                    document_id=doc_id,
                    content=results["documents"][i],
                    metadata=results["metadatas"][i] or {},
                    score=1.0,  # No similarity score for metadata search
                    embedding=np.array(results["embeddings"][i])
                )
                search_results.append(result)
            
            logger.debug(f"Found {len(search_results)} vectors matching metadata filters")
            return search_results
            
        except Exception as e:
            raise StorageError(f"Failed to search by metadata: {e}") from e
    
    def count_vectors(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors matching optional filters in ChromaDB"""
        try:
            if not filters:
                return self.collection.count()
            
            where_clause = self._build_where_clause(filters)
            results = self.collection.get(where=where_clause, include=[])
            return len(results["ids"])
            
        except Exception as e:
            logger.error(f"Failed to count vectors: {e}")
            return 0
    
    def get_stats(self) -> VectorStoreStats:
        """Get vector store statistics from ChromaDB"""
        try:
            total_vectors = self.collection.count()
            vector_dimension = 0
            storage_size = 0
            
            if total_vectors > 0:
                # Get a sample to determine dimension
                sample = self.collection.peek(limit=1)
                if sample["embeddings"]:
                    vector_dimension = len(sample["embeddings"][0])
                
                # Estimate storage size (rough calculation)
                storage_size = total_vectors * vector_dimension * 4  # 4 bytes per float
            
            return VectorStoreStats(
                total_vectors=total_vectors,
                vector_dimension=vector_dimension,
                storage_size_bytes=storage_size,
                index_type="chroma_hnsw"
            )
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return VectorStoreStats(
                total_vectors=0,
                vector_dimension=0,
                storage_size_bytes=0,
                index_type="chroma_hnsw"
            )
    
    def clear(self) -> bool:
        """Clear all vectors from ChromaDB collection"""
        try:
            # ChromaDB doesn't have a direct clear method
            # We need to delete the collection and recreate it
            collection_name = self.config["collection_name"]
            similarity_metric = self.config["similarity_metric"]
            
            # Delete the collection
            self.client.delete_collection(name=collection_name)
            
            # Recreate the collection
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"similarity_metric": similarity_metric}
            )
            
            logger.info("Cleared all vectors from ChromaDB collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear vectors: {e}")
            return False
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters"""
        where_clause = {}
        
        for key, value in filters.items():
            if isinstance(value, dict):
                # Handle operator-based filters
                if "$eq" in value:
                    where_clause[key] = {"$eq": value["$eq"]}
                elif "$ne" in value:
                    where_clause[key] = {"$ne": value["$ne"]}
                elif "$in" in value:
                    where_clause[key] = {"$in": value["$in"]}
                elif "$nin" in value:
                    where_clause[key] = {"$nin": value["$nin"]}
                elif "$gt" in value:
                    where_clause[key] = {"$gt": value["$gt"]}
                elif "$gte" in value:
                    where_clause[key] = {"$gte": value["$gte"]}
                elif "$lt" in value:
                    where_clause[key] = {"$lt": value["$lt"]}
                elif "$lte" in value:
                    where_clause[key] = {"$lte": value["$lte"]}
            else:
                # Simple equality
                where_clause[key] = {"$eq": value}
        
        return where_clause
```
## Project Structure / プロジェクト構造

For plugin development, we recommend the following project structure:

プラグイン開発には、以下のプロジェクト構造を推奨します：

```
my-refinire-rag-plugin/
├── pyproject.toml
├── README.md
├── src/
│   └── my_refinire_plugin/
│       ├── __init__.py
│       ├── keyword_search.py        # KeywordSearch implementation
│       ├── vector_store.py          # VectorStore implementation
│       └── config.py               # Configuration classes
└── tests/
    ├── __init__.py
    ├── test_keyword_search.py
    ├── test_vector_store.py
    └── test_integration.py
```

## プロジェクト設定 / Project Configuration

### 必須pyproject.toml設定

```toml
[project]
name = "my-refinire-rag-plugin"
version = "0.1.0"
dependencies = [
    "refinire-rag>=0.1.0",
    # プラグイン固有の依存関係のみ追加
]

# 必須：エントリーポイント設定
[project.entry-points."refinire_rag.keyword_stores"]
elasticsearch = "my_refinire_plugin:ElasticsearchKeywordStore"

[project.entry-points."refinire_rag.vector_stores"] 
chroma = "my_refinire_plugin:ChromaVectorStore"
```

### 必須パッケージ初期化

```python
# src/my_refinire_plugin/__init__.py

from .keyword_search import ElasticsearchKeywordStore
from .vector_store import ChromaVectorStore

# 必須：プラグインクラスのエクスポート
__all__ = [
    "ElasticsearchKeywordStore", 
    "ChromaVectorStore"
]
```

## プラグインテスト / Plugin Testing

### 必須テスト項目

#### 1. 基本機能テスト

```python
# tests/test_plugin.py
import pytest
from my_refinire_plugin import ElasticsearchKeywordStore

def test_plugin_initialization():
    """必須：プラグイン初期化テスト"""
    plugin = ElasticsearchKeywordStore()
    assert plugin is not None
    
def test_get_config():
    """必須：設定取得テスト"""
    plugin = ElasticsearchKeywordStore()
    config = plugin.get_config()
    assert isinstance(config, dict)

def test_environment_variables():
    """必須：環境変数サポートテスト"""
    import os
    os.environ['REFINIRE_RAG_ELASTICSEARCH_HOST'] = 'test-host'
    plugin = ElasticsearchKeywordStore()
    assert plugin.host == 'test-host'
```

#### 2. エラーハンドリングテスト

```python
def test_search_error_handling():
    """必須：エラー時の空リスト返却テスト"""
    plugin = ElasticsearchKeywordStore()
    # 接続不可能な設定でテスト
    results = plugin.search("test query")
    assert results == []  # エラー時は空リスト
```

## プラグイン開発チェックリスト / Development Checklist

### ✅ 必須実装項目

#### KeywordSearchプラグイン
- [ ] `KeywordSearch` 基底クラスを継承
- [ ] `__init__(**kwargs)` で環境変数サポート
- [ ] `get_config()` メソッド実装
- [ ] `search()`, `add_document()`, `retrieve()` メソッド実装
- [ ] エラー時に空リスト返却
- [ ] 適切なログ出力

#### VectorStoreプラグイン  
- [ ] `VectorStore` 基底クラスを継承
- [ ] `__init__(**kwargs)` で環境変数サポート
- [ ] `get_config()` メソッド実装
- [ ] `add_vector()`, `search_similar()` メソッド実装
- [ ] エラー時に空リスト返却
- [ ] 適切なログ出力

#### プロジェクト設定
- [ ] pyproject.tomlにentry points設定
- [ ] __init__.pyでクラスエクスポート
- [ ] 環境変数命名規則遵守 (`REFINIRE_RAG_{PLUGIN}_{SETTING}`)
- [ ] 基本テストケース作成

---

この簡略化ガイドに従って、プラグインで守らなければならない必須事項に焦点を当てて開発を進めてください。詳細な実装は各プラグインの要件に応じて調整してください。

