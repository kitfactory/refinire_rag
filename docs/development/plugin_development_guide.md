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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with DocumentProcessor integration"""
        super().__init__(config or {})
        # Your initialization code here
    
    @classmethod
    def get_config_class(cls) -> Type[Dict]:
        """Get the configuration class for this keyword search"""
        return Dict
    
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

```python
import logging
import time
from typing import List, Optional, Dict, Any, Type
from elasticsearch import Elasticsearch

from refinire_rag.retrieval.base import KeywordSearch, SearchResult
from refinire_rag.models.document import Document
from refinire_rag.exceptions import StorageError

logger = logging.getLogger(__name__)


class ElasticsearchKeywordStore(KeywordSearch):
    """Elasticsearch-based keyword search implementation
    Elasticsearchベースのキーワード検索実装
    
    Provides full-text search capabilities using Elasticsearch with
    support for complex queries, aggregations, and filters.
    
    複雑なクエリ、集約、フィルタをサポートするElasticsearchを使用した
    全文検索機能を提供します。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Elasticsearch keyword store
        
        Args:
            config: Configuration including ES connection details
                   ES接続詳細を含む設定
        """
        super().__init__(config or {})
        
        # Default configuration
        self.config = {
            "host": "localhost",
            "port": 9200,
            "index_name": "documents",
            "similarity_threshold": 0.0,
            "top_k": 10,
            **self.config
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
    
    @classmethod
    def get_config_class(cls) -> Type[Dict]:
        """Get the configuration class for this keyword search"""
        return Dict
    
    def add_document(self, document: Document) -> None:
        """Add a document to the Elasticsearch index"""
        try:
            doc_body = {
                "content": document.content,
                "metadata": document.metadata,
                "timestamp": time.time()
            }
            
            self.es_client.index(
                index=self.config["index_name"],
                id=document.id,
                body=doc_body
            )
            
            logger.debug(f"Added document to Elasticsearch: {document.id}")
            
        except Exception as e:
            raise StorageError(f"Failed to add document {document.id}: {e}") from e
    
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search for documents using Elasticsearch full-text search"""
        start_time = time.time()
        
        try:
            # Build Elasticsearch query
            es_query = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "metadata.*"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "size": limit,
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"timestamp": {"order": "desc"}}
                ]
            }
            
            # Execute search
            response = self.es_client.search(
                index=self.config["index_name"],
                body=es_query
            )
            
            # Process results
            search_results = []
            for hit in response["hits"]["hits"]:
                doc_id = hit["_id"]
                score = float(hit["_score"])
                source = hit["_source"]
                
                # Apply similarity threshold
                if score < self.config["similarity_threshold"]:
                    continue
                
                # Create Document object
                document = Document(
                    id=doc_id,
                    content=source["content"],
                    metadata=source.get("metadata", {})
                )
                
                # Create SearchResult
                search_result = SearchResult(
                    document_id=doc_id,
                    document=document,
                    score=score,
                    metadata={
                        "retrieval_method": "elasticsearch",
                        "algorithm": "multi_match",
                        "query_length": len(query),
                        "keyword_store": "ElasticsearchKeywordStore"
                    }
                )
                search_results.append(search_result)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats["queries_processed"] += 1
            self.processing_stats["processing_time"] += processing_time
            
            logger.debug(f"Elasticsearch search completed: {len(search_results)} results in {processing_time:.3f}s")
            return search_results
            
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            logger.error(f"Elasticsearch search failed: {e}")
            return []
    
    def retrieve(self, 
                 query: str, 
                 limit: Optional[int] = None,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Retrieve documents with optional metadata filtering"""
        limit = limit or self.config["top_k"]
        
        try:
            # Build base query
            query_clause = {
                "multi_match": {
                    "query": query,
                    "fields": ["content^2", "metadata.*"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
            
            # Add metadata filters if provided
            if metadata_filter:
                filter_clauses = []
                for key, value in metadata_filter.items():
                    if isinstance(value, list):
                        filter_clauses.append({
                            "terms": {f"metadata.{key}": value}
                        })
                    elif isinstance(value, dict):
                        # Range queries
                        range_query = {}
                        if "$gte" in value:
                            range_query["gte"] = value["$gte"]
                        if "$lte" in value:
                            range_query["lte"] = value["$lte"]
                        if range_query:
                            filter_clauses.append({
                                "range": {f"metadata.{key}": range_query}
                            })
                    else:
                        filter_clauses.append({
                            "term": {f"metadata.{key}": value}
                        })
                
                # Combine query with filters
                es_query = {
                    "query": {
                        "bool": {
                            "must": query_clause,
                            "filter": filter_clauses
                        }
                    },
                    "size": limit
                }
            else:
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

### Interface Requirements / インターフェース要件

All VectorStore plugins must implement the `VectorStore` interface:

```python
from refinire_rag.storage.vector_store import VectorStore, VectorEntry, VectorSearchResult, VectorStoreStats
from refinire_rag.models.document import Document
from typing import List, Optional, Dict, Any, Type
import numpy as np

class CustomVectorStore(VectorStore):
    """Custom vector store implementation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with DocumentProcessor integration"""
        super().__init__(config or {})
        # Your initialization code here
    
    @classmethod
    def get_config_class(cls) -> Type[Dict]:
        """Get the configuration class for this vector store"""
        return Dict
    
    # Core VectorStore methods
    def add_vector(self, entry: VectorEntry) -> str:
        """Add a vector entry to the store"""
        pass
    
    def add_vectors(self, entries: List[VectorEntry]) -> List[str]:
        """Add multiple vector entries to the store"""
        pass
    
    def get_vector(self, document_id: str) -> Optional[VectorEntry]:
        """Retrieve vector entry by document ID"""
        pass
    
    def update_vector(self, entry: VectorEntry) -> bool:
        """Update an existing vector entry"""
        pass
    
    def delete_vector(self, document_id: str) -> bool:
        """Delete vector entry by document ID"""
        pass
    
    def search_similar(
        self, 
        query_vector: np.ndarray, 
        limit: int = 10,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors"""
        pass
    
    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 100
    ) -> List[VectorSearchResult]:
        """Search vectors by metadata filters"""
        pass
    
    def count_vectors(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors matching optional filters"""
        pass
    
    def get_stats(self) -> VectorStoreStats:
        """Get vector store statistics"""
        pass
    
    def clear(self) -> bool:
        """Clear all vectors from the store"""
        pass
```

### Example Implementation: ChromaVectorStore / 実装例

```python
import logging
import time
from typing import List, Optional, Dict, Any, Type
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

from refinire_rag.storage.vector_store import VectorStore, VectorEntry, VectorSearchResult, VectorStoreStats
from refinire_rag.exceptions import StorageError

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store implementation
    ChromaDBベースのベクトルストア実装
    
    Provides persistent vector storage with efficient similarity search
    using ChromaDB as the backend database.
    
    ChromaDBをバックエンドデータベースとして使用した効率的な類似検索を備えた
    永続的なベクトルストレージを提供します。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ChromaDB vector store
        
        Args:
            config: Configuration including ChromaDB settings
                   ChromaDB設定を含む設定
        """
        if chromadb is None:
            raise ImportError("chromadb package is required for ChromaVectorStore")
        
        super().__init__(config or {})
        
        # Default configuration
        self.config = {
            "collection_name": "documents",
            "persist_directory": "./chroma_db",
            "similarity_metric": "cosine",
            "host": None,  # For remote ChromaDB
            "port": None,
            **self.config
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
    
    @classmethod
    def get_config_class(cls) -> Type[Dict]:
        """Get the configuration class for this vector store"""
        return Dict
    
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

## Project Configuration / プロジェクト設定

### pyproject.toml Setup

Your plugin's `pyproject.toml` should include:

```toml
[project]
name = "my-refinire-rag-plugin"
version = "0.1.0"
description = "Custom KeywordSearch and VectorStore implementations for refinire-rag"
requires-python = ">=3.10"
dependencies = [
    "refinire-rag>=0.1.0",
    "numpy>=1.24.0",
    # Add your specific dependencies here
    "elasticsearch>=8.0.0",  # For Elasticsearch example
    "chromadb>=0.4.0",       # For ChromaDB example
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
]

[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = [
    "--import-mode=importlib",
    "--cov=my_refinire_plugin",
    "--cov-report=term-missing",
]
```

## Plugin Registration / プラグイン登録

### Using Plugin Discovery

Create an `__init__.py` file in your plugin package to register your implementations:

```python
# src/my_refinire_plugin/__init__.py

"""My Refinire RAG Plugin

Custom KeywordSearch and VectorStore implementations for specific use cases.
"""

from .keyword_search import ElasticsearchKeywordStore
from .vector_store import ChromaVectorStore

__version__ = "0.1.0"

# Plugin registration
__all__ = [
    "ElasticsearchKeywordStore",
    "ChromaVectorStore",
]

# Plugin metadata
PLUGIN_INFO = {
    "name": "my-refinire-rag-plugin",
    "version": __version__,
    "description": "Custom search implementations",
    "implementations": {
        "keyword_search": {
            "elasticsearch": ElasticsearchKeywordStore,
        },
        "vector_store": {
            "chroma": ChromaVectorStore,
        }
    }
}
```

## Testing Your Plugins / プラグインのテスト

### KeywordSearch Testing

```python
# tests/test_keyword_search.py

import pytest
import numpy as np
from refinire_rag.models.document import Document
from refinire_rag.retrieval.base import SearchResult

from my_refinire_plugin.keyword_search import ElasticsearchKeywordStore


class TestElasticsearchKeywordStore:
    """Test suite for ElasticsearchKeywordStore"""
    
    @pytest.fixture
    def keyword_store(self):
        """Create a test keyword store instance"""
        config = {
            "host": "localhost",
            "port": 9200,
            "index_name": "test_documents",
            "similarity_threshold": 0.1
        }
        return ElasticsearchKeywordStore(config)
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                id="doc1",
                content="Machine learning algorithms for data analysis",
                metadata={"category": "AI", "year": 2024}
            ),
            Document(
                id="doc2",
                content="Deep learning and neural networks",
                metadata={"category": "AI", "year": 2024}
            ),
            Document(
                id="doc3",
                content="Database optimization techniques",
                metadata={"category": "DB", "year": 2023}
            )
        ]
    
    def test_add_document(self, keyword_store, sample_documents):
        """Test adding a single document"""
        doc = sample_documents[0]
        keyword_store.add_document(doc)
        
        # Verify document was added
        assert keyword_store.get_document_count() == 1
    
    def test_index_documents(self, keyword_store, sample_documents):
        """Test indexing multiple documents"""
        keyword_store.index_documents(sample_documents)
        
        # Verify all documents were indexed
        assert keyword_store.get_document_count() == len(sample_documents)
    
    def test_search(self, keyword_store, sample_documents):
        """Test basic search functionality"""
        # Index test documents
        keyword_store.index_documents(sample_documents)
        
        # Search for relevant documents
        results = keyword_store.search("machine learning", limit=5)
        
        # Verify search results
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        
        # Check that most relevant document is first
        assert results[0].document_id == "doc1"
        assert results[0].score > 0.1
    
    def test_retrieve_with_metadata_filter(self, keyword_store, sample_documents):
        """Test retrieval with metadata filtering"""
        # Index test documents
        keyword_store.index_documents(sample_documents)
        
        # Search with metadata filter
        metadata_filter = {"category": "AI"}
        results = keyword_store.retrieve(
            "learning", 
            limit=10, 
            metadata_filter=metadata_filter
        )
        
        # Verify filtered results
        assert len(results) == 2  # Only AI category documents
        for result in results:
            assert result.document.metadata["category"] == "AI"
    
    def test_remove_document(self, keyword_store, sample_documents):
        """Test document removal"""
        # Index documents
        keyword_store.index_documents(sample_documents)
        
        # Remove a document
        success = keyword_store.remove_document("doc1")
        assert success is True
        
        # Verify document count decreased
        assert keyword_store.get_document_count() == len(sample_documents) - 1
        
        # Verify document is no longer found in search
        results = keyword_store.search("machine learning", limit=5)
        doc_ids = [r.document_id for r in results]
        assert "doc1" not in doc_ids
    
    def test_clear_index(self, keyword_store, sample_documents):
        """Test clearing the entire index"""
        # Index documents
        keyword_store.index_documents(sample_documents)
        assert keyword_store.get_document_count() > 0
        
        # Clear index
        keyword_store.clear_index()
        
        # Verify index is empty
        assert keyword_store.get_document_count() == 0
```

### VectorStore Testing

```python
# tests/test_vector_store.py

import pytest
import numpy as np
from refinire_rag.storage.vector_store import VectorEntry, VectorSearchResult
from refinire_rag.models.document import Document

from my_refinire_plugin.vector_store import ChromaVectorStore


class TestChromaVectorStore:
    """Test suite for ChromaVectorStore"""
    
    @pytest.fixture
    def vector_store(self):
        """Create a test vector store instance"""
        config = {
            "collection_name": "test_collection",
            "persist_directory": "./test_chroma_db",
            "similarity_metric": "cosine"
        }
        return ChromaVectorStore(config)
    
    @pytest.fixture
    def sample_vector_entries(self):
        """Create sample vector entries for testing"""
        return [
            VectorEntry(
                document_id="doc1",
                content="Machine learning algorithms",
                embedding=np.array([1.0, 0.0, 0.0, 0.5]),
                metadata={"category": "AI", "year": 2024}
            ),
            VectorEntry(
                document_id="doc2",
                content="Deep learning networks",
                embedding=np.array([0.8, 0.2, 0.0, 0.3]),
                metadata={"category": "AI", "year": 2024}
            ),
            VectorEntry(
                document_id="doc3",
                content="Database optimization",
                embedding=np.array([0.0, 0.0, 1.0, 0.0]),
                metadata={"category": "DB", "year": 2023}
            )
        ]
    
    def test_add_vector(self, vector_store, sample_vector_entries):
        """Test adding a single vector"""
        entry = sample_vector_entries[0]
        result_id = vector_store.add_vector(entry)
        
        assert result_id == entry.document_id
        
        # Verify vector was added
        stats = vector_store.get_stats()
        assert stats.total_vectors == 1
    
    def test_add_vectors_batch(self, vector_store, sample_vector_entries):
        """Test adding multiple vectors in batch"""
        result_ids = vector_store.add_vectors(sample_vector_entries)
        
        assert len(result_ids) == len(sample_vector_entries)
        
        # Verify all vectors were added
        stats = vector_store.get_stats()
        assert stats.total_vectors == len(sample_vector_entries)
    
    def test_get_vector(self, vector_store, sample_vector_entries):
        """Test retrieving a specific vector"""
        # Add vectors
        vector_store.add_vectors(sample_vector_entries)
        
        # Retrieve a vector
        retrieved = vector_store.get_vector("doc1")
        
        assert retrieved is not None
        assert retrieved.document_id == "doc1"
        assert retrieved.content == "Machine learning algorithms"
        np.testing.assert_array_equal(retrieved.embedding, sample_vector_entries[0].embedding)
    
    def test_search_similar(self, vector_store, sample_vector_entries):
        """Test similarity search"""
        # Add vectors
        vector_store.add_vectors(sample_vector_entries)
        
        # Search with a query vector similar to doc1
        query_vector = np.array([0.9, 0.1, 0.0, 0.4])
        results = vector_store.search_similar(query_vector, limit=3)
        
        # Verify search results
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        
        # Results should be sorted by similarity
        assert results[0].score >= results[1].score
        
        # Most similar should be doc1 or doc2 (both AI-related)
        assert results[0].document_id in ["doc1", "doc2"]
    
    def test_search_with_metadata_filter(self, vector_store, sample_vector_entries):
        """Test similarity search with metadata filtering"""
        # Add vectors
        vector_store.add_vectors(sample_vector_entries)
        
        # Search with metadata filter
        query_vector = np.array([0.5, 0.5, 0.5, 0.5])
        filters = {"category": "AI"}
        results = vector_store.search_similar(
            query_vector, 
            limit=5, 
            filters=filters
        )
        
        # Verify filtered results
        assert len(results) == 2  # Only AI category vectors
        for result in results:
            assert result.metadata["category"] == "AI"
    
    def test_count_vectors(self, vector_store, sample_vector_entries):
        """Test counting vectors"""
        # Initially empty
        assert vector_store.count_vectors() == 0
        
        # Add vectors
        vector_store.add_vectors(sample_vector_entries)
        
        # Count all vectors
        assert vector_store.count_vectors() == len(sample_vector_entries)
        
        # Count with filter
        ai_count = vector_store.count_vectors({"category": "AI"})
        assert ai_count == 2
    
    def test_delete_vector(self, vector_store, sample_vector_entries):
        """Test vector deletion"""
        # Add vectors
        vector_store.add_vectors(sample_vector_entries)
        
        # Delete a vector
        success = vector_store.delete_vector("doc1")
        assert success is True
        
        # Verify vector was deleted
        assert vector_store.get_vector("doc1") is None
        assert vector_store.count_vectors() == len(sample_vector_entries) - 1
    
    def test_clear(self, vector_store, sample_vector_entries):
        """Test clearing all vectors"""
        # Add vectors
        vector_store.add_vectors(sample_vector_entries)
        assert vector_store.count_vectors() > 0
        
        # Clear all vectors
        success = vector_store.clear()
        assert success is True
        
        # Verify store is empty
        assert vector_store.count_vectors() == 0
```

## Integration Testing / 統合テスト

```python
# tests/test_integration.py

import pytest
import numpy as np
from refinire_rag.models.document import Document
from refinire_rag.storage.vector_store import VectorEntry

from my_refinire_plugin.keyword_search import ElasticsearchKeywordStore
from my_refinire_plugin.vector_store import ChromaVectorStore


class TestPluginIntegration:
    """Integration tests for plugin components"""
    
    @pytest.fixture
    def documents(self):
        """Sample documents for integration testing"""
        return [
            Document(
                id="doc1",
                content="Machine learning algorithms for data analysis and prediction",
                metadata={"category": "AI", "difficulty": "intermediate"}
            ),
            Document(
                id="doc2",
                content="Deep learning neural networks and backpropagation",
                metadata={"category": "AI", "difficulty": "advanced"}
            ),
            Document(
                id="doc3",
                content="Database query optimization and indexing strategies",
                metadata={"category": "DB", "difficulty": "intermediate"}
            )
        ]
    
    def test_hybrid_search_workflow(self, documents):
        """Test combining keyword and vector search"""
        # Initialize both stores
        keyword_store = ElasticsearchKeywordStore({
            "index_name": "test_hybrid",
            "similarity_threshold": 0.1
        })
        
        vector_store = ChromaVectorStore({
            "collection_name": "test_hybrid",
            "persist_directory": "./test_hybrid_db"
        })
        
        # Index documents in keyword store
        keyword_store.index_documents(documents)
        
        # Create vector entries (simulate embedder)
        vector_entries = [
            VectorEntry(
                document_id=doc.id,
                content=doc.content,
                embedding=np.random.rand(384),  # Simulate embedding
                metadata=doc.metadata
            )
            for doc in documents
        ]
        
        # Index vectors
        vector_store.add_vectors(vector_entries)
        
        # Perform hybrid search
        query = "machine learning"
        
        # Get keyword results
        keyword_results = keyword_store.search(query, limit=5)
        
        # Get vector results (using first document's embedding as query)
        query_vector = vector_entries[0].embedding
        vector_results = vector_store.search_similar(query_vector, limit=5)
        
        # Verify both searches return results
        assert len(keyword_results) > 0
        assert len(vector_results) > 0
        
        # Verify results contain expected documents
        keyword_doc_ids = {r.document_id for r in keyword_results}
        vector_doc_ids = {r.document_id for r in vector_results}
        
        assert "doc1" in keyword_doc_ids  # Should match "machine learning"
        assert "doc1" in vector_doc_ids   # Should be similar to itself
    
    def test_document_processor_integration(self, documents):
        """Test DocumentProcessor integration"""
        keyword_store = ElasticsearchKeywordStore()
        vector_store = ChromaVectorStore()
        
        # Test processing documents through the stores
        keyword_processed = list(keyword_store.process(documents))
        vector_processed = list(vector_store.process(documents))
        
        # Verify documents pass through unchanged
        assert len(keyword_processed) == len(documents)
        assert len(vector_processed) == len(documents)
        
        # Verify documents have been indexed
        assert keyword_store.get_document_count() == len(documents)
        # Note: vector_store would need an embedder set to actually store vectors
```

## Best Practices / ベストプラクティス

### 1. Error Handling / エラーハンドリング

```python
from refinire_rag.exceptions import StorageError

class MyKeywordStore(KeywordSearch):
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        try:
            # Your search implementation
            results = self._perform_search(query, limit)
            return results
        except ConnectionError as e:
            logger.error(f"Connection failed: {e}")
            raise StorageError(f"Search backend unavailable: {e}") from e
        except ValueError as e:
            logger.error(f"Invalid query: {e}")
            return []  # Return empty results for invalid queries
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise StorageError(f"Search failed: {e}") from e
```

### 2. Configuration Management / 設定管理

```python
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ElasticsearchConfig:
    """Configuration for Elasticsearch KeywordStore"""
    host: str = "localhost"
    port: int = 9200
    index_name: str = "documents"
    similarity_threshold: float = 0.0
    top_k: int = 10
    max_retries: int = 3
    timeout: int = 30
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ElasticsearchConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})

class ElasticsearchKeywordStore(KeywordSearch):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self.es_config = ElasticsearchConfig.from_dict(self.config)
        # Use self.es_config for typed access to configuration
```

### 3. Performance Optimization / パフォーマンス最適化

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedVectorStore(VectorStore):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._cache = {}  # Simple cache for frequent queries
    
    def add_vectors(self, entries: List[VectorEntry]) -> List[str]:
        """Optimized batch addition"""
        # Process in chunks for better memory usage
        chunk_size = 100
        result_ids = []
        
        for i in range(0, len(entries), chunk_size):
            chunk = entries[i:i + chunk_size]
            chunk_ids = self._add_vector_chunk(chunk)
            result_ids.extend(chunk_ids)
        
        return result_ids
    
    def search_similar_async(self, query_vector: np.ndarray, **kwargs):
        """Async version of similarity search"""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(
            self.executor, 
            self.search_similar, 
            query_vector, 
            **kwargs
        )
```

### 4. Logging and Monitoring / ログとモニタリング

```python
import logging
import time
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor method performance"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time
            
            # Update statistics
            if hasattr(self, 'processing_stats'):
                self.processing_stats['processing_time'] += duration
                self.processing_stats[f'{func.__name__}_calls'] = \
                    self.processing_stats.get(f'{func.__name__}_calls', 0) + 1
            
            logger.debug(f"{func.__name__} completed in {duration:.3f}s")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    return wrapper

class MonitoredKeywordStore(KeywordSearch):
    @monitor_performance
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        # Your implementation here
        pass
```

### 5. Testing Strategies / テスト戦略

```python
# Use pytest fixtures for consistent test setup
@pytest.fixture(scope="session")
def test_elasticsearch():
    """Start test Elasticsearch instance"""
    # Setup test instance (e.g., using testcontainers)
    yield es_instance
    # Cleanup

@pytest.fixture
def clean_index(test_elasticsearch):
    """Ensure clean index for each test"""
    index_name = "test_index"
    # Clean up before test
    yield index_name
    # Clean up after test

# Use property-based testing for edge cases
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=1000))
def test_search_with_random_queries(keyword_store, query):
    """Test search with various query inputs"""
    # Should not crash on any valid text input
    results = keyword_store.search(query, limit=5)
    assert isinstance(results, list)
```

This comprehensive plugin development guide provides everything needed to create robust, production-ready KeywordSearch and VectorStore plugins for the refinire-rag system. The examples demonstrate real-world implementations with proper error handling, testing, and best practices.