# DocumentStoreLoader設計書

## 1. 概要

DocumentStoreLoaderは、既存のDocumentStoreから文書をロードし、Document Processingパイプラインに供給するLoaderクラスです。統一された例外処理により、堅牢で保守性の高い実装を提供します。

## 2. 例外設計の統合

### 使用する例外クラス

| 例外クラス | 使用場面 | 継承関係 |
|-----------|---------|---------|
| `DocumentStoreError` | DocumentStore操作全般 | `StorageError` -> `RefinireRAGError` |
| `LoaderError` | ローダー固有のエラー | `RefinireRAGError` |
| `ValidationError` | 設定・データ検証エラー | `RefinireRAGError` |
| `ConfigurationError` | 設定エラー | `RefinireRAGError` |
| `ProcessingError` | 処理パイプライン全般 | `RefinireRAGError` |

### エラーハンドリング戦略

```python
from refinire_rag.exceptions import (
    DocumentStoreError, LoaderError, ValidationError, 
    ConfigurationError, wrap_exception
)

# 例外の適切な使い分け
try:
    documents = document_store.search_by_metadata(filters)
except Exception as e:
    # 外部ライブラリの例外を統一例外にラップ
    raise wrap_exception(e, "Failed to search documents in store")
```

## 3. クラス設計

### DocumentStoreLoader

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Iterator, Iterable
from datetime import datetime

from refinire_rag.loader.loader import Loader
from refinire_rag.storage.document_store import DocumentStore
from refinire_rag.models.document import Document
from refinire_rag.metadata.metadata import Metadata
from refinire_rag.exceptions import (
    DocumentStoreError, LoaderError, ValidationError, 
    ConfigurationError, wrap_exception
)

class LoadStrategy(Enum):
    """Document loading strategies"""
    FULL = "full"           # Load all documents
    FILTERED = "filtered"   # Load with metadata/content filters  
    INCREMENTAL = "incremental"  # Load based on timestamps
    ID_LIST = "id_list"     # Load specific document IDs
    PAGINATED = "paginated" # Load in batches

@dataclass
class DocumentLoadConfig:
    """
    Configuration for document loading from DocumentStore
    DocumentStoreからの文書ロード設定
    """
    # Loading strategy
    strategy: LoadStrategy = LoadStrategy.FULL
    
    # Filtering options
    metadata_filters: Optional[Dict[str, Any]] = None
    content_query: Optional[str] = None
    document_ids: Optional[List[str]] = None
    
    # Date-based filtering
    modified_after: Optional[datetime] = None
    modified_before: Optional[datetime] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    
    # Pagination
    batch_size: int = 100
    max_documents: Optional[int] = None
    
    # Sorting
    sort_by: str = "created_at"
    sort_order: str = "desc"
    
    # Processing options
    include_deleted: bool = False
    validate_documents: bool = True
    
    def validate(self) -> None:
        """
        Validate configuration settings
        設定の妥当性を検証
        """
        if self.batch_size <= 0:
            raise ValidationError("batch_size must be positive")
        
        if self.max_documents is not None and self.max_documents <= 0:
            raise ValidationError("max_documents must be positive")
        
        if self.strategy == LoadStrategy.ID_LIST and not self.document_ids:
            raise ConfigurationError("document_ids required for ID_LIST strategy")
        
        if self.modified_after and self.modified_before:
            if self.modified_after >= self.modified_before:
                raise ValidationError("modified_after must be before modified_before")

@dataclass
class LoadResult:
    """
    Result of document loading operation
    文書ロード操作の結果
    """
    loaded_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)
    total_processed: int = 0
    
    def add_error(self, error_message: str):
        """Add error message and increment error count"""
        self.errors.append(error_message)
        self.error_count += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_processed == 0:
            return 1.0
        return (self.loaded_count + self.skipped_count) / self.total_processed

class DocumentStoreLoader(Loader):
    """
    Loader for documents from DocumentStore
    DocumentStoreからの文書ローダー
    """
    
    def __init__(self, 
                 document_store: DocumentStore,
                 load_config: Optional[DocumentLoadConfig] = None,
                 metadata_processors: Optional[List[Metadata]] = None):
        """
        Initialize DocumentStore loader
        DocumentStoreローダーを初期化
        
        Args:
            document_store: DocumentStore instance
            load_config: Loading configuration
            metadata_processors: Optional metadata processors
        """
        super().__init__(metadata_processors)
        
        if document_store is None:
            raise ConfigurationError("document_store cannot be None")
        
        self.document_store = document_store
        self.load_config = load_config or DocumentLoadConfig()
        
        # Validate configuration
        try:
            self.load_config.validate()
        except (ValidationError, ConfigurationError) as e:
            raise ConfigurationError(f"Invalid load configuration: {e}")
    
    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """
        DocumentProcessor interface implementation
        DocumentProcessorインターフェース実装
        
        This method loads documents from the store and yields them,
        ignoring the input documents parameter.
        このメソッドはストアから文書をロードしてyieldし、
        入力documentsパラメータは無視します。
        """
        try:
            for document in self._load_documents():
                yield document
        except Exception as e:
            if isinstance(e, (DocumentStoreError, LoaderError)):
                raise
            else:
                raise wrap_exception(e, "Error in document processing")
    
    def load_all(self) -> LoadResult:
        """
        Load all documents matching configuration
        設定にマッチするすべての文書をロード
        """
        result = LoadResult()
        
        try:
            for document in self._load_documents():
                try:
                    if self._validate_document(document):
                        result.loaded_count += 1
                    else:
                        result.skipped_count += 1
                except Exception as e:
                    result.add_error(f"Error processing document {document.id}: {str(e)}")
                finally:
                    result.total_processed += 1
                    
        except Exception as e:
            error_msg = f"Failed to load documents: {str(e)}"
            result.add_error(error_msg)
            if isinstance(e, (DocumentStoreError, LoaderError)):
                raise
            else:
                raise wrap_exception(e, "Document loading failed")
        
        return result
    
    def _load_documents(self) -> Iterator[Document]:
        """
        Load documents based on configuration strategy
        設定戦略に基づいて文書をロード
        """
        try:
            if self.load_config.strategy == LoadStrategy.FULL:
                yield from self._load_all_documents()
            elif self.load_config.strategy == LoadStrategy.FILTERED:
                yield from self._load_filtered_documents()
            elif self.load_config.strategy == LoadStrategy.INCREMENTAL:
                yield from self._load_incremental_documents()
            elif self.load_config.strategy == LoadStrategy.ID_LIST:
                yield from self._load_by_ids()
            elif self.load_config.strategy == LoadStrategy.PAGINATED:
                yield from self._load_paginated_documents()
            else:
                raise LoaderError(f"Unsupported load strategy: {self.load_config.strategy}")
                
        except DocumentStoreError:
            # Re-raise DocumentStore errors as-is
            raise
        except Exception as e:
            raise wrap_exception(e, f"Error loading documents with strategy {self.load_config.strategy}")
    
    def _load_all_documents(self) -> Iterator[Document]:
        """Load all documents from store"""
        try:
            offset = 0
            while True:
                documents = self.document_store.list_documents(
                    limit=self.load_config.batch_size,
                    offset=offset,
                    sort_by=self.load_config.sort_by,
                    sort_order=self.load_config.sort_order
                )
                
                if not documents:
                    break
                
                for doc in documents:
                    yield doc
                    
                offset += len(documents)
                
                # Check max_documents limit
                if (self.load_config.max_documents and 
                    offset >= self.load_config.max_documents):
                    break
                    
        except Exception as e:
            raise wrap_exception(e, "Error loading all documents")
    
    def _load_filtered_documents(self) -> Iterator[Document]:
        """Load documents with metadata/content filters"""
        try:
            if self.load_config.metadata_filters:
                # Build complete filter including date filters
                filters = self._build_metadata_filters()
                search_results = self.document_store.search_by_metadata(
                    filters=filters,
                    limit=self.load_config.max_documents or 1000000
                )
                for result in search_results:
                    yield result.document
                    
            elif self.load_config.content_query:
                search_results = self.document_store.search_by_content(
                    query=self.load_config.content_query,
                    limit=self.load_config.max_documents or 1000000
                )
                for result in search_results:
                    yield result.document
            else:
                # No specific filters, load all
                yield from self._load_all_documents()
                
        except Exception as e:
            raise wrap_exception(e, "Error loading filtered documents")
    
    def _load_incremental_documents(self) -> Iterator[Document]:
        """Load documents based on modification timestamps"""
        try:
            filters = {}
            
            if self.load_config.modified_after:
                filters['modified_at'] = {'$gte': self.load_config.modified_after.isoformat()}
            
            if self.load_config.modified_before:
                if 'modified_at' in filters:
                    filters['modified_at']['$lte'] = self.load_config.modified_before.isoformat()
                else:
                    filters['modified_at'] = {'$lte': self.load_config.modified_before.isoformat()}
            
            if not filters:
                raise LoaderError("No timestamp filters specified for incremental loading")
            
            search_results = self.document_store.search_by_metadata(
                filters=filters,
                limit=self.load_config.max_documents or 1000000
            )
            
            for result in search_results:
                yield result.document
                
        except LoaderError:
            raise
        except Exception as e:
            raise wrap_exception(e, "Error loading incremental documents")
    
    def _load_by_ids(self) -> Iterator[Document]:
        """Load specific documents by IDs"""
        if not self.load_config.document_ids:
            raise LoaderError("No document IDs specified for ID list loading")
        
        try:
            for doc_id in self.load_config.document_ids:
                document = self.document_store.get_document(doc_id)
                if document:
                    yield document
                elif self.load_config.validate_documents:
                    raise LoaderError(f"Document not found: {doc_id}")
                    
        except LoaderError:
            raise
        except Exception as e:
            raise wrap_exception(e, "Error loading documents by IDs")
    
    def _load_paginated_documents(self) -> Iterator[Document]:
        """Load documents in paginated fashion"""
        try:
            offset = 0
            total_loaded = 0
            
            while True:
                documents = self.document_store.list_documents(
                    limit=self.load_config.batch_size,
                    offset=offset,
                    sort_by=self.load_config.sort_by,
                    sort_order=self.load_config.sort_order
                )
                
                if not documents:
                    break
                
                for doc in documents:
                    yield doc
                    total_loaded += 1
                    
                    if (self.load_config.max_documents and 
                        total_loaded >= self.load_config.max_documents):
                        return
                
                offset += len(documents)
                
        except Exception as e:
            raise wrap_exception(e, "Error loading paginated documents")
    
    def _build_metadata_filters(self) -> Dict[str, Any]:
        """Build complete metadata filters from configuration"""
        filters = {}
        
        if self.load_config.metadata_filters:
            filters.update(self.load_config.metadata_filters)
        
        # Add date-based filters
        if self.load_config.modified_after or self.load_config.modified_before:
            date_filter = {}
            if self.load_config.modified_after:
                date_filter['$gte'] = self.load_config.modified_after.isoformat()
            if self.load_config.modified_before:
                date_filter['$lte'] = self.load_config.modified_before.isoformat()
            filters['modified_at'] = date_filter
        
        return filters
    
    def _validate_document(self, document: Document) -> bool:
        """
        Validate document before yielding
        文書をyieldする前に検証
        """
        if not self.load_config.validate_documents:
            return True
        
        try:
            # Basic validation
            if not document.id:
                raise ValidationError("Document missing ID")
            
            if not document.content and not document.metadata:
                raise ValidationError("Document has no content or metadata")
            
            return True
            
        except ValidationError as e:
            if self.load_config.validate_documents:
                raise LoaderError(f"Document validation failed: {e}")
            return False
    
    def count_matching_documents(self) -> int:
        """
        Count documents that would be loaded with current configuration
        現在の設定でロードされる文書数をカウント
        """
        try:
            if self.load_config.strategy == LoadStrategy.FULL:
                return self.document_store.count_documents()
            elif self.load_config.strategy == LoadStrategy.FILTERED:
                if self.load_config.metadata_filters:
                    filters = self._build_metadata_filters()
                    return self.document_store.count_documents(filters)
                else:
                    return self.document_store.count_documents()
            elif self.load_config.strategy == LoadStrategy.ID_LIST:
                return len(self.load_config.document_ids or [])
            else:
                # For other strategies, we'd need to actually query
                return -1  # Unknown
                
        except Exception as e:
            raise wrap_exception(e, "Error counting matching documents")
```

## 4. 使用例

```python
from refinire_rag.loader.document_store_loader import DocumentStoreLoader, DocumentLoadConfig, LoadStrategy
from refinire_rag.storage.document_store import DocumentStore
from refinire_rag.exceptions import DocumentStoreError, LoaderError, ConfigurationError
from datetime import datetime, timedelta

# Example 1: Load all documents with error handling
try:
    loader = DocumentStoreLoader(
        document_store=my_store,
        load_config=DocumentLoadConfig(strategy=LoadStrategy.FULL)
    )
    result = loader.load_all()
    print(f"Loaded: {result.loaded_count}, Errors: {result.error_count}")
    
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except DocumentStoreError as e:
    print(f"Document store error: {e}")
except LoaderError as e:
    print(f"Loader error: {e}")

# Example 2: Incremental loading with validation
try:
    config = DocumentLoadConfig(
        strategy=LoadStrategy.INCREMENTAL,
        modified_after=datetime.now() - timedelta(days=7),
        validate_documents=True,
        batch_size=50
    )
    
    loader = DocumentStoreLoader(my_store, config)
    
    # Count before loading
    count = loader.count_matching_documents()
    print(f"Will load {count} documents")
    
    # Process documents
    for document in loader.process([]):
        process_document(document)
        
except ValidationError as e:
    print(f"Validation error: {e}")
except Exception as e:
    # Handle unexpected errors
    print(f"Unexpected error: {e}")
```

この設計により、統一された例外処理を持つ堅牢なDocumentStoreLoaderが実現できます。