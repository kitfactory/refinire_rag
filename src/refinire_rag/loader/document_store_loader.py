"""
DocumentStore Loader for loading documents from existing DocumentStore
既存のDocumentStoreから文書をロードするDocumentStoreLoader
"""

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
    """
    Document loading strategies
    文書ロード戦略
    """
    FULL = "full"           # Load all documents / すべての文書をロード
    FILTERED = "filtered"   # Load with metadata/content filters / メタデータ・内容フィルターでロード
    INCREMENTAL = "incremental"  # Load based on timestamps / タイムスタンプベースでロード
    ID_LIST = "id_list"     # Load specific document IDs / 特定の文書IDをロード
    PAGINATED = "paginated" # Load in batches / バッチでロード


@dataclass
class DocumentLoadConfig:
    """
    Configuration for document loading from DocumentStore
    DocumentStoreからの文書ロード設定
    """
    # Loading strategy
    # ロード戦略
    strategy: LoadStrategy = LoadStrategy.FULL
    
    # Filtering options
    # フィルタリングオプション
    metadata_filters: Optional[Dict[str, Any]] = None
    content_query: Optional[str] = None
    document_ids: Optional[List[str]] = None
    
    # Date-based filtering
    # 日付ベースフィルタリング
    modified_after: Optional[datetime] = None
    modified_before: Optional[datetime] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    
    # Pagination
    # ページング
    batch_size: int = 100
    max_documents: Optional[int] = None
    
    # Sorting
    # ソート
    sort_by: str = "created_at"
    sort_order: str = "desc"
    
    # Processing options
    # 処理オプション
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
        
        if self.created_after and self.created_before:
            if self.created_after >= self.created_before:
                raise ValidationError("created_after must be before created_before")
        
        if self.sort_order not in ["asc", "desc"]:
            raise ValidationError("sort_order must be 'asc' or 'desc'")


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
        """
        Add error message and increment error count
        エラーメッセージを追加してエラー数を増加
        """
        self.errors.append(error_message)
        self.error_count += 1
    
    @property
    def success_rate(self) -> float:
        """
        Calculate success rate
        成功率を計算
        """
        if self.total_processed == 0:
            return 1.0
        return (self.loaded_count + self.skipped_count) / self.total_processed
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of loading results
        ロード結果のサマリーを取得
        """
        return {
            "loaded": self.loaded_count,
            "skipped": self.skipped_count,
            "errors": self.error_count,
            "total_processed": self.total_processed,
            "success_rate": self.success_rate,
            "error_messages": self.errors[-5:] if self.errors else []  # Last 5 errors
        }


class DocumentStoreLoader(Loader):
    """
    Loader for documents from DocumentStore
    DocumentStoreからの文書ローダー
    
    This loader can load documents from an existing DocumentStore using various
    strategies including full load, filtered load, incremental load, and more.
    このローダーは既存のDocumentStoreからフルロード、フィルタードロード、
    インクリメンタルロードなどの様々な戦略を使用して文書をロードできます。
    """
    
    def __init__(self, 
                 document_store: DocumentStore,
                 load_config: Optional[DocumentLoadConfig] = None,
                 metadata_processors: Optional[List[Metadata]] = None):
        """
        Initialize DocumentStore loader
        DocumentStoreローダーを初期化
        
        Args:
            document_store: DocumentStore instance to load from
            load_config: Loading configuration
            metadata_processors: Optional metadata processors
            document_store: ロード元のDocumentStoreインスタンス
            load_config: ロード設定
            metadata_processors: オプションのメタデータプロセッサー
        """
        super().__init__(metadata_processors)
        
        if document_store is None:
            raise ConfigurationError("document_store cannot be None")
        
        self.document_store = document_store
        self.load_config = load_config or DocumentLoadConfig()
        self.metadata_processors = metadata_processors or []
        
        # Validate configuration
        # 設定を検証
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
        
        Args:
            documents: Input documents (ignored)
            config: Optional configuration (ignored)
            documents: 入力文書（無視される）
            config: オプション設定（無視される）
            
        Yields:
            Documents loaded from the store
            ストアからロードされた文書
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
        
        Returns:
            LoadResult with loading statistics
            ロード統計を含むLoadResult
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
        
        Yields:
            Documents from the store
            ストアからの文書
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
            # DocumentStoreエラーはそのまま再発生
            raise
        except Exception as e:
            raise wrap_exception(e, f"Error loading documents with strategy {self.load_config.strategy}")
    
    def _load_all_documents(self) -> Iterator[Document]:
        """
        Load all documents from store
        ストアからすべての文書をロード
        """
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
                    
                    # Check max_documents limit
                    # max_documents制限をチェック
                    if (self.load_config.max_documents and 
                        total_loaded >= self.load_config.max_documents):
                        return
                    
                offset += len(documents)
                
        except Exception as e:
            raise wrap_exception(e, "Error loading all documents")
    
    def _load_filtered_documents(self) -> Iterator[Document]:
        """
        Load documents with metadata/content filters
        メタデータ・内容フィルターで文書をロード
        """
        try:
            if self.load_config.metadata_filters or self._has_date_filters():
                # Build complete filter including date filters
                # 日付フィルターを含む完全なフィルターを構築
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
                # 特定のフィルターなし、すべてをロード
                yield from self._load_all_documents()
                
        except Exception as e:
            raise wrap_exception(e, "Error loading filtered documents")
    
    def _load_incremental_documents(self) -> Iterator[Document]:
        """
        Load documents based on modification timestamps
        更新タイムスタンプに基づいて文書をロード
        """
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
        """
        Load specific documents by IDs
        特定のIDで文書をロード
        """
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
        """
        Load documents in paginated fashion
        ページング方式で文書をロード
        """
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
    
    def _has_date_filters(self) -> bool:
        """
        Check if any date filters are configured
        日付フィルターが設定されているかチェック
        """
        return any([
            self.load_config.modified_after,
            self.load_config.modified_before,
            self.load_config.created_after,
            self.load_config.created_before
        ])
    
    def _build_metadata_filters(self) -> Dict[str, Any]:
        """
        Build complete metadata filters from configuration
        設定から完全なメタデータフィルターを構築
        """
        filters = {}
        
        if self.load_config.metadata_filters:
            filters.update(self.load_config.metadata_filters)
        
        # Add date-based filters
        # 日付ベースフィルターを追加
        if self.load_config.modified_after or self.load_config.modified_before:
            date_filter = {}
            if self.load_config.modified_after:
                date_filter['$gte'] = self.load_config.modified_after.isoformat()
            if self.load_config.modified_before:
                date_filter['$lte'] = self.load_config.modified_before.isoformat()
            filters['modified_at'] = date_filter
        
        if self.load_config.created_after or self.load_config.created_before:
            date_filter = {}
            if self.load_config.created_after:
                date_filter['$gte'] = self.load_config.created_after.isoformat()
            if self.load_config.created_before:
                date_filter['$lte'] = self.load_config.created_before.isoformat()
            filters['created_at'] = date_filter
        
        return filters
    
    def _validate_document(self, document: Document) -> bool:
        """
        Validate document before yielding
        文書をyieldする前に検証
        
        Args:
            document: Document to validate
            document: 検証する文書
            
        Returns:
            True if document is valid, False otherwise
            文書が有効な場合True、そうでなければFalse
        """
        if not self.load_config.validate_documents:
            return True
        
        try:
            # Basic validation
            # 基本的な検証
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
        
        Returns:
            Number of matching documents, or -1 if unknown
            マッチする文書数、不明な場合は-1
        """
        try:
            if self.load_config.strategy == LoadStrategy.FULL:
                return self.document_store.count_documents()
            elif self.load_config.strategy == LoadStrategy.FILTERED:
                if self.load_config.metadata_filters or self._has_date_filters():
                    filters = self._build_metadata_filters()
                    return self.document_store.count_documents(filters)
                else:
                    return self.document_store.count_documents()
            elif self.load_config.strategy == LoadStrategy.ID_LIST:
                return len(self.load_config.document_ids or [])
            else:
                # For other strategies, we'd need to actually query
                # 他の戦略では実際にクエリが必要
                return -1  # Unknown
                
        except Exception as e:
            raise wrap_exception(e, "Error counting matching documents")
    
    def get_load_summary(self) -> Dict[str, Any]:
        """
        Get summary of loader configuration and capabilities
        ローダー設定と機能のサマリーを取得
        
        Returns:
            Dictionary with loader information
            ローダー情報を含む辞書
        """
        try:
            return {
                "strategy": self.load_config.strategy.value,
                "batch_size": self.load_config.batch_size,
                "max_documents": self.load_config.max_documents,
                "has_metadata_filters": bool(self.load_config.metadata_filters),
                "has_content_query": bool(self.load_config.content_query),
                "has_date_filters": self._has_date_filters(),
                "validate_documents": self.load_config.validate_documents,
                "estimated_count": self.count_matching_documents()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def __str__(self) -> str:
        """
        String representation of the loader
        ローダーの文字列表現
        """
        return f"DocumentStoreLoader(strategy={self.load_config.strategy.value})"
    
    def __repr__(self) -> str:
        """
        Developer representation of the loader
        ローダーの開発者向け表現
        """
        return (f"DocumentStoreLoader(strategy={self.load_config.strategy.value}, "
                f"batch_size={self.load_config.batch_size}, "
                f"validate={self.load_config.validate_documents})")