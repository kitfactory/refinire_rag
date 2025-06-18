# インクリメンタルローダー詳細設計書

## モジュール構成

```
src/refinire_rag/
├── models/
│   └── (既存のモデル)
└── loader/
    ├── models/
    │   ├── __init__.py
    │   ├── file_info.py
    │   ├── change_set.py
    │   ├── sync_result.py
    │   └── filter_config.py
    ├── file_tracker.py
    ├── incremental_directory_loader.py
    └── filters/
        ├── __init__.py
        ├── base_filter.py
        ├── extension_filter.py
        ├── size_filter.py
        ├── date_filter.py
        └── path_filter.py
```

## 1. フィルター設計

### BaseFilter
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path
from refinire_rag.loader.models.file_info import FileInfo

class BaseFilter(ABC):
    """
    ファイルフィルターの基底クラス
    Base class for file filters
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize filter with configuration
        設定でフィルターを初期化
        
        Args:
            config: Filter-specific configuration
            config: フィルター固有の設定
        """
        self.config = config or {}
    
    @abstractmethod
    def should_include(self, file_info: FileInfo) -> bool:
        """
        Determine if a file should be included
        ファイルを含めるかどうかを判定
        
        Args:
            file_info: File information to evaluate
            file_info: 評価するファイル情報
            
        Returns:
            True if file should be included
            ファイルを含める場合はTrue
        """
        pass
    
    @abstractmethod
    def get_additional_metadata(self, file_info: FileInfo) -> Dict[str, Any]:
        """
        Get additional metadata to attach to documents
        文書に添付する追加メタデータを取得
        
        Args:
            file_info: File information
            file_info: ファイル情報
            
        Returns:
            Additional metadata dictionary
            追加メタデータ辞書
        """
        pass
```

### 具体的なフィルター実装

#### ExtensionFilter
```python
class ExtensionFilter(BaseFilter):
    """
    ファイル拡張子によるフィルター
    Filter files by extension
    """
    
    def __init__(self, allowed_extensions: List[str] = None, blocked_extensions: List[str] = None):
        """
        Initialize extension filter
        拡張子フィルターを初期化
        
        Args:
            allowed_extensions: List of allowed extensions (e.g., ['.txt', '.md'])
            allowed_extensions: 許可する拡張子のリスト
            blocked_extensions: List of blocked extensions
            blocked_extensions: ブロックする拡張子のリスト
        """
        super().__init__()
        self.allowed_extensions = set(allowed_extensions or [])
        self.blocked_extensions = set(blocked_extensions or [])
    
    def should_include(self, file_info: FileInfo) -> bool:
        ext = Path(file_info.path).suffix.lower()
        
        if self.blocked_extensions and ext in self.blocked_extensions:
            return False
        
        if self.allowed_extensions and ext not in self.allowed_extensions:
            return False
        
        return True
    
    def get_additional_metadata(self, file_info: FileInfo) -> Dict[str, Any]:
        return {
            'file_extension': Path(file_info.path).suffix.lower(),
            'filter_type': 'extension'
        }
```

#### DateFilter
```python
class DateFilter(BaseFilter):
    """
    更新日時によるフィルター
    Filter files by modification date
    """
    
    def __init__(self, after: datetime = None, before: datetime = None):
        """
        Initialize date filter
        日付フィルターを初期化
        
        Args:
            after: Include files modified after this date
            after: この日付以降に更新されたファイルを含める
            before: Include files modified before this date
            before: この日付以前に更新されたファイルを含める
        """
        super().__init__()
        self.after = after
        self.before = before
    
    def should_include(self, file_info: FileInfo) -> bool:
        if self.after and file_info.modified_at < self.after:
            return False
        
        if self.before and file_info.modified_at > self.before:
            return False
        
        return True
    
    def get_additional_metadata(self, file_info: FileInfo) -> Dict[str, Any]:
        return {
            'last_modified': file_info.modified_at.isoformat(),
            'filter_type': 'date'
        }
```

#### PathFilter
```python
class PathFilter(BaseFilter):
    """
    パスパターンによるフィルター
    Filter files by path patterns
    """
    
    def __init__(self, include_patterns: List[str] = None, exclude_patterns: List[str] = None):
        """
        Initialize path filter
        パスフィルターを初期化
        
        Args:
            include_patterns: Glob patterns to include (e.g., ['**/docs/**'])
            include_patterns: 含めるglobパターン
            exclude_patterns: Glob patterns to exclude (e.g., ['**/.git/**'])
            exclude_patterns: 除外するglobパターン
        """
        super().__init__()
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
    
    def should_include(self, file_info: FileInfo) -> bool:
        from fnmatch import fnmatch
        
        path = file_info.path
        
        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if fnmatch(path, pattern):
                return False
        
        # If include patterns are specified, file must match at least one
        if self.include_patterns:
            return any(fnmatch(path, pattern) for pattern in self.include_patterns)
        
        return True
    
    def get_additional_metadata(self, file_info: FileInfo) -> Dict[str, Any]:
        path_obj = Path(file_info.path)
        return {
            'directory': str(path_obj.parent),
            'filename': path_obj.name,
            'depth': len(path_obj.parts),
            'filter_type': 'path'
        }
```

### FilterConfig
```python
@dataclass
class FilterConfig:
    """
    フィルター設定のコンテナ
    Container for filter configuration
    """
    extension_filter: ExtensionFilter = None
    date_filter: DateFilter = None
    path_filter: PathFilter = None
    size_filter: SizeFilter = None
    custom_filters: List[BaseFilter] = field(default_factory=list)
    
    def get_all_filters(self) -> List[BaseFilter]:
        """
        Get all configured filters
        設定されたすべてのフィルターを取得
        """
        filters = []
        
        if self.extension_filter:
            filters.append(self.extension_filter)
        if self.date_filter:
            filters.append(self.date_filter)
        if self.path_filter:
            filters.append(self.path_filter)
        if self.size_filter:
            filters.append(self.size_filter)
        
        filters.extend(self.custom_filters)
        return filters
```

## 2. FileTracker (更新版)

```python
class FileTracker:
    """
    ファイルの状態を追跡・比較し、フィルタリングとメタデータ付与を行う
    Track and compare file states with filtering and metadata enhancement
    """
    
    def __init__(self, filter_config: FilterConfig = None):
        """
        Initialize file tracker
        ファイルトラッカーを初期化
        
        Args:
            filter_config: Configuration for file filtering
            filter_config: ファイルフィルタリングの設定
        """
        self.filter_config = filter_config or FilterConfig()
        self.filters = self.filter_config.get_all_filters()
    
    def scan_directory(self, path: Path, recursive: bool = True) -> Dict[str, FileInfo]:
        """
        ディレクトリ内のファイル情報を収集（フィルタリング適用）
        Scan directory for file information with filtering applied
        
        Args:
            path: Directory path to scan
            path: スキャンするディレクトリパス
            recursive: Whether to scan recursively
            recursive: 再帰的にスキャンするかどうか
            
        Returns:
            Dictionary mapping file paths to FileInfo objects
            ファイルパスからFileInfoオブジェクトへのマッピング辞書
        """
        files = {}
        
        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if not file_path.is_file():
                continue
            
            file_info = FileInfo.from_file(file_path)
            
            # Apply filters
            if self._should_include_file(file_info):
                # Add additional metadata from filters
                file_info.additional_metadata = self._get_additional_metadata(file_info)
                files[str(file_path)] = file_info
        
        return files
    
    def _should_include_file(self, file_info: FileInfo) -> bool:
        """
        Check if file should be included based on filters
        フィルターに基づいてファイルを含めるかどうかをチェック
        """
        return all(filter.should_include(file_info) for filter in self.filters)
    
    def _get_additional_metadata(self, file_info: FileInfo) -> Dict[str, Any]:
        """
        Get additional metadata from all filters
        すべてのフィルターから追加メタデータを取得
        """
        metadata = {}
        for filter in self.filters:
            filter_metadata = filter.get_additional_metadata(file_info)
            metadata.update(filter_metadata)
        
        return metadata
    
    def get_corpus_files(self, corpus_store: CorpusStore) -> Dict[str, FileInfo]:
        """
        CorpusStore内のファイル情報を取得
        Get file information from CorpusStore
        """
        stored_files = {}
        
        # CorpusStoreから全文書のメタデータを取得
        documents_metadata = corpus_store.get_file_metadata()
        
        for doc_metadata in documents_metadata:
            if 'file_path' in doc_metadata:
                file_info = FileInfo.from_document_metadata(doc_metadata)
                stored_files[doc_metadata['file_path']] = file_info
        
        return stored_files
    
    def compare(self, current: Dict[str, FileInfo], stored: Dict[str, FileInfo]) -> ChangeSet:
        """
        ファイル情報を比較して変更を検出
        Compare file information to detect changes
        """
        current_paths = set(current.keys())
        stored_paths = set(stored.keys())
        
        added = current_paths - stored_paths
        deleted = stored_paths - current_paths
        common = current_paths & stored_paths
        
        # Check for modifications in common files
        modified = []
        unchanged = []
        
        for path in common:
            current_file = current[path]
            stored_file = stored[path]
            
            # Compare hash, size, and modified_at
            if (current_file.hash_md5 != stored_file.hash_md5 or
                current_file.size != stored_file.size or
                current_file.modified_at != stored_file.modified_at):
                modified.append(path)
            else:
                unchanged.append(path)
        
        return ChangeSet(
            added=list(added),
            modified=modified,
            deleted=list(deleted),
            unchanged=unchanged
        )
```

## 3. IncrementalDirectoryLoader (更新版)

```python
class IncrementalDirectoryLoader:
    """
    ディレクトリの変更を検出し、フィルタリングとメタデータ付与を行いながらインクリメンタルにロードする
    Detect directory changes and incrementally load with filtering and metadata enhancement
    """
    
    def __init__(self, 
                 directory_path: str, 
                 corpus_store: CorpusStore,
                 filter_config: FilterConfig = None,
                 additional_metadata: Dict[str, Any] = None):
        """
        Initialize incremental directory loader
        インクリメンタルディレクトリローダーを初期化
        
        Args:
            directory_path: Path to monitor
            directory_path: 監視するパス
            corpus_store: Storage for documents
            corpus_store: 文書の保存先
            filter_config: Configuration for file filtering
            filter_config: ファイルフィルタリングの設定
            additional_metadata: Additional metadata to add to all documents
            additional_metadata: 全文書に追加するメタデータ
        """
        self.directory_path = Path(directory_path)
        self.corpus_store = corpus_store
        self.filter_config = filter_config or FilterConfig()
        self.additional_metadata = additional_metadata or {}
        
        self.file_tracker = FileTracker(filter_config)
        self.directory_loader = DirectoryLoader(str(directory_path))
    
    def detect_changes(self) -> ChangeSet:
        """
        ディレクトリとCorpusStoreの差分を検出（フィルタリング適用）
        Detect differences between directory and CorpusStore with filtering
        """
        current_files = self.file_tracker.scan_directory(self.directory_path)
        stored_files = self.file_tracker.get_corpus_files(self.corpus_store)
        
        changes = self.file_tracker.compare(current_files, stored_files)
        return changes
    
    def sync(self) -> SyncResult:
        """
        差分に基づいてCorpusStoreを同期（メタデータ付与）
        Sync CorpusStore based on differences with metadata enhancement
        """
        changes = self.detect_changes()
        result = SyncResult()
        
        # Get current file info for metadata enhancement
        current_files = self.file_tracker.scan_directory(self.directory_path)
        
        # Process added files
        for file_path in changes.added:
            try:
                docs = self._load_file_with_metadata(file_path, current_files.get(file_path))
                self.corpus_store.add_documents(docs)
                result.added_documents.extend(docs)
            except Exception as e:
                result.errors.append(f"Error adding {file_path}: {e}")
        
        # Process modified files
        for file_path in changes.modified:
            try:
                # Delete old documents
                old_docs = self.corpus_store.get_documents_by_path(file_path)
                self.corpus_store.delete_documents([doc.id for doc in old_docs])
                
                # Add new documents with metadata
                docs = self._load_file_with_metadata(file_path, current_files.get(file_path))
                self.corpus_store.add_documents(docs)
                result.updated_documents.extend(docs)
            except Exception as e:
                result.errors.append(f"Error updating {file_path}: {e}")
        
        # Process deleted files
        for file_path in changes.deleted:
            try:
                docs = self.corpus_store.get_documents_by_path(file_path)
                doc_ids = [doc.id for doc in docs]
                self.corpus_store.delete_documents(doc_ids)
                result.deleted_document_ids.extend(doc_ids)
            except Exception as e:
                result.errors.append(f"Error deleting {file_path}: {e}")
        
        return result
    
    def _load_file_with_metadata(self, file_path: str, file_info: FileInfo = None) -> List[Document]:
        """
        ファイルをロードし、フィルターとローダー設定からのメタデータを付与
        Load file and enhance with metadata from filters and loader configuration
        """
        # Load documents using directory loader
        docs = list(self.directory_loader.load_file(file_path))
        
        # Enhance each document with additional metadata
        for doc in docs:
            # Add file tracking metadata
            if file_info:
                doc.metadata.update({
                    'file_size': file_info.size,
                    'file_modified_at': file_info.modified_at.isoformat(),
                    'file_hash': file_info.hash_md5,
                    'loaded_at': datetime.now().isoformat(),
                })
                
                # Add filter-generated metadata
                if hasattr(file_info, 'additional_metadata'):
                    doc.metadata.update(file_info.additional_metadata)
            
            # Add user-specified additional metadata
            doc.metadata.update(self.additional_metadata)
            
            # Add loader identification
            doc.metadata.update({
                'loader_type': 'incremental_directory',
                'loader_version': '1.0.0'
            })
        
        return docs
```

## 使用例

```python
from refinire_rag.loader.incremental_directory_loader import IncrementalDirectoryLoader
from refinire_rag.loader.models.filter_config import FilterConfig
from refinire_rag.loader.filters.extension_filter import ExtensionFilter
from refinire_rag.loader.filters.date_filter import DateFilter
from refinire_rag.loader.filters.path_filter import PathFilter

# フィルター設定
filter_config = FilterConfig(
    extension_filter=ExtensionFilter(
        allowed_extensions=['.txt', '.md', '.pdf']
    ),
    date_filter=DateFilter(
        after=datetime(2024, 1, 1)  # 2024年以降のファイルのみ
    ),
    path_filter=PathFilter(
        exclude_patterns=['**/.git/**', '**/node_modules/**']
    )
)

# 追加メタデータ
additional_metadata = {
    'project': 'documentation',
    'department': 'engineering',
    'classification': 'public'
}

# ローダー初期化
loader = IncrementalDirectoryLoader(
    directory_path="/path/to/documents",
    corpus_store=corpus_store,
    filter_config=filter_config,
    additional_metadata=additional_metadata
)

# 同期実行
sync_result = loader.sync()
print(f"処理結果: 追加{len(sync_result.added_documents)}, 更新{len(sync_result.updated_documents)}")
```

この設計により、柔軟なフィルタリングと豊富なメタデータ付与が可能になります。