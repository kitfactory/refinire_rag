# インクリメンタルローダー設計書（フィルターメタデータ除去版）

## フィルター設計の修正

フィルターは**ファイルの包含/除外判定のみ**を行い、メタデータは追加しません。

### 1. BaseFilter（修正版）

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
    
    def get_filter_name(self) -> str:
        """
        Get the name of this filter for logging/debugging
        ログ/デバッグ用のフィルター名を取得
        
        Returns:
            Filter name
            フィルター名
        """
        return self.__class__.__name__
```

### 2. 具体的なフィルター実装（修正版）

#### ExtensionFilter
```python
class ExtensionFilter(BaseFilter):
    """
    ファイル拡張子によるフィルター（メタデータ追加なし）
    Filter files by extension (no metadata addition)
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
        self.allowed_extensions = set(ext.lower() for ext in (allowed_extensions or []))
        self.blocked_extensions = set(ext.lower() for ext in (blocked_extensions or []))
    
    def should_include(self, file_info: FileInfo) -> bool:
        """
        Check if file should be included based on extension
        拡張子に基づいてファイルを含めるかチェック
        """
        ext = Path(file_info.path).suffix.lower()
        
        # Check blocked extensions first
        if self.blocked_extensions and ext in self.blocked_extensions:
            return False
        
        # If allowed extensions are specified, file must match
        if self.allowed_extensions and ext not in self.allowed_extensions:
            return False
        
        return True
```

#### DateFilter
```python
class DateFilter(BaseFilter):
    """
    更新日時によるフィルター（メタデータ追加なし）
    Filter files by modification date (no metadata addition)
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
        """
        Check if file should be included based on modification date
        更新日時に基づいてファイルを含めるかチェック
        """
        if self.after and file_info.modified_at < self.after:
            return False
        
        if self.before and file_info.modified_at > self.before:
            return False
        
        return True
```

#### PathFilter
```python
class PathFilter(BaseFilter):
    """
    パスパターンによるフィルター（メタデータ追加なし）
    Filter files by path patterns (no metadata addition)
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
        """
        Check if file should be included based on path patterns
        パスパターンに基づいてファイルを含めるかチェック
        """
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
```

#### SizeFilter
```python
class SizeFilter(BaseFilter):
    """
    ファイルサイズによるフィルター（メタデータ追加なし）
    Filter files by size (no metadata addition)
    """
    
    def __init__(self, min_size: int = None, max_size: int = None):
        """
        Initialize size filter
        サイズフィルターを初期化
        
        Args:
            min_size: Minimum file size in bytes
            min_size: 最小ファイルサイズ（バイト）
            max_size: Maximum file size in bytes
            max_size: 最大ファイルサイズ（バイト）
        """
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
    
    def should_include(self, file_info: FileInfo) -> bool:
        """
        Check if file should be included based on size
        サイズに基づいてファイルを含めるかチェック
        """
        if self.min_size is not None and file_info.size < self.min_size:
            return False
        
        if self.max_size is not None and file_info.size > self.max_size:
            return False
        
        return True
```

## 3. FileTracker（修正版）

```python
class FileTracker:
    """
    ファイルの状態を追跡・比較し、フィルタリングを行う（メタデータ追加なし）
    Track and compare file states with filtering (no metadata addition)
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
        ディレクトリ内のファイル情報を収集（フィルタリング適用、メタデータ追加なし）
        Scan directory for file information with filtering applied (no metadata addition)
        
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
            
            # Apply filters (only for inclusion/exclusion decision)
            # フィルターを適用（包含/除外の判定のみ）
            if self._should_include_file(file_info):
                files[str(file_path)] = file_info
        
        return files
    
    def _should_include_file(self, file_info: FileInfo) -> bool:
        """
        Check if file should be included based on filters
        フィルターに基づいてファイルを含めるかどうかをチェック
        """
        return all(filter.should_include(file_info) for filter in self.filters)
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about applied filters
        適用されたフィルターの統計を取得
        
        Returns:
            Dictionary with filter statistics
            フィルター統計の辞書
        """
        return {
            'total_filters': len(self.filters),
            'filter_names': [filter.get_filter_name() for filter in self.filters],
            'filter_types': {
                'extension': any(isinstance(f, ExtensionFilter) for f in self.filters),
                'date': any(isinstance(f, DateFilter) for f in self.filters),
                'path': any(isinstance(f, PathFilter) for f in self.filters),
                'size': any(isinstance(f, SizeFilter) for f in self.filters),
            }
        }
    
    # 他のメソッドは変更なし...
```

## 4. IncrementalDirectoryLoader（修正版）

```python
class IncrementalDirectoryLoader(Loader):
    """
    Directory loader with incremental loading capabilities (filters don't add metadata)
    インクリメンタルローディング機能付きディレクトリローダー（フィルターはメタデータを追加しない）
    """
    
    def _load_file_with_metadata(self, file_path: str, file_info: 'FileInfo' = None) -> List[Document]:
        """
        Load file and enhance with metadata (only from loader configuration, not filters)
        ファイルをロードし、メタデータを付与（ローダー設定からのみ、フィルターからは追加しない）
        
        Args:
            file_path: Path to the file to load
            file_path: ロードするファイルのパス
            file_info: File information (used only for basic file metadata)
            file_info: ファイル情報（基本的なファイルメタデータのみに使用）
            
        Returns:
            List of documents with enhanced metadata
            拡張メタデータ付きの文書リスト
        """
        # Load documents using directory loader
        temp_loader = DirectoryLoader(
            directory_path=str(self.directory_path.parent),
            recursive=False
        )
        
        # Create a mock document for the specific file
        mock_doc = Document(
            id="temp_id",
            content="",
            metadata={'file_path': file_path}
        )
        
        # Load the specific file
        docs = list(temp_loader.process([mock_doc]))
        
        # Enhance each document with metadata (no filter metadata)
        for doc in docs:
            self._enhance_document_metadata(doc, file_info)
        
        return docs
    
    def _enhance_document_metadata(self, doc: Document, file_info: 'FileInfo' = None):
        """
        Enhance document with metadata (excluding filter-generated metadata)
        文書にメタデータを付与（フィルター生成メタデータは除外）
        
        Args:
            doc: Document to enhance
            doc: 拡張する文書
            file_info: File information (used only for basic file metadata)
            file_info: ファイル情報（基本的なファイルメタデータのみに使用）
        """
        # Add basic file tracking metadata (not from filters)
        # 基本的なファイル追跡メタデータを追加（フィルターからではない）
        if file_info:
            doc.metadata.update({
                'file_size': file_info.size,
                'file_modified_at': file_info.modified_at.isoformat(),
                'file_hash': file_info.hash_md5,
            })
        
        # Add loader metadata
        doc.metadata.update({
            'loaded_at': datetime.now().isoformat(),
            'loader_type': 'incremental_directory',
            'loader_version': '1.0.0',
            'directory_path': str(self.directory_path),
            'recursive_scan': self.recursive,
        })
        
        # Add user-specified additional metadata only
        # ユーザー指定の追加メタデータのみを追加
        doc.metadata.update(self.additional_metadata)
    
    def get_scan_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the last directory scan (including filter info)
        最後のディレクトリスキャンの統計を取得（フィルター情報を含む）
        
        Returns:
            Dictionary with scan statistics
            スキャン統計の辞書
        """
        stats = {
            'last_scan_time': self._last_scan_time.isoformat() if self._last_scan_time else None,
            'total_files_found': len(self._current_files),
            'directory_path': str(self.directory_path),
            'recursive': self.recursive,
        }
        
        # Add filter statistics
        if self.file_tracker:
            stats.update({
                'filter_statistics': self.file_tracker.get_filter_statistics()
            })
        
        return stats
```

## 5. メタデータ構成（修正版）

フィルター情報は含まれず、以下のメタデータのみが文書に追加されます：

```python
document.metadata = {
    # 既存のローダーメタデータ（DirectoryLoaderから）
    'file_path': '/path/to/file.txt',
    'file_type': 'text',
    
    # 基本ファイル情報（FileInfoから）
    'file_size': 1024,
    'file_modified_at': '2025-06-14T18:30:00',
    'file_hash': 'abc123...',
    
    # ローダー識別情報
    'loaded_at': '2025-06-14T18:30:05',
    'loader_type': 'incremental_directory',
    'loader_version': '1.0.0',
    'directory_path': '/path/to/documents',
    'recursive_scan': True,
    
    # ユーザー指定の追加メタデータのみ
    'project': 'documentation',
    'department': 'engineering',
}
```

## 6. 使用例（修正版）

```python
from refinire_rag.loader.incremental_directory_loader import IncrementalDirectoryLoader
from refinire_rag.loader.models.filter_config import FilterConfig
from refinire_rag.loader.filters.extension_filter import ExtensionFilter
from refinire_rag.loader.filters.path_filter import PathFilter

# フィルター設定（メタデータ追加なし）
filter_config = FilterConfig(
    extension_filter=ExtensionFilter(
        allowed_extensions=['.txt', '.md', '.pdf']
    ),
    path_filter=PathFilter(
        exclude_patterns=['**/.git/**', '**/node_modules/**']
    )
)

# ユーザー指定の追加メタデータ（フィルターとは独立）
additional_metadata = {
    'project': 'documentation',
    'department': 'engineering',
    'classification': 'public'
}

# ローダー初期化
loader = IncrementalDirectoryLoader(
    directory_path="/path/to/documents",
    filter_config=filter_config,
    additional_metadata=additional_metadata  # これらのメタデータのみ追加される
)

# フィルター統計の確認（デバッグ用）
stats = loader.get_scan_statistics()
print("Filter statistics:", stats['filter_statistics'])
```

この修正により、フィルターは**ファイルの包含/除外判定のみ**を行い、メタデータは追加されません。メタデータはユーザーが明示的に指定したもののみが文書に追加されます。