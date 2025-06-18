# IncrementalDirectoryLoader 修正設計書

## アーキテクチャの修正

`IncrementalDirectoryLoader`を`Loader`クラスのサブクラスとして設計し直します。

## 1. 基底Loaderクラスの拡張

まず、`Loader`基底クラスにインクリメンタル機能用のインターフェースを追加：

```python
# src/refinire_rag/loader/loader.py

from abc import ABC, abstractmethod
from typing import Iterator, Iterable, Optional, Any
from refinire_rag.models.document import Document
from refinire_rag.document_processor import DocumentProcessor

class Loader(DocumentProcessor):
    """
    Base class for all document loaders
    すべてのドキュメントローダーの基底クラス
    """
    
    # 既存のメソッド...
    
    def supports_incremental_loading(self) -> bool:
        """
        Check if this loader supports incremental loading
        このローダーがインクリメンタルローディングをサポートするかチェック
        
        Returns:
            True if incremental loading is supported
            インクリメンタルローディングがサポートされている場合True
        """
        return False
    
    def detect_changes(self, corpus_store) -> Optional['ChangeSet']:
        """
        Detect changes for incremental loading
        インクリメンタルローディングのための変更を検出
        
        Args:
            corpus_store: Storage to compare against
            corpus_store: 比較対象のストレージ
            
        Returns:
            ChangeSet if incremental loading is supported, None otherwise
            インクリメンタルローディングがサポートされている場合はChangeSet、そうでなければNone
        """
        if not self.supports_incremental_loading():
            return None
        return self._detect_changes_impl(corpus_store)
    
    def _detect_changes_impl(self, corpus_store) -> 'ChangeSet':
        """
        Implementation of change detection
        変更検出の実装
        """
        raise NotImplementedError("Incremental loaders must implement _detect_changes_impl")
```

## 2. IncrementalDirectoryLoader の設計

```python
# src/refinire_rag/loader/incremental_directory_loader.py

from typing import Iterator, Iterable, Optional, Any, Dict, List
from pathlib import Path
from datetime import datetime

from refinire_rag.loader.loader import Loader
from refinire_rag.loader.directory_loader import DirectoryLoader
from refinire_rag.loader.file_tracker import FileTracker
from refinire_rag.loader.models.filter_config import FilterConfig
from refinire_rag.loader.models.change_set import ChangeSet
from refinire_rag.loader.models.sync_result import SyncResult
from refinire_rag.models.document import Document

class IncrementalDirectoryLoader(Loader):
    """
    Directory loader with incremental loading capabilities
    インクリメンタルローディング機能付きディレクトリローダー
    
    This loader extends the base Loader functionality to support:
    このローダーは基底Loaderの機能を拡張して以下をサポート：
    - Change detection (added, modified, deleted files)
    - 変更検出（追加、更新、削除されたファイル）
    - Filtering based on file attributes
    - ファイル属性に基づくフィルタリング
    - Enhanced metadata attachment
    - 拡張メタデータの付与
    """
    
    def __init__(self, 
                 directory_path: str,
                 recursive: bool = True,
                 filter_config: FilterConfig = None,
                 additional_metadata: Dict[str, Any] = None):
        """
        Initialize incremental directory loader
        インクリメンタルディレクトリローダーを初期化
        
        Args:
            directory_path: Path to monitor
            directory_path: 監視するパス
            recursive: Whether to scan recursively
            recursive: 再帰的にスキャンするかどうか
            filter_config: Configuration for file filtering
            filter_config: ファイルフィルタリングの設定
            additional_metadata: Additional metadata to add to all documents
            additional_metadata: 全文書に追加するメタデータ
        """
        super().__init__()
        self.directory_path = Path(directory_path)
        self.recursive = recursive
        self.filter_config = filter_config or FilterConfig()
        self.additional_metadata = additional_metadata or {}
        
        # Internal components
        self.file_tracker = FileTracker(filter_config)
        self.directory_loader = DirectoryLoader(str(directory_path), recursive=recursive)
        
        # State tracking
        self._last_scan_time = None
        self._current_files = {}
    
    def supports_incremental_loading(self) -> bool:
        """
        This loader supports incremental loading
        このローダーはインクリメンタルローディングをサポート
        """
        return True
    
    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """
        Process documents (standard Loader interface)
        文書を処理（標準Loaderインターフェース）
        
        For incremental loader, this method loads all files from directory
        インクリメンタルローダーでは、このメソッドはディレクトリからすべてのファイルをロード
        
        Args:
            documents: Input documents (ignored for directory loading)
            documents: 入力文書（ディレクトリローディングでは無視）
            config: Optional configuration
            config: オプション設定
            
        Yields:
            Document objects with enhanced metadata
            拡張メタデータ付きのDocumentオブジェクト
        """
        # Scan directory and load all filtered files
        current_files = self.file_tracker.scan_directory(self.directory_path, self.recursive)
        self._current_files = current_files
        self._last_scan_time = datetime.now()
        
        for file_path, file_info in current_files.items():
            try:
                docs = self._load_file_with_metadata(file_path, file_info)
                for doc in docs:
                    yield doc
            except Exception as e:
                # Log error but continue processing other files
                # エラーをログに記録するが、他のファイルの処理は継続
                print(f"Error loading {file_path}: {e}")
    
    def _detect_changes_impl(self, corpus_store) -> ChangeSet:
        """
        Implementation of change detection for incremental loading
        インクリメンタルローディングのための変更検出の実装
        
        Args:
            corpus_store: Storage to compare against
            corpus_store: 比較対象のストレージ
            
        Returns:
            ChangeSet containing detected changes
            検出された変更を含むChangeSet
        """
        # Get current state of directory
        current_files = self.file_tracker.scan_directory(self.directory_path, self.recursive)
        
        # Get stored state from corpus
        stored_files = self.file_tracker.get_corpus_files(corpus_store)
        
        # Compare and detect changes
        changes = self.file_tracker.compare(current_files, stored_files)
        
        # Update internal state
        self._current_files = current_files
        self._last_scan_time = datetime.now()
        
        return changes
    
    def sync_with_corpus(self, corpus_store) -> SyncResult:
        """
        Synchronize directory changes with corpus store
        ディレクトリの変更をコーパスストアと同期
        
        Args:
            corpus_store: Target corpus store
            corpus_store: 対象のコーパスストア
            
        Returns:
            SyncResult containing sync statistics
            同期統計を含むSyncResult
        """
        changes = self.detect_changes(corpus_store)
        if not changes:
            return SyncResult()  # No changes to sync
        
        result = SyncResult()
        
        # Process added files
        for file_path in changes.added:
            try:
                file_info = self._current_files.get(file_path)
                docs = self._load_file_with_metadata(file_path, file_info)
                corpus_store.add_documents(docs)
                result.added_documents.extend(docs)
            except Exception as e:
                result.errors.append(f"Error adding {file_path}: {e}")
        
        # Process modified files
        for file_path in changes.modified:
            try:
                # Remove old documents
                old_docs = corpus_store.get_documents_by_path(file_path)
                if old_docs:
                    corpus_store.delete_documents([doc.id for doc in old_docs])
                
                # Add updated documents
                file_info = self._current_files.get(file_path)
                docs = self._load_file_with_metadata(file_path, file_info)
                corpus_store.add_documents(docs)
                result.updated_documents.extend(docs)
            except Exception as e:
                result.errors.append(f"Error updating {file_path}: {e}")
        
        # Process deleted files
        for file_path in changes.deleted:
            try:
                docs = corpus_store.get_documents_by_path(file_path)
                if docs:
                    doc_ids = [doc.id for doc in docs]
                    corpus_store.delete_documents(doc_ids)
                    result.deleted_document_ids.extend(doc_ids)
            except Exception as e:
                result.errors.append(f"Error deleting {file_path}: {e}")
        
        return result
    
    def _load_file_with_metadata(self, file_path: str, file_info: 'FileInfo' = None) -> List[Document]:
        """
        Load file and enhance with metadata from filters and loader configuration
        ファイルをロードし、フィルターとローダー設定からのメタデータを付与
        
        Args:
            file_path: Path to the file to load
            file_path: ロードするファイルのパス
            file_info: File information with filter metadata
            file_info: フィルターメタデータ付きのファイル情報
            
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
        
        # Enhance each document with additional metadata
        for doc in docs:
            self._enhance_document_metadata(doc, file_info)
        
        return docs
    
    def _enhance_document_metadata(self, doc: Document, file_info: 'FileInfo' = None):
        """
        Enhance document with comprehensive metadata
        文書に包括的なメタデータを付与
        
        Args:
            doc: Document to enhance
            doc: 拡張する文書
            file_info: File information with filter metadata
            file_info: フィルターメタデータ付きのファイル情報
        """
        # Add file tracking metadata
        if file_info:
            doc.metadata.update({
                'file_size': file_info.size,
                'file_modified_at': file_info.modified_at.isoformat(),
                'file_hash': file_info.hash_md5,
            })
            
            # Add filter-generated metadata
            if hasattr(file_info, 'additional_metadata'):
                doc.metadata.update(file_info.additional_metadata)
        
        # Add loader metadata
        doc.metadata.update({
            'loaded_at': datetime.now().isoformat(),
            'loader_type': 'incremental_directory',
            'loader_version': '1.0.0',
            'directory_path': str(self.directory_path),
            'recursive_scan': self.recursive,
        })
        
        # Add user-specified additional metadata
        doc.metadata.update(self.additional_metadata)
    
    def get_scan_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the last directory scan
        最後のディレクトリスキャンの統計を取得
        
        Returns:
            Dictionary with scan statistics
            スキャン統計の辞書
        """
        return {
            'last_scan_time': self._last_scan_time.isoformat() if self._last_scan_time else None,
            'total_files_found': len(self._current_files),
            'directory_path': str(self.directory_path),
            'recursive': self.recursive,
            'filters_applied': len(self.filter_config.get_all_filters()),
        }
```

## 3. 使用例

### 標準的なLoader使用（一括ロード）
```python
from refinire_rag.loader.incremental_directory_loader import IncrementalDirectoryLoader
from refinire_rag.loader.models.filter_config import FilterConfig
from refinire_rag.loader.filters.extension_filter import ExtensionFilter

# フィルター設定
filter_config = FilterConfig(
    extension_filter=ExtensionFilter(allowed_extensions=['.txt', '.md'])
)

# ローダー作成
loader = IncrementalDirectoryLoader(
    directory_path="/path/to/documents",
    filter_config=filter_config,
    additional_metadata={'project': 'docs'}
)

# 標準的なLoader使用方法
mock_input = [Document(id="dummy", content="", metadata={})]
all_documents = list(loader.process(mock_input))
print(f"Loaded {len(all_documents)} documents")
```

### インクリメンタルローディング使用
```python
# 変更検出
if loader.supports_incremental_loading():
    changes = loader.detect_changes(corpus_store)
    
    if changes and changes.has_changes():
        print(f"Changes detected:")
        print(f"  Added: {len(changes.added)}")
        print(f"  Modified: {len(changes.modified)}")
        print(f"  Deleted: {len(changes.deleted)}")
        
        # 同期実行
        sync_result = loader.sync_with_corpus(corpus_store)
        print(f"Sync completed: {sync_result.total_processed} documents processed")
```

### DirectoryLoaderとの使い分け
```python
# 基本的なディレクトリローディング
basic_loader = DirectoryLoader("/path/to/docs")

# 高機能なインクリメンタルローディング
incremental_loader = IncrementalDirectoryLoader(
    directory_path="/path/to/docs",
    filter_config=filter_config,
    additional_metadata={'source': 'monitoring'}
)

# 両方ともLoaderサブクラスなので同じインターフェース
for loader in [basic_loader, incremental_loader]:
    docs = list(loader.process([mock_doc]))
    print(f"{loader.__class__.__name__}: {len(docs)} documents")
```

## 4. クラス階層

```
DocumentProcessor
└── Loader
    ├── TextLoader
    ├── CSVLoader
    ├── JSONLoader
    ├── HTMLLoader
    ├── DirectoryLoader
    └── IncrementalDirectoryLoader  # 新規追加
```

この設計により、`IncrementalDirectoryLoader`は標準的な`Loader`インターフェースを維持しながら、インクリメンタルローディング機能を提供できます。