"""
Incremental Directory Loader for detecting and loading changed files
変更されたファイルを検出してロードするインクリメンタルディレクトリローダー
"""

from pathlib import Path
from typing import Iterable, Iterator, Optional, Any, List, Dict, Union
from refinire_rag.loader.loader import Loader
from refinire_rag.loader.file_tracker import FileTracker
from refinire_rag.loader.models.filter_config import FilterConfig
from refinire_rag.loader.models.sync_result import SyncResult
from refinire_rag.loader.models.change_set import ChangeSet
from refinire_rag.models.document import Document
from refinire_rag.metadata.metadata import Metadata
from refinire_rag.utils import generate_document_id
from refinire_rag.storage.document_store import DocumentStore


class IncrementalDirectoryLoader(Loader):
    """
    Loader that detects and processes only changed files in a directory
    ディレクトリ内の変更されたファイルのみを検出して処理するローダー
    
    This loader extends the base Loader class to provide incremental loading
    functionality. It tracks file changes between scans and only processes
    files that have been added, modified, or deleted.
    このローダーは基底Loaderクラスを拡張してインクリメンタルローディング
    機能を提供します。スキャン間のファイル変更を追跡し、追加、更新、
    削除されたファイルのみを処理します。
    """
    
    def __init__(self, 
                 directory_path: Union[str, Path],
                 document_store: DocumentStore,
                 filter_config: Optional[FilterConfig] = None,
                 tracking_file_path: Optional[Path] = None,
                 recursive: bool = True,
                 metadata_processors: Optional[List[Metadata]] = None,
                 additional_metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the incremental directory loader
        インクリメンタルディレクトリローダーを初期化
        
        Args:
            directory_path: Path to the directory to monitor
            document_store: DocumentStore instance for document storage operations
            filter_config: Optional filter configuration for file inclusion/exclusion
            tracking_file_path: Optional path to persist file tracking data
            recursive: Whether to scan subdirectories recursively
            metadata_processors: Optional metadata processors
            additional_metadata: Additional metadata to add to all documents
            directory_path: 監視するディレクトリのパス
            document_store: 文書ストレージ操作用のDocumentStoreインスタンス
            filter_config: ファイル包含/除外用のオプションフィルター設定
            tracking_file_path: ファイル追跡データを永続化するオプションのパス
            recursive: サブディレクトリを再帰的にスキャンするかどうか
            metadata_processors: オプションのメタデータプロセッサー
            additional_metadata: すべての文書に追加する追加メタデータ
        """
        super().__init__(metadata_processors)
        
        self.directory_path = Path(directory_path)
        self.document_store = document_store
        self.filter_config = filter_config
        self.recursive = recursive
        self.additional_metadata = additional_metadata or {}
        
        # Initialize file tracker
        # ファイルトラッカーを初期化
        self.file_tracker = FileTracker(tracking_file_path)
        
        # Validate directory
        # ディレクトリを検証
        if not self.directory_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.directory_path}")
        if not self.directory_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.directory_path}")
    
    def sync_with_store(self) -> SyncResult:
        """
        Synchronize directory changes with the document store
        ディレクトリの変更をドキュメントストアと同期
        
        This method detects changes in the directory and applies them to the
        document store, including adding new documents, updating existing ones,
        and removing deleted files.
        このメソッドはディレクトリ内の変更を検出し、新しい文書の追加、
        既存文書の更新、削除されたファイルの削除を含めてドキュメントストア
        に適用します。
        
        Returns:
            SyncResult with information about the sync operation
            同期操作に関する情報を含むSyncResult
        """
        sync_result = SyncResult()
        
        try:
            # Detect changes in the directory
            # ディレクトリ内の変更を検出
            change_set = self.file_tracker.scan_directory(
                self.directory_path, 
                self.filter_config, 
                self.recursive
            )
            
            # Process added files
            # 追加されたファイルを処理
            sync_result = self._process_added_files(change_set.added, sync_result)
            
            # Process modified files
            # 更新されたファイルを処理
            sync_result = self._process_modified_files(change_set.modified, sync_result)
            
            # Process deleted files
            # 削除されたファイルを処理
            sync_result = self._process_deleted_files(change_set.deleted, sync_result)
            
        except Exception as e:
            sync_result.add_error(f"Sync operation failed: {str(e)}")
        
        return sync_result
    
    def _process_added_files(self, file_paths: List[str], sync_result: SyncResult) -> SyncResult:
        """
        Process newly added files
        新しく追加されたファイルを処理
        
        Args:
            file_paths: List of file paths that were added
            sync_result: SyncResult to update
            file_paths: 追加されたファイルパスのリスト
            sync_result: 更新するSyncResult
            
        Returns:
            Updated SyncResult
            更新されたSyncResult
        """
        for file_path in file_paths:
            try:
                # Load and process the file
                # ファイルをロードして処理
                documents = list(self._load_file(Path(file_path)))
                
                # Add documents to document store
                # 文書をドキュメントストアに追加
                for doc in documents:
                    self.document_store.store_document(doc)
                    sync_result.added_documents.append(doc)
                    
            except Exception as e:
                sync_result.add_error(f"Failed to process added file {file_path}: {str(e)}")
        
        return sync_result
    
    def _process_modified_files(self, file_paths: List[str], sync_result: SyncResult) -> SyncResult:
        """
        Process modified files
        更新されたファイルを処理
        
        Args:
            file_paths: List of file paths that were modified
            sync_result: SyncResult to update
            file_paths: 更新されたファイルパスのリスト
            sync_result: 更新するSyncResult
            
        Returns:
            Updated SyncResult
            更新されたSyncResult
        """
        for file_path in file_paths:
            try:
                # Remove existing documents for this file
                # このファイルの既存文書を削除
                search_results = self.document_store.search_by_metadata(
                    {self.METADATA_ABSOLUTE_PATH: file_path}
                )
                
                for search_result in search_results:
                    self.document_store.delete_document(search_result.document.id)
                
                # Load and add updated documents
                # 更新された文書をロードして追加
                documents = list(self._load_file(Path(file_path)))
                
                for doc in documents:
                    self.document_store.store_document(doc)
                    sync_result.updated_documents.append(doc)
                    
            except Exception as e:
                sync_result.add_error(f"Failed to process modified file {file_path}: {str(e)}")
        
        return sync_result
    
    def _process_deleted_files(self, file_paths: List[str], sync_result: SyncResult) -> SyncResult:
        """
        Process deleted files
        削除されたファイルを処理
        
        Args:
            file_paths: List of file paths that were deleted
            sync_result: SyncResult to update
            file_paths: 削除されたファイルパスのリスト
            sync_result: 更新するSyncResult
            
        Returns:
            Updated SyncResult
            更新されたSyncResult
        """
        for file_path in file_paths:
            try:
                # Find and remove documents for this file
                # このファイルの文書を見つけて削除
                search_results = self.document_store.search_by_metadata(
                    {self.METADATA_ABSOLUTE_PATH: file_path}
                )
                
                for search_result in search_results:
                    self.document_store.delete_document(search_result.document.id)
                    sync_result.deleted_document_ids.append(search_result.document.id)
                    
            except Exception as e:
                sync_result.add_error(f"Failed to process deleted file {file_path}: {str(e)}")
        
        return sync_result
    
    def _load_file(self, file_path: Path) -> Iterator[Document]:
        """
        Load a single file and create Document objects
        単一ファイルをロードしてDocumentオブジェクトを作成
        
        Args:
            file_path: Path to the file to load
            file_path: ロードするファイルのパス
            
        Yields:
            Document objects created from the file
            ファイルから作成されたDocumentオブジェクト
        """
        try:
            # Read file content
            # ファイル内容を読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get metadata for the file
            # ファイルのメタデータを取得
            metadata = self._get_metadata(file_path)
            
            # Add additional metadata if specified
            # 指定されている場合は追加メタデータを追加
            metadata.update(self.additional_metadata)
            
            # Create and yield document
            # 文書を作成してyield
            document = Document(
                id=generate_document_id(),
                content=content,
                metadata=metadata
            )
            
            yield document
            
        except Exception as e:
            raise Exception(f"Failed to load file {file_path}: {str(e)}")
    
    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """
        Process documents (required by DocumentProcessor interface)
        文書を処理（DocumentProcessorインターフェースで必要）
        
        This method processes the provided documents and applies metadata.
        For incremental loading, use sync_with_corpus() instead.
        このメソッドは提供された文書を処理してメタデータを適用します。
        インクリメンタルローディングには、代わりにsync_with_corpus()を使用してください。
        
        Args:
            documents: Documents to process
            config: Optional configuration
            documents: 処理する文書
            config: オプション設定
            
        Yields:
            Processed documents
            処理された文書
        """
        for document in documents:
            # Apply metadata processors if configured
            # 設定されている場合はメタデータプロセッサーを適用
            if self.metadata_processors:
                metadata = document.metadata.copy()
                
                # Note: This assumes the document has a file path in metadata
                # 注意: これは文書のメタデータにファイルパスがあることを想定
                file_path = Path(metadata.get(self.METADATA_ABSOLUTE_PATH, ""))
                
                if file_path.exists():
                    updated_metadata = self._get_metadata(file_path)
                    updated_metadata.update(self.additional_metadata)
                    
                    # Create new document with updated metadata
                    # 更新されたメタデータで新しい文書を作成
                    yield Document(
                        id=document.id,
                        content=document.content,
                        metadata=updated_metadata
                    )
                else:
                    yield document
            else:
                yield document
    
    def get_change_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the last change detection
        最後の変更検出のサマリーを取得
        
        Returns:
            Dictionary with change summary information
            変更サマリー情報を含む辞書
        """
        change_set = self.file_tracker.scan_directory(
            self.directory_path, 
            self.filter_config, 
            self.recursive
        )
        
        return {
            'directory': str(self.directory_path),
            'recursive': self.recursive,
            'tracked_files': self.file_tracker.get_file_count(),
            'changes': change_set.get_summary(),
            'filters_configured': self.filter_config.has_filters() if self.filter_config else False
        }
    
    def reset_tracking(self):
        """
        Reset file tracking data
        ファイル追跡データをリセット
        
        This will clear all tracking information and treat the next sync
        as if it's the first scan of the directory.
        これは全ての追跡情報をクリアし、次の同期を
        ディレクトリの最初のスキャンとして扱います。
        """
        self.file_tracker.clear_tracking_data()
    
    def __str__(self) -> str:
        """
        String representation of the loader
        ローダーの文字列表現
        """
        return f"IncrementalDirectoryLoader(dir={self.directory_path}, tracked={self.file_tracker.get_file_count()})"
    
    def __repr__(self) -> str:
        """
        Developer representation of the loader
        ローダーの開発者向け表現
        """
        return (f"IncrementalDirectoryLoader(directory_path={self.directory_path}, "
                f"recursive={self.recursive}, "
                f"filter_config={self.filter_config is not None})")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        base_config = super().get_config()
        
        config_dict = {
            **base_config,
            'directory_path': str(self.directory_path),
            'recursive': self.recursive,
            'has_filter_config': self.filter_config is not None,
            'document_store_type': type(self.document_store).__name__ if self.document_store else None,
            'file_tracker_type': type(self.file_tracker).__name__
        }
        
        # Add filter config details if available
        if self.filter_config:
            config_dict['filter_config'] = {
                'has_filters': self.filter_config.has_filters(),
                'extensions': getattr(self.filter_config, 'allowed_extensions', None),
                'patterns': getattr(self.filter_config, 'include_patterns', None)
            }
        
        return config_dict