from refinire_rag.document_processor import DocumentProcessor
from refinire_rag.models.document import Document
from refinire_rag.metadata.metadata import Metadata
from typing import Iterable, Iterator, Optional, Any, List, Dict
from pathlib import Path
from abc import abstractmethod
from datetime import datetime
import os

class Loader(DocumentProcessor):
    """
    Loader class for loading documents.
    ドキュメントをロードするためのLoaderクラス。

    Args:
        metadata_processors (List[Metadata]): List of metadata processors to apply to documents.
                                            ドキュメントに適用するメタデータプロセッサーのリスト。
    Returns:
        Iterator[Document]: Loaded documents
        Iterator[Document]: ロードされたドキュメント
    """

    # Standard metadata keys
    # 標準メタデータのキー
    METADATA_FILE_NAME = "file_name"  # ファイル名
    METADATA_FILE_SIZE = "file_size"  # ファイルサイズ（バイト）
    METADATA_MODIFIED_AT = "modified_at"  # 最終更新日時
    METADATA_IMPORTED_AT = "imported_at"  # インポート日時
    METADATA_FILE_EXTENSION = "file_extension"  # ファイル拡張子
    METADATA_ABSOLUTE_PATH = "absolute_path"  # 絶対パス

    def __init__(self, metadata_processors: Optional[List[Metadata]] = None):
        """
        Initialize the Loader with metadata processors.
        
        メタデータプロセッサーを使用してLoaderを初期化します。
        
        Args:
            metadata_processors (Optional[List[Metadata]]): List of metadata processors to apply to documents.
                                                          ドキュメントに適用するメタデータプロセッサーのリスト。
                                                          If None, standard metadata will be applied.
                                                          Noneの場合、標準メタデータが適用されます。
        """
        # Initialize DocumentProcessor with a config containing metadata_processors
        super().__init__(config={'metadata_processors': metadata_processors})
        self.metadata_processors = metadata_processors or []

    def _get_metadata(self, file: Path) -> dict:
        """
        Get metadata from metadata processors or apply standard metadata.
        
        メタデータプロセッサーからメタデータを取得するか、標準メタデータを適用します。
        
        Rules:
        1. If metadata_processors is None (not specified), apply standard metadata
           メタデータプロセッサーがNone（未指定）の場合、標準メタデータを適用
        2. If metadata_processors is specified (not None), use only those processors
           メタデータプロセッサーが指定されている（Noneでない）場合、それらのプロセッサーのみを使用
        3. If metadata_processors is empty list ([]), return empty metadata
           メタデータプロセッサーが空リスト（[]）の場合、空のメタデータを返す
        
        Args:
            file (Path): The file to get metadata for.
                        メタデータを取得するファイル。
        
        Returns:
            dict: Metadata from processors or standard metadata.
                 プロセッサーからのメタデータまたは標準メタデータ。
        """
        # Case 1: metadata_processors is None - apply standard metadata
        # ケース1: メタデータプロセッサーがNone - 標準メタデータを適用
        if self.metadata_processors is None:
            return self._get_standard_metadata(file)
        
        # Case 2 & 3: metadata_processors is specified (empty or not)
        # ケース2 & 3: メタデータプロセッサーが指定されている（空かどうか）
        metadata = {}
        for processor in self.metadata_processors:
            metadata = processor.get_metadata(metadata, file)
        
        return metadata

    def _get_standard_metadata(self, file: Path) -> Dict[str, Any]:
        """
        Get standard metadata for a file.
        
        ファイルの標準メタデータを取得します。
        
        Args:
            file (Path): The file to get metadata for.
                        メタデータを取得するファイル。
        
        Returns:
            Dict[str, Any]: Standard metadata for the file.
                           ファイルの標準メタデータ。
        """
        try:
            stat = file.stat()
            return {
                self.METADATA_FILE_NAME: file.name,
                self.METADATA_FILE_SIZE: stat.st_size,
                self.METADATA_MODIFIED_AT: datetime.fromtimestamp(stat.st_mtime).isoformat(),
                self.METADATA_IMPORTED_AT: datetime.now().isoformat(),
                self.METADATA_FILE_EXTENSION: file.suffix[1:] if file.suffix else "",
                self.METADATA_ABSOLUTE_PATH: str(file.absolute())
            }
        except OSError as e:
            # Return basic metadata if file stats cannot be accessed
            # ファイルの統計情報にアクセスできない場合は基本的なメタデータを返す
            return {
                self.METADATA_FILE_NAME: file.name,
                self.METADATA_FILE_EXTENSION: file.suffix[1:] if file.suffix else "",
                self.METADATA_ABSOLUTE_PATH: str(file.absolute()),
                self.METADATA_IMPORTED_AT: datetime.now().isoformat()
            }

    @abstractmethod
    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """
        Process documents and apply metadata.
        This method must be implemented by subclasses.
        
        ドキュメントを処理し、メタデータを適用します。
        このメソッドはサブクラスで実装する必要があります。
        
        Args:
            documents: Iterable of Document objects
            documents: ドキュメントのイテラブル
            config: Optional configuration
            config: オプション設定
        
        Returns:
            Iterator of processed Document objects with metadata
            メタデータが適用された処理済みドキュメントのイテレータ
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        return {
            'metadata_processors': [type(proc).__name__ for proc in self.metadata_processors] if self.metadata_processors else []
        } 