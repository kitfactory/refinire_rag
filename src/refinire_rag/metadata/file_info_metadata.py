from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import os
from datetime import datetime
from enum import Enum, auto
from .metadata import Metadata
from ..models.document import Document

class FileAttribute(Enum):
    """
    Available file attributes for metadata.
    
    メタデータとして利用可能なファイル属性。
    """
    FILE_NAME = auto()  # ファイル名
    FILE_EXTENSION = auto()  # ファイル拡張子
    FILE_SIZE = auto()  # ファイルサイズ
    CREATED_AT = auto()  # 作成日時
    MODIFIED_AT = auto()  # 最終更新日時
    IMPORTED_AT = auto()  # 文書取り込み日時
    PARENT_FOLDER = auto()  # 親フォルダ名
    RELATIVE_PATH = auto()  # 相対パス
    ABSOLUTE_PATH = auto()  # 絶対パス
    IS_HIDDEN = auto()  # 隠しファイルかどうか
    IS_SYMLINK = auto()  # シンボリックリンクかどうか

class FileInfoMetadata(Metadata):
    """
    Metadata processor that adds file information as metadata.
    Attributes to be added can be specified in the constructor.
    
    ファイル情報をメタデータとして付与するプロセッサー。
    付与する属性はコンストラクタで指定可能。
    """
    
    # Available file attributes
    AVAILABLE_ATTRIBUTES = {
        FileAttribute.FILE_NAME: lambda p: p.name,
        FileAttribute.FILE_EXTENSION: lambda p: p.suffix[1:] if p.suffix else "",
        FileAttribute.FILE_SIZE: lambda p: p.stat().st_size,
        FileAttribute.CREATED_AT: lambda p: p.stat().st_ctime,
        FileAttribute.MODIFIED_AT: lambda p: p.stat().st_mtime,
        FileAttribute.IMPORTED_AT: lambda p: datetime.now().timestamp(),
        FileAttribute.PARENT_FOLDER: lambda p: p.parent.name,
        FileAttribute.RELATIVE_PATH: lambda p: str(p),
        FileAttribute.ABSOLUTE_PATH: lambda p: str(p.absolute()),
        FileAttribute.IS_HIDDEN: lambda p: p.name.startswith("."),
        FileAttribute.IS_SYMLINK: lambda p: p.is_symlink()
    }
    
    # 属性名とEnumのマッピング
    ATTRIBUTE_NAMES = {
        FileAttribute.FILE_NAME: "file_name",
        FileAttribute.FILE_EXTENSION: "file_extension",
        FileAttribute.FILE_SIZE: "file_size",
        FileAttribute.CREATED_AT: "created_at",
        FileAttribute.MODIFIED_AT: "modified_at",
        FileAttribute.IMPORTED_AT: "imported_at",
        FileAttribute.PARENT_FOLDER: "parent_folder",
        FileAttribute.RELATIVE_PATH: "relative_path",
        FileAttribute.ABSOLUTE_PATH: "absolute_path",
        FileAttribute.IS_HIDDEN: "is_hidden",
        FileAttribute.IS_SYMLINK: "is_symlink"
    }
    
    def __init__(self, attributes: Optional[Set[FileAttribute]] = None):
        """
        Initialize the file info metadata processor.
        
        ファイル情報メタデータプロセッサーを初期化します。
        
        Args:
            attributes (Optional[Set[FileAttribute]]): Set of file attributes to add as metadata.
                                                      メタデータとして追加するファイル属性のセット。
                                                      If None, all available attributes will be added.
                                                      Noneの場合、利用可能なすべての属性が追加されます。
        
        Examples:
            >>> # すべての属性を使用
            >>> metadata = FileInfoMetadata()
            
            >>> # 特定の属性のみを使用
            >>> metadata = FileInfoMetadata({
            ...     FileAttribute.FILE_NAME,
            ...     FileAttribute.FILE_SIZE,
            ...     FileAttribute.MODIFIED_AT
            ... })
        """
        self.attributes = attributes or set(FileAttribute)
    
    def get_metadata(self, metadata: Dict[str, Any], file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Add file information as metadata.
        
        ファイル情報をメタデータとして追加します。
        
        Args:
            metadata (Dict[str, Any]): Current metadata to process.
                                      処理する現在のメタデータ。
            file (Optional[Path]): File object associated with the metadata.
                                 メタデータに関連付けられたファイルオブジェクト。
        
        Returns:
            Dict[str, Any]: Processed metadata with added file information.
                           ファイル情報が追加された処理済みのメタデータ。
        """
        if not file and not metadata.get("path"):
            return metadata

        try:
            file_path = file or Path(metadata["path"])
            if not file_path.exists():
                return metadata

            # Get file attributes
            file_metadata = {}
            for attr in self.attributes:
                try:
                    value = self.AVAILABLE_ATTRIBUTES[attr](file_path)
                    file_metadata[self.ATTRIBUTE_NAMES[attr]] = value
                except (OSError, PermissionError):
                    continue

            # Merge with existing metadata
            return {**metadata, **file_metadata}
        except Exception:
            return metadata 