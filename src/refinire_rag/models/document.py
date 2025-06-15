"""
Document model implementation
"""

import logging
from typing import List, Optional
from datetime import datetime

# from refinire_rag.models.config import DocumentConfig  # 不要なので削除
"""
Document data model for refinire-rag
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Document:
    """
    A document with content and metadata.
    / コンテンツとメタデータを持つ文書
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Set default values for required metadata fields.
        / 必須メタデータフィールドのデフォルト値を設定する
        """
        # Required metadata fields
        # / 必須メタデータフィールド
        required_fields = {
            "path": str,
            "file_type": str,
            "size_bytes": int,
            "created_at": str,  # ISO 8601 format
            "updated_at": str   # ISO 8601 format
        }

        # Set default values for missing required fields
        # / 不足している必須フィールドにデフォルト値を設定
        for field_name, field_type in required_fields.items():
            if field_name not in self.metadata:
                if field_name == "path":
                    self.metadata[field_name] = f"unknown_{self.id}"
                elif field_name == "file_type":
                    self.metadata[field_name] = "unknown"
                elif field_name == "size_bytes":
                    self.metadata[field_name] = len(self.content.encode('utf-8'))
                elif field_name in ["created_at", "updated_at"]:
                    # If not provided, use current time
                    # / 提供されていない場合は現在時刻を使用
                    self.metadata[field_name] = datetime.now().isoformat()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata value by key.
        / キーでメタデータ値を取得する
        """
        return self.metadata.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set a metadata value.
        / メタデータ値を設定する
        """
        self.metadata[key] = value

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Update multiple metadata values.
        / 複数のメタデータ値を更新する
        """
        self.metadata.update(metadata)

    @property
    def path(self) -> str:
        """
        Get the document path.
        / 文書のパスを取得する
        """
        return self.metadata["path"]

    @property
    def file_type(self) -> str:
        """
        Get the document file type.
        / 文書のファイルタイプを取得する
        """
        return self.metadata["file_type"]

    @property
    def size_bytes(self) -> int:
        """
        Get the document size in bytes.
        / 文書のサイズ（バイト）を取得する
        """
        return self.metadata["size_bytes"]

    @property
    def created_at(self) -> str:
        """
        Get the document creation timestamp.
        / 文書の作成タイムスタンプを取得する
        """
        return self.metadata["created_at"]

    @property
    def updated_at(self) -> str:
        """
        Get the document last update timestamp.
        / 文書の最終更新タイムスタンプを取得する
        """
        return self.metadata["updated_at"]