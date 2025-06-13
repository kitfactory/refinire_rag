"""Document class
文書クラス

This module provides a Document class for representing text documents with metadata.
このモジュールは、メタデータを持つテキスト文書を表現するDocumentクラスを提供します。
"""

from typing import Any, Dict, Optional


class Document:
    """A class representing a text document with metadata
    メタデータを持つテキスト文書を表現するクラス
    
    This class holds the document's ID, content, and metadata.
    このクラスは文書のID、内容、メタデータを保持します。
    """

    def __init__(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a document
        文書を初期化
        
        Args:
            id: Unique identifier for the document
            ID: 文書の一意の識別子
            content: The text content of the document
            内容: 文書のテキスト内容
            metadata: Optional metadata associated with the document
            メタデータ: 文書に関連付けられたオプションのメタデータ
        """
        self.id = id
        self.content = content
        self.metadata = metadata or {} 