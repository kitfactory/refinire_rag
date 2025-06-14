from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

class Metadata(ABC):
    """
    Base class for metadata processors.
    This class defines the interface for adding metadata to a single document.
    
    メタデータプロセッサーの基底クラス。
    ドキュメント1件にメタデータを追加するためのインターフェースを定義します。
    """

    @abstractmethod
    def get_metadata(self, metadata: Dict[str, Any], file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Process metadata and add new metadata.
        
        メタデータを処理し、新しいメタデータを追加します。
        
        Args:
            metadata (Dict[str, Any]): Current metadata to process.
                                      処理する現在のメタデータ。
            file (Optional[Path]): File object associated with the metadata.
                                 メタデータに関連付けられたファイルオブジェクト。
        
        Returns:
            Dict[str, Any]: Processed metadata with added metadata.
                           メタデータが追加された処理済みのメタデータ。
        """
        pass 