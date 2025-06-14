from typing import Dict, Any, Optional
from pathlib import Path
from .metadata import Metadata

class ConstantMetadata(Metadata):
    """
    Metadata processor that adds constant metadata to documents.
    
    定数のメタデータをドキュメントに追加するメタデータプロセッサー。
    """

    def __init__(self, metadata: Dict[str, Any]):
        """
        Initialize the ConstantMetadata processor.
        
        ConstantMetadataプロセッサーを初期化します。
        
        Args:
            metadata (Dict[str, Any]): Constant metadata to add to documents.
                                      ドキュメントに追加する定数のメタデータ。
        """
        self.metadata = metadata

    def get_metadata(self, metadata: Dict[str, Any], file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Add constant metadata.
        
        定数のメタデータを追加します。
        
        Args:
            metadata (Dict[str, Any]): Current metadata to process.
                                      処理する現在のメタデータ。
            file (Optional[Path]): File object associated with the metadata.
                                 メタデータに関連付けられたファイルオブジェクト。
        
        Returns:
            Dict[str, Any]: Processed metadata with added constant metadata.
                           定数のメタデータが追加された処理済みのメタデータ。
        """
        return {**metadata, **self.metadata} 