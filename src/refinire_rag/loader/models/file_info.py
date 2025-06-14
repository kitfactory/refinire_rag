"""
FileInfo model for tracking file state
ファイル状態追跡用のFileInfoモデル
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from refinire_rag.models.document import Document

@dataclass
class FileInfo:
    """
    Information about a file for incremental loading
    インクリメンタルローディング用のファイル情報
    
    This class holds the essential information needed to track changes
    in files between scans.
    このクラスは、スキャン間でファイルの変更を追跡するために
    必要な基本情報を保持します。
    """
    path: str
    size: int  
    modified_at: datetime
    hash_md5: str
    file_type: str
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'FileInfo':
        """
        Create FileInfo from a file on disk
        ディスク上のファイルからFileInfoを作成
        
        Args:
            file_path: Path to the file
            file_path: ファイルのパス
            
        Returns:
            FileInfo object with file information
            ファイル情報を含むFileInfoオブジェクト
        """
        stat = file_path.stat()
        
        # Calculate MD5 hash of file content
        # ファイル内容のMD5ハッシュを計算
        hash_md5 = cls._calculate_file_hash(file_path)
        
        # Determine file type from extension
        # 拡張子からファイルタイプを決定
        file_type = cls._determine_file_type(file_path)
        
        return cls(
            path=str(file_path),
            size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            hash_md5=hash_md5,
            file_type=file_type
        )
    
    @classmethod
    def from_document_metadata(cls, metadata: Dict[str, Any]) -> 'FileInfo':
        """
        Create FileInfo from document metadata
        文書メタデータからFileInfoを作成
        
        Args:
            metadata: Document metadata dictionary
            metadata: 文書メタデータ辞書
            
        Returns:
            FileInfo object reconstructed from metadata
            メタデータから再構築されたFileInfoオブジェクト
        """
        return cls(
            path=metadata.get('file_path', ''),
            size=metadata.get('file_size', 0),
            modified_at=datetime.fromisoformat(metadata.get('file_modified_at', datetime.now().isoformat())),
            hash_md5=metadata.get('file_hash', ''),
            file_type=metadata.get('file_type', 'unknown')
        )
    
    @staticmethod
    def _calculate_file_hash(file_path: Path) -> str:
        """
        Calculate MD5 hash of file content
        ファイル内容のMD5ハッシュを計算
        
        Args:
            file_path: Path to the file
            file_path: ファイルのパス
            
        Returns:
            MD5 hash as hexadecimal string
            16進文字列としてのMD5ハッシュ
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                # 大きなファイルを効率的に処理するためにチャンクで読み込み
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except (OSError, IOError):
            # If file cannot be read, return empty hash
            # ファイルが読み込めない場合は空のハッシュを返す
            return ""
        
        return hash_md5.hexdigest()
    
    @staticmethod
    def _determine_file_type(file_path: Path) -> str:
        """
        Determine file type from extension
        拡張子からファイルタイプを決定
        
        Args:
            file_path: Path to the file
            file_path: ファイルのパス
            
        Returns:
            File type string
            ファイルタイプ文字列
        """
        extension = file_path.suffix.lower()
        
        # Map common extensions to file types
        # 一般的な拡張子をファイルタイプにマッピング
        extension_map = {
            '.txt': 'text',
            '.md': 'markdown',
            '.csv': 'csv',
            '.json': 'json',
            '.html': 'html',
            '.htm': 'html',
            '.pdf': 'pdf',
            '.doc': 'document',
            '.docx': 'document',
            '.py': 'code',
            '.js': 'code',
            '.ts': 'code',
            '.java': 'code',
            '.cpp': 'code',
            '.c': 'code',
        }
        
        return extension_map.get(extension, 'unknown')
    
    def __eq__(self, other) -> bool:
        """
        Check equality based on hash, size, and modification time
        ハッシュ、サイズ、更新時間に基づく等価性をチェック
        """
        if not isinstance(other, FileInfo):
            return False
        
        return (
            self.hash_md5 == other.hash_md5 and
            self.size == other.size and
            self.modified_at == other.modified_at
        )
    
    def __hash__(self) -> int:
        """
        Generate hash for use in sets and dictionaries
        セットや辞書で使用するためのハッシュを生成
        """
        return hash((self.path, self.hash_md5, self.size, self.modified_at))