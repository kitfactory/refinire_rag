"""
Size-based file filter
サイズベースのファイルフィルター
"""

from pathlib import Path
from typing import Optional
from refinire_rag.loader.filters.base_filter import BaseFilter
from refinire_rag.loader.models.file_info import FileInfo


class SizeFilter(BaseFilter):
    """
    Filter files based on file size
    ファイルサイズに基づいてファイルをフィルタリング
    
    This filter can include files within a size range, files larger than
    a threshold, or files smaller than a threshold.
    このフィルターはサイズ範囲内のファイル、閾値より大きいファイル、
    または閾値より小さいファイルを含めることができます。
    """
    
    def __init__(self, 
                 min_size: Optional[int] = None,
                 max_size: Optional[int] = None):
        """
        Initialize the size filter
        サイズフィルターを初期化
        
        Args:
            min_size: Minimum file size in bytes (inclusive)
            max_size: Maximum file size in bytes (inclusive)
            min_size: 最小ファイルサイズ（バイト、包含）
            max_size: 最大ファイルサイズ（バイト、包含）
            
        Note:
            At least one of min_size or max_size must be specified.
            Both can be specified to create a size range.
            min_sizeまたはmax_sizeの少なくとも一方を指定する必要があります。
            両方を指定してサイズ範囲を作成することもできます。
        """
        if min_size is None and max_size is None:
            raise ValueError("At least one of min_size or max_size must be specified")
        
        if min_size is not None and min_size < 0:
            raise ValueError("min_size must be non-negative")
        
        if max_size is not None and max_size < 0:
            raise ValueError("max_size must be non-negative")
        
        if (min_size is not None and max_size is not None and min_size > max_size):
            raise ValueError("min_size must be less than or equal to max_size")
        
        self.min_size = min_size
        self.max_size = max_size
    
    @classmethod
    def min_size_filter(cls, min_size: int) -> 'SizeFilter':
        """
        Create filter for files at least min_size bytes
        最低min_sizeバイトのファイル用のフィルターを作成
        
        Args:
            min_size: Minimum file size in bytes
            min_size: 最小ファイルサイズ（バイト）
            
        Returns:
            SizeFilter instance
            SizeFilterインスタンス
        """
        return cls(min_size=min_size)
    
    @classmethod
    def max_size_filter(cls, max_size: int) -> 'SizeFilter':
        """
        Create filter for files at most max_size bytes
        最大max_sizeバイトのファイル用のフィルターを作成
        
        Args:
            max_size: Maximum file size in bytes
            max_size: 最大ファイルサイズ（バイト）
            
        Returns:
            SizeFilter instance
            SizeFilterインスタンス
        """
        return cls(max_size=max_size)
    
    @classmethod
    def size_range_filter(cls, min_size: int, max_size: int) -> 'SizeFilter':
        """
        Create filter for files within a size range
        サイズ範囲内のファイル用のフィルターを作成
        
        Args:
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes
            min_size: 最小ファイルサイズ（バイト）
            max_size: 最大ファイルサイズ（バイト）
            
        Returns:
            SizeFilter instance
            SizeFilterインスタンス
        """
        return cls(min_size=min_size, max_size=max_size)
    
    def should_include(self, file_path: Path, file_info: FileInfo = None) -> bool:
        """
        Determine if file should be included based on size
        サイズに基づいてファイルが含まれるべきかどうかを判定
        
        Args:
            file_path: Path to the file being evaluated
            file_info: Optional FileInfo object with file size
            file_path: 評価対象のファイルパス
            file_info: ファイルサイズを含むオプションのFileInfoオブジェクト
            
        Returns:
            True if file should be included, False otherwise
            ファイルが含まれるべき場合True、そうでなければFalse
        """
        # Get file size from FileInfo if available, otherwise from file
        # FileInfoからファイルサイズを取得、利用できない場合はファイルから取得
        if file_info is not None:
            file_size = file_info.size
        else:
            try:
                stat = file_path.stat()
                file_size = stat.st_size
            except (OSError, IOError):
                # If file cannot be accessed, exclude it
                # ファイルにアクセスできない場合は除外
                return False
        
        # Check minimum size
        # 最小サイズをチェック
        if self.min_size is not None and file_size < self.min_size:
            return False
        
        # Check maximum size
        # 最大サイズをチェック
        if self.max_size is not None and file_size > self.max_size:
            return False
        
        return True
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format
        ファイルサイズを人間が読みやすい形式でフォーマット
        
        Args:
            size_bytes: Size in bytes
            size_bytes: バイト単位のサイズ
            
        Returns:
            Formatted size string
            フォーマット済みサイズ文字列
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}PB"
    
    def get_filter_name(self) -> str:
        """
        Get descriptive name for this filter
        このフィルターの説明的な名前を取得
        """
        if self.min_size is not None and self.max_size is not None:
            min_str = self._format_size(self.min_size)
            max_str = self._format_size(self.max_size)
            return f"SizeFilter({min_str} - {max_str})"
        elif self.min_size is not None:
            min_str = self._format_size(self.min_size)
            return f"SizeFilter(>= {min_str})"
        else:
            max_str = self._format_size(self.max_size)
            return f"SizeFilter(<= {max_str})"
    
    def get_description(self) -> str:
        """
        Get detailed description of this filter
        このフィルターの詳細説明を取得
        """
        if self.min_size is not None and self.max_size is not None:
            min_str = self._format_size(self.min_size)
            max_str = self._format_size(self.max_size)
            return f"Include files between {min_str} and {max_str}"
        elif self.min_size is not None:
            min_str = self._format_size(self.min_size)
            return f"Include files at least {min_str}"
        else:
            max_str = self._format_size(self.max_size)
            return f"Include files at most {max_str}"
    
    def __repr__(self) -> str:
        """
        Developer representation of the filter
        フィルターの開発者向け表現
        """
        return f"SizeFilter(min_size={self.min_size}, max_size={self.max_size})"