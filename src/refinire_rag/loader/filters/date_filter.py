"""
Date-based file filter
日付ベースのファイルフィルター
"""

from pathlib import Path
from datetime import datetime, date
from typing import Optional
from refinire_rag.loader.filters.base_filter import BaseFilter
from refinire_rag.loader.models.file_info import FileInfo


class DateFilter(BaseFilter):
    """
    Filter files based on modification date
    更新日時に基づいてファイルをフィルタリング
    
    This filter can include files modified after a certain date,
    before a certain date, or within a date range.
    このフィルターは特定の日付以降に更新されたファイル、
    特定の日付以前に更新されたファイル、または日付範囲内の
    ファイルを含めることができます。
    """
    
    def __init__(self, 
                 modified_after: Optional[datetime] = None,
                 modified_before: Optional[datetime] = None):
        """
        Initialize the date filter
        日付フィルターを初期化
        
        Args:
            modified_after: Include files modified after this date
            modified_before: Include files modified before this date
            modified_after: この日付以降に更新されたファイルを含める
            modified_before: この日付以前に更新されたファイルを含める
            
        Note:
            At least one of modified_after or modified_before must be specified.
            Both can be specified to create a date range.
            modified_afterまたはmodified_beforeの少なくとも一方を指定する必要があります。
            両方を指定して日付範囲を作成することもできます。
        """
        if modified_after is None and modified_before is None:
            raise ValueError("At least one of modified_after or modified_before must be specified")
        
        self.modified_after = modified_after
        self.modified_before = modified_before
        
        # Validate date range
        # 日付範囲を検証
        if (self.modified_after is not None and 
            self.modified_before is not None and 
            self.modified_after >= self.modified_before):
            raise ValueError("modified_after must be before modified_before")
    
    @classmethod
    def modified_since(cls, since_date: datetime) -> 'DateFilter':
        """
        Create filter for files modified since a specific date
        特定の日付以降に更新されたファイル用のフィルターを作成
        
        Args:
            since_date: Include files modified after this date
            since_date: この日付以降に更新されたファイルを含める
            
        Returns:
            DateFilter instance
            DateFilterインスタンス
        """
        return cls(modified_after=since_date)
    
    @classmethod
    def modified_until(cls, until_date: datetime) -> 'DateFilter':
        """
        Create filter for files modified until a specific date
        特定の日付まで更新されたファイル用のフィルターを作成
        
        Args:
            until_date: Include files modified before this date
            until_date: この日付以前に更新されたファイルを含める
            
        Returns:
            DateFilter instance
            DateFilterインスタンス
        """
        return cls(modified_before=until_date)
    
    @classmethod
    def modified_between(cls, start_date: datetime, end_date: datetime) -> 'DateFilter':
        """
        Create filter for files modified within a date range
        日付範囲内で更新されたファイル用のフィルターを作成
        
        Args:
            start_date: Include files modified after this date
            end_date: Include files modified before this date
            start_date: この日付以降に更新されたファイルを含める
            end_date: この日付以前に更新されたファイルを含める
            
        Returns:
            DateFilter instance
            DateFilterインスタンス
        """
        return cls(modified_after=start_date, modified_before=end_date)
    
    def should_include(self, file_path: Path, file_info: FileInfo = None) -> bool:
        """
        Determine if file should be included based on modification date
        更新日時に基づいてファイルが含まれるべきかどうかを判定
        
        Args:
            file_path: Path to the file being evaluated
            file_info: Optional FileInfo object with modification time
            file_path: 評価対象のファイルパス
            file_info: 更新時刻を含むオプションのFileInfoオブジェクト
            
        Returns:
            True if file should be included, False otherwise
            ファイルが含まれるべき場合True、そうでなければFalse
        """
        # Get modification time from FileInfo if available, otherwise from file
        # FileInfoから更新時刻を取得、利用できない場合はファイルから取得
        if file_info is not None:
            modified_time = file_info.modified_at
        else:
            try:
                stat = file_path.stat()
                modified_time = datetime.fromtimestamp(stat.st_mtime)
            except (OSError, IOError):
                # If file cannot be accessed, exclude it
                # ファイルにアクセスできない場合は除外
                return False
        
        # Check against after date
        # 以降日付をチェック
        if self.modified_after is not None and modified_time <= self.modified_after:
            return False
        
        # Check against before date
        # 以前日付をチェック
        if self.modified_before is not None and modified_time >= self.modified_before:
            return False
        
        return True
    
    def get_filter_name(self) -> str:
        """
        Get descriptive name for this filter
        このフィルターの説明的な名前を取得
        """
        if self.modified_after is not None and self.modified_before is not None:
            after_str = self.modified_after.strftime("%Y-%m-%d")
            before_str = self.modified_before.strftime("%Y-%m-%d")
            return f"DateFilter(between {after_str} and {before_str})"
        elif self.modified_after is not None:
            after_str = self.modified_after.strftime("%Y-%m-%d")
            return f"DateFilter(after {after_str})"
        else:
            before_str = self.modified_before.strftime("%Y-%m-%d")
            return f"DateFilter(before {before_str})"
    
    def get_description(self) -> str:
        """
        Get detailed description of this filter
        このフィルターの詳細説明を取得
        """
        if self.modified_after is not None and self.modified_before is not None:
            after_str = self.modified_after.strftime("%Y-%m-%d %H:%M:%S")
            before_str = self.modified_before.strftime("%Y-%m-%d %H:%M:%S")
            return f"Include files modified between {after_str} and {before_str}"
        elif self.modified_after is not None:
            after_str = self.modified_after.strftime("%Y-%m-%d %H:%M:%S")
            return f"Include files modified after {after_str}"
        else:
            before_str = self.modified_before.strftime("%Y-%m-%d %H:%M:%S")
            return f"Include files modified before {before_str}"
    
    def __repr__(self) -> str:
        """
        Developer representation of the filter
        フィルターの開発者向け表現
        """
        return f"DateFilter(modified_after={self.modified_after}, modified_before={self.modified_before})"