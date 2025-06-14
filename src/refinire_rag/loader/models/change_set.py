"""
ChangeSet model for representing detected file changes
検出されたファイル変更を表現するChangeSetモデル
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class ChangeSet:
    """
    Represents the set of changes detected between directory scans
    ディレクトリスキャン間で検出された変更のセットを表現
    
    This class categorizes files into different change types to enable
    efficient incremental processing.
    このクラスはファイルを異なる変更タイプに分類し、
    効率的なインクリメンタル処理を可能にします。
    """
    added: List[str] = field(default_factory=list)      # New files / 新規追加されたファイル
    modified: List[str] = field(default_factory=list)   # Updated files / 更新されたファイル  
    deleted: List[str] = field(default_factory=list)    # Removed files / 削除されたファイル
    unchanged: List[str] = field(default_factory=list)  # Unchanged files / 変更されていないファイル
    
    def has_changes(self) -> bool:
        """
        Check if there are any changes to process
        処理すべき変更があるかどうかをチェック
        
        Returns:
            True if there are files to add, modify, or delete
            追加、更新、削除するファイルがある場合True
        """
        return bool(self.added or self.modified or self.deleted)
    
    @property
    def total_changes(self) -> int:
        """
        Get the total number of files that have changes
        変更があるファイルの総数を取得
        
        Returns:
            Total count of changed files
            変更されたファイルの総数
        """
        return len(self.added) + len(self.modified) + len(self.deleted)
    
    @property 
    def total_files(self) -> int:
        """
        Get the total number of files processed
        処理されたファイルの総数を取得
        
        Returns:
            Total count of all files
            全ファイルの総数
        """
        return len(self.added) + len(self.modified) + len(self.deleted) + len(self.unchanged)
    
    def get_summary(self) -> dict:
        """
        Get a summary of the changes
        変更のサマリーを取得
        
        Returns:
            Dictionary with change statistics
            変更統計を含む辞書
        """
        return {
            'added': len(self.added),
            'modified': len(self.modified),
            'deleted': len(self.deleted),
            'unchanged': len(self.unchanged),
            'total_changes': self.total_changes,
            'total_files': self.total_files,
            'has_changes': self.has_changes()
        }
    
    def __str__(self) -> str:
        """
        String representation of the change set
        変更セットの文字列表現
        """
        summary = self.get_summary()
        return (
            f"ChangeSet(added={summary['added']}, "
            f"modified={summary['modified']}, "
            f"deleted={summary['deleted']}, "
            f"unchanged={summary['unchanged']})"
        )