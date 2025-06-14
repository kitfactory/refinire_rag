"""
Base filter class for file filtering
ファイルフィルタリング用の基底フィルタークラス
"""

from abc import ABC, abstractmethod
from pathlib import Path
from refinire_rag.loader.models.file_info import FileInfo


class BaseFilter(ABC):
    """
    Abstract base class for file filters
    ファイルフィルター用の抽象基底クラス
    
    Filters are used only for inclusion/exclusion decisions and do not add
    metadata to documents. They should implement the should_include method
    to determine whether a file should be processed.
    フィルターは包含/除外の判定にのみ使用され、文書にメタデータを
    追加しません。ファイルが処理されるべきかどうかを判定するために
    should_includeメソッドを実装する必要があります。
    """
    
    @abstractmethod
    def should_include(self, file_path: Path, file_info: FileInfo = None) -> bool:
        """
        Determine whether a file should be included in processing
        ファイルが処理に含まれるべきかどうかを判定
        
        Args:
            file_path: Path to the file being evaluated
            file_info: Optional FileInfo object with file details
            file_path: 評価対象のファイルパス
            file_info: ファイル詳細を含むオプションのFileInfoオブジェクト
            
        Returns:
            True if the file should be included, False otherwise
            ファイルが含まれるべき場合True、そうでなければFalse
        """
        pass
    
    @abstractmethod
    def get_filter_name(self) -> str:
        """
        Get a descriptive name for this filter
        このフィルターの説明的な名前を取得
        
        Returns:
            Human-readable name for the filter
            フィルターの人間が読みやすい名前
        """
        pass
    
    def get_description(self) -> str:
        """
        Get a description of what this filter does
        このフィルターが何をするかの説明を取得
        
        Returns:
            Description of the filter's behavior
            フィルターの動作の説明
        """
        return f"Filter: {self.get_filter_name()}"
    
    def __str__(self) -> str:
        """
        String representation of the filter
        フィルターの文字列表現
        """
        return self.get_filter_name()
    
    def __repr__(self) -> str:
        """
        Developer representation of the filter
        フィルターの開発者向け表現
        """
        return f"{self.__class__.__name__}()"