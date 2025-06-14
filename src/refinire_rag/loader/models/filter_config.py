"""
FilterConfig model for managing file filters
ファイルフィルター管理用のFilterConfigモデル
"""

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

# Forward declarations to avoid circular imports
# 循環インポートを避けるための前方宣言
if TYPE_CHECKING:
    from refinire_rag.loader.filters.base_filter import BaseFilter
    from refinire_rag.loader.filters.extension_filter import ExtensionFilter
    from refinire_rag.loader.filters.date_filter import DateFilter
    from refinire_rag.loader.filters.path_filter import PathFilter
    from refinire_rag.loader.filters.size_filter import SizeFilter

@dataclass
class FilterConfig:
    """
    Configuration container for file filters
    ファイルフィルターの設定コンテナ
    
    This class manages the collection of filters to be applied
    during directory scanning. Filters are used only for inclusion/exclusion
    decisions and do not add metadata to documents.
    このクラスは、ディレクトリスキャン中に適用される
    フィルターのコレクションを管理します。フィルターは包含/除外の
    判定にのみ使用され、文書にメタデータを追加しません。
    """
    extension_filter: 'ExtensionFilter' = None
    date_filter: 'DateFilter' = None
    path_filter: 'PathFilter' = None
    size_filter: 'SizeFilter' = None
    custom_filters: List['BaseFilter'] = field(default_factory=list)
    
    def get_all_filters(self) -> List['BaseFilter']:
        """
        Get all configured filters as a list
        設定されたすべてのフィルターをリストとして取得
        
        Returns:
            List of all active filters
            すべてのアクティブなフィルターのリスト
        """
        filters = []
        
        if self.extension_filter is not None:
            filters.append(self.extension_filter)
        if self.date_filter is not None:
            filters.append(self.date_filter)
        if self.path_filter is not None:
            filters.append(self.path_filter)
        if self.size_filter is not None:
            filters.append(self.size_filter)
        
        filters.extend(self.custom_filters)
        return filters
    
    def add_custom_filter(self, filter_instance: 'BaseFilter'):
        """
        Add a custom filter to the configuration
        設定にカスタムフィルターを追加
        
        Args:
            filter_instance: Custom filter instance
            filter_instance: カスタムフィルターインスタンス
        """
        self.custom_filters.append(filter_instance)
    
    def has_filters(self) -> bool:
        """
        Check if any filters are configured
        フィルターが設定されているかどうかをチェック
        
        Returns:
            True if at least one filter is configured
            少なくとも1つのフィルターが設定されている場合True
        """
        return len(self.get_all_filters()) > 0
    
    def get_filter_summary(self) -> dict:
        """
        Get a summary of configured filters
        設定されたフィルターのサマリーを取得
        
        Returns:
            Dictionary with filter information
            フィルター情報を含む辞書
        """
        all_filters = self.get_all_filters()
        
        return {
            'total_filters': len(all_filters),
            'has_extension_filter': self.extension_filter is not None,
            'has_date_filter': self.date_filter is not None,
            'has_path_filter': self.path_filter is not None,
            'has_size_filter': self.size_filter is not None,
            'custom_filters_count': len(self.custom_filters),
            'filter_names': [filter.get_filter_name() for filter in all_filters] if hasattr(all_filters[0], 'get_filter_name') and all_filters else []
        }
    
    def __str__(self) -> str:
        """
        String representation of the filter configuration
        フィルター設定の文字列表現
        """
        summary = self.get_filter_summary()
        return f"FilterConfig(total_filters={summary['total_filters']})"