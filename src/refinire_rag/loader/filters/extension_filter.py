"""
Extension-based file filter
拡張子ベースのファイルフィルター
"""

from pathlib import Path
from typing import Set, List, Union
from refinire_rag.loader.filters.base_filter import BaseFilter
from refinire_rag.loader.models.file_info import FileInfo


class ExtensionFilter(BaseFilter):
    """
    Filter files based on file extensions
    ファイル拡張子に基づいてファイルをフィルタリング
    
    This filter can be configured to include only specific extensions
    or to exclude specific extensions.
    このフィルターは特定の拡張子のみを含むか、
    特定の拡張子を除外するように設定できます。
    """
    
    def __init__(self, 
                 include_extensions: Union[List[str], Set[str]] = None,
                 exclude_extensions: Union[List[str], Set[str]] = None):
        """
        Initialize the extension filter
        拡張子フィルターを初期化
        
        Args:
            include_extensions: Extensions to include (e.g., ['.txt', '.md'])
            exclude_extensions: Extensions to exclude (e.g., ['.tmp', '.log'])
            include_extensions: 含める拡張子（例：['.txt', '.md']）
            exclude_extensions: 除外する拡張子（例：['.tmp', '.log']）
            
        Note:
            If include_extensions is specified, only those extensions will be included.
            If exclude_extensions is specified, all extensions except those will be included.
            If both are specified, include_extensions takes precedence.
            include_extensionsが指定された場合、それらの拡張子のみが含まれます。
            exclude_extensionsが指定された場合、それら以外のすべての拡張子が含まれます。
            両方が指定された場合、include_extensionsが優先されます。
        """
        self.include_extensions = set(self._normalize_extensions(include_extensions or []))
        self.exclude_extensions = set(self._normalize_extensions(exclude_extensions or []))
        
        if not self.include_extensions and not self.exclude_extensions:
            raise ValueError("At least one of include_extensions or exclude_extensions must be specified")
    
    def _normalize_extensions(self, extensions: Union[List[str], Set[str]]) -> List[str]:
        """
        Normalize extensions to ensure they start with a dot and are lowercase
        拡張子を正規化してドットで始まり小文字であることを確認
        
        Args:
            extensions: List or set of extensions
            extensions: 拡張子のリストまたはセット
            
        Returns:
            Normalized extensions list
            正規化された拡張子のリスト
        """
        normalized = []
        for ext in extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            normalized.append(ext.lower())
        return normalized
    
    def should_include(self, file_path: Path, file_info: FileInfo = None) -> bool:
        """
        Determine if file should be included based on extension
        拡張子に基づいてファイルが含まれるべきかどうかを判定
        
        Args:
            file_path: Path to the file being evaluated
            file_info: Optional FileInfo object (not used for extension filtering)
            file_path: 評価対象のファイルパス
            file_info: オプションのFileInfoオブジェクト（拡張子フィルタリングには使用されない）
            
        Returns:
            True if file should be included, False otherwise
            ファイルが含まれるべき場合True、そうでなければFalse
        """
        file_extension = file_path.suffix.lower()
        
        # If include_extensions is specified, only include those
        # include_extensionsが指定されている場合、それらのみを含める
        if self.include_extensions:
            return file_extension in self.include_extensions
        
        # Otherwise, exclude specified extensions
        # そうでなければ、指定された拡張子を除外
        return file_extension not in self.exclude_extensions
    
    def get_filter_name(self) -> str:
        """
        Get descriptive name for this filter
        このフィルターの説明的な名前を取得
        """
        if self.include_extensions:
            extensions = ', '.join(sorted(self.include_extensions))
            return f"ExtensionFilter(include: {extensions})"
        else:
            extensions = ', '.join(sorted(self.exclude_extensions))
            return f"ExtensionFilter(exclude: {extensions})"
    
    def get_description(self) -> str:
        """
        Get detailed description of this filter
        このフィルターの詳細説明を取得
        """
        if self.include_extensions:
            extensions = ', '.join(sorted(self.include_extensions))
            return f"Include only files with extensions: {extensions}"
        else:
            extensions = ', '.join(sorted(self.exclude_extensions))
            return f"Exclude files with extensions: {extensions}"
    
    def __repr__(self) -> str:
        """
        Developer representation of the filter
        フィルターの開発者向け表現
        """
        if self.include_extensions:
            return f"ExtensionFilter(include_extensions={list(self.include_extensions)})"
        else:
            return f"ExtensionFilter(exclude_extensions={list(self.exclude_extensions)})"