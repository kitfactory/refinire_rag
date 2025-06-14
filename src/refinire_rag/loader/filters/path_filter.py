"""
Path-based file filter
パスベースのファイルフィルター
"""

import re
from pathlib import Path
from typing import List, Union, Pattern
from refinire_rag.loader.filters.base_filter import BaseFilter
from refinire_rag.loader.models.file_info import FileInfo


class PathFilter(BaseFilter):
    """
    Filter files based on file paths using patterns
    パターンを使用してファイルパスに基づいてファイルをフィルタリング
    
    This filter supports glob patterns, regular expressions, and exact path matching
    for both inclusion and exclusion of files.
    このフィルターはglobパターン、正規表現、および完全パスマッチングを
    サポートして、ファイルの包含と除外の両方を行います。
    """
    
    def __init__(self, 
                 include_patterns: Union[List[str], str] = None,
                 exclude_patterns: Union[List[str], str] = None,
                 use_regex: bool = False):
        """
        Initialize the path filter
        パスフィルターを初期化
        
        Args:
            include_patterns: Patterns for paths to include
            exclude_patterns: Patterns for paths to exclude
            use_regex: If True, patterns are treated as regex, otherwise as glob
            include_patterns: 含めるパスのパターン
            exclude_patterns: 除外するパスのパターン
            use_regex: Trueの場合、パターンを正規表現として扱い、そうでなければglobとして扱う
            
        Note:
            If include_patterns is specified, only matching paths will be included.
            If exclude_patterns is specified, matching paths will be excluded.
            If both are specified, include_patterns takes precedence.
            include_patternsが指定された場合、マッチするパスのみが含まれます。
            exclude_patternsが指定された場合、マッチするパスが除外されます。
            両方が指定された場合、include_patternsが優先されます。
        """
        if include_patterns is None and exclude_patterns is None:
            raise ValueError("At least one of include_patterns or exclude_patterns must be specified")
        
        # Convert single patterns to lists
        # 単一パターンをリストに変換
        self.include_patterns = self._ensure_list(include_patterns)
        self.exclude_patterns = self._ensure_list(exclude_patterns)
        self.use_regex = use_regex
        
        # Compile regex patterns if needed
        # 必要に応じて正規表現パターンをコンパイル
        if self.use_regex:
            self.compiled_include = [re.compile(pattern) for pattern in self.include_patterns]
            self.compiled_exclude = [re.compile(pattern) for pattern in self.exclude_patterns]
        else:
            self.compiled_include = None
            self.compiled_exclude = None
    
    def _ensure_list(self, patterns: Union[List[str], str, None]) -> List[str]:
        """
        Ensure patterns is a list
        パターンがリストであることを確認
        
        Args:
            patterns: Single pattern string or list of patterns
            patterns: 単一パターン文字列またはパターンのリスト
            
        Returns:
            List of pattern strings
            パターン文字列のリスト
        """
        if patterns is None:
            return []
        elif isinstance(patterns, str):
            return [patterns]
        else:
            return list(patterns)
    
    def _matches_glob_pattern(self, path_str: str, pattern: str) -> bool:
        """
        Check if path matches glob pattern
        パスがglobパターンにマッチするかチェック
        
        Args:
            path_str: Path string to check
            pattern: Glob pattern
            path_str: チェックするパス文字列
            pattern: Globパターン
            
        Returns:
            True if path matches pattern
            パスがパターンにマッチする場合True
        """
        from fnmatch import fnmatch
        return fnmatch(path_str, pattern)
    
    def _matches_regex_pattern(self, path_str: str, compiled_pattern: Pattern) -> bool:
        """
        Check if path matches regex pattern
        パスが正規表現パターンにマッチするかチェック
        
        Args:
            path_str: Path string to check
            compiled_pattern: Compiled regex pattern
            path_str: チェックするパス文字列
            compiled_pattern: コンパイル済み正規表現パターン
            
        Returns:
            True if path matches pattern
            パスがパターンにマッチする場合True
        """
        return bool(compiled_pattern.search(path_str))
    
    def should_include(self, file_path: Path, file_info: FileInfo = None) -> bool:
        """
        Determine if file should be included based on path patterns
        パスパターンに基づいてファイルが含まれるべきかどうかを判定
        
        Args:
            file_path: Path to the file being evaluated
            file_info: Optional FileInfo object (not used for path filtering)
            file_path: 評価対象のファイルパス
            file_info: オプションのFileInfoオブジェクト（パスフィルタリングには使用されない）
            
        Returns:
            True if file should be included, False otherwise
            ファイルが含まれるべき場合True、そうでなければFalse
        """
        path_str = str(file_path)
        
        # If include patterns are specified, file must match at least one
        # includeパターンが指定されている場合、ファイルは少なくとも1つにマッチする必要がある
        if self.include_patterns:
            include_match = False
            
            if self.use_regex:
                include_match = any(
                    self._matches_regex_pattern(path_str, pattern) 
                    for pattern in self.compiled_include
                )
            else:
                include_match = any(
                    self._matches_glob_pattern(path_str, pattern) 
                    for pattern in self.include_patterns
                )
            
            if not include_match:
                return False
        
        # Check exclude patterns - file must not match any
        # excludeパターンをチェック - ファイルはいずれにもマッチしてはいけない
        if self.exclude_patterns:
            if self.use_regex:
                exclude_match = any(
                    self._matches_regex_pattern(path_str, pattern) 
                    for pattern in self.compiled_exclude
                )
            else:
                exclude_match = any(
                    self._matches_glob_pattern(path_str, pattern) 
                    for pattern in self.exclude_patterns
                )
            
            if exclude_match:
                return False
        
        return True
    
    def get_filter_name(self) -> str:
        """
        Get descriptive name for this filter
        このフィルターの説明的な名前を取得
        """
        pattern_type = "regex" if self.use_regex else "glob"
        
        if self.include_patterns and self.exclude_patterns:
            return f"PathFilter({pattern_type}, include: {len(self.include_patterns)}, exclude: {len(self.exclude_patterns)})"
        elif self.include_patterns:
            return f"PathFilter({pattern_type}, include: {len(self.include_patterns)} patterns)"
        else:
            return f"PathFilter({pattern_type}, exclude: {len(self.exclude_patterns)} patterns)"
    
    def get_description(self) -> str:
        """
        Get detailed description of this filter
        このフィルターの詳細説明を取得
        """
        pattern_type = "regex" if self.use_regex else "glob"
        parts = []
        
        if self.include_patterns:
            patterns = ', '.join(f"'{p}'" for p in self.include_patterns[:3])
            if len(self.include_patterns) > 3:
                patterns += f" and {len(self.include_patterns) - 3} more"
            parts.append(f"Include paths matching {pattern_type} patterns: {patterns}")
        
        if self.exclude_patterns:
            patterns = ', '.join(f"'{p}'" for p in self.exclude_patterns[:3])
            if len(self.exclude_patterns) > 3:
                patterns += f" and {len(self.exclude_patterns) - 3} more"
            parts.append(f"Exclude paths matching {pattern_type} patterns: {patterns}")
        
        return "; ".join(parts)
    
    def __repr__(self) -> str:
        """
        Developer representation of the filter
        フィルターの開発者向け表現
        """
        return (f"PathFilter(include_patterns={self.include_patterns}, "
                f"exclude_patterns={self.exclude_patterns}, "
                f"use_regex={self.use_regex})")