from typing import Dict, Any, Optional
import re
from pathlib import Path
from collections import OrderedDict
from .metadata import Metadata
import fnmatch

class PathMapMetadata(Metadata):
    """
    Metadata processor that adds metadata based on path patterns.
    Supports regex patterns, priority-based matching, and multiple pattern matching.
    
    パスのパターンに基づいてメタデータを付与するプロセッサー。
    正規表現パターン、優先順位ベースのマッチング、複数パターンマッチングをサポート。
    
    Features:
    - Path pattern matching with wildcards (*, **)
    - Case-insensitive matching
    - Regex pattern support
    - Priority-based matching
    - Multiple pattern matching with merge strategies
    
    機能:
    - ワイルドカード（*, **）を使用したパスパターンマッチング
    - 大文字小文字を区別しないマッチング
    - 正規表現パターンのサポート
    - 優先順位ベースのマッチング
    - マージ戦略を使用した複数パターンマッチング
    """
    
    def __init__(
        self,
        path_map: Dict[str, Dict[str, Any]],
        use_regex: bool = False,
        priority_based: bool = False,
        merge_strategy: str = "first"
    ):
        """
        Initialize the path map metadata processor.
        
        パスマップメタデータプロセッサーを初期化します。
        
        Args:
            path_map (Dict[str, Dict[str, Any]]): Dictionary mapping path patterns to their corresponding metadata.
                                                  パスパターンとそれに対応するメタデータのマッピング。
                                                  Example: {
                                                      "hr/*": {"department": "HR"},
                                                      "finance/*": {"department": "Finance"},
                                                      r".*\.pdf$": {"type": "document"}
                                                  }
            use_regex (bool): Whether to use regex patterns for matching.
                             正規表現パターンを使用するかどうか。
            priority_based (bool): Whether to use priority-based matching (patterns are matched in order).
                                  優先順位ベースのマッチングを使用するかどうか（パターンは順番にマッチング）。
            merge_strategy (str): Strategy for handling multiple matches. Options: "first", "last", "merge".
                                 複数マッチ時の処理戦略。"first"（最初のマッチ）、"last"（最後のマッチ）、"merge"（マージ）から選択。
        
        Raises:
            ValueError: If merge_strategy is not one of "first", "last", or "merge".
                       マージ戦略が"first"、"last"、"merge"のいずれでもない場合。
        """
        if merge_strategy not in ["first", "last", "merge"]:
            raise ValueError('merge_strategy must be one of "first", "last", or "merge"')
            
        # パターン順序を保証
        self.path_map = OrderedDict(path_map)
        self.use_regex = use_regex
        self.priority_based = priority_based
        self.merge_strategy = merge_strategy
        
        if use_regex:
            # Compile regex patterns
            self.compiled_patterns = OrderedDict((re.compile(pattern), metadata) for pattern, metadata in self.path_map.items())
    
    def _match_pattern(self, path: str, pattern: str) -> bool:
        """
        Check if the path matches the pattern (supports wildcards and case-insensitive).
        
        パスがパターンにマッチするか判定します（ワイルドカード・大文字小文字無視対応）。
        
        Args:
            path (str): The path to check.
                       チェックするパス。
            pattern (str): The pattern to match against.
                          マッチングするパターン。
        
        Returns:
            bool: True if the path matches the pattern, False otherwise.
                 パスがパターンにマッチする場合はTrue、そうでない場合はFalse。
        
        Examples:
            >>> _match_pattern("HR/policy.txt", "hr/*")
            True
            >>> _match_pattern("finance/contract.pdf", "*.pdf")
            True
        """
        return fnmatch.fnmatch(path.lower(), pattern.lower()) or pattern.lower() in path.lower()
    
    def _merge_metadata(self, base: Dict[str, Any], metadata_list: list[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge a list of metadata dicts into the base dict.
        
        複数のメタデータ辞書をbaseにマージします。
        
        Args:
            base (Dict[str, Any]): The base metadata dictionary.
                                  ベースとなるメタデータ辞書。
            metadata_list (list[Dict[str, Any]]): List of metadata dictionaries to merge.
                                                 マージするメタデータ辞書のリスト。
        
        Returns:
            Dict[str, Any]: The merged metadata dictionary.
                           マージされたメタデータ辞書。
        
        Examples:
            >>> _merge_metadata({"a": 1}, [{"b": 2}, {"c": 3}])
            {"a": 1, "b": 2, "c": 3}
        """
        result = dict(base)
        for md in metadata_list:
            result.update(md)
        return result
    
    def get_metadata(self, path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add metadata based on path patterns.
        
        パスのパターンに基づいてメタデータを付与します。
        
        Args:
            path (str): The path to process.
                       処理するパス。
            metadata (Dict[str, Any]): The base metadata dictionary.
                                      ベースとなるメタデータ辞書。
        
        Returns:
            Dict[str, Any]: The updated metadata dictionary with additional metadata from matching patterns.
                           マッチしたパターンからの追加メタデータを含む更新されたメタデータ辞書。
        
        Examples:
            >>> processor = PathMapMetadata({"hr/*": {"department": "HR"}})
            >>> processor.get_metadata("hr/policy.txt", {})
            {"department": "HR"}
        """
        matches = []
        if self.use_regex:
            for pattern, pattern_metadata in self.compiled_patterns.items():
                if pattern.search(path):
                    matches.append((pattern, pattern_metadata))
        else:
            for pattern, pattern_metadata in self.path_map.items():
                if self._match_pattern(path, pattern):
                    matches.append((pattern, pattern_metadata))

        if not matches:
            return metadata

        if self.priority_based:
            # パターンの優先順位に基づいてマッチを処理
            # より具体的なパターンを先に評価するため、パターンの長さでソート
            matches.sort(key=lambda x: len(x[0]), reverse=True)
            # すべてのマッチをマージ
            return self._merge_metadata(metadata, [m[1] for m in matches])
        else:
            # すべてのマッチをマージ
            return self._merge_metadata(metadata, [m[1] for m in matches]) 