"""
FileTracker for detecting changes between directory scans
ディレクトリスキャン間の変更を検出するFileTracker
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from refinire_rag.loader.models.file_info import FileInfo
from refinire_rag.loader.models.change_set import ChangeSet
from refinire_rag.loader.models.filter_config import FilterConfig


class FileTracker:
    """
    Tracks file changes between directory scans
    ディレクトリスキャン間のファイル変更を追跡
    
    This class maintains a registry of files and their metadata to detect
    when files have been added, modified, or deleted between scans.
    このクラスはファイルとそのメタデータのレジストリを維持し、
    スキャン間でファイルが追加、更新、削除されたことを検出します。
    """
    
    def __init__(self, tracking_file_path: Optional[Path] = None):
        """
        Initialize the file tracker
        ファイルトラッカーを初期化
        
        Args:
            tracking_file_path: Optional path to persist tracking data
            tracking_file_path: 追跡データを永続化するオプションのパス
        """
        self.tracking_file_path = tracking_file_path
        self.file_registry: Dict[str, FileInfo] = {}
        
        # Load existing tracking data if available
        # 利用可能な場合、既存の追跡データを読み込む
        if self.tracking_file_path and self.tracking_file_path.exists():
            self._load_tracking_data()
    
    def scan_directory(self, 
                      directory_path: Path, 
                      filter_config: FilterConfig = None,
                      recursive: bool = True) -> ChangeSet:
        """
        Scan directory and detect changes from previous scan
        ディレクトリをスキャンし、前回のスキャンからの変更を検出
        
        Args:
            directory_path: Directory to scan
            filter_config: Optional filter configuration
            recursive: Whether to scan subdirectories recursively
            directory_path: スキャンするディレクトリ
            filter_config: オプションのフィルター設定
            recursive: サブディレクトリを再帰的にスキャンするかどうか
            
        Returns:
            ChangeSet with detected changes
            検出された変更を含むChangeSet
        """
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory_path}")
        
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory_path}")
        
        # Get current file information
        # 現在のファイル情報を取得
        current_files = self._scan_files(directory_path, filter_config, recursive)
        
        # Compare with previous scan
        # 前回のスキャンと比較
        change_set = self._compare_file_states(current_files)
        
        # Update registry with current files
        # 現在のファイルでレジストリを更新
        self.file_registry = current_files
        
        # Persist tracking data if path is configured
        # パスが設定されている場合、追跡データを永続化
        if self.tracking_file_path:
            self._save_tracking_data()
        
        return change_set
    
    def _scan_files(self, 
                   directory_path: Path, 
                   filter_config: FilterConfig = None,
                   recursive: bool = True) -> Dict[str, FileInfo]:
        """
        Scan directory and collect file information
        ディレクトリをスキャンしてファイル情報を収集
        
        Args:
            directory_path: Directory to scan
            filter_config: Optional filter configuration
            recursive: Whether to scan subdirectories recursively
            directory_path: スキャンするディレクトリ
            filter_config: オプションのフィルター設定
            recursive: サブディレクトリを再帰的にスキャンするかどうか
            
        Returns:
            Dictionary mapping file paths to FileInfo objects
            ファイルパスをFileInfoオブジェクトにマッピングする辞書
        """
        files = {}
        
        # Get file pattern for scanning
        # スキャン用のファイルパターンを取得
        pattern = "**/*" if recursive else "*"
        
        # Get all filters if configured
        # 設定されている場合、すべてのフィルターを取得
        filters = filter_config.get_all_filters() if filter_config else []
        
        for file_path in directory_path.glob(pattern):
            if not file_path.is_file():
                continue
            
            # Apply filters if configured
            # 設定されている場合、フィルターを適用
            if filters:
                should_include = True
                for filter_instance in filters:
                    if not filter_instance.should_include(file_path):
                        should_include = False
                        break
                
                if not should_include:
                    continue
            
            try:
                # Create FileInfo for this file
                # このファイル用のFileInfoを作成
                file_info = FileInfo.from_file(file_path)
                files[str(file_path)] = file_info
            except (OSError, IOError) as e:
                # Skip files that cannot be accessed
                # アクセスできないファイルをスキップ
                print(f"Warning: Could not access file {file_path}: {e}")
                continue
        
        return files
    
    def _compare_file_states(self, current_files: Dict[str, FileInfo]) -> ChangeSet:
        """
        Compare current files with previous registry to detect changes
        現在のファイルを前回のレジストリと比較して変更を検出
        
        Args:
            current_files: Current file information
            current_files: 現在のファイル情報
            
        Returns:
            ChangeSet with detected changes
            検出された変更を含むChangeSet
        """
        change_set = ChangeSet()
        
        # Get file path sets for comparison
        # 比較用のファイルパスセットを取得
        current_paths = set(current_files.keys())
        previous_paths = set(self.file_registry.keys())
        
        # Find added files (in current but not in previous)
        # 追加されたファイルを見つける（現在にあるが前回にない）
        added_paths = current_paths - previous_paths
        change_set.added.extend(sorted(added_paths))
        
        # Find deleted files (in previous but not in current)
        # 削除されたファイルを見つける（前回にあるが現在にない）
        deleted_paths = previous_paths - current_paths
        change_set.deleted.extend(sorted(deleted_paths))
        
        # Find potentially modified files (in both)
        # 潜在的に更新されたファイルを見つける（両方にある）
        common_paths = current_paths & previous_paths
        
        for file_path in common_paths:
            current_info = current_files[file_path]
            previous_info = self.file_registry[file_path]
            
            # Compare file information to detect changes
            # ファイル情報を比較して変更を検出
            if current_info != previous_info:
                change_set.modified.append(file_path)
            else:
                change_set.unchanged.append(file_path)
        
        return change_set
    
    def _save_tracking_data(self):
        """
        Save tracking data to file
        追跡データをファイルに保存
        """
        if not self.tracking_file_path:
            return
        
        # Ensure parent directory exists
        # 親ディレクトリが存在することを確認
        self.tracking_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert FileInfo objects to dictionaries for JSON serialization
        # JSON直列化のためにFileInfoオブジェクトを辞書に変換
        data = {}
        for file_path, file_info in self.file_registry.items():
            data[file_path] = {
                'path': file_info.path,
                'size': file_info.size,
                'modified_at': file_info.modified_at.isoformat(),
                'hash_md5': file_info.hash_md5,
                'file_type': file_info.file_type
            }
        
        try:
            with open(self.tracking_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except (OSError, IOError) as e:
            print(f"Warning: Could not save tracking data to {self.tracking_file_path}: {e}")
    
    def _load_tracking_data(self):
        """
        Load tracking data from file
        ファイルから追跡データを読み込む
        """
        if not self.tracking_file_path or not self.tracking_file_path.exists():
            return
        
        try:
            with open(self.tracking_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert dictionaries back to FileInfo objects
            # 辞書をFileInfoオブジェクトに戻す
            self.file_registry = {}
            for file_path, file_data in data.items():
                file_info = FileInfo(
                    path=file_data['path'],
                    size=file_data['size'],
                    modified_at=datetime.fromisoformat(file_data['modified_at']),
                    hash_md5=file_data['hash_md5'],
                    file_type=file_data['file_type']
                )
                self.file_registry[file_path] = file_info
                
        except (OSError, IOError, json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Could not load tracking data from {self.tracking_file_path}: {e}")
            self.file_registry = {}
    
    def get_file_count(self) -> int:
        """
        Get the number of files currently tracked
        現在追跡されているファイル数を取得
        
        Returns:
            Number of tracked files
            追跡されているファイル数
        """
        return len(self.file_registry)
    
    def get_tracked_files(self) -> List[str]:
        """
        Get list of all tracked file paths
        追跡されているすべてのファイルパスのリストを取得
        
        Returns:
            List of file paths
            ファイルパスのリスト
        """
        return sorted(self.file_registry.keys())
    
    def is_file_tracked(self, file_path: str) -> bool:
        """
        Check if a file is currently being tracked
        ファイルが現在追跡されているかどうかをチェック
        
        Args:
            file_path: Path to check
            file_path: チェックするパス
            
        Returns:
            True if file is tracked, False otherwise
            ファイルが追跡されている場合True、そうでなければFalse
        """
        return file_path in self.file_registry
    
    def get_file_info(self, file_path: str) -> Optional[FileInfo]:
        """
        Get FileInfo for a tracked file
        追跡されているファイルのFileInfoを取得
        
        Args:
            file_path: Path to get info for
            file_path: 情報を取得するパス
            
        Returns:
            FileInfo object or None if not tracked
            FileInfoオブジェクト、または追跡されていない場合None
        """
        return self.file_registry.get(file_path)
    
    def clear_tracking_data(self):
        """
        Clear all tracking data
        すべての追跡データをクリア
        """
        self.file_registry.clear()
        
        # Remove tracking file if it exists
        # 追跡ファイルが存在する場合は削除
        if self.tracking_file_path and self.tracking_file_path.exists():
            try:
                self.tracking_file_path.unlink()
            except (OSError, IOError) as e:
                print(f"Warning: Could not remove tracking file {self.tracking_file_path}: {e}")
    
    def __str__(self) -> str:
        """
        String representation of the file tracker
        ファイルトラッカーの文字列表現
        """
        return f"FileTracker(tracked_files={self.get_file_count()})"
    
    def __repr__(self) -> str:
        """
        Developer representation of the file tracker
        ファイルトラッカーの開発者向け表現
        """
        return f"FileTracker(tracking_file_path={self.tracking_file_path}, tracked_files={self.get_file_count()})"