"""
Incremental Loader - Processes only new and updated documents

新規・更新された文書のみを処理する増分ローダー。
処理済み文書はスキップし、効率的なバッチ処理を実現します。
"""

import json
import hashlib
import os
from typing import List, Dict, Optional, Union, Set, Tuple
from pathlib import Path
from datetime import datetime
import logging

from .base import Loader, MetadataGenerator
from .universal import UniversalLoader
from ..models.document import Document
from ..models.config import LoadingConfig
from ..storage.document_store import DocumentStore
from ..exceptions import LoaderError

logger = logging.getLogger(__name__)


class IncrementalLoader:
    """
    増分ローダー
    
    ファイルの変更時刻とハッシュを追跡し、新規・更新された文書のみを処理します。
    """
    
    def __init__(
        self,
        document_store: DocumentStore,
        base_loader: Optional[Loader] = None,
        cache_file: Optional[str] = None,
        metadata_generator: Optional[MetadataGenerator] = None,
        config: Optional[LoadingConfig] = None
    ):
        """
        増分ローダーを初期化
        
        Args:
            document_store: 処理済み文書を保存するストア
            base_loader: 実際の文書読み込みを行うローダー
            cache_file: ファイル状態キャッシュのパス
            metadata_generator: メタデータ生成器
            config: ローディング設定
        """
        self.document_store = document_store
        self.base_loader = base_loader or UniversalLoader(metadata_generator, config)
        self.cache_file = Path(cache_file) if cache_file else Path(".refinire_cache.json")
        self.config = config or LoadingConfig()
        
        # ファイル状態キャッシュ
        self.file_cache: Dict[str, Dict] = self._load_cache()
        
        logger.info(f"Initialized IncrementalLoader with cache: {self.cache_file}")
    
    def _load_cache(self) -> Dict[str, Dict]:
        """ファイル状態キャッシュを読み込み"""
        
        if not self.cache_file.exists():
            logger.info("No cache file found, starting fresh")
            return {}
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            logger.info(f"Loaded cache with {len(cache)} entries")
            return cache
        except Exception as e:
            logger.warning(f"Failed to load cache file: {e}")
            return {}
    
    def _save_cache(self):
        """ファイル状態キャッシュを保存"""
        
        try:
            # キャッシュディレクトリを作成
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.file_cache, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved cache with {len(self.file_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache file: {e}")
    
    def _get_file_info(self, file_path: Path) -> Dict:
        """ファイルの情報（サイズ、更新時刻、ハッシュ）を取得"""
        
        try:
            stat = file_path.stat()
            
            # ファイルハッシュを計算（変更検出用）
            content_hash = self._calculate_file_hash(file_path)
            
            return {
                "path": str(file_path),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "content_hash": content_hash,
                "last_processed": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return {}
    
    def _calculate_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """ファイルのSHA256ハッシュを計算"""
        
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def _is_file_changed(self, file_path: Path) -> bool:
        """ファイルが変更されているかチェック"""
        
        file_key = str(file_path)
        
        # キャッシュにない場合は新規ファイル
        if file_key not in self.file_cache:
            logger.debug(f"New file detected: {file_path}")
            return True
        
        cached_info = self.file_cache[file_key]
        current_info = self._get_file_info(file_path)
        
        # ファイルサイズまたは更新時刻が変更されている場合
        if (cached_info.get("size") != current_info.get("size") or 
            cached_info.get("mtime") != current_info.get("mtime")):
            logger.debug(f"File modified (size/mtime): {file_path}")
            return True
        
        # ハッシュが変更されている場合
        if cached_info.get("content_hash") != current_info.get("content_hash"):
            logger.debug(f"File modified (content): {file_path}")
            return True
        
        return False
    
    def _should_process_document(self, document_id: str, file_path: Path) -> bool:
        """文書を処理すべきかどうかを判定"""
        
        # ファイルが変更されている場合は処理
        if self._is_file_changed(file_path):
            return True
        
        # ドキュメントストアに存在しない場合は処理
        try:
            existing_doc = self.document_store.get_document(document_id)
            if existing_doc is None:
                logger.debug(f"Document not in store, processing: {document_id}")
                return True
        except Exception as e:
            logger.warning(f"Error checking document store for {document_id}: {e}")
            return True
        
        logger.debug(f"Document unchanged, skipping: {file_path}")
        return False
    
    def scan_directory(
        self, 
        directory: Union[str, Path], 
        pattern: str = "*",
        recursive: bool = True
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        ディレクトリをスキャンして新規・更新・未変更ファイルを分類
        
        Args:
            directory: スキャン対象ディレクトリ
            pattern: ファイルパターン（*.pdf, *.txt など）
            recursive: 再帰的にスキャンするか
            
        Returns:
            (新規ファイル, 更新ファイル, 未変更ファイル)
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise LoaderError(f"Directory does not exist: {directory}")
        
        # ファイルを検索
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        # ディレクトリを除外
        files = [f for f in files if f.is_file()]
        
        new_files = []
        updated_files = []
        unchanged_files = []
        
        for file_path in files:
            file_key = str(file_path)
            
            if file_key not in self.file_cache:
                new_files.append(file_path)
            elif self._is_file_changed(file_path):
                updated_files.append(file_path)
            else:
                unchanged_files.append(file_path)
        
        logger.info(f"Scanned {directory}: {len(new_files)} new, {len(updated_files)} updated, {len(unchanged_files)} unchanged")
        
        return new_files, updated_files, unchanged_files
    
    def process_incremental(
        self, 
        sources: Union[str, Path, List[Union[str, Path]]],
        force_reload: Optional[Set[str]] = None
    ) -> Dict[str, List[Document]]:
        """
        増分処理を実行
        
        Args:
            sources: 処理対象のファイル・ディレクトリパス
            force_reload: 強制的に再処理するファイルパスのセット
            
        Returns:
            処理結果辞書 {
                'new': 新規処理された文書,
                'updated': 更新処理された文書, 
                'skipped': スキップされた文書,
                'errors': エラーが発生した文書
            }
        """
        if not isinstance(sources, list):
            sources = [sources]
        
        force_reload = force_reload or set()
        
        results = {
            'new': [],
            'updated': [],
            'skipped': [],
            'errors': []
        }
        
        # 処理対象ファイルを収集
        files_to_process = []
        
        for source in sources:
            source_path = Path(source)
            
            if source_path.is_file():
                files_to_process.append(source_path)
            elif source_path.is_dir():
                new_files, updated_files, unchanged_files = self.scan_directory(source_path)
                files_to_process.extend(new_files + updated_files)
                
                # 強制再処理ファイルを追加
                for file_path in unchanged_files:
                    if str(file_path) in force_reload:
                        files_to_process.append(file_path)
                        logger.info(f"Force reloading: {file_path}")
        
        logger.info(f"Processing {len(files_to_process)} files")
        
        # ファイルを処理
        for file_path in files_to_process:
            try:
                # 文書IDを生成
                document_id = self._generate_document_id(file_path)
                
                # 処理判定
                is_new = str(file_path) not in self.file_cache
                should_process = (
                    is_new or 
                    self._is_file_changed(file_path) or
                    str(file_path) in force_reload or
                    not self._document_exists_in_store(document_id)
                )
                
                if not should_process:
                    results['skipped'].append(file_path)
                    continue
                
                # 文書を読み込み
                document = self.base_loader.load_single(file_path)
                
                # 既存文書の削除（更新の場合）
                if not is_new and self._document_exists_in_store(document_id):
                    self._remove_old_document(document_id)
                
                # 文書を保存
                self.document_store.add_document(document)
                
                # キャッシュを更新
                self.file_cache[str(file_path)] = self._get_file_info(file_path)
                
                # 結果を分類
                if is_new:
                    results['new'].append(document)
                    logger.info(f"Processed new document: {file_path}")
                else:
                    results['updated'].append(document)
                    logger.info(f"Updated document: {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results['errors'].append({
                    'file_path': str(file_path),
                    'error': str(e)
                })
        
        # キャッシュを保存
        self._save_cache()
        
        logger.info(f"Incremental processing completed: {len(results['new'])} new, {len(results['updated'])} updated, {len(results['skipped'])} skipped, {len(results['errors'])} errors")
        
        return results
    
    def _generate_document_id(self, file_path: Path) -> str:
        """ファイルパスから文書IDを生成"""
        return hashlib.md5(str(file_path).encode()).hexdigest()
    
    def _document_exists_in_store(self, document_id: str) -> bool:
        """文書がストアに存在するかチェック"""
        try:
            return self.document_store.get_document(document_id) is not None
        except Exception:
            return False
    
    def _remove_old_document(self, document_id: str):
        """古い文書をストアから削除"""
        try:
            # ドキュメントストアから削除
            self.document_store.delete_document(document_id)
            logger.debug(f"Removed old document: {document_id}")
        except Exception as e:
            logger.warning(f"Failed to remove old document {document_id}: {e}")
    
    def cleanup_deleted_files(self, source_directories: List[Union[str, Path]]) -> List[str]:
        """
        削除されたファイルに対応する文書をクリーンアップ
        
        Args:
            source_directories: チェック対象のディレクトリリスト
            
        Returns:
            削除された文書IDのリスト
        """
        deleted_documents = []
        
        # キャッシュ内のファイルで実際に存在しないものを特定
        files_to_remove = []
        
        for file_path in list(self.file_cache.keys()):
            path_obj = Path(file_path)
            
            # ソースディレクトリ内のファイルかチェック
            is_in_source = any(
                path_obj.is_relative_to(Path(source_dir)) 
                for source_dir in source_directories
            )
            
            if is_in_source and not path_obj.exists():
                files_to_remove.append(file_path)
        
        # 削除されたファイルに対応する文書をクリーンアップ
        for file_path in files_to_remove:
            try:
                document_id = self._generate_document_id(Path(file_path))
                
                # ドキュメントストアから削除
                if self._document_exists_in_store(document_id):
                    self._remove_old_document(document_id)
                    deleted_documents.append(document_id)
                
                # キャッシュから削除
                del self.file_cache[file_path]
                
                logger.info(f"Cleaned up deleted file: {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to cleanup {file_path}: {e}")
        
        # キャッシュを保存
        if files_to_remove:
            self._save_cache()
        
        logger.info(f"Cleanup completed: {len(deleted_documents)} documents removed")
        return deleted_documents
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        
        if not self.file_cache:
            return {"total_files": 0, "cache_size": 0}
        
        total_files = len(self.file_cache)
        cache_size = sum(info.get("size", 0) for info in self.file_cache.values())
        
        # 最新・最古の処理時刻
        processed_times = [
            info.get("last_processed") 
            for info in self.file_cache.values() 
            if info.get("last_processed")
        ]
        
        stats = {
            "total_files": total_files,
            "total_size_bytes": cache_size,
            "cache_file": str(self.cache_file),
            "cache_exists": self.cache_file.exists()
        }
        
        if processed_times:
            stats["earliest_processed"] = min(processed_times)
            stats["latest_processed"] = max(processed_times)
        
        return stats
    
    def reset_cache(self, file_pattern: Optional[str] = None):
        """
        キャッシュをリセット
        
        Args:
            file_pattern: 特定パターンのファイルのみリセット（Noneで全リセット）
        """
        if file_pattern is None:
            # 全リセット
            self.file_cache = {}
            logger.info("Reset entire cache")
        else:
            # パターンマッチするファイルのみリセット
            import fnmatch
            keys_to_remove = [
                key for key in self.file_cache.keys() 
                if fnmatch.fnmatch(key, file_pattern)
            ]
            
            for key in keys_to_remove:
                del self.file_cache[key]
            
            logger.info(f"Reset cache for {len(keys_to_remove)} files matching '{file_pattern}'")
        
        self._save_cache()