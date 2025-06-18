# バックエンドモジュール インターフェース定義書

## 概要

本文書では、refinire-ragの各バックエンドモジュールの詳細なインターフェースを定義します。各モジュールは単一責任の原則に従い、依存性注入によって交換可能な設計となっています。

## 1. DocumentProcessor（基底クラス）

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Any
from dataclasses import dataclass

class DocumentProcessor(ABC):
    """Base interface for document processing
    文書処理の基底インターフェース"""
    
    @abstractmethod
    def process(self, document: Document, config: Optional[Any] = None) -> List[Document]:
        """Process a document and return list of resulting documents
        文書を処理して結果文書のリストを返す
        
        Args:
            document: Input document to process
            config: Optional configuration for processing
            
        Returns:
            List of processed documents (could be 1 for normalization, many for chunking)
        """
        pass

class DocumentPipeline:
    """Pipeline for chaining multiple document processors
    複数の文書プロセッサーをチェーンするパイプライン"""
    
    def __init__(self, processors: List[DocumentProcessor], document_store: DocumentStore):
        self.processors = processors
        self.document_store = document_store
    
    def process_document(self, document: Document) -> List[Document]:
        """Process document through the entire pipeline
        文書をパイプライン全体で処理
        
        Returns:
            All documents created during processing
        """
        current_docs = [document]
        all_results = []
        
        for processor in self.processors:
            next_docs = []
            for doc in current_docs:
                processed = processor.process(doc)
                next_docs.extend(processed)
                
                # Store each processed document
                for processed_doc in processed:
                    self.document_store.store_document(processed_doc)
                    all_results.append(processed_doc)
            
            current_docs = next_docs
        
        return all_results
```

## 2. MetadataGenerator

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Union
from pathlib import Path

class MetadataGenerator(ABC):
    """Interface for generating additional metadata
    追加メタデータ生成のインターフェース"""
    
    @abstractmethod
    def generate_metadata(
        self,
        required_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate additional metadata from required fields
        必須フィールドから追加メタデータを生成
        
        Args:
            required_metadata: Required metadata fields (path, created_at, file_type, size_bytes)
            
        Returns:
            Additional metadata to be merged with required fields
        """
        pass
```

### 実装例

```python
class PathBasedMetadataGenerator(MetadataGenerator):
    """Generate metadata based on file path patterns
    ファイルパスパターンに基づくメタデータ生成"""
    
    def __init__(self, path_rules: Dict[str, Dict[str, Any]]):
        """Initialize with path-based rules
        パスベースのルールで初期化
        
        Args:
            path_rules: Dict mapping path patterns to metadata
                例: {
                    "/docs/public/*": {"access_group": "public", "classification": "open"},
                    "/docs/internal/*": {"access_group": "employees", "classification": "internal"},
                    "/docs/confidential/*": {"access_group": "managers", "classification": "confidential"}
                }
        """
        self.path_rules = path_rules
    
    def generate_metadata(self, required_metadata: Dict[str, Any]) -> Dict[str, Any]:
        path = required_metadata["path"]
        additional_metadata = {}
        
        # Apply path-based rules
        for pattern, metadata in self.path_rules.items():
            if self._matches_pattern(path, pattern):
                additional_metadata.update(metadata)
                break
        
        # Extract folder-based information
        path_obj = Path(path)
        additional_metadata.update({
            "filename": path_obj.name,
            "directory": str(path_obj.parent),
            "folder_name": path_obj.parent.name
        })
        
        # Add file type specific metadata
        file_type = required_metadata["file_type"]
        if file_type == ".pdf":
            additional_metadata["document_type"] = "pdf_document"
        elif file_type in [".md", ".txt"]:
            additional_metadata["document_type"] = "text_document"
        
        return additional_metadata
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Simple pattern matching (can be enhanced with regex)
        シンプルなパターンマッチング"""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)

class ContentBasedMetadataGenerator(MetadataGenerator):
    """Generate metadata based on content analysis
    コンテンツ解析に基づくメタデータ生成"""
    
    def generate_metadata(self, required_metadata: Dict[str, Any]) -> Dict[str, Any]:
        # This would analyze file content to extract metadata
        # コンテンツを解析してメタデータを抽出
        return {
            "language": "ja",  # Language detection
            "estimated_reading_time": 5,  # Based on content length
            "content_type": "technical"  # Based on content analysis
        }
```

## 2. DocumentStore

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Document search result
    文書検索結果"""
    document: Document
    score: Optional[float] = None
    rank: Optional[int] = None

@dataclass
class StorageStats:
    """Storage statistics
    ストレージ統計情報"""
    total_documents: int
    total_chunks: int
    storage_size_bytes: int
    oldest_document: Optional[str]
    newest_document: Optional[str]

class DocumentStore(ABC):
    """Interface for document storage and retrieval
    文書の保存・取得インターフェース"""
    
    @abstractmethod
    def store_document(self, document: Document) -> str:
        """Store a document and return its ID
        文書を保存してIDを返却
        
        Args:
            document: Document to store
            
        Returns:
            Document ID
        """
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by ID
        IDで文書を取得
        
        Args:
            document_id: Document ID to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update_document(self, document: Document) -> bool:
        """Update an existing document
        既存文書を更新
        
        Args:
            document: Document with updated content/metadata
            
        Returns:
            True if updated successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID
        IDで文書を削除
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if deleted successfully, False if not found
        """
        pass
    
    @abstractmethod
    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[SearchResult]:
        """Search documents by metadata filters
        メタデータフィルターで文書を検索
        
        Args:
            filters: Metadata filters (supports operators like $gte, $contains, $in)
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    def search_by_content(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[SearchResult]:
        """Search documents by content (full-text search)
        コンテンツで文書を検索（全文検索）
        
        Args:
            query: Text query to search for
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of search results with relevance scores
        """
        pass
    
    @abstractmethod
    def get_documents_by_lineage(
        self,
        original_document_id: str
    ) -> List[Document]:
        """Get all documents derived from an original document
        オリジナル文書から派生した全ての文書を取得
        
        Args:
            original_document_id: ID of the original document
            
        Returns:
            List of all derived documents
        """
        pass
    
    @abstractmethod
    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> List[Document]:
        """List documents with pagination and sorting
        ページネーションとソート付きで文書をリスト表示
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            sort_by: Field to sort by
            sort_order: Sort order ("asc" or "desc")
            
        Returns:
            List of documents
        """
        pass
    
    @abstractmethod
    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count documents matching optional filters
        オプションのフィルターに一致する文書数をカウント
        
        Args:
            filters: Optional metadata filters
            
        Returns:
            Number of matching documents
        """
        pass
    
    @abstractmethod
    def get_storage_stats(self) -> StorageStats:
        """Get storage statistics
        ストレージ統計情報を取得
        
        Returns:
            Storage statistics
        """
        pass
    
    @abstractmethod
    def cleanup_orphaned_documents(self) -> int:
        """Clean up orphaned documents (no references)
        孤立した文書をクリーンアップ（参照がない文書）
        
        Returns:
            Number of documents cleaned up
        """
        pass
    
    @abstractmethod
    def backup_to_file(self, backup_path: str) -> bool:
        """Backup all documents to a file
        全文書をファイルにバックアップ
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if backup successful, False otherwise
        """
        pass
    
    @abstractmethod
    def restore_from_file(self, backup_path: str) -> bool:
        """Restore documents from a backup file
        バックアップファイルから文書を復元
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if restore successful, False otherwise
        """
        pass
```

### 実装例（SQLiteDocumentStore）

```python
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

class SQLiteDocumentStore(DocumentStore):
    """SQLite-based document storage implementation
    SQLiteベースの文書ストレージ実装"""
    
    def __init__(self, db_path: str = "./data/documents.db"):
        """Initialize SQLite document store
        SQLite文書ストアを初期化
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        # Check for JSON1 extension
        try:
            self.conn.execute("SELECT json('{}')")
            self.json_enabled = True
        except sqlite3.OperationalError:
            self.json_enabled = False
        
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema
        データベーススキーマを初期化"""
        
        schema_sql = """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents (created_at);
            CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents (updated_at);
        """
        
        if self.json_enabled:
            # Add generated columns for common metadata fields
            schema_sql += """
                -- Generated columns for fast metadata search
                ALTER TABLE documents ADD COLUMN file_type TEXT 
                    GENERATED ALWAYS AS (json_extract(metadata, '$.file_type')) STORED;
                ALTER TABLE documents ADD COLUMN original_document_id TEXT 
                    GENERATED ALWAYS AS (json_extract(metadata, '$.original_document_id')) STORED;
                ALTER TABLE documents ADD COLUMN size_bytes INTEGER 
                    GENERATED ALWAYS AS (json_extract(metadata, '$.size_bytes')) STORED;
                    
                CREATE INDEX IF NOT EXISTS idx_file_type ON documents (file_type);
                CREATE INDEX IF NOT EXISTS idx_original_doc_id ON documents (original_document_id);
                CREATE INDEX IF NOT EXISTS idx_size_bytes ON documents (size_bytes);
            """
        
        # FTS5 for full-text search
        schema_sql += """
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                id UNINDEXED,
                content,
                content='documents',
                content_rowid='rowid'
            );
            
            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, id, content) VALUES (new.rowid, new.id, new.content);
            END;
            
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, id, content) VALUES('delete', old.rowid, old.id, old.content);
            END;
            
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, id, content) VALUES('delete', old.rowid, old.id, old.content);
                INSERT INTO documents_fts(rowid, id, content) VALUES (new.rowid, new.id, new.content);
            END;
        """
        
        try:
            self.conn.executescript(schema_sql)
            self.conn.commit()
        except sqlite3.OperationalError as e:
            # Handle case where generated columns already exist
            if "duplicate column name" not in str(e):
                raise
    
    def store_document(self, document: Document) -> str:
        """Store a document in SQLite
        SQLiteに文書を保存"""
        
        self.conn.execute(
            """INSERT OR REPLACE INTO documents (id, content, metadata, updated_at) 
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
            (document.id, document.content, json.dumps(document.metadata))
        )
        self.conn.commit()
        return document.id
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve document by ID
        IDで文書を取得"""
        
        cursor = self.conn.execute(
            "SELECT id, content, metadata FROM documents WHERE id = ?",
            (document_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return Document(
                id=row["id"],
                content=row["content"],
                metadata=json.loads(row["metadata"])
            )
        return None
    
    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[SearchResult]:
        """Search documents by metadata
        メタデータで文書を検索"""
        
        if self.json_enabled:
            return self._search_with_json(filters, limit, offset)
        else:
            return self._search_with_like(filters, limit, offset)
    
    def search_by_content(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[SearchResult]:
        """Full-text search using FTS5
        FTS5を使用した全文検索"""
        
        cursor = self.conn.execute(
            """SELECT d.id, d.content, d.metadata, bm25(documents_fts) as score
               FROM documents_fts 
               JOIN documents d ON documents_fts.id = d.id
               WHERE documents_fts MATCH ?
               ORDER BY bm25(documents_fts)
               LIMIT ? OFFSET ?""",
            (query, limit, offset)
        )
        
        results = []
        for row in cursor.fetchall():
            document = Document(
                id=row["id"],
                content=row["content"],
                metadata=json.loads(row["metadata"])
            )
            results.append(SearchResult(document=document, score=row["score"]))
        
        return results
    
    def get_documents_by_lineage(self, original_document_id: str) -> List[Document]:
        """Get all documents derived from original
        オリジナルから派生した全文書を取得"""
        
        if self.json_enabled:
            cursor = self.conn.execute(
                """SELECT id, content, metadata FROM documents 
                   WHERE json_extract(metadata, '$.original_document_id') = ?
                   OR id = ?
                   ORDER BY created_at""",
                (original_document_id, original_document_id)
            )
        else:
            cursor = self.conn.execute(
                """SELECT id, content, metadata FROM documents 
                   WHERE metadata LIKE ? OR id = ?
                   ORDER BY created_at""",
                (f'%"original_document_id":"{original_document_id}"%', original_document_id)
            )
        
        documents = []
        for row in cursor.fetchall():
            documents.append(Document(
                id=row["id"],
                content=row["content"],
                metadata=json.loads(row["metadata"])
            ))
        
        return documents
    
    def get_storage_stats(self) -> StorageStats:
        """Get storage statistics
        ストレージ統計を取得"""
        
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_docs,
                SUM(LENGTH(content)) as total_size,
                MIN(created_at) as oldest,
                MAX(created_at) as newest
            FROM documents
        """)
        row = cursor.fetchone()
        
        return StorageStats(
            total_documents=row["total_docs"],
            total_chunks=0,  # Will be implemented when chunks are added
            storage_size_bytes=row["total_size"] or 0,
            oldest_document=row["oldest"],
            newest_document=row["newest"]
        )
    
    def _search_with_json(self, filters: Dict[str, Any], limit: int, offset: int) -> List[SearchResult]:
        """Search using JSON1 extension
        JSON1拡張を使用した検索"""
        
        where_clauses = []
        params = []
        
        for key, value in filters.items():
            if isinstance(value, dict):
                if "$gte" in value:
                    where_clauses.append("CAST(json_extract(metadata, ?) AS REAL) >= ?")
                    params.extend([f"$.{key}", value["$gte"]])
                elif "$contains" in value:
                    where_clauses.append("json_extract(metadata, ?) LIKE ?")
                    params.extend([f"$.{key}", f"%{value['$contains']}%"])
                elif "$in" in value:
                    placeholders = ",".join("?" * len(value["$in"]))
                    where_clauses.append(f"json_extract(metadata, ?) IN ({placeholders})")
                    params.extend([f"$.{key}"] + value["$in"])
            else:
                where_clauses.append("json_extract(metadata, ?) = ?")
                params.extend([f"$.{key}", value])
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        cursor = self.conn.execute(
            f"""SELECT id, content, metadata FROM documents 
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?""",
            params + [limit, offset]
        )
        
        results = []
        for row in cursor.fetchall():
            document = Document(
                id=row["id"],
                content=row["content"],
                metadata=json.loads(row["metadata"])
            )
            results.append(SearchResult(document=document))
        
        return results
```

## 3. Loader

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

@dataclass
class LoadingConfig:
    """Configuration for document loading
    文書読み込み設定"""
    parallel: bool = True
    max_workers: Optional[int] = None  # None = CPU count
    use_multiprocessing: bool = False  # Thread vs Process
    chunk_size: int = 10  # Batch size for parallel processing
    timeout_per_file: Optional[float] = None  # Timeout per file in seconds
    skip_errors: bool = True  # Continue on individual file errors

@dataclass
class LoadingResult:
    """Result of loading operation
    読み込み操作の結果"""
    documents: List[Document]
    failed_paths: List[str]
    errors: List[Exception]
    total_time_seconds: float
    successful_count: int
    failed_count: int

class Loader(ABC):
    """Base interface for document loading
    文書読み込みの基底インターフェース"""
    
    def __init__(
        self, 
        metadata_generator: Optional[MetadataGenerator] = None,
        config: Optional[LoadingConfig] = None
    ):
        """Initialize loader with optional metadata generator and config
        オプションのメタデータジェネレータと設定でローダーを初期化"""
        self.metadata_generator = metadata_generator
        self.config = config or LoadingConfig()
    
    @abstractmethod
    def load_single(self, path: Union[str, Path]) -> Document:
        """Load a single document from path (must be implemented by subclasses)
        パスから単一の文書を読み込む（サブクラスで実装必須）"""
        pass
    
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Get list of supported file formats
        サポートされているファイル形式のリストを取得"""
        pass
    
    def load(self, path: Union[str, Path]) -> Document:
        """Load a single document from path
        パスから単一の文書を読み込む"""
        return self.load_single(path)
    
    def load_batch(
        self, 
        paths: List[Union[str, Path]], 
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> LoadingResult:
        """Load multiple documents with optional parallel processing
        並列処理オプション付きで複数の文書を読み込む
        
        Args:
            paths: List of file paths to load
            progress_callback: Optional callback function(completed, total)
        
        Returns:
            LoadingResult with documents and statistics
        """
        import time
        
        start_time = time.time()
        documents = []
        failed_paths = []
        errors = []
        
        if self.config.parallel and len(paths) > 1:
            result = self._load_parallel(paths, progress_callback)
        else:
            result = self._load_sequential(paths, progress_callback)
        
        result.total_time_seconds = time.time() - start_time
        return result
    
    async def load_batch_async(
        self,
        paths: List[Union[str, Path]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> LoadingResult:
        """Async version of batch loading
        バッチ読み込みの非同期版"""
        import time
        
        start_time = time.time()
        
        if self.config.parallel and len(paths) > 1:
            result = await self._load_async_parallel(paths, progress_callback)
        else:
            result = self._load_sequential(paths, progress_callback)
        
        result.total_time_seconds = time.time() - start_time
        return result
    
    def _load_sequential(
        self,
        paths: List[Union[str, Path]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> LoadingResult:
        """Sequential loading implementation
        順次読み込み実装"""
        documents = []
        failed_paths = []
        errors = []
        
        for i, path in enumerate(paths):
            try:
                doc = self.load_single(path)
                documents.append(doc)
                if progress_callback:
                    progress_callback(i + 1, len(paths))
            except Exception as e:
                failed_paths.append(str(path))
                errors.append(e)
                if not self.config.skip_errors:
                    raise
        
        return LoadingResult(
            documents=documents,
            failed_paths=failed_paths,
            errors=errors,
            total_time_seconds=0,  # Will be set by caller
            successful_count=len(documents),
            failed_count=len(failed_paths)
        )
    
    def _load_parallel(
        self,
        paths: List[Union[str, Path]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> LoadingResult:
        """Parallel loading using threads or processes
        スレッドまたはプロセスを使用した並列読み込み"""
        documents = []
        failed_paths = []
        errors = []
        completed = 0
        
        ExecutorClass = ProcessPoolExecutor if self.config.use_multiprocessing else ThreadPoolExecutor
        
        with ExecutorClass(max_workers=self.config.max_workers) as executor:
            # Submit all jobs
            future_to_path = {
                executor.submit(self._safe_load_single, path): path 
                for path in paths
            }
            
            # Collect results
            for future in future_to_path:
                path = future_to_path[future]
                try:
                    result = future.result(timeout=self.config.timeout_per_file)
                    if result:
                        documents.append(result)
                    else:
                        failed_paths.append(str(path))
                except Exception as e:
                    failed_paths.append(str(path))
                    errors.append(e)
                    if not self.config.skip_errors:
                        # Cancel remaining futures
                        for f in future_to_path:
                            f.cancel()
                        raise
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(paths))
        
        return LoadingResult(
            documents=documents,
            failed_paths=failed_paths,
            errors=errors,
            total_time_seconds=0,  # Will be set by caller
            successful_count=len(documents),
            failed_count=len(failed_paths)
        )
    
    async def _load_async_parallel(
        self,
        paths: List[Union[str, Path]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> LoadingResult:
        """Async parallel loading
        非同期並列読み込み"""
        documents = []
        failed_paths = []
        errors = []
        
        semaphore = asyncio.Semaphore(self.config.max_workers or 10)
        
        async def load_with_semaphore(path):
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._safe_load_single, path)
        
        # Create tasks
        tasks = [load_with_semaphore(path) for path in paths]
        
        # Execute with progress tracking
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await coro
                if result:
                    documents.append(result)
                else:
                    failed_paths.append(str(paths[i]))
            except Exception as e:
                failed_paths.append(str(paths[i]))
                errors.append(e)
                if not self.config.skip_errors:
                    # Cancel remaining tasks
                    for task in tasks:
                        task.cancel()
                    raise
            
            if progress_callback:
                progress_callback(i + 1, len(paths))
        
        return LoadingResult(
            documents=documents,
            failed_paths=failed_paths,
            errors=errors,
            total_time_seconds=0,  # Will be set by caller
            successful_count=len(documents),
            failed_count=len(failed_paths)
        )
    
    def _safe_load_single(self, path: Union[str, Path]) -> Optional[Document]:
        """Safe wrapper for load_single that handles errors
        エラーを処理するload_singleの安全なラッパー"""
        try:
            return self.load_single(path)
        except Exception:
            if self.config.skip_errors:
                return None
            raise
    
    def _generate_base_metadata(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Generate required metadata fields
        必須メタデータフィールドを生成"""
        path_obj = Path(path)
        stat = path_obj.stat()
        
        base_metadata = {
            "path": str(path),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "file_type": path_obj.suffix,
            "size_bytes": stat.st_size
        }
        
        # Generate additional metadata if generator is provided
        if self.metadata_generator:
            additional_metadata = self.metadata_generator.generate_metadata(base_metadata)
            base_metadata.update(additional_metadata)
        
        return base_metadata
    
    def _generate_document_id(self, path: Union[str, Path]) -> str:
        """Generate unique document ID from path
        パスから一意の文書IDを生成"""
        import hashlib
        return hashlib.md5(str(path).encode()).hexdigest()
```

### 汎用ローダー（デフォルト実装）

```python
class UniversalLoader(Loader):
    """Universal loader that delegates to specialized loaders based on file extension
    ファイル拡張子に基づいて専用ローダーに委譲する汎用ローダー"""
    
    def __init__(
        self,
        metadata_generator: Optional[MetadataGenerator] = None,
        config: Optional[LoadingConfig] = None,
        custom_loaders: Optional[Dict[str, Loader]] = None
    ):
        """Initialize universal loader with custom loader mappings
        カスタムローダーマッピングで汎用ローダーを初期化
        
        Args:
            metadata_generator: Optional metadata generator
            config: Loading configuration
            custom_loaders: Dict mapping file extensions to loader instances
        """
        super().__init__(metadata_generator, config)
        
        # Default loaders for common formats
        self._default_loaders = {
            '.txt': TextLoader(metadata_generator, config),
            '.md': MarkdownLoader(metadata_generator, config),
            '.pdf': PDFLoader(metadata_generator, config),
            '.docx': DocxLoader(metadata_generator, config),
            '.html': HTMLLoader(metadata_generator, config),
            '.json': JSONLoader(metadata_generator, config),
            '.csv': CSVLoader(metadata_generator, config),
        }
        
        # Custom loaders override defaults
        if custom_loaders:
            self._default_loaders.update(custom_loaders)
        
        # Available loaders (can be extended by packages)
        self._available_loaders = {}
        self._register_available_loaders()
    
    def register_loader(self, extension: str, loader_class: type):
        """Register a new loader for a file extension
        ファイル拡張子用の新しいローダーを登録
        
        Args:
            extension: File extension (e.g., '.xml')
            loader_class: Loader class to use for this extension
        """
        self._available_loaders[extension] = loader_class
        
        # Create instance if not already present
        if extension not in self._default_loaders:
            self._default_loaders[extension] = loader_class(
                self.metadata_generator, 
                self.config
            )
    
    def _register_available_loaders(self):
        """Register loaders from available packages
        利用可能なパッケージからローダーを登録"""
        
        # Try to import and register docling loader
        try:
            from refinire_rag.loaders.docling import DoclingLoader
            self.register_loader('.pdf', DoclingLoader)
            self.register_loader('.docx', DoclingLoader)
            self.register_loader('.pptx', DoclingLoader)
        except ImportError:
            pass  # Docling not available
        
        # Try to import and register unstructured loader
        try:
            from refinire_rag.loaders.unstructured import UnstructuredLoader
            self.register_loader('.pdf', UnstructuredLoader)
            self.register_loader('.docx', UnstructuredLoader)
        except ImportError:
            pass  # Unstructured not available
        
        # Try to import other specialized loaders
        try:
            from refinire_rag.loaders.excel import ExcelLoader
            self.register_loader('.xlsx', ExcelLoader)
            self.register_loader('.xls', ExcelLoader)
        except ImportError:
            pass
    
    def load_single(self, path: Union[str, Path]) -> Document:
        """Load single document using appropriate loader
        適切なローダーを使用して単一文書を読み込む"""
        path_obj = Path(path)
        extension = path_obj.suffix.lower()
        
        # Find appropriate loader
        loader = self._default_loaders.get(extension)
        if not loader:
            raise ValueError(f"No loader available for file type: {extension}")
        
        # Load document
        document = loader.load_single(path)
        
        # Add universal loader metadata
        document.metadata.update({
            "loader_used": loader.__class__.__name__,
            "loader_type": "universal"
        })
        
        return document
    
    def supported_formats(self) -> List[str]:
        """Get all supported formats from registered loaders
        登録されたローダーからすべてのサポート形式を取得"""
        return list(self._default_loaders.keys())
    
    def get_loader_for_extension(self, extension: str) -> Optional[Loader]:
        """Get the loader that will be used for a given extension
        指定された拡張子に使用されるローダーを取得"""
        return self._default_loaders.get(extension.lower())
    
    def list_available_loaders(self) -> Dict[str, str]:
        """List all available loaders and their extensions
        利用可能なすべてのローダーとその拡張子をリスト表示"""
        return {
            ext: loader.__class__.__name__ 
            for ext, loader in self._default_loaders.items()
        }

### 専用ローダーの基底クラス

class SpecializedLoader(Loader):
    """Base class for specialized loaders
    専用ローダーの基底クラス"""
    
    def __init__(
        self,
        metadata_generator: Optional[MetadataGenerator] = None,
        config: Optional[LoadingConfig] = None
    ):
        super().__init__(metadata_generator, config)
    
    def _extract_content(self, path: Union[str, Path]) -> str:
        """Extract text content from file (must be implemented by subclasses)
        ファイルからテキストコンテンツを抽出（サブクラスで実装必須）"""
        raise NotImplementedError("Subclasses must implement _extract_content")
    
    def load_single(self, path: Union[str, Path]) -> Document:
        """Standard implementation using _extract_content
        _extract_contentを使用した標準実装"""
        
        # Extract content using specialized method
        content = self._extract_content(path)
        
        # Generate metadata
        metadata = self._generate_base_metadata(path)
        
        # Generate document ID
        document_id = self._generate_document_id(path)
        
        # Add loader-specific metadata
        metadata.update({
            "loader_used": self.__class__.__name__,
            "content_length": len(content)
        })
        
        return Document(
            id=document_id,
            content=content,
            metadata=metadata
        )

### 具体的なローダー実装例

class TextLoader(SpecializedLoader):
    """Loader for plain text files
    プレーンテキストファイル用ローダー"""
    
    def _extract_content(self, path: Union[str, Path]) -> str:
        """Extract content from text file
        テキストファイルからコンテンツを抽出"""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def supported_formats(self) -> List[str]:
        return ['.txt']

class MarkdownLoader(SpecializedLoader):
    """Loader for Markdown files
    Markdownファイル用ローダー"""
    
    def _extract_content(self, path: Union[str, Path]) -> str:
        """Extract content from Markdown file
        Markdownファイルからコンテンツを抽出"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add markdown-specific metadata extraction here
        return content
    
    def supported_formats(self) -> List[str]:
        return ['.md', '.markdown']

class PDFLoader(SpecializedLoader):
    """Basic PDF loader using PyPDF2
    PyPDF2を使用した基本PDFローダー"""
    
    def _extract_content(self, path: Union[str, Path]) -> str:
        """Extract text from PDF file
        PDFファイルからテキストを抽出"""
        try:
            import PyPDF2
            
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except ImportError:
            raise ImportError("PyPDF2 required for PDF loading: pip install PyPDF2")
    
    def supported_formats(self) -> List[str]:
        return ['.pdf']

### 拡張ローダーの例（サブパッケージで提供）

class DoclingLoader(SpecializedLoader):
    """Advanced document loader using Docling
    Doclingを使用した高度な文書ローダー"""
    
    def __init__(
        self,
        metadata_generator: Optional[MetadataGenerator] = None,
        config: Optional[LoadingConfig] = None,
        docling_config: Optional[Dict] = None
    ):
        super().__init__(metadata_generator, config)
        self.docling_config = docling_config or {}
    
    def _extract_content(self, path: Union[str, Path]) -> str:
        """Extract content using Docling
        Doclingを使用してコンテンツを抽出"""
        try:
            # This would use the actual Docling library
            # from docling import DocumentLoader
            # loader = DocumentLoader(**self.docling_config)
            # result = loader.load(path)
            # return result.text
            
            # Placeholder implementation
            return f"Content extracted from {path} using Docling"
        except ImportError:
            raise ImportError("Docling package required: pip install docling")
    
    def load_single(self, path: Union[str, Path]) -> Document:
        """Enhanced loading with Docling-specific features
        Docling固有機能を含む拡張読み込み"""
        
        # Extract content and metadata using Docling
        content = self._extract_content(path)
        metadata = self._generate_base_metadata(path)
        
        # Add Docling-specific metadata
        metadata.update({
            "loader_used": "DoclingLoader",
            "extraction_method": "docling_advanced",
            "supports_tables": True,
            "supports_images": True,
            "layout_analysis": True
        })
        
        document_id = self._generate_document_id(path)
        
        return Document(
            id=document_id,
            content=content,
            metadata=metadata
        )
    
    def supported_formats(self) -> List[str]:
        return ['.pdf', '.docx', '.pptx', '.html']
```

### 使用例

```python
class DefaultLoader(Loader):
    """Default implementation supporting common formats
    一般的な形式をサポートするデフォルト実装"""
    
    def __init__(self, metadata_generator: Optional[MetadataGenerator] = None):
        super().__init__(metadata_generator)
        self._parsers = {
            '.pdf': PDFParser(),
            '.docx': DocxParser(),
            '.md': MarkdownParser(),
            '.txt': TextParser()
        }
    
    def load(self, path: Union[str, Path]) -> Document:
        """Load a single document from path
        パスから単一の文書を読み込む"""
        path_obj = Path(path)
        
        # Generate metadata (including additional metadata from generator)
        metadata = self._generate_base_metadata(path)
        
        # Load content using appropriate parser
        parser = self._parsers.get(path_obj.suffix)
        if not parser:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")
        
        content = parser.parse(path)
        
        # Generate document ID
        document_id = self._generate_document_id(path)
        
        return Document(
            id=document_id,
            content=content,
            metadata=metadata
        )
    
    def load_batch(self, paths: List[Union[str, Path]]) -> List[Document]:
        """Load multiple documents in batch
        複数の文書をバッチで読み込む"""
        return [self.load(path) for path in paths]
    
    def supported_formats(self) -> List[str]:
        return list(self._parsers.keys())
    
    def _generate_document_id(self, path: Union[str, Path]) -> str:
        """Generate unique document ID from path
        パスから一意の文書IDを生成"""
        import hashlib
        return hashlib.md5(str(path).encode()).hexdigest()
```

### 使用例

```python
# パスベースのメタデータ生成ルールを定義
path_rules = {
    "/docs/public/*": {
        "access_group": "public",
        "classification": "open",
        "department": "general"
    },
    "/docs/internal/*": {
        "access_group": "employees", 
        "classification": "internal",
        "department": "company"
    },
    "/docs/confidential/*": {
        "access_group": "managers",
        "classification": "confidential", 
        "department": "executive"
    },
    "/docs/engineering/*": {
        "access_group": "engineers",
        "classification": "technical",
        "department": "engineering",
        "tags": ["technical", "engineering"]
    }
}

# MetadataGeneratorを作成
metadata_gen = PathBasedMetadataGenerator(path_rules)

# LoaderにMetadataGeneratorを注入
loader = DefaultLoader(metadata_generator=metadata_gen)

# 文書を読み込み（自動的にメタデータが付与される）
doc = loader.load("/docs/engineering/api_spec.pdf")

# 生成されたメタデータ例
print(doc.metadata)
# {
#     "path": "/docs/engineering/api_spec.pdf",
#     "created_at": "2024-01-15T10:30:00",
#     "file_type": ".pdf", 
#     "size_bytes": 1024000,
#     "access_group": "engineers",
#     "classification": "technical",
#     "department": "engineering",
#     "tags": ["technical", "engineering"],
#     "filename": "api_spec.pdf",
#     "directory": "/docs/engineering",
#     "folder_name": "engineering",
#     "document_type": "pdf_document"
# }
```

## 2. DictionaryMaker

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Dictionary:
    """Dictionary data model
    辞書データモデル"""
    terms: List[str]
    abbreviations: Dict[str, str]  # abbr -> full form
    synonyms: Dict[str, List[str]]  # term -> synonyms
    definitions: Dict[str, str]  # term -> definition

@dataclass 
class DictionaryConfig:
    """Configuration for dictionary generation
    辞書生成の設定"""
    min_term_frequency: int = 2
    extract_abbreviations: bool = True
    extract_synonyms: bool = True
    language: str = "ja"

class DictionaryMaker(DocumentProcessor):
    """Interface for dictionary generation
    辞書生成のインターフェース"""
    
    @abstractmethod
    def process(self, document: Document, config: Optional[DictionaryConfig] = None) -> List[Document]:
        """Generate dictionary document from input document
        入力文書から辞書文書を生成
        
        Returns:
            List containing a single Document with dictionary content, with metadata:
            - original_document_id: ID of the source document
            - processing_stage: "dictionary"
            - dictionary_type: "terms" | "abbreviations" | "full"
            - term_count: Number of extracted terms
            - language: Language of the dictionary
        """
        pass
    
    def extract_terms(self, document: Document) -> Dictionary:
        """Extract terms from a document (utility method)
        文書から用語を抽出（ユーティリティメソッド）"""
        pass
    
    def merge_dictionaries(self, dictionaries: List[Dictionary]) -> Dictionary:
        """Merge multiple dictionaries (utility method)
        複数の辞書を統合（ユーティリティメソッド）"""
        pass
```

## 3. Normalizer

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class NormalizationResult:
    """Result of normalization (utility data)
    正規化の結果（ユーティリティデータ）"""
    normalized_content: str
    replacements_made: Dict[str, int]  # term -> count
    tags_added: int

@dataclass
class NormalizationConfig:
    """Configuration for text normalization
    テキスト正規化の設定"""
    use_dictionary: bool = True
    expand_abbreviations: bool = True
    add_semantic_tags: bool = True
    preserve_formatting: bool = False
    language: str = "ja"

class Normalizer(DocumentProcessor):
    """Interface for text normalization
    テキスト正規化のインターフェース"""
    
    @abstractmethod
    def process(
        self, 
        document: Document, 
        config: Optional[NormalizationConfig] = None
    ) -> List[Document]:
        """Normalize document content
        文書内容を正規化
        
        Returns:
            List containing a single normalized Document with metadata:
            - original_document_id: ID of the original document
            - parent_document_id: ID of the immediate parent
            - processing_stage: "normalized"
            - replacements_made: Number of term replacements
            - tags_added: Number of semantic tags added
            - normalization_rules: Rules applied during normalization
        """
        pass
    
    def set_dictionary(self, dictionary: Dictionary) -> None:
        """Set dictionary for normalization
        正規化用の辞書を設定"""
        pass
    
    def set_rules(self, rules: Dict[str, Any]) -> None:
        """Set normalization rules
        正規化ルールを設定"""
        pass
```

## 4. GraphBuilder

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class Node:
    """Graph node
    グラフノード"""
    id: str
    label: str
    type: str  # "entity", "concept", etc.
    properties: Dict[str, Any]

@dataclass
class Edge:
    """Graph edge
    グラフエッジ"""
    source_id: str
    target_id: str
    relationship: str
    weight: float

@dataclass
class Graph:
    """Knowledge graph
    知識グラフ"""
    nodes: List[Node]
    edges: List[Edge]
    metadata: Dict[str, Any]

@dataclass
class GraphConfig:
    """Configuration for graph building
    グラフ構築の設定"""
    extract_entities: bool = True
    extract_relationships: bool = True
    min_entity_confidence: float = 0.7
    language: str = "ja"
    graph_format: str = "networkx"  # "networkx", "json", "rdf"

class GraphBuilder(DocumentProcessor):
    """Interface for knowledge graph construction
    知識グラフ構築のインターフェース"""
    
    @abstractmethod
    def process(self, document: Document, config: Optional[GraphConfig] = None) -> List[Document]:
        """Build knowledge graph document from input document
        入力文書から知識グラフ文書を構築
        
        Returns:
            List containing a single Document with graph content, with metadata:
            - original_document_id: ID of the source document
            - processing_stage: "graph"
            - graph_format: Format of the graph data
            - node_count: Number of nodes in graph
            - edge_count: Number of edges in graph
            - extraction_confidence: Average confidence of extractions
        """
        pass
    
    def build_graph(self, document: Document) -> Graph:
        """Build knowledge graph from document (utility method)
        文書から知識グラフを構築（ユーティリティメソッド）"""
        pass
    
    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract subject-predicate-object triples (utility method)
        主語-述語-目的語の三つ組を抽出（ユーティリティメソッド）"""
        pass
    
    def merge_graphs(self, graphs: List[Graph]) -> Graph:
        """Merge multiple graphs (utility method)
        複数のグラフを統合（ユーティリティメソッド）"""
        pass
```

## 5. Chunker

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

# DocumentProcessor base class is defined above

@dataclass
class ChunkingConfig:
    """Configuration for chunking
    チャンキングの設定"""
    chunk_size: int = 512
    overlap: int = 50
    split_by_sentence: bool = True
    preserve_formatting: bool = False
    min_chunk_size: int = 50
    max_chunk_size: int = 1024

class Chunker(DocumentProcessor):
    """Interface for document chunking
    文書チャンキングのインターフェース"""
    
    @abstractmethod
    def process(
        self, 
        document: Document, 
        config: Optional[ChunkingConfig] = None
    ) -> List[Document]:
        """Split document into chunk documents
        文書をチャンク文書に分割
        
        Returns:
            List of Document objects representing chunks, with metadata:
            - original_document_id: ID of the original document
            - parent_document_id: ID of the immediate parent
            - processing_stage: "chunked"
            - chunk_position: Position in sequence (0-based)
            - chunk_total: Total number of chunks
            - token_count: Number of tokens in this chunk
            - start_char: Start character position in parent
            - end_char: End character position in parent
            - overlap_previous: Token overlap with previous chunk
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text
        テキストのトークン数を推定"""
        pass
```

## 6. Embedder

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class Vector:
    """Vector representation
    ベクトル表現"""
    document_id: str
    embedding: np.ndarray
    model_name: str
    dimension: int

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation
    埋め込み生成の設定"""
    model_name: str = "text-embedding-3-small"
    batch_size: int = 100
    max_retries: int = 3
    timeout_seconds: float = 30.0
    normalize_embeddings: bool = True

class Embedder(ABC):
    """Interface for text embedding
    テキスト埋め込みのインターフェース"""
    
    @abstractmethod
    def embed(self, text: Union[str, Document]) -> Vector:
        """Embed single text or document
        単一のテキストまたは文書を埋め込む"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[Union[str, Document]]) -> List[Vector]:
        """Embed multiple texts in batch
        複数のテキストをバッチで埋め込む"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension
        埋め込みの次元数を取得"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name
        モデル名を取得"""
        pass

### 実装例（OpenAI Embeddings）

```python
import openai
import numpy as np
from typing import List, Union, Optional
from dataclasses import dataclass
import time

class OpenAIEmbedder(Embedder):
    """OpenAI Embeddings implementation
    OpenAI Embeddingsの実装"""
    
    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        api_key: Optional[str] = None
    ):
        """Initialize OpenAI embedder
        OpenAI embedderを初期化
        
        Args:
            config: Embedding configuration
            api_key: OpenAI API key (if not set in environment)
        """
        self.config = config or EmbeddingConfig()
        
        # Initialize OpenAI client
        if api_key:
            openai.api_key = api_key
        
        self.client = openai.OpenAI()
        
        # Cache model dimensions
        self._dimension_cache = {}
    
    def embed(self, text: Union[str, Document]) -> Vector:
        """Embed single text or document using OpenAI
        OpenAIを使用して単一のテキストまたは文書を埋め込む"""
        
        # Extract text content
        if isinstance(text, Document):
            content = text.content
            doc_id = text.id
        else:
            content = text
            doc_id = f"text_{hash(text)}"
        
        # Call OpenAI API with retry logic
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.config.model_name,
                    input=content,
                    timeout=self.config.timeout_seconds
                )
                
                embedding = np.array(response.data[0].embedding)
                
                # Normalize if requested
                if self.config.normalize_embeddings:
                    embedding = embedding / np.linalg.norm(embedding)
                
                return Vector(
                    document_id=doc_id,
                    embedding=embedding,
                    model_name=self.config.model_name,
                    dimension=len(embedding)
                )
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise EmbeddingError(f"Failed to generate embedding after {self.config.max_retries} attempts: {e}")
    
    def embed_batch(self, texts: List[Union[str, Document]]) -> List[Vector]:
        """Embed multiple texts in batch using OpenAI
        OpenAIを使用して複数のテキストをバッチで埋め込む"""
        
        vectors = []
        
        # Process in batches to respect API limits
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            # Extract content and IDs
            batch_content = []
            batch_ids = []
            
            for item in batch:
                if isinstance(item, Document):
                    batch_content.append(item.content)
                    batch_ids.append(item.id)
                else:
                    batch_content.append(item)
                    batch_ids.append(f"text_{hash(item)}")
            
            # Call OpenAI API with retry logic
            for attempt in range(self.config.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.config.model_name,
                        input=batch_content,
                        timeout=self.config.timeout_seconds
                    )
                    
                    # Process response
                    for j, embedding_data in enumerate(response.data):
                        embedding = np.array(embedding_data.embedding)
                        
                        # Normalize if requested
                        if self.config.normalize_embeddings:
                            embedding = embedding / np.linalg.norm(embedding)
                        
                        vectors.append(Vector(
                            document_id=batch_ids[j],
                            embedding=embedding,
                            model_name=self.config.model_name,
                            dimension=len(embedding)
                        ))
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < self.config.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    raise EmbeddingError(f"Failed to generate batch embeddings after {self.config.max_retries} attempts: {e}")
        
        return vectors
    
    def get_dimension(self) -> int:
        """Get embedding dimension for the current model
        現在のモデルの埋め込み次元を取得"""
        
        if self.config.model_name in self._dimension_cache:
            return self._dimension_cache[self.config.model_name]
        
        # Known dimensions for OpenAI models
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        if self.config.model_name in model_dimensions:
            dimension = model_dimensions[self.config.model_name]
            self._dimension_cache[self.config.model_name] = dimension
            return dimension
        
        # For unknown models, make a test call
        try:
            test_response = self.client.embeddings.create(
                model=self.config.model_name,
                input="test"
            )
            dimension = len(test_response.data[0].embedding)
            self._dimension_cache[self.config.model_name] = dimension
            return dimension
        except Exception as e:
            raise EmbeddingError(f"Cannot determine dimension for model {self.config.model_name}: {e}")
    
    def get_model_name(self) -> str:
        """Get current model name
        現在のモデル名を取得"""
        return self.config.model_name

### 使用例

```python
# OpenAI Embedder の使用例
config = EmbeddingConfig(
    model_name="text-embedding-3-small",
    batch_size=50,
    normalize_embeddings=True
)

embedder = OpenAIEmbedder(config=config)

# 単一文書の埋め込み
document = Document(
    id="doc_001",
    content="RAGシステムについての技術文書です。",
    metadata={"type": "technical"}
)

vector = embedder.embed(document)
print(f"Embedding dimension: {vector.dimension}")
print(f"Model used: {vector.model_name}")

# バッチ埋め込み
documents = [
    Document(id="doc_001", content="技術文書1", metadata={}),
    Document(id="doc_002", content="技術文書2", metadata={}),
    Document(id="doc_003", content="技術文書3", metadata={})
]

vectors = embedder.embed_batch(documents)
print(f"Generated {len(vectors)} embeddings")

# テキストからの直接埋め込み
text_vector = embedder.embed("テスト用のテキスト")
```
```

## 7. StoreAdapter

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class QueryResult:
    """Result of vector search
    ベクトル検索の結果"""
    document_id: str
    score: float
    metadata: Dict[str, Any]

class StoreAdapter(ABC):
    """Interface for vector storage
    ベクトルストレージのインターフェース"""
    
    @abstractmethod
    def upsert(self, vectors: List[Vector], documents: List[Document]) -> None:
        """Insert or update vectors and documents
        ベクトルと文書を挿入または更新"""
        pass
    
    @abstractmethod
    def query(
        self, 
        query_vector: np.ndarray, 
        k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """Search similar vectors
        類似ベクトルを検索"""
        pass
    
    @abstractmethod
    def delete(self, document_ids: List[str]) -> None:
        """Delete vectors by document IDs
        文書IDでベクトルを削除"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics
        ストレージの統計情報を取得"""
        pass
```

## 8. Retriever

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    """Result of retrieval
    検索結果"""
    documents: List[Document]
    scores: List[float]
    query_time_ms: float

class Retriever(ABC):
    """Interface for document retrieval
    文書検索のインターフェース"""
    
    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Retrieve relevant documents for query
        クエリに関連する文書を検索"""
        pass
    
    @abstractmethod
    def set_embedder(self, embedder: Embedder) -> None:
        """Set embedder for query encoding
        クエリエンコーディング用のEmbedderを設定"""
        pass
    
    @abstractmethod
    def set_store(self, store: StoreAdapter) -> None:
        """Set vector store
        ベクトルストアを設定"""
        pass
```

## 9. Reranker

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class RerankResult:
    """Result of reranking
    再ランキングの結果"""
    documents: List[Document]
    scores: List[float]
    rerank_time_ms: float

class Reranker(ABC):
    """Interface for result reranking
    結果再ランキングのインターフェース"""
    
    @abstractmethod
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        initial_scores: List[float],
        k: Optional[int] = None
    ) -> RerankResult:
        """Rerank documents based on query
        クエリに基づいて文書を再ランキング"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get reranker model name
        再ランキングモデル名を取得"""
        pass
```

## 10. Reader

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Answer:
    """Generated answer
    生成された回答"""
    text: str
    confidence: float
    sources: List[str]  # document IDs used
    reasoning: Optional[str] = None

class Reader(ABC):
    """Interface for answer generation
    回答生成のインターフェース"""
    
    @abstractmethod
    def read(
        self, 
        query: str, 
        documents: List[Document],
        context: Optional[Dict[str, Any]] = None
    ) -> Answer:
        """Generate answer from query and documents
        クエリと文書から回答を生成"""
        pass
    
    @abstractmethod
    def set_prompt_template(self, template: str) -> None:
        """Set prompt template for answer generation
        回答生成用のプロンプトテンプレートを設定"""
        pass
```

## 11. TestSuite

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

@dataclass
class TestCase:
    """Test case for evaluation
    評価用テストケース"""
    case_id: str
    query: str
    expected_answer: Optional[str]
    relevant_documents: List[str]
    metadata: Dict[str, Any]

@dataclass
class TestResult:
    """Result of test execution
    テスト実行の結果"""
    case_id: str
    actual_answer: str
    expected_answer: Optional[str]
    metrics: Dict[str, float]
    execution_time_ms: float
    success: bool

class TestSuite(ABC):
    """Interface for test execution
    テスト実行のインターフェース"""
    
    @abstractmethod
    def load_test_set(self, path: str) -> List[TestCase]:
        """Load test set from file
        ファイルからテストセットを読み込む"""
        pass
    
    @abstractmethod
    def run_tests(
        self, 
        test_cases: List[TestCase],
        pipeline: Callable[[str], Answer]
    ) -> List[TestResult]:
        """Run tests with given pipeline
        指定されたパイプラインでテストを実行"""
        pass
```

## 12. Evaluator

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Metrics:
    """Evaluation metrics
    評価メトリクス"""
    precision: float
    recall: float
    f1_score: float
    avg_latency_ms: float
    coherence_score: float
    relevance_score: float

class Evaluator(ABC):
    """Interface for metrics evaluation
    メトリクス評価のインターフェース"""
    
    @abstractmethod
    def evaluate(self, test_results: List[TestResult]) -> Metrics:
        """Calculate metrics from test results
        テスト結果からメトリクスを計算"""
        pass
    
    @abstractmethod
    def calculate_precision_recall(
        self, 
        predicted: List[str], 
        expected: List[str]
    ) -> Tuple[float, float]:
        """Calculate precision and recall
        精度と再現率を計算"""
        pass
```

## 13. ContradictionDetector

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Claim:
    """Extracted claim
    抽出されたクレーム"""
    text: str
    source_document_id: str
    confidence: float

@dataclass
class Contradiction:
    """Detected contradiction
    検出された矛盾"""
    claim1: Claim
    claim2: Claim
    contradiction_score: float
    type: str  # "direct", "implicit", etc.

class ContradictionDetector(ABC):
    """Interface for contradiction detection
    矛盾検出のインターフェース"""
    
    @abstractmethod
    def extract_claims(self, documents: List[Document]) -> List[Claim]:
        """Extract claims from documents
        文書からクレームを抽出"""
        pass
    
    @abstractmethod
    def detect_contradictions(
        self, 
        claims: List[Claim]
    ) -> List[Contradiction]:
        """Detect contradictions between claims
        クレーム間の矛盾を検出"""
        pass
```

## 14. InsightReporter

### インターフェース定義

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Insight:
    """Analysis insight
    分析の洞察"""
    category: str
    finding: str
    severity: str  # "low", "medium", "high"
    evidence: Dict[str, Any]
    recommendation: str

@dataclass
class Report:
    """Analysis report
    分析レポート"""
    report_id: str
    generated_at: datetime
    summary: str
    insights: List[Insight]
    visualizations: Dict[str, Any]

class InsightReporter(ABC):
    """Interface for insight reporting
    洞察レポート生成のインターフェース"""
    
    @abstractmethod
    def analyze_metrics(
        self, 
        metrics: Metrics,
        thresholds: Dict[str, float]
    ) -> List[Insight]:
        """Analyze metrics and generate insights
        メトリクスを分析し洞察を生成"""
        pass
    
    @abstractmethod
    def generate_report(
        self,
        insights: List[Insight],
        metrics: Metrics,
        conflicts: Optional[List[Contradiction]] = None
    ) -> Report:
        """Generate comprehensive report
        包括的なレポートを生成"""
        pass
    
    @abstractmethod
    def export_report(self, report: Report, format: str, path: str) -> None:
        """Export report in specified format
        指定された形式でレポートをエクスポート"""
        pass
```

## メタデータ仕様

Documentクラスのmetadataは完全に自由な辞書型です。必須フィールドのみが定義され、それ以外は任意のキー・値を設定できます。

### 必須フィールド（Loaderが設定）

```python
# これらのフィールドのみが必須
required_metadata = {
    "path": str,           # ファイルパス
    "created_at": str,     # ISO 8601形式の日時文字列
    "file_type": str,      # ファイル拡張子
    "size_bytes": int,     # ファイルサイズ（バイト単位）
}
```

### 推奨フィールド例（任意）

```python
# 以下は推奨例ですが、完全に任意です
optional_metadata_examples = {
    # Loaderが設定可能
    "filename": "document.pdf",
    "modified_at": "2024-01-15T11:00:00Z",
    "title": "技術仕様書",
    "author": "開発チーム",
    
    # 処理システムが設定
    "original_document_id": "doc_001",
    "parent_document_id": "doc_001_normalized",
    
    # ユーザーが自由に設定
    "dataset_name": "technical_docs",
    "tags": ["重要", "API", "v2.0"],
    "department": "エンジニアリング",
    "project": "RAGシステム",
    "priority": "high",
    "review_status": "approved",
    "expiry_date": "2025-12-31",
    
    # 完全にカスタム
    "任意のキー": "任意の値",
    "custom_score": 95.5,
    "related_docs": ["doc_002", "doc_003"],
    "metadata_version": "1.2"
}
```

### 使用例：完全に自由なメタデータ

```python
# Loaderがオリジナル文書を作成（必須フィールドのみ）
original_doc = Document(
    id="doc_001",
    content="RAGシステムは...",
    metadata={
        # 必須フィールド（Loaderが設定）
        "path": "/docs/rag_guide.pdf",
        "created_at": "2024-01-15T10:30:00Z",
        "file_type": ".pdf",
        "size_bytes": 1024000,
        
        # 以下は完全に自由（ユーザーやシステムが任意に設定）
        "dataset": "rag_docs_v2",
        "stage": "original",
        "category": "技術文書",
        "importance": "high",
        "team": "AI開発",
        "version": "1.0"
    }
)

# ユーザーが独自のメタデータを追加
user_doc = Document(
    id="doc_002", 
    content="...",
    metadata={
        # 必須フィールド
        "path": "/docs/manual.md",
        "created_at": "2024-01-15T11:00:00Z", 
        "file_type": ".md",
        "size_bytes": 512000,
        
        # ユーザー独自のメタデータ（完全に自由）
        "プロジェクト名": "新機能開発",
        "承認者": "田中部長", 
        "期限": "2024-03-31",
        "状態": "レビュー中",
        "関連文書": ["doc_001", "doc_003"],
        "重要度": 8.5,
        "キーワード": ["API", "認証", "セキュリティ"],
        "社内分類": "機密",
        "custom_field_1": "any_value",
        "custom_field_2": {"nested": "object", "allowed": True}
    }
)

# 処理システムが任意のメタデータを追加
processed_doc = Document(
    id="doc_001_processed",
    content="正規化済みコンテンツ...",
    metadata={
        # 必須フィールド
        "path": "/docs/rag_guide.pdf",
        "created_at": "2024-01-15T10:35:00Z",
        "file_type": ".pdf", 
        "size_bytes": 1024000,
        
        # 処理システムが独自に設定
        "source_doc": "doc_001",
        "processing": ["normalized", "dictionary_applied"],
        "processor_version": "v2.1.0",
        "quality_score": 0.95,
        "extracted_entities": ["RAG", "AI", "機械学習"],
        
        # 元のメタデータを継承
        "dataset": "rag_docs_v2",
        "category": "技術文書",
        "importance": "high"
    }
)
```

### 柔軟なメタデータ検索の例

```python
# 任意のフィールドで検索可能
filters = {
    "importance": "high",
    "category": "技術文書",
    "stage": "original"
}

# 日本語キーでも検索可能
filters = {
    "プロジェクト名": "新機能開発",
    "状態": "レビュー中",
    "重要度": {"$gte": 8.0}
}

# リスト型フィールドの検索
filters = {
    "processing": {"$contains": "normalized"},
    "キーワード": {"$contains": "API"}
}

# 複雑な条件の組み合わせ
filters = {
    "file_type": ".pdf",
    "created_at": {"$gte": "2024-01-01"},
    "size_bytes": {"$gte": 1000000},
    "dataset": "rag_docs_v2",
    "importance": {"$in": ["high", "critical"]}
}

# ネストされたオブジェクトの検索（ベクトルDBがサポートしている場合）
filters = {
    "custom_field_2.nested": "object",
    "custom_field_2.allowed": True
}

# 数値範囲検索
filters = {
    "quality_score": {"$gte": 0.9},
    "重要度": {"$between": [7.0, 10.0]}
}

# 存在チェック
filters = {
    "承認者": {"$exists": True},
    "source_doc": {"$exists": True}
}

# ユーザー独自の検索ロジック
user_filters = {
    "team": "AI開発",
    "期限": {"$lte": "2024-12-31"},
    "社内分類": {"$ne": "機密"}
}

results = store_adapter.query(
    query_vector=vector,
    k=10,
    filters=user_filters
)
```

## 実装ガイドライン

### 1. エラーハンドリング

各実装は適切な例外処理を行い、以下の基底例外クラスを継承した例外を発生させる：

```python
class RefinireRAGError(Exception):
    """Base exception for refinire-rag"""
    pass

class LoaderError(RefinireRAGError):
    """Error in document loading"""
    pass

class EmbeddingError(RefinireRAGError):
    """Error in embedding generation"""
    pass

class StorageError(RefinireRAGError):
    """Error in vector storage operations"""
    pass
```

### 2. ロギング

各実装は適切なロギングを行う：

```python
import logging

logger = logging.getLogger(__name__)

class SomeImplementation(SomeInterface):
    def some_method(self):
        logger.info("Starting operation...")
        try:
            # implementation
            logger.debug("Operation details...")
        except Exception as e:
            logger.error(f"Operation failed: {e}")
            raise
```

### 3. 設定管理

各実装は設定可能なパラメータを持つ：

```python
from pydantic import BaseModel

class SomeConfig(BaseModel):
    param1: str = "default"
    param2: int = 100
    
class SomeImplementation(SomeInterface):
    def __init__(self, config: SomeConfig):
        self.config = config
```