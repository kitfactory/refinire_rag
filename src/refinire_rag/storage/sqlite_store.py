"""
SQLite-based DocumentStore implementation
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

from .document_store import DocumentStore, SearchResult, StorageStats
from ..models.document import Document
from ..exceptions import StorageError

logger = logging.getLogger(__name__)


class SQLiteDocumentStore(DocumentStore):
    """SQLite-based document storage implementation"""
    
    def __init__(self, **kwargs):
        """Initialize SQLite document store
        
        Args:
            **kwargs: Configuration parameters, environment variables used as fallback
                     設定パラメータ、環境変数をフォールバックとして使用
                db_path (str): Path to SQLite database file
                              SQLiteデータベースファイルのパス
                timeout (float): Database timeout in seconds
                               データベースタイムアウト（秒）
                enable_wal (bool): Enable WAL mode for better performance
                                 パフォーマンス向上のWALモードを有効にするか
                auto_vacuum (bool): Enable automatic database vacuuming
                                  自動データベースバキュームを有効にするか
        """
        # Environment variable support with priority: kwargs > env vars > defaults
        import os
        
        db_path = kwargs.get('db_path', os.getenv('REFINIRE_RAG_SQLITE_PATH', './data/documents.db'))
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Additional configuration parameters
        self.timeout = float(kwargs.get('timeout', os.getenv('REFINIRE_RAG_SQLITE_TIMEOUT', '30.0')))
        self.enable_wal = kwargs.get('enable_wal', os.getenv('REFINIRE_RAG_SQLITE_ENABLE_WAL', 'true').lower() == 'true')
        self.auto_vacuum = kwargs.get('auto_vacuum', os.getenv('REFINIRE_RAG_SQLITE_AUTO_VACUUM', 'false').lower() == 'true')
        self.page_size = int(kwargs.get('page_size', os.getenv('REFINIRE_RAG_SQLITE_PAGE_SIZE', '4096')))
        self.cache_size = int(kwargs.get('cache_size', os.getenv('REFINIRE_RAG_SQLITE_CACHE_SIZE', '2000')))  # Pages
        self.synchronous = kwargs.get('synchronous', os.getenv('REFINIRE_RAG_SQLITE_SYNCHRONOUS', 'NORMAL'))
        self.journal_mode = kwargs.get('journal_mode', os.getenv('REFINIRE_RAG_SQLITE_JOURNAL_MODE', 'WAL' if self.enable_wal else 'DELETE'))
        
        # Configure connection with timeout and other settings
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=self.timeout)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        # Apply database configuration
        self._configure_database()
        
        # Check for JSON1 extension
        try:
            self.conn.execute("SELECT json('{}')")
            self.json_enabled = True
            logger.info("SQLite JSON1 extension available")
        except sqlite3.OperationalError:
            self.json_enabled = False
            logger.warning("SQLite JSON1 extension not available - using fallback search")
        
        self._init_schema()
        logger.info(f"Initialized SQLiteDocumentStore at {self.db_path}")
    
    def _configure_database(self):
        """Configure SQLite database settings"""
        try:
            # Set performance and reliability configurations
            self.conn.execute(f"PRAGMA journal_mode = {self.journal_mode}")
            self.conn.execute(f"PRAGMA synchronous = {self.synchronous}")
            self.conn.execute(f"PRAGMA cache_size = -{self.cache_size}")
            self.conn.execute(f"PRAGMA page_size = {self.page_size}")
            
            if self.auto_vacuum:
                self.conn.execute("PRAGMA auto_vacuum = FULL")
            
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
            
            self.conn.commit()
            logger.info(f"SQLite configured: journal_mode={self.journal_mode}, synchronous={self.synchronous}")
            
        except sqlite3.Error as e:
            logger.warning(f"Failed to configure SQLite settings: {e}")
    
    def _init_schema(self):
        """Initialize database schema"""
        
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
                -- Create metadata search columns if they don't exist
                CREATE TABLE IF NOT EXISTS temp_check AS SELECT * FROM documents LIMIT 0;
            """
            
            # Check if columns exist and add if needed
            cursor = self.conn.execute("PRAGMA table_info(documents)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'file_type' not in columns:
                schema_sql += """
                    ALTER TABLE documents ADD COLUMN file_type TEXT 
                        GENERATED ALWAYS AS (json_extract(metadata, '$.file_type')) STORED;
                    CREATE INDEX IF NOT EXISTS idx_file_type ON documents (file_type);
                """
            
            if 'original_document_id' not in columns:
                schema_sql += """
                    ALTER TABLE documents ADD COLUMN original_document_id TEXT 
                        GENERATED ALWAYS AS (json_extract(metadata, '$.original_document_id')) STORED;
                    CREATE INDEX IF NOT EXISTS idx_original_doc_id ON documents (original_document_id);
                """
            
            if 'size_bytes' not in columns:
                schema_sql += """
                    ALTER TABLE documents ADD COLUMN size_bytes INTEGER 
                        GENERATED ALWAYS AS (json_extract(metadata, '$.size_bytes')) STORED;
                    CREATE INDEX IF NOT EXISTS idx_size_bytes ON documents (size_bytes);
                """
        
        # FTS5 initialization will be done lazily when needed
        self.fts_initialized = False
        
        try:
            self.conn.executescript(schema_sql)
            self.conn.commit()
            logger.info("Database schema initialized successfully")
        except sqlite3.OperationalError as e:
            # Handle case where some operations fail
            if "duplicate column name" not in str(e) and "already exists" not in str(e):
                raise StorageError(f"Failed to initialize database schema: {e}") from e
            logger.debug(f"Schema initialization warning (expected): {e}")
    
    def _init_fts(self):
        """Initialize FTS5 search (lazy initialization)"""
        if self.fts_initialized:
            return
            
        fts_sql = """
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
            self.conn.executescript(fts_sql)
            
            # Populate FTS with existing documents
            cursor = self.conn.execute("SELECT rowid, id, content FROM documents")
            for row in cursor.fetchall():
                self.conn.execute(
                    "INSERT INTO documents_fts(rowid, id, content) VALUES (?, ?, ?)",
                    (row[0], row[1], row[2])
                )
            
            self.conn.commit()
            self.fts_initialized = True
            logger.info("FTS5 search initialized and populated with existing documents")
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS5 initialization failed: {e}")
            self.fts_initialized = False
    
    def store_document(self, document: Document) -> str:
        """Store a document in SQLite"""
        
        try:
            self.conn.execute(
                """INSERT OR REPLACE INTO documents (id, content, metadata, updated_at) 
                   VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                (document.id, document.content, json.dumps(document.metadata))
            )
            self.conn.commit()
            logger.debug(f"Stored document {document.id}")
            return document.id
        except Exception as e:
            raise StorageError(f"Failed to store document {document.id}: {e}") from e
    
    def add_document(self, document: Document) -> str:
        """Alias for store_document for compatibility"""
        return self.store_document(document)
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve document by ID"""
        
        try:
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
        except Exception as e:
            raise StorageError(f"Failed to get document {document_id}: {e}") from e
    
    def update_document(self, document: Document) -> bool:
        """Update an existing document"""
        
        try:
            cursor = self.conn.execute(
                """UPDATE documents SET content = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP 
                   WHERE id = ?""",
                (document.content, json.dumps(document.metadata), document.id)
            )
            self.conn.commit()
            
            updated = cursor.rowcount > 0
            if updated:
                logger.debug(f"Updated document {document.id}")
            else:
                logger.warning(f"Document {document.id} not found for update")
            
            return updated
        except Exception as e:
            raise StorageError(f"Failed to update document {document.id}: {e}") from e
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID"""
        
        try:
            cursor = self.conn.execute(
                "DELETE FROM documents WHERE id = ?",
                (document_id,)
            )
            self.conn.commit()
            
            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug(f"Deleted document {document_id}")
            else:
                logger.warning(f"Document {document_id} not found for deletion")
            
            return deleted
        except Exception as e:
            raise StorageError(f"Failed to delete document {document_id}: {e}") from e
    
    def document_exists(self, document_id: str) -> bool:
        """Check if a document exists by ID"""
        
        try:
            cursor = self.conn.execute(
                "SELECT 1 FROM documents WHERE id = ?",
                (document_id,)
            )
            return cursor.fetchone() is not None
        except Exception as e:
            raise StorageError(f"Failed to check document existence {document_id}: {e}") from e
    
    def clear_all_documents(self) -> int:
        """Clear all documents from the store"""
        
        try:
            cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            
            self.conn.execute("DELETE FROM documents")
            self.conn.commit()
            
            logger.info(f"Cleared {count} documents from store")
            return count
        except Exception as e:
            raise StorageError(f"Failed to clear all documents: {e}") from e
    
    def clear_all(self) -> int:
        """Alias for clear_all_documents for compatibility"""
        return self.clear_all_documents()
    
    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[SearchResult]:
        """Search documents by metadata"""
        
        try:
            if self.json_enabled:
                return self._search_with_json(filters, limit, offset)
            else:
                return self._search_with_like(filters, limit, offset)
        except Exception as e:
            raise StorageError(f"Failed to search by metadata: {e}") from e
    
    def search_by_content(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[SearchResult]:
        """Full-text search using FTS5"""
        
        # Initialize FTS5 if not already done
        self._init_fts()
        
        try:
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
            
            logger.debug(f"Found {len(results)} documents for content search: '{query}'")
            return results
            
        except Exception as e:
            raise StorageError(f"Failed to search by content: {e}") from e
    
    def get_documents_by_lineage(self, original_document_id: str) -> List[Document]:
        """Get all documents derived from original"""
        
        try:
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
                    (f'%"original_document_id": "{original_document_id}"%', original_document_id)
                )
            
            documents = []
            for row in cursor.fetchall():
                documents.append(Document(
                    id=row["id"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"])
                ))
            
            logger.debug(f"Found {len(documents)} documents in lineage for {original_document_id}")
            return documents
            
        except Exception as e:
            raise StorageError(f"Failed to get documents by lineage: {e}") from e
    
    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> List[Document]:
        """List documents with pagination and sorting"""
        
        try:
            # Validate sort order
            if sort_order.lower() not in ["asc", "desc"]:
                sort_order = "desc"
            
            # Validate sort field
            valid_fields = ["created_at", "updated_at", "id"]
            if sort_by not in valid_fields:
                sort_by = "created_at"
            
            cursor = self.conn.execute(
                f"""SELECT id, content, metadata FROM documents 
                    ORDER BY {sort_by} {sort_order.upper()}
                    LIMIT ? OFFSET ?""",
                (limit, offset)
            )
            
            documents = []
            for row in cursor.fetchall():
                documents.append(Document(
                    id=row["id"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"])
                ))
            
            logger.debug(f"Listed {len(documents)} documents (limit={limit}, offset={offset})")
            return documents
            
        except Exception as e:
            raise StorageError(f"Failed to list documents: {e}") from e
    
    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count documents matching optional filters"""
        
        try:
            if filters:
                # Use the same filter logic as search_by_metadata but just count
                if self.json_enabled:
                    where_clauses, params = self._build_json_where_clause(filters)
                    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
                    
                    cursor = self.conn.execute(
                        f"SELECT COUNT(*) FROM documents WHERE {where_clause}",
                        params
                    )
                else:
                    where_clauses, params = self._build_like_where_clause(filters)
                    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
                    
                    cursor = self.conn.execute(
                        f"SELECT COUNT(*) FROM documents WHERE {where_clause}",
                        params
                    )
            else:
                cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
            
            count = cursor.fetchone()[0]
            logger.debug(f"Counted {count} documents with filters: {filters}")
            return count
            
        except Exception as e:
            raise StorageError(f"Failed to count documents: {e}") from e
    
    def get_document_count(self) -> int:
        """Get total document count (alias for count_documents)"""
        return self.count_documents()
    
    def search_documents(self, query: str, limit: int = 100, offset: int = 0) -> List[SearchResult]:
        """Search documents by content or metadata"""
        try:
            # Try content search first
            if query.strip():
                return self.search_by_content(query, limit, offset)
            else:
                # If empty query, return recent documents
                documents = self.list_documents(limit, offset)
                return [SearchResult(document=doc) for doc in documents]
        except Exception:
            # Fallback to metadata search
            return self.search_by_metadata({}, limit, offset)
    
    def batch_store_documents(self, documents: List[Document]) -> List[str]:
        """Store multiple documents in a batch operation"""
        try:
            stored_ids = []
            for document in documents:
                stored_id = self.store_document(document)
                stored_ids.append(stored_id)
            logger.debug(f"Batch stored {len(documents)} documents")
            return stored_ids
        except Exception as e:
            raise StorageError(f"Failed to batch store documents: {e}") from e
    
    def search_documents_by_metadata(self, filters: Dict[str, Any], limit: int = 100, offset: int = 0) -> List[SearchResult]:
        """Search documents by metadata (alias for search_by_metadata)"""
        return self.search_by_metadata(filters, limit, offset)
    
    def store_documents(self, documents: List[Document]) -> List[str]:
        """Store multiple documents (alias for batch_store_documents)"""
        return self.batch_store_documents(documents)
    
    def get_documents(self, document_ids: List[str]) -> List[Document]:
        """Get multiple documents by their IDs"""
        try:
            documents = []
            for doc_id in document_ids:
                doc = self.get_document(doc_id)
                if doc:
                    documents.append(doc)
            logger.debug(f"Retrieved {len(documents)} documents out of {len(document_ids)} requested")
            return documents
        except Exception as e:
            raise StorageError(f"Failed to get documents: {e}") from e
    
    def get_storage_stats(self) -> StorageStats:
        """Get storage statistics"""
        
        try:
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
        except Exception as e:
            raise StorageError(f"Failed to get storage stats: {e}") from e
    
    def cleanup_orphaned_documents(self) -> int:
        """Clean up orphaned documents"""
        
        try:
            # For now, this is a placeholder - in a real implementation,
            # we'd check for documents that have no references from other systems
            logger.info("Orphaned document cleanup not yet implemented")
            return 0
        except Exception as e:
            raise StorageError(f"Failed to cleanup orphaned documents: {e}") from e
    
    def backup_to_file(self, backup_path: str) -> bool:
        """Backup all documents to a file"""
        
        try:
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup database
            backup_conn = sqlite3.connect(str(backup_file))
            self.conn.backup(backup_conn)
            backup_conn.close()
            
            logger.info(f"Backup created at {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def restore_from_file(self, backup_path: str) -> bool:
        """Restore documents from a backup file"""
        
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                logger.error(f"Backup file {backup_path} does not exist")
                return False
            
            # Close current connection
            self.conn.close()
            
            # Replace database with backup
            import shutil
            shutil.copy2(backup_path, self.db_path)
            
            # Reconnect
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            logger.info(f"Restored from backup {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False
    
    def _search_with_json(self, filters: Dict[str, Any], limit: int, offset: int) -> List[SearchResult]:
        """Search using JSON1 extension"""
        
        where_clauses, params = self._build_json_where_clause(filters)
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
        
        logger.debug(f"Found {len(results)} documents with JSON search")
        return results
    
    def _search_with_like(self, filters: Dict[str, Any], limit: int, offset: int) -> List[SearchResult]:
        """Search using LIKE patterns (fallback)"""
        
        where_clauses, params = self._build_like_where_clause(filters)
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
        
        logger.debug(f"Found {len(results)} documents with LIKE search")
        return results
    
    def _build_json_where_clause(self, filters: Dict[str, Any]) -> tuple[List[str], List[Any]]:
        """Build WHERE clause using JSON1 functions"""
        
        where_clauses = []
        params = []
        
        for key, value in filters.items():
            if isinstance(value, dict):
                if "$gte" in value:
                    where_clauses.append("CAST(json_extract(metadata, ?) AS REAL) >= ?")
                    params.extend([f"$.{key}", value["$gte"]])
                elif "$lte" in value:
                    where_clauses.append("CAST(json_extract(metadata, ?) AS REAL) <= ?")
                    params.extend([f"$.{key}", value["$lte"]])
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
        
        return where_clauses, params
    
    def _build_like_where_clause(self, filters: Dict[str, Any]) -> tuple[List[str], List[Any]]:
        """Build WHERE clause using LIKE patterns (fallback)"""
        
        where_clauses = []
        params = []
        
        for key, value in filters.items():
            if isinstance(value, dict):
                if "$contains" in value:
                    where_clauses.append("metadata LIKE ?")
                    params.append(f'%"{key}"%{value["$contains"]}%')
                elif "$in" in value:
                    like_clauses = []
                    for val in value["$in"]:
                        like_clauses.append("metadata LIKE ?")
                        params.append(f'%"{key}": "{val}"%')
                    where_clauses.append(f"({' OR '.join(like_clauses)})")
            else:
                where_clauses.append("metadata LIKE ?")
                params.append(f'%"{key}": "{value}"%')
        
        return where_clauses, params
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close()
        except:
            pass
    
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary
        
        Returns:
            Dict[str, Any]: Current configuration parameters
        """
        return {
            'db_path': str(self.db_path),
            'timeout': self.timeout,
            'enable_wal': self.enable_wal,
            'auto_vacuum': self.auto_vacuum,
            'page_size': self.page_size,
            'cache_size': self.cache_size,
            'synchronous': self.synchronous,
            'journal_mode': self.journal_mode,
            'json_enabled': self.json_enabled,
            'fts_initialized': getattr(self, 'fts_initialized', False),
            'connection_open': self.conn is not None
        }