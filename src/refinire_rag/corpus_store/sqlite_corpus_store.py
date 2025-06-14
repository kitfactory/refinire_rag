import json
import sqlite3
from typing import List, Optional, Dict, Any
from ..models import Document
from ..corpusstore import CorpusStore
from pathlib import Path

class SQLiteCorpusStore(CorpusStore):
    """
    SQLite-based implementation of CorpusStore.
    CorpusStoreのSQLiteベースの実装
    """
    def __init__(self, db_path: str):
        """
        Initialize the SQLite store.
        SQLiteストアを初期化する
        
        Args:
            db_path: Path to the SQLite database file
            / db_path: SQLiteデータベースファイルのパス
        """
        self.db_path = db_path
        self._create_tables()

    def _create_tables(self) -> None:
        """
        Create necessary database tables.
        必要なデータベーステーブルを作成する
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def add_document(self, document: Document) -> str:
        """
        Add a document to the SQLite store.
        SQLiteストアに文書を追加する
        
        Args:
            document: Document to add
            / document: 追加する文書
            
        Returns:
            str: ID of the added document
            / str: 追加された文書のID
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO documents (id, content, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.content,
                    json.dumps(document.metadata),
                    document.created_at,
                    document.updated_at
                )
            )
        return document.id

    def update_document(self, document_id: str, document: Document) -> bool:
        """
        Update an existing document in the store.
        SQLiteストア内の既存の文書を更新する
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    UPDATE documents
                    SET content = ?, metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        document.content,
                        json.dumps(document.metadata),
                        document.updated_at,
                        document_id
                    )
                )
                return cursor.rowcount > 0
        except sqlite3.Error:
            return False

    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Get a document by its ID from the SQLite store.
        SQLiteストアからIDによって文書を取得する
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, content, metadata FROM documents WHERE id = ?",
                (document_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
                
            id, content, metadata_json = row
            metadata = json.loads(metadata_json)
            
            return Document(
                id=id,
                content=content,
                metadata=metadata
            )

    def list_documents(self, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        List documents with optional metadata filtering from the SQLite store.
        SQLiteストアからオプションのメタデータフィルタリングを使用して文書をリストする
        """
        with sqlite3.connect(self.db_path) as conn:
            if metadata_filter:
                # Build WHERE clause for metadata filtering
                conditions = []
                params = []
                for key, value in metadata_filter.items():
                    conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                    params.append(str(value))
                
                where_clause = " AND ".join(conditions)
                query = f"SELECT id, content, metadata FROM documents WHERE {where_clause}"
            else:
                query = "SELECT id, content, metadata FROM documents"
                params = []
            
            cursor = conn.execute(query, params)
            documents = []
            
            for row in cursor:
                id, content, metadata_json = row
                metadata = json.loads(metadata_json)
                documents.append(Document(
                    id=id,
                    content=content,
                    metadata=metadata
                ))
            
            return documents

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by its ID from the SQLite store.
        SQLiteストアからIDによって文書を削除する
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                return cursor.rowcount > 0
        except sqlite3.Error:
            return False

    def export_documents(
        self,
        export_dir: Path,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Path]:
        """
        Export documents to files.
        文書をファイルにエクスポートする
        
        Args:
            export_dir: Directory to export documents to
            / export_dir: 文書をエクスポートするディレクトリ
            metadata_filter: Optional metadata filter
            / metadata_filter: オプションのメタデータフィルター
            include_metadata: Whether to include metadata in exported files
            / include_metadata: エクスポートされたファイルにメタデータを含めるかどうか
            
        Returns:
            List[Path]: List of exported file paths
            / List[Path]: エクスポートされたファイルパスのリスト
        """
        export_dir.mkdir(parents=True, exist_ok=True)
        documents = self.list_documents(metadata_filter)
        exported_paths = []
        
        for doc in documents:
            # Create file path
            # / ファイルパスを作成
            file_path = export_dir / f"{doc.id}.{doc.file_type}"
            
            # Write content
            # / コンテンツを書き込む
            with open(file_path, 'w', encoding='utf-8') as f:
                if include_metadata:
                    # Write metadata as YAML front matter
                    # / メタデータをYAMLフロントマターとして書き込む
                    f.write("---\n")
                    for key, value in doc.metadata.items():
                        f.write(f"{key}: {value}\n")
                    f.write("---\n\n")
                f.write(doc.content)
            
            exported_paths.append(file_path)
        
        return exported_paths 