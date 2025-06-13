import json
import sqlite3
from typing import List, Optional, Dict, Any
from ..models import Document
from ..corpusstore import CorpusStore

class SQLiteCorpusStore(CorpusStore):
    """
    SQLite-based implementation of CorpusStore.
    CorpusStoreのSQLiteベースの実装
    """
    def __init__(self, db_path: str):
        """
        Initialize the SQLiteCorpusStore.
        SQLiteCorpusStoreを初期化する
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """
        Initialize the database with required tables.
        必要なテーブルでデータベースを初期化する
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT
                )
            """)

    def add_document(self, document: Document) -> None:
        """
        Add a document to the SQLite store.
        SQLiteストアに文書を追加する
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents (id, content, metadata, embedding) VALUES (?, ?, ?, ?)",
                (
                    document.id,
                    document.content,
                    json.dumps(document.metadata),
                    json.dumps(document.embedding) if document.embedding else None
                )
            )

    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Get a document by its ID from the SQLite store.
        SQLiteストアからIDによって文書を取得する
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT content, metadata, embedding FROM documents WHERE id = ?",
                (document_id,)
            )
            row = cursor.fetchone()
            if row:
                return Document(
                    id=document_id,
                    content=row[0],
                    metadata=json.loads(row[1]),
                    embedding=json.loads(row[2]) if row[2] else None
                )
            return None

    def list_documents(self, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        List documents with optional metadata filtering from the SQLite store.
        SQLiteストアからオプションのメタデータフィルタリングを使用して文書をリストする
        """
        with sqlite3.connect(self.db_path) as conn:
            if metadata_filter:
                # メタデータフィルタリングの実装
                # この実装は簡略化されており、実際の使用ではより複雑なクエリが必要になる可能性があります
                cursor = conn.execute("SELECT id, content, metadata, embedding FROM documents")
            else:
                cursor = conn.execute("SELECT id, content, metadata, embedding FROM documents")
            
            documents = []
            for row in cursor:
                doc_metadata = json.loads(row[2])
                if metadata_filter:
                    if all(doc_metadata.get(k) == v for k, v in metadata_filter.items()):
                        documents.append(Document(
                            id=row[0],
                            content=row[1],
                            metadata=doc_metadata,
                            embedding=json.loads(row[3]) if row[3] else None
                        ))
                else:
                    documents.append(Document(
                        id=row[0],
                        content=row[1],
                        metadata=doc_metadata,
                        embedding=json.loads(row[3]) if row[3] else None
                    ))
            return documents

    def delete_document(self, document_id: str) -> None:
        """
        Delete a document by its ID from the SQLite store.
        SQLiteストアからIDによって文書を削除する
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,)) 