import os
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from refinire_rag.models import Document
from refinire_rag.corpusstore import CorpusStore
from refinire_rag.corpus_store.sqlite_corpus_store import SQLiteCorpusStore

@pytest.fixture
def test_db_path(tmp_path):
    """
    Create a temporary database path for testing.
    テスト用の一時的なデータベースパスを作成する
    """
    return str(tmp_path / "test.db")

@pytest.fixture
def temp_db():
    """
    Create a temporary SQLite database for testing.
    / テスト用の一時的なSQLiteデータベースを作成する
    """
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)

@pytest.fixture
def corpus_store(test_db_path):
    """
    Create a SQLiteCorpusStore instance for testing.
    テスト用のSQLiteCorpusStoreインスタンスを作成する
    """
    store = SQLiteCorpusStore(test_db_path)
    yield store
    # Cleanup
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

@pytest.fixture
def store(temp_db):
    """
    Create a SQLiteCorpusStore instance for testing.
    / テスト用のSQLiteCorpusStoreインスタンスを作成する
    """
    return SQLiteCorpusStore(temp_db)

@pytest.fixture
def sample_document():
    """
    Create a sample document for testing.
    / テスト用のサンプル文書を作成する
    """
    now = datetime.now().isoformat()
    return Document(
        id="test_doc_1",
        content="This is a test document.",
        metadata={
            "author": "Test Author",
            "category": "Test",
            "created_at": now,
            "updated_at": now,
            "path": "/test/path/doc1.txt",
            "file_type": "txt",
            "size_bytes": 100
        }
    )

def test_add_and_get_document(store):
    """
    Test adding and retrieving a document.
    / 文書の追加と取得をテストする
    """
    now = datetime.now().isoformat()
    doc = Document(
        id="test_id",
        content="test content",
        metadata={
            "source": "test",
            "created_at": now,
            "updated_at": now,
            "path": "/test/path/test.txt",
            "file_type": "txt",
            "size_bytes": len("test content".encode('utf-8'))
        }
    )
    
    # Add document
    # / 文書を追加
    doc_id = store.add_document(doc)
    assert doc_id == doc.id
    
    # Get document
    # / 文書を取得
    retrieved_doc = store.get_document(doc_id)
    assert retrieved_doc is not None
    assert retrieved_doc.id == doc.id
    assert retrieved_doc.content == doc.content
    assert retrieved_doc.metadata == doc.metadata

def test_list_documents(store):
    """
    Test listing documents with and without metadata filtering.
    メタデータフィルタリングの有無で文書をリストするテスト
    """
    doc1 = Document(
        id="test_id_1",
        content="test content 1",
        metadata={"source": "test1"},
        embedding=[0.1, 0.2, 0.3]
    )
    doc2 = Document(
        id="test_id_2",
        content="test content 2",
        metadata={"source": "test2"},
        embedding=[0.4, 0.5, 0.6]
    )
    store.add_document(doc1)
    store.add_document(doc2)

    # Test listing all documents
    all_docs = store.list_documents()
    assert len(all_docs) == 2

    # Test listing documents with metadata filter
    filtered_docs = store.list_documents({"source": "test1"})
    assert len(filtered_docs) == 1
    assert filtered_docs[0].id == "test_id_1"

def test_delete_document(store):
    """
    Test deleting a document.
    文書の削除をテストする
    """
    doc = Document(
        id="test_id",
        content="test content",
        metadata={"source": "test"},
        embedding=[0.1, 0.2, 0.3]
    )
    store.add_document(doc)
    store.delete_document("test_id")
    assert store.get_document("test_id") is None

def test_add_document(store, sample_document):
    """
    Test adding a document to the store.
    / ストアに文書を追加するテスト
    """
    doc_id = store.add_document(sample_document)
    assert doc_id == sample_document.id
    
    # Verify the document was added
    retrieved_doc = store.get_document(doc_id)
    assert retrieved_doc is not None
    assert retrieved_doc.id == sample_document.id
    assert retrieved_doc.content == sample_document.content
    assert retrieved_doc.metadata == sample_document.metadata
    assert retrieved_doc.created_at == sample_document.created_at
    assert retrieved_doc.updated_at == sample_document.updated_at

def test_update_document(store, sample_document):
    """
    Test updating a document in the store.
    / ストア内の文書を更新するテスト
    """
    # Add the document first
    store.add_document(sample_document)
    
    # Update the document
    updated_at = datetime.now().isoformat()
    updated_doc = Document(
        id=sample_document.id,
        content="Updated content",
        metadata={
            "author": "Updated Author",
            "created_at": sample_document.created_at,
            "updated_at": updated_at,
            "path": sample_document.path,
            "file_type": sample_document.file_type,
            "size_bytes": len("Updated content".encode('utf-8'))
        }
    )
    
    success = store.update_document(sample_document.id, updated_doc)
    assert success is True
    
    # Verify the update
    retrieved_doc = store.get_document(sample_document.id)
    assert retrieved_doc is not None
    assert retrieved_doc.content == updated_doc.content
    assert retrieved_doc.metadata == updated_doc.metadata
    assert retrieved_doc.created_at == sample_document.created_at
    assert retrieved_doc.updated_at == updated_at

def test_get_nonexistent_document(store):
    """
    Test retrieving a non-existent document.
    / 存在しない文書を取得するテスト
    """
    doc = store.get_document("nonexistent_id")
    assert doc is None

def test_delete_document(store, sample_document):
    """
    Test deleting a document from the store.
    / ストアから文書を削除するテスト
    """
    # Add the document first
    store.add_document(sample_document)
    
    # Delete the document
    success = store.delete_document(sample_document.id)
    assert success is True
    
    # Verify the document was deleted
    doc = store.get_document(sample_document.id)
    assert doc is None

def test_list_documents(store, sample_document):
    """
    Test listing documents with metadata filtering.
    / メタデータフィルタリングを使用して文書をリストするテスト
    """
    # Add multiple documents
    doc1 = sample_document
    now = datetime.now().isoformat()
    doc2 = Document(
        id="test_doc_2",
        content="Another test document",
        metadata={
            "author": "Different Author",
            "category": "Test",
            "created_at": now,
            "updated_at": now,
            "path": "/test/path/doc2.txt",
            "file_type": "txt",
            "size_bytes": 150
        }
    )
    
    store.add_document(doc1)
    store.add_document(doc2)
    
    # Test listing all documents
    all_docs = store.list_documents()
    assert len(all_docs) == 2
    
    # Test filtering by author
    filtered_docs = store.list_documents({"author": "Test Author"})
    assert len(filtered_docs) == 1
    assert filtered_docs[0].id == doc1.id
    
    # Test filtering by category
    filtered_docs = store.list_documents({"category": "Test"})
    assert len(filtered_docs) == 2

def test_export_documents(store, sample_document, tmp_path):
    """
    Test exporting documents to files.
    / 文書をファイルにエクスポートするテスト
    """
    # Add the document
    store.add_document(sample_document)
    
    # Export with metadata
    export_dir = tmp_path / "export_with_metadata"
    exported_paths = store.export_documents(
        export_dir,
        include_metadata=True
    )
    
    assert len(exported_paths) == 1
    content = exported_paths[0].read_text(encoding='utf-8')
    assert "---" in content
    assert "author: Test Author" in content
    assert f"created_at: {sample_document.created_at}" in content
    assert f"updated_at: {sample_document.updated_at}" in content
    assert sample_document.content in content
    
    # Export without metadata
    export_dir = tmp_path / "export_without_metadata"
    exported_paths = store.export_documents(
        export_dir,
        include_metadata=False
    )
    
    assert len(exported_paths) == 1
    content = exported_paths[0].read_text(encoding='utf-8')
    assert "---" not in content
    assert content.strip() == sample_document.content

def test_export_documents_with_filter(store, tmp_path):
    """
    Test exporting documents with metadata filtering.
    / メタデータフィルタリングを使用して文書をエクスポートするテスト
    """
    # Add multiple documents
    now = datetime.now().isoformat()
    doc1 = Document(
        id="test_doc_1",
        content="First test document",
        metadata={
            "category": "A",
            "created_at": now,
            "updated_at": now,
            "path": "/test/path/doc1.txt",
            "file_type": "txt",
            "size_bytes": 100
        }
    )
    doc2 = Document(
        id="test_doc_2",
        content="Second test document",
        metadata={
            "category": "B",
            "created_at": now,
            "updated_at": now,
            "path": "/test/path/doc2.txt",
            "file_type": "txt",
            "size_bytes": 150
        }
    )
    
    store.add_document(doc1)
    store.add_document(doc2)
    
    # Export with filter
    export_dir = tmp_path / "filtered_export"
    exported_paths = store.export_documents(
        export_dir,
        metadata_filter={"category": "A"}
    )
    
    assert len(exported_paths) == 1
    content = exported_paths[0].read_text(encoding='utf-8')
    assert doc1.content in content
    assert doc2.content not in content 