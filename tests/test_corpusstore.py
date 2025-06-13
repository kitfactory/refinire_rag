import os
import pytest
from refinire.rag.models import Document
from refinire.rag.corpusstore import CorpusStore
from refinire.rag.corpus_store.sqlite_corpus_store import SQLiteCorpusStore

@pytest.fixture
def test_db_path(tmp_path):
    """
    Create a temporary database path for testing.
    テスト用の一時的なデータベースパスを作成する
    """
    return str(tmp_path / "test.db")

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

def test_add_and_get_document(corpus_store):
    """
    Test adding and retrieving a document.
    文書の追加と取得をテストする
    """
    doc = Document(
        id="test_id",
        content="test content",
        metadata={"source": "test"},
        embedding=[0.1, 0.2, 0.3]
    )
    corpus_store.add_document(doc)
    retrieved_doc = corpus_store.get_document("test_id")
    assert retrieved_doc is not None
    assert retrieved_doc.id == doc.id
    assert retrieved_doc.content == doc.content
    assert retrieved_doc.metadata == doc.metadata
    assert retrieved_doc.embedding == doc.embedding

def test_list_documents(corpus_store):
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
    corpus_store.add_document(doc1)
    corpus_store.add_document(doc2)

    # Test listing all documents
    all_docs = corpus_store.list_documents()
    assert len(all_docs) == 2

    # Test listing documents with metadata filter
    filtered_docs = corpus_store.list_documents({"source": "test1"})
    assert len(filtered_docs) == 1
    assert filtered_docs[0].id == "test_id_1"

def test_delete_document(corpus_store):
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
    corpus_store.add_document(doc)
    corpus_store.delete_document("test_id")
    assert corpus_store.get_document("test_id") is None 