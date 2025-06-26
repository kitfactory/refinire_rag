import pytest
from refinire_rag.models import Document, QAPair, EvaluationResult

def test_document_creation():
    """
    Test the creation of a Document instance.
    Documentインスタンスの作成をテストする
    """
    doc = Document(
        id="test_id",
        content="test content",
        metadata={"source": "test"}
    )
    assert doc.id == "test_id"
    assert doc.content == "test content"
    assert "source" in doc.metadata
    assert doc.metadata["source"] == "test"
    
    # Test that required metadata fields are auto-populated
    assert "path" in doc.metadata
    assert "file_type" in doc.metadata
    assert "size_bytes" in doc.metadata
    assert "created_at" in doc.metadata
    assert "updated_at" in doc.metadata

def test_qa_pair_creation():
    """
    Test the creation of a QAPair instance.
    QAPairインスタンスの作成をテストする
    """
    qa = QAPair(
        question="What is the test?",
        answer="This is a test.",
        document_id="test_id",
        metadata={"source": "test"}
    )
    assert qa.question == "What is the test?"
    assert qa.answer == "This is a test."
    assert qa.document_id == "test_id"
    assert qa.metadata == {"source": "test"}

def test_evaluation_result_creation():
    """
    Test the creation of an EvaluationResult instance.
    EvaluationResultインスタンスの作成をテストする
    """
    result = EvaluationResult(
        precision=0.8,
        recall=0.7,
        f1_score=0.75,
        metadata={"model": "test"}
    )
    assert result.precision == 0.8
    assert result.recall == 0.7
    assert result.f1_score == 0.75
    assert result.metadata == {"model": "test"}

def test_document_metadata_methods():
    """
    Test Document metadata manipulation methods.
    Documentのメタデータ操作メソッドをテストする
    """
    doc = Document(id="test", content="content")
    
    # Test get_metadata
    assert doc.get_metadata("path") == "unknown_test"
    assert doc.get_metadata("nonexistent", "default") == "default"
    
    # Test set_metadata
    doc.set_metadata("custom_field", "custom_value")
    assert doc.get_metadata("custom_field") == "custom_value"
    
    # Test update_metadata
    doc.update_metadata({"field1": "value1", "field2": "value2"})
    assert doc.get_metadata("field1") == "value1"
    assert doc.get_metadata("field2") == "value2"

def test_document_properties():
    """
    Test Document property accessors.
    Documentのプロパティアクセサをテストする
    """
    doc = Document(id="test", content="test content")
    
    # Test properties
    assert doc.path == "unknown_test"
    assert doc.file_type == "unknown"
    assert doc.size_bytes == len("test content".encode('utf-8'))
    assert isinstance(doc.created_at, str)
    assert isinstance(doc.updated_at, str)

def test_document_with_custom_metadata():
    """
    Test Document creation with custom metadata.
    カスタムメタデータでのDocument作成をテストする
    """
    custom_metadata = {
        "path": "/custom/path.txt",
        "file_type": "text",
        "size_bytes": 1024,
        "created_at": "2023-01-01T00:00:00",
        "updated_at": "2023-01-02T00:00:00",
        "author": "test_author"
    }
    
    doc = Document(id="custom", content="content", metadata=custom_metadata)
    
    assert doc.path == "/custom/path.txt"
    assert doc.file_type == "text"
    assert doc.size_bytes == 1024
    assert doc.created_at == "2023-01-01T00:00:00"
    assert doc.updated_at == "2023-01-02T00:00:00"
    assert doc.get_metadata("author") == "test_author" 