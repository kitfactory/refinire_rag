import pytest
from refinire.rag.models import Document, QAPair, EvaluationResult

def test_document_creation():
    """
    Test the creation of a Document instance.
    Documentインスタンスの作成をテストする
    """
    doc = Document(
        id="test_id",
        content="test content",
        metadata={"source": "test"},
        embedding=[0.1, 0.2, 0.3]
    )
    assert doc.id == "test_id"
    assert doc.content == "test content"
    assert doc.metadata == {"source": "test"}
    assert doc.embedding == [0.1, 0.2, 0.3]

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