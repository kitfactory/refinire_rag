"""
Tests for the CodeTextSplitter class.
"""

import pytest
from refinire_rag.splitter import CodeTextSplitter
from refinire_rag.models.document import Document


def test_basic_code_splitting():
    """Test basic code splitting functionality."""
    splitter = CodeTextSplitter(chunk_size=200, overlap_size=10)
    text = """
def hello_world():
    print("Hello, World!")
    return True

def another_function():
    print("Another function")
    return False
"""
    docs = splitter.split([Document(content=text, id="test1")])
    assert len(docs) == 2
    assert "def hello_world()" in docs[0].content
    assert "def another_function()" in docs[1].content


def test_code_overlap():
    """Test code splitting with overlap."""
    splitter = CodeTextSplitter(chunk_size=200, overlap_size=20)
    text = """
def function_one():
    print("First function")
    return True

def function_two():
    print("Second function")
    return False
"""
    docs = splitter.split([Document(content=text, id="test2")])
    assert len(docs) == 2
    # Check that overlap is preserved
    assert docs[0].content.endswith("return True\n")
    assert docs[1].content[splitter.overlap_size:].lstrip().startswith("def function_two")


def test_small_code():
    """Test splitting of small code blocks."""
    splitter = CodeTextSplitter(chunk_size=100, overlap_size=20)
    text = "def small(): return True"
    docs = splitter.split([Document(content=text, id="test3")])
    assert len(docs) == 1
    assert docs[0].content == text


def test_empty_code():
    """Test handling of empty code."""
    splitter = CodeTextSplitter()
    docs = splitter.split([Document(content="", id="test4")])
    assert len(docs) == 0


def test_multiple_code_documents():
    """Test processing of multiple code documents."""
    splitter = CodeTextSplitter(chunk_size=50, overlap_size=10)
    docs = splitter.split([
        Document(content="def doc1(): return True", id="test5"),
        Document(content="def doc2(): return False", id="test6")
    ])
    assert len(docs) == 2
    assert "doc1" in docs[0].content
    assert "doc2" in docs[1].content


def test_code_metadata_preservation():
    """Test that document metadata is preserved during splitting."""
    splitter = CodeTextSplitter(chunk_size=50, overlap_size=10)
    metadata = {"language": "python", "file": "test.py"}
    docs = splitter.split([
        Document(
            content="def test(): return True",
            metadata=metadata,
            id="test7"
        )
    ])
    assert len(docs) == 1
    assert docs[0].metadata == metadata


def test_language_specific_splitting():
    """Test splitting with language-specific settings."""
    splitter = CodeTextSplitter(
        chunk_size=200,
        overlap_size=10,
        language="python"
    )
    text = """
def python_function():
    print("Python code")
    return True
"""
    docs = splitter.split([Document(content=text, id="test8")])
    assert len(docs) == 1
    assert "python_function" in docs[0].content 