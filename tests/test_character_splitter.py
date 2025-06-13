"""Tests for character text splitter
文字ベースのテキスト分割プロセッサーのテスト"""

import pytest

from refinire.rag.models.document import Document
from refinire.rag.splitter.character_splitter import CharacterTextSplitter


def test_basic_splitting():
    """Test basic text splitting
    基本的なテキスト分割のテスト"""
    splitter = CharacterTextSplitter(chunk_size=10, overlap_size=0)
    doc = Document(id="test1", content="This is a test document. It has multiple sentences.")
    docs = list(splitter.process([doc]))
    
    assert len(docs) == 6
    assert docs[0].content == "This is a "
    assert docs[1].content == "test docum"
    assert docs[2].content == "ent. It ha"
    assert docs[3].content == "s multiple"
    assert docs[4].content == " sentences"
    assert docs[5].content == "."


def test_chunk_overlap():
    """Test chunk overlap
    チャンクのオーバーラップのテスト"""
    splitter = CharacterTextSplitter(chunk_size=10, overlap_size=2)
    doc = Document(id="test2", content="This is a test document. It has multiple sentences.")
    docs = list(splitter.process([doc]))
    
    assert len(docs) == 7
    assert docs[0].content == "This is a "
    assert docs[1].content == "a test doc"
    assert docs[2].content == "ocument. I"
    assert docs[3].content == " It has mu"
    assert docs[4].content == "multiple s"
    assert docs[5].content == " sentences"
    assert docs[6].content == "es."


def test_small_text():
    """Test splitting small text
    小さなテキストの分割テスト"""
    splitter = CharacterTextSplitter(chunk_size=20, overlap_size=0)
    doc = Document(id="test3", content="Short text")
    docs = list(splitter.process([doc]))
    
    assert len(docs) == 1
    assert docs[0].content == "Short text"


def test_empty_text():
    """Test splitting empty text
    空のテキストの分割テスト"""
    splitter = CharacterTextSplitter(chunk_size=10, overlap_size=0)
    doc = Document(id="test4", content="")
    docs = list(splitter.process([doc]))
    
    assert len(docs) == 0


def test_multiple_documents():
    """Test processing multiple documents
    複数ドキュメントの処理テスト"""
    splitter = CharacterTextSplitter(chunk_size=10, overlap_size=0)
    docs = [
        Document(id="test5", content="First document"),
        Document(id="test6", content="Second document")
    ]
    result = list(splitter.process(docs))
    
    assert len(result) == 4
    assert result[0].content == "First docu"
    assert result[1].content == "ment"
    assert result[2].content == "Second doc"
    assert result[3].content == "ument" 