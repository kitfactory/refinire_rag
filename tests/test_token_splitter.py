"""
Tests for TokenTextSplitter
TokenTextSplitterのテスト

This module contains tests for the TokenTextSplitter class.
このモジュールはTokenTextSplitterクラスのテストを含みます。
"""

import pytest
from refinire.rag.splitter.token_splitter import TokenTextSplitter
from refinire.rag.models.document import Document

def test_basic_splitting():
    """
    Test basic text splitting functionality
    基本的なテキスト分割機能のテスト
    """
    splitter = TokenTextSplitter(chunk_size=5, overlap_size=0)
    text = "This is a test. This is another test. This is a third test."
    doc = Document(id="test1", content=text)
    
    docs = list(splitter.process([doc]))
    assert len(docs) == 3  # 5トークンごとに分割されるので3チャンク
    assert docs[0].content == "This is a test. This"
    assert docs[1].content == "is another test. This is"
    assert docs[2].content == "a third test."

def test_chunk_overlap():
    """
    Test text splitting with overlap
    オーバーラップ付きのテキスト分割のテスト
    """
    splitter = TokenTextSplitter(chunk_size=5, overlap_size=2)
    text = "This is a test. This is another test. This is a third test."
    doc = Document(id="test2", content=text)
    
    docs = list(splitter.process([doc]))
    assert len(docs) == 4  # 5トークンごと、2トークン重複で4チャンク
    assert docs[0].content == "This is a test. This"
    assert docs[1].content == "test. This is another test."
    assert docs[2].content == "another test. This is a"
    assert docs[3].content == "is a third test."

def test_small_text():
    """
    Test splitting of small text
    小さいテキストの分割テスト
    """
    splitter = TokenTextSplitter(chunk_size=100)
    text = "This is a small text."
    doc = Document(id="test3", content=text)
    
    docs = list(splitter.process([doc]))
    assert len(docs) == 1  # 1つのチャンクのままであることを確認
    assert docs[0].content == text

def test_empty_text():
    """
    Test handling of empty text
    空のテキストの処理テスト
    """
    splitter = TokenTextSplitter()
    doc = Document(id="test4", content="")
    
    docs = list(splitter.process([doc]))
    assert len(docs) == 0  # 空のテキストからはチャンクが生成されないことを確認

def test_multiple_documents():
    """
    Test processing of multiple documents
    複数ドキュメントの処理テスト
    """
    splitter = TokenTextSplitter(chunk_size=5)
    docs = [
        Document(id="test5", content="First document. With multiple sentences."),
        Document(id="test6", content="Second document. Also with multiple sentences.")
    ]
    
    results = list(splitter.process(docs))
    assert len(results) == 3  # 1つ目は1チャンク、2つ目は2チャンク
    assert results[0].content == "First document. With multiple sentences."
    assert results[1].content == "Second document. Also with multiple"
    assert results[2].content == "sentences."

def test_metadata_preservation():
    """
    Test preservation of document metadata
    ドキュメントメタデータの保持テスト
    """
    splitter = TokenTextSplitter(chunk_size=5)
    metadata = {"source": "test", "author": "tester"}
    doc = Document(id="test8", content="This is a test document.", metadata=metadata)
    
    docs = list(splitter.process([doc]))
    assert len(docs) == 1  # 5トークン以下なので1チャンク
    assert docs[0].metadata["source"] == "test"
    assert docs[0].metadata["author"] == "tester" 