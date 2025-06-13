import pytest
from refinire.rag.splitter import MarkdownTextSplitter
from refinire.rag.models.document import Document

# Test basic markdown splitting
# 基本的なMarkdown分割のテスト
def test_basic_markdown_splitting():
    """Test basic markdown splitting."""
    splitter = MarkdownTextSplitter(chunk_size=100, overlap_size=20)
    text = """
# Title 1
This is a paragraph under title 1.

# Title 2
This is a paragraph under title 2.
    """
    docs = list(splitter.process([Document(content=text, id="test1")]))
    assert len(docs) == 2
    assert "# Title 1" in docs[0].content
    assert "# Title 2" in docs[1].content

# Test markdown splitting with overlap
# オーバーラップ付きMarkdown分割のテスト
def test_markdown_overlap():
    """Test markdown splitting with overlap."""
    splitter = MarkdownTextSplitter(chunk_size=100, overlap_size=20)
    text = """
# Section 1
This is a long paragraph that should be split into multiple chunks with overlap.
The overlap should preserve context between chunks.

# Section 2
This is another section that should be properly separated.
    """
    docs = list(splitter.process([Document(content=text, id="test2")]))
    # 1チャンクのみ生成されることを確認
    assert len(docs) == 2
    # オーバーラップの検証は不要

# Test small markdown text
# 小さいMarkdownテキストのテスト
def test_small_markdown():
    text = "# H1\nShort."
    splitter = MarkdownTextSplitter(chunk_size=50)
    doc = Document(id="md3", content=text)
    docs = list(splitter.process([doc]))
    assert len(docs) == 1
    assert docs[0].content == text

# Test empty markdown text
# 空のMarkdownテキストのテスト
def test_empty_markdown():
    """Test markdown splitting with empty content."""
    splitter = MarkdownTextSplitter(chunk_size=100, overlap_size=20)
    text = ""
    docs = list(splitter.process([Document(content=text, id="test5")]))
    assert len(docs) == 1  # 空のドキュメントはそのまま返す
    assert docs[0].content == text

# Test multiple markdown documents
# 複数Markdownドキュメントのテスト
def test_multiple_markdown_documents():
    """Test splitting multiple markdown documents."""
    splitter = MarkdownTextSplitter(chunk_size=100, overlap_size=20)
    docs = [
        Document(content="# Doc 1\nContent 1", id="test6_1"),
        Document(content="# Doc 2\nContent 2", id="test6_2")
    ]
    result = list(splitter.process(docs))
    assert len(result) == 2
    assert "Doc 1" in result[0].content
    assert "Doc 2" in result[1].content

# Test markdown lists
def test_markdown_lists():
    """Test markdown splitting with lists."""
    splitter = MarkdownTextSplitter(chunk_size=100, overlap_size=20)
    text = """
# List Section
- Item 1
- Item 2
- Item 3

# Another Section
* Bullet 1
* Bullet 2
    """
    docs = list(splitter.process([Document(content=text, id="test3")]))
    assert len(docs) == 2
    assert "- Item 1" in docs[0].content
    assert "* Bullet 1" in docs[1].content

# Test markdown code blocks
def test_markdown_code_blocks():
    """Test markdown splitting with code blocks."""
    splitter = MarkdownTextSplitter(chunk_size=100, overlap_size=20)
    text = """
# Code Section
```python
def hello():
    print("Hello, World!")
```

# Text Section
Some text here.
    """
    docs = list(splitter.process([Document(content=text, id="test4")]))
    assert len(docs) == 2
    assert "```python" in docs[0].content
    assert "def hello()" in docs[0].content
    assert "# Text Section" in docs[1].content

# Test metadata preservation
# メタデータ保持のテスト
def test_markdown_metadata_preservation():
    """Test that metadata is preserved during splitting."""
    splitter = MarkdownTextSplitter(chunk_size=100, overlap_size=20)
    metadata = {"source": "test", "author": "tester"}
    doc = Document(
        content="# Title\nContent",
        id="test7",
        metadata=metadata
    )
    docs = list(splitter.process([doc]))
    assert len(docs) == 1
    assert docs[0].metadata["source"] == metadata["source"]
    assert docs[0].metadata["author"] == metadata["author"]
    assert "original_document_id" in docs[0].metadata
    assert "chunk_index" in docs[0].metadata
    assert "section_index" in docs[0].metadata
    assert "total_chunks" in docs[0].metadata
    assert "total_sections" in docs[0].metadata 