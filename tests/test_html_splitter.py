import pytest
from refinire.rag.splitter.html_splitter import HTMLTextSplitter
from refinire.rag.models.document import Document

# Test basic HTML splitting
# 基本的なHTML分割のテスト
def test_basic_html_splitting():
    text = """<h1>Heading 1</h1><p>This is a paragraph.</p><h2>Heading 2</h2><ul><li>List item 1</li><li>List item 2</li></ul><p>Another paragraph.</p>"""
    splitter = HTMLTextSplitter(chunk_size=30, overlap_size=0)
    doc = Document(id="html1", content=text)
    docs = list(splitter.process([doc]))
    assert len(docs) >= 2
    assert docs[0].content.startswith("<h1>")
    assert any("<li>List item 1</li>" in d.content for d in docs)

# Test HTML splitting with overlap
# オーバーラップ付きHTML分割のテスト
def test_html_overlap():
    text = """<h1>H1</h1><p>A paragraph.</p><h2>H2</h2><p>Another paragraph.</p><ul><li>Item 1</li><li>Item 2</li></ul>"""
    splitter = HTMLTextSplitter(chunk_size=25, overlap_size=5)
    doc = Document(id="html2", content=text)
    docs = list(splitter.process([doc]))
    assert len(docs) >= 2
    # オーバーラップ部分が次のチャンクの先頭に含まれる
    if len(docs) > 1:
        overlap = docs[0].content[-5:]
        assert overlap in docs[1].content

# Test small HTML text
# 小さいHTMLテキストのテスト
def test_small_html():
    text = "<h1>H1</h1><p>Short.</p>"
    splitter = HTMLTextSplitter(chunk_size=50)
    doc = Document(id="html3", content=text)
    docs = list(splitter.process([doc]))
    assert len(docs) == 1
    assert docs[0].content == "<h1>H1</h1>\n<p>Short.</p>"

# Test empty HTML text
# 空のHTMLテキストのテスト
def test_empty_html():
    text = ""
    splitter = HTMLTextSplitter(chunk_size=50)
    doc = Document(id="html4", content=text)
    docs = list(splitter.process([doc]))
    assert len(docs) == 0

# Test multiple HTML documents
# 複数HTMLドキュメントのテスト
def test_multiple_html_documents():
    text1 = "<h1>H1</h1><p>Para1.</p>"
    text2 = "<h2>H2</h2><p>Para2.</p>"
    splitter = HTMLTextSplitter(chunk_size=10)
    docs = [Document(id="html5", content=text1), Document(id="html6", content=text2)]
    results = list(splitter.process(docs))
    assert len(results) == 4  # 各ドキュメント2チャンクずつ
    assert results[0].content.startswith("<h1>")
    assert results[2].content.startswith("<h2>")

# Test metadata preservation
# メタデータ保持のテスト
def test_html_metadata_preservation():
    text = "<h1>H1</h1><p>Meta test.</p>"
    metadata = {"source": "test", "author": "tester"}
    splitter = HTMLTextSplitter(chunk_size=10)
    doc = Document(id="html7", content=text, metadata=metadata)
    docs = list(splitter.process([doc]))
    assert len(docs) >= 1
    for d in docs:
        assert d.metadata["source"] == "test"
        assert d.metadata["author"] == "tester" 