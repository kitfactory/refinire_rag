import pytest
from pathlib import Path
from refinire_rag.loader.html_loader import HTMLLoader
from refinire_rag.models.document import Document

@pytest.fixture
def html_file(tmp_path):
    """
    Create a temporary HTML file for testing.
    テスト用の一時的なHTMLファイルを作成する。
    """
    file_path = tmp_path / "test.html"
    content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta charset="utf-8">
    <style>
        body { color: black; }
    </style>
    <script>
        console.log("Hello");
    </script>
</head>
<body>
    <h1>Main Title</h1>
    <p>This is a paragraph with <strong>bold</strong> text.</p>
    <div>
        <h2>Subtitle</h2>
        <p>Another paragraph with <em>italic</em> text.</p>
        <ul>
            <li>First item</li>
            <li>Second item</li>
        </ul>
    </div>
</body>
</html>"""
    file_path.write_text(content, encoding='utf-8')
    return file_path

@pytest.fixture
def html_loader():
    """
    Create an HTMLLoader instance for testing.
    テスト用のHTMLLoaderインスタンスを作成する。
    """
    return HTMLLoader()

def test_html_loader_loads_file(html_loader, html_file):
    """
    Test that HTMLLoader correctly loads an HTML file.
    HTMLLoaderがHTMLファイルを正しく読み込むことをテストする。
    """
    doc = Document(
        id="test_html_1",
        content="",
        metadata={'file_path': str(html_file)}
    )

    loaded_docs = list(html_loader.process([doc]))

    assert len(loaded_docs) == 1
    assert loaded_docs[0].metadata['content_type'] == 'html'
    assert loaded_docs[0].metadata['file_encoding'] == 'utf-8'

    content = loaded_docs[0].content
    # Check that script and style tags are removed
    # スクリプトとスタイルタグが削除されていることを確認
    assert 'console.log' not in content
    assert 'body { color: black; }' not in content

    # Check that text content is preserved
    # テキストコンテンツが保持されていることを確認
    assert 'Main Title' in content
    assert 'This is a paragraph with bold text' in content
    assert 'Subtitle' in content
    assert 'Another paragraph with italic text' in content
    assert 'First item' in content
    assert 'Second item' in content

def test_html_loader_nonexistent_file(html_loader):
    """
    Test that HTMLLoader raises FileNotFoundError for nonexistent files.
    HTMLLoaderが存在しないファイルに対してFileNotFoundErrorを発生させることをテストする。
    """
    doc = Document(
        id="test_html_2",
        content="",
        metadata={'file_path': 'nonexistent.html'}
    )

    with pytest.raises(FileNotFoundError):
        list(html_loader.process([doc]))

def test_html_loader_custom_encoding(tmp_path):
    """
    Test that HTMLLoader works with custom encoding.
    HTMLLoaderがカスタムエンコーディングで動作することをテストする。
    """
    # Create a file with Shift-JIS encoding
    # Shift-JISエンコーディングでファイルを作成
    file_path = tmp_path / "sjis.html"
    content = """<!DOCTYPE html>
<html>
<body>
    <h1>日本語のタイトル</h1>
    <p>これは日本語の段落です。</p>
</body>
</html>"""
    file_path.write_text(content, encoding='shift_jis')

    # Create loader with Shift-JIS encoding
    # Shift-JISエンコーディングでローダーを作成
    loader = HTMLLoader(encoding='shift_jis')
    doc = Document(
        id="test_html_3",
        content="",
        metadata={'file_path': str(file_path)}
    )

    # Load the file
    # ファイルを読み込む
    loaded_docs = list(loader.process([doc]))

    # Check results
    # 結果を確認
    assert len(loaded_docs) == 1
    content = loaded_docs[0].content
    assert '日本語のタイトル' in content
    assert 'これは日本語の段落です' in content

def test_html_loader_empty_file(tmp_path):
    """
    Test that HTMLLoader handles empty HTML files correctly.
    HTMLLoaderが空のHTMLファイルを適切に処理することをテストする。
    """
    # Create an empty HTML file
    # 空のHTMLファイルを作成
    file_path = tmp_path / "empty.html"
    content = "<!DOCTYPE html><html><body></body></html>"
    file_path.write_text(content, encoding='utf-8')

    loader = HTMLLoader()
    doc = Document(
        id="test_html_4",
        content="",
        metadata={'file_path': str(file_path)}
    )

    # Load the file
    # ファイルを読み込む
    loaded_docs = list(loader.process([doc]))

    # Check results
    # 結果を確認
    assert len(loaded_docs) == 1
    assert loaded_docs[0].content.strip() == "" 