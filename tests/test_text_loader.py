import pytest
from pathlib import Path
from refinire_rag.loader.text_loader import TextLoader
from refinire_rag.models.document import Document

@pytest.fixture
def text_file(tmp_path):
    """
    Create a temporary text file for testing.
    テスト用の一時的なテキストファイルを作成する。
    """
    file_path = tmp_path / "test.txt"
    content = "This is a test file.\nIt has multiple lines.\n日本語のテストも含まれています。"
    file_path.write_text(content, encoding='utf-8')
    return file_path

@pytest.fixture
def text_loader():
    """
    Create a TextLoader instance for testing.
    テスト用のTextLoaderインスタンスを作成する。
    """
    return TextLoader()

def test_text_loader_loads_file(text_loader, text_file):
    """
    Test that TextLoader correctly loads a text file.
    TextLoaderがテキストファイルを正しく読み込むことをテストする。
    """
    # Create a Document with file path
    # ファイルパスを含むDocumentを作成
    doc = Document(
        id="test_doc_1",
        content="",
        metadata={'file_path': str(text_file)}
    )

    # Load the file
    # ファイルを読み込む
    loaded_docs = list(text_loader.process([doc]))

    # Check results
    # 結果を確認
    assert len(loaded_docs) == 1
    loaded_doc = loaded_docs[0]
    assert "This is a test file." in loaded_doc.content
    assert "日本語のテストも含まれています。" in loaded_doc.content
    assert loaded_doc.metadata['file_path'] == str(text_file)
    assert loaded_doc.metadata['encoding'] == 'utf-8'
    assert loaded_doc.metadata['file_type'] == 'text'

def test_text_loader_nonexistent_file(text_loader):
    """
    Test that TextLoader raises FileNotFoundError for nonexistent files.
    TextLoaderが存在しないファイルに対してFileNotFoundErrorを発生させることをテストする。
    """
    doc = Document(
        id="test_doc_2",
        content="",
        metadata={'file_path': 'nonexistent.txt'}
    )

    with pytest.raises(FileNotFoundError):
        list(text_loader.process([doc]))

def test_text_loader_custom_encoding(tmp_path):
    """
    Test that TextLoader works with custom encoding.
    TextLoaderがカスタムエンコーディングで動作することをテストする。
    """
    # Create a file with Shift-JIS encoding
    # Shift-JISエンコーディングでファイルを作成
    file_path = tmp_path / "sjis.txt"
    content = "これはShift-JISでエンコードされたファイルです。"
    file_path.write_text(content, encoding='shift_jis')

    # Create loader with Shift-JIS encoding
    # Shift-JISエンコーディングでローダーを作成
    loader = TextLoader(encoding='shift_jis')
    doc = Document(
        id="test_doc_3",
        content="",
        metadata={'file_path': str(file_path)}
    )

    # Load the file
    # ファイルを読み込む
    loaded_docs = list(loader.process([doc]))

    # Check results
    # 結果を確認
    assert len(loaded_docs) == 1
    assert loaded_docs[0].content == content
    assert loaded_docs[0].metadata['encoding'] == 'shift_jis'

def test_text_loader_wrong_encoding(text_loader, tmp_path):
    """
    Test that TextLoader raises UnicodeDecodeError for wrong encoding.
    TextLoaderが間違ったエンコーディングに対してUnicodeDecodeErrorを発生させることをテストする。
    """
    # Create a file with Shift-JIS encoding
    # Shift-JISエンコーディングでファイルを作成
    file_path = tmp_path / "sjis.txt"
    content = "これはShift-JISでエンコードされたファイルです。"
    file_path.write_text(content, encoding='shift_jis')

    # Try to load with UTF-8 encoding
    # UTF-8エンコーディングで読み込もうとする
    doc = Document(
        id="test_doc_4",
        content="",
        metadata={'file_path': str(file_path)}
    )

    with pytest.raises(UnicodeDecodeError):
        list(text_loader.process([doc])) 