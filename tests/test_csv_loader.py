import pytest
from pathlib import Path
from refinire_rag.loader.csv_loader import CSVLoader
from refinire_rag.models.document import Document

@pytest.fixture
def csv_file(tmp_path):
    """
    Create a temporary CSV file for testing.
    テスト用の一時的なCSVファイルを作成する。
    """
    file_path = tmp_path / "test.csv"
    content = """name,age,city
John,30,Tokyo
Alice,25,Osaka
Bob,35,Fukuoka"""
    file_path.write_text(content, encoding='utf-8')
    return file_path

@pytest.fixture
def csv_loader():
    """
    Create a CSVLoader instance for testing.
    テスト用のCSVLoaderインスタンスを作成する。
    """
    return CSVLoader()

def test_csv_loader_loads_file(csv_loader, csv_file):
    """
    Test that CSVLoader correctly loads a CSV file.
    CSVLoaderがCSVファイルを正しく読み込むことをテストする。
    """
    # Create a Document with file path
    # ファイルパスを含むDocumentを作成
    doc = Document(
        id="test_csv_1",
        content="",
        metadata={'file_path': str(csv_file)}
    )

    # Load the file
    # ファイルを読み込む
    loaded_docs = list(csv_loader.process([doc]))

    # Check results
    # 結果を確認
    assert len(loaded_docs) == 3  # 3 rows of data
    assert loaded_docs[0].metadata['columns'] == ['name', 'age', 'city']
    assert loaded_docs[0].metadata['row_index'] == 0
    assert loaded_docs[1].metadata['row_index'] == 1
    assert loaded_docs[2].metadata['row_index'] == 2

    # Check content of first row
    # 最初の行の内容を確認
    first_row = eval(loaded_docs[0].content)  # str(dict)をdictに変換
    assert first_row['name'] == 'John'
    assert first_row['age'] == '30'
    assert first_row['city'] == 'Tokyo'

def test_csv_loader_nonexistent_file(csv_loader):
    """
    Test that CSVLoader raises FileNotFoundError for nonexistent files.
    CSVLoaderが存在しないファイルに対してFileNotFoundErrorを発生させることをテストする。
    """
    doc = Document(
        id="test_csv_2",
        content="",
        metadata={'file_path': 'nonexistent.csv'}
    )

    with pytest.raises(FileNotFoundError):
        list(csv_loader.process([doc]))

def test_csv_loader_custom_encoding(tmp_path):
    """
    Test that CSVLoader works with custom encoding.
    CSVLoaderがカスタムエンコーディングで動作することをテストする。
    """
    # Create a file with Shift-JIS encoding
    # Shift-JISエンコーディングでファイルを作成
    file_path = tmp_path / "sjis.csv"
    content = """name,age,city
山田,30,東京
鈴木,25,大阪
佐藤,35,福岡"""
    file_path.write_text(content, encoding='shift_jis')

    # Create loader with Shift-JIS encoding
    # Shift-JISエンコーディングでローダーを作成
    loader = CSVLoader(encoding='shift_jis')
    doc = Document(
        id="test_csv_3",
        content="",
        metadata={'file_path': str(file_path)}
    )

    # Load the file
    # ファイルを読み込む
    loaded_docs = list(loader.process([doc]))

    # Check results
    # 結果を確認
    assert len(loaded_docs) == 3
    first_row = eval(loaded_docs[0].content)
    assert first_row['name'] == '山田'
    assert first_row['city'] == '東京'

def test_csv_loader_empty_file(tmp_path):
    """
    Test that CSVLoader handles empty CSV files correctly.
    CSVLoaderが空のCSVファイルを適切に処理することをテストする。
    """
    # Create an empty CSV file
    # 空のCSVファイルを作成
    file_path = tmp_path / "empty.csv"
    content = "name,age,city\n"  # ヘッダーのみ
    file_path.write_text(content, encoding='utf-8')

    loader = CSVLoader()
    doc = Document(
        id="test_csv_4",
        content="",
        metadata={'file_path': str(file_path)}
    )

    # Load the file
    # ファイルを読み込む
    loaded_docs = list(loader.process([doc]))

    # Check results
    # 結果を確認
    assert len(loaded_docs) == 0  # データ行がない 