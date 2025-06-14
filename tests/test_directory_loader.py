import pytest
from pathlib import Path
from refinire_rag.loader.directory_loader import DirectoryLoader
from refinire_rag.loader.text_loader import TextLoader
from refinire_rag.loader.csv_loader import CSVLoader
from refinire_rag.loader.json_loader import JSONLoader
from refinire_rag.models.document import Document

@pytest.fixture
def test_dir(tmp_path):
    """
    Create a temporary directory with test files.
    テスト用の一時ディレクトリとファイルを作成する。
    """
    # Create test files
    # テストファイルを作成
    (tmp_path / "test.txt").write_text("This is a text file.", encoding='utf-8')
    (tmp_path / "test.csv").write_text("col1,col2\n1,2", encoding='utf-8')
    (tmp_path / "test.json").write_text('{"key": "value"}', encoding='utf-8')
    (tmp_path / "test.unknown").write_text("Unknown file type", encoding='utf-8')

    # Create a subdirectory with more files
    # サブディレクトリとその中のファイルを作成
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "sub.txt").write_text("This is a subdirectory text file.", encoding='utf-8')
    (subdir / "sub.csv").write_text("col1,col2\n3,4", encoding='utf-8')

    return tmp_path

@pytest.fixture
def directory_loader():
    """
    Create a DirectoryLoader instance for testing.
    テスト用のDirectoryLoaderインスタンスを作成する。
    """
    return DirectoryLoader()

def test_directory_loader_loads_all_files(directory_loader, test_dir):
    """
    Test that DirectoryLoader loads all supported files in the directory.
    DirectoryLoaderがディレクトリ内の対応ファイルを全て読み込むことをテストする。
    """
    # Create a Document with directory path
    # ディレクトリパスを含むDocumentを作成
    doc = Document(
        id="test_dir_1",
        content="",
        metadata={'dir_path': str(test_dir)}
    )

    # Load all files
    # 全てのファイルを読み込む
    loaded_docs = list(directory_loader.process([doc]))

    # Check results
    # 結果を確認
    assert len(loaded_docs) == 5  # 5 supported files (including subdirectory)
    
    # Check file types
    # ファイルタイプを確認
    file_types = {doc.metadata['file_ext'] for doc in loaded_docs}
    assert '.txt' in file_types
    assert '.csv' in file_types
    assert '.json' in file_types
    assert '.unknown' not in file_types  # 未対応の拡張子は含まれない

def test_directory_loader_non_recursive(directory_loader, test_dir):
    """
    Test that DirectoryLoader works in non-recursive mode.
    DirectoryLoaderが非再帰モードで動作することをテストする。
    """
    # Create a non-recursive loader
    # 非再帰のローダーを作成
    non_recursive_loader = DirectoryLoader(recursive=False)
    
    doc = Document(
        id="test_dir_2",
        content="",
        metadata={'dir_path': str(test_dir)}
    )

    # Load files
    # ファイルを読み込む
    loaded_docs = list(non_recursive_loader.process([doc]))

    # Check results (should not include subdirectory files)
    # 結果を確認（サブディレクトリのファイルは含まれない）
    assert len(loaded_docs) == 3  # 3 files in root directory (excluding .unknown)
    subdir_files = [doc for doc in loaded_docs if 'subdir' in doc.metadata['file_path']]
    assert len(subdir_files) == 0

def test_directory_loader_custom_mapping(test_dir):
    """
    Test that DirectoryLoader works with custom extension mapping.
    DirectoryLoaderがカスタム拡張子マッピングで動作することをテストする。
    """
    # Create a custom loader mapping
    # カスタムローダーマッピングを作成
    custom_mapping = {
        '.txt': TextLoader(),
        '.csv': CSVLoader(),
        # .json is intentionally omitted
    }
    
    custom_loader = DirectoryLoader(extension_loader_map=custom_mapping)
    
    doc = Document(
        id="test_dir_3",
        content="",
        metadata={'dir_path': str(test_dir)}
    )

    # Load files
    # ファイルを読み込む
    loaded_docs = list(custom_loader.process([doc]))

    # Check results (should not include .json files)
    # 結果を確認（.jsonファイルは含まれない）
    assert len(loaded_docs) == 4  # 4 files (excluding .json)
    json_files = [doc for doc in loaded_docs if doc.metadata['file_ext'] == '.json']
    assert len(json_files) == 0

def test_directory_loader_nonexistent_directory(directory_loader):
    """
    Test that DirectoryLoader handles nonexistent directories gracefully.
    DirectoryLoaderが存在しないディレクトリを適切に処理することをテストする。
    """
    doc = Document(
        id="test_dir_4",
        content="",
        metadata={'dir_path': 'nonexistent_dir'}
    )

    # Should not raise an error
    # エラーを発生させない
    loaded_docs = list(directory_loader.process([doc]))
    assert len(loaded_docs) == 0 