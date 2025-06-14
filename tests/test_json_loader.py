import pytest
import json
from pathlib import Path
from refinire_rag.loader.json_loader import JSONLoader
from refinire_rag.models.document import Document

@pytest.fixture
def json_file_list(tmp_path):
    """
    Create a temporary JSON file with a list of objects.
    オブジェクトのリストを含む一時的なJSONファイルを作成する。
    """
    file_path = tmp_path / "test_list.json"
    content = """[
        {"name": "John", "age": 30, "city": "Tokyo"},
        {"name": "Alice", "age": 25, "city": "Osaka"},
        {"name": "Bob", "age": 35, "city": "Fukuoka"}
    ]"""
    file_path.write_text(content, encoding='utf-8')
    return file_path

@pytest.fixture
def json_file_object(tmp_path):
    """
    Create a temporary JSON file with a single object.
    単一オブジェクトを含む一時的なJSONファイルを作成する。
    """
    file_path = tmp_path / "test_object.json"
    content = """{
        "name": "John",
        "age": 30,
        "city": "Tokyo",
        "hobbies": ["reading", "gaming", "coding"]
    }"""
    file_path.write_text(content, encoding='utf-8')
    return file_path

@pytest.fixture
def json_file_nested(tmp_path):
    """
    Create a temporary JSON file with nested objects.
    ネストされたオブジェクトを含む一時的なJSONファイルを作成する。
    """
    file_path = tmp_path / "test_nested.json"
    content = """{
        "users": [
            {"name": "John", "age": 30},
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 35}
        ]
    }"""
    file_path.write_text(content, encoding='utf-8')
    return file_path

@pytest.fixture
def json_loader():
    """
    Create a JSONLoader instance for testing.
    テスト用のJSONLoaderインスタンスを作成する。
    """
    return JSONLoader()

def test_json_loader_list(json_loader, json_file_list):
    """
    Test that JSONLoader correctly loads a JSON file containing a list of objects.
    JSONLoaderがオブジェクトのリストを含むJSONファイルを正しく読み込むことをテストする。
    """
    doc = Document(
        id="test_json_1",
        content="",
        metadata={'file_path': str(json_file_list)}
    )

    loaded_docs = list(json_loader.process([doc]))

    assert len(loaded_docs) == 1
    assert loaded_docs[0].metadata['content_type'] == 'json'
    
    # Parse the JSON content
    # JSONコンテンツを解析
    json_content = json.loads(loaded_docs[0].content)
    assert isinstance(json_content, list)
    assert len(json_content) == 3
    
    first_item = json_content[0]
    assert first_item['name'] == 'John'
    assert first_item['age'] == 30
    assert first_item['city'] == 'Tokyo'

def test_json_loader_object(json_loader, json_file_object):
    """
    Test that JSONLoader correctly loads a JSON file containing a single object.
    JSONLoaderが単一オブジェクトを含むJSONファイルを正しく読み込むことをテストする。
    """
    doc = Document(
        id="test_json_2",
        content="",
        metadata={'file_path': str(json_file_object)}
    )

    loaded_docs = list(json_loader.process([doc]))

    assert len(loaded_docs) == 1
    assert loaded_docs[0].metadata['content_type'] == 'json'

    obj = json.loads(loaded_docs[0].content)
    assert obj['name'] == 'John'
    assert obj['age'] == 30
    assert obj['city'] == 'Tokyo'
    assert obj['hobbies'] == ['reading', 'gaming', 'coding']

def test_json_loader_nested(json_loader, json_file_nested):
    """
    Test that JSONLoader correctly loads a JSON file containing nested objects.
    JSONLoaderがネストされたオブジェクトを含むJSONファイルを正しく読み込むことをテストする。
    """
    doc = Document(
        id="test_json_3",
        content="",
        metadata={'file_path': str(json_file_nested)}
    )

    loaded_docs = list(json_loader.process([doc]))

    assert len(loaded_docs) == 1
    assert loaded_docs[0].metadata['content_type'] == 'json'
    
    # Parse the JSON content
    # JSONコンテンツを解析
    json_content = json.loads(loaded_docs[0].content)
    assert 'users' in json_content
    assert len(json_content['users']) == 3

    first_user = json_content['users'][0]
    assert first_user['name'] == 'John'
    assert first_user['age'] == 30

def test_json_loader_nonexistent_file(json_loader):
    """
    Test that JSONLoader raises FileNotFoundError for nonexistent files.
    JSONLoaderが存在しないファイルに対してFileNotFoundErrorを発生させることをテストする。
    """
    doc = Document(
        id="test_json_4",
        content="",
        metadata={'file_path': 'nonexistent.json'}
    )

    with pytest.raises(FileNotFoundError):
        list(json_loader.process([doc]))

def test_json_loader_invalid_json(tmp_path, json_loader):
    """
    Test that JSONLoader raises JSONDecodeError for invalid JSON files.
    JSONLoaderが不正なJSONファイルに対してJSONDecodeErrorを発生させることをテストする。
    """
    file_path = tmp_path / "invalid.json"
    content = "{invalid json}"
    file_path.write_text(content, encoding='utf-8')

    doc = Document(
        id="test_json_5",
        content="",
        metadata={'file_path': str(file_path)}
    )

    with pytest.raises(Exception) as exc_info:
        list(json_loader.process([doc]))
    assert "Invalid JSON" in str(exc_info.value) 