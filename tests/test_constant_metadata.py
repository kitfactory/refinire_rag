import pytest
from refinire_rag.metadata.constant_metadata import ConstantMetadata

@pytest.fixture
def constant_metadata():
    """
    Create constant metadata for testing.
    
    テスト用の定数のメタデータを作成。
    """
    return {
        "department": "HR",
        "category": "policy",
        "version": "1.0",
        "tags": ["important", "internal"]
    }

@pytest.fixture
def metadata_processor(constant_metadata):
    """
    Create a ConstantMetadata processor instance.
    
    ConstantMetadataプロセッサーのインスタンスを作成。
    """
    return ConstantMetadata(constant_metadata)

def test_constant_metadata_adds_metadata(metadata_processor, constant_metadata):
    """
    Test that constant metadata is correctly added.
    
    定数のメタデータが正しく追加されることをテスト。
    """
    metadata = {"path": "/test/path.txt"}
    
    processed_metadata = metadata_processor.get_metadata(metadata)
    
    # Check that all constant metadata is present
    for key, value in constant_metadata.items():
        assert processed_metadata[key] == value
    
    # Check that existing metadata is preserved
    assert processed_metadata["path"] == "/test/path.txt"

def test_constant_metadata_overwrites_existing(metadata_processor):
    """
    Test that constant metadata overwrites existing metadata with same keys.
    
    定数のメタデータが同じキーの既存のメタデータを上書きすることをテスト。
    """
    metadata = {
        "department": "Finance",
        "path": "/test/path.txt"
    }
    
    processed_metadata = metadata_processor.get_metadata(metadata)
    
    # Check that constant metadata overwrites existing
    assert processed_metadata["department"] == "HR"
    # Check that other existing metadata is preserved
    assert processed_metadata["path"] == "/test/path.txt"

def test_constant_metadata_empty_metadata(metadata_processor, constant_metadata):
    """
    Test that constant metadata is added to empty metadata.
    
    空のメタデータに定数のメタデータが追加されることをテスト。
    """
    metadata = {}
    
    processed_metadata = metadata_processor.get_metadata(metadata)
    
    # Check that all constant metadata is present
    for key, value in constant_metadata.items():
        assert processed_metadata[key] == value

def test_constant_metadata_empty_metadata_dict():
    """
    Test that empty metadata dictionary is handled correctly.
    
    空のメタデータ辞書が正しく処理されることをテスト。
    """
    metadata_processor = ConstantMetadata({})
    metadata = {"path": "/test/path.txt"}
    
    processed_metadata = metadata_processor.get_metadata(metadata)
    
    # Check that metadata is unchanged
    assert processed_metadata == metadata 