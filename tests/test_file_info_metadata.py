import pytest
from pathlib import Path
import os
import time
from refinire_rag.metadata.file_info_metadata import FileInfoMetadata, FileAttribute
from refinire_rag.models.document import Document

@pytest.fixture
def test_file(tmp_path):
    """
    Create a test file for metadata testing.
    
    メタデータテスト用のテストファイルを作成。
    """
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")
    return file_path

@pytest.fixture
def test_hidden_file(tmp_path):
    """
    Create a hidden test file.
    
    隠しファイルのテストファイルを作成。
    """
    hidden_file = tmp_path / ".hidden.txt"
    hidden_file.write_text("hidden content")
    return hidden_file

@pytest.fixture
def test_symlink(tmp_path):
    """
    Create a symlink for testing.
    
    シンボリックリンクのテストファイルを作成。
    """
    target_file = tmp_path / "target.txt"
    target_file.write_text("target content")
    symlink_path = tmp_path / "symlink.txt"
    symlink_path.symlink_to(target_file)
    return symlink_path

@pytest.fixture
def metadata_processor():
    """
    Create a FileInfoMetadata processor with all attributes.
    
    すべての属性を持つFileInfoMetadataプロセッサーを作成。
    """
    return FileInfoMetadata()

def test_file_info_metadata_all_attributes(metadata_processor, test_file):
    """
    Test that all file attributes are correctly added as metadata.
    
    すべてのファイル属性が正しくメタデータとして追加されることをテスト。
    """
    metadata = {}
    processed_metadata = metadata_processor.get_metadata(metadata, test_file)
    
    # Check basic file attributes
    assert processed_metadata["file_name"] == "test.txt"
    assert processed_metadata["file_extension"] == "txt"
    assert processed_metadata["file_size"] > 0
    assert processed_metadata["parent_folder"] == test_file.parent.name
    assert processed_metadata["relative_path"] == str(test_file)
    assert processed_metadata["absolute_path"] == str(test_file.absolute())
    assert processed_metadata["is_hidden"] is False
    assert processed_metadata["is_symlink"] is False
    
    # Check timestamps
    assert "created_at" in processed_metadata
    assert "modified_at" in processed_metadata
    assert "imported_at" in processed_metadata
    assert processed_metadata["imported_at"] > processed_metadata["created_at"]

def test_file_info_metadata_selected_attributes(test_file):
    """
    Test that only selected attributes are added as metadata.
    
    選択された属性のみがメタデータとして追加されることをテスト。
    """
    selected_attributes = {
        FileAttribute.FILE_NAME,
        FileAttribute.FILE_SIZE,
        FileAttribute.IMPORTED_AT
    }
    metadata_processor = FileInfoMetadata(selected_attributes)
    
    metadata = {}
    processed_metadata = metadata_processor.get_metadata(metadata, test_file)
    
    # Check that only selected attributes are present
    assert set(processed_metadata.keys()) == {"file_name", "file_size", "imported_at"}
    assert processed_metadata["file_name"] == "test.txt"
    assert processed_metadata["file_size"] > 0
    assert "imported_at" in processed_metadata

def test_file_info_metadata_hidden_file(test_hidden_file):
    """
    Test that hidden file attribute is correctly detected.
    
    隠しファイル属性が正しく検出されることをテスト。
    """
    metadata_processor = FileInfoMetadata({FileAttribute.IS_HIDDEN})
    metadata = {}
    
    processed_metadata = metadata_processor.get_metadata(metadata, test_hidden_file)
    
    assert processed_metadata["is_hidden"] is True

def test_file_info_metadata_symlink(test_symlink):
    """
    Test that symlink attribute is correctly detected.
    
    シンボリックリンク属性が正しく検出されることをテスト。
    """
    metadata_processor = FileInfoMetadata({FileAttribute.IS_SYMLINK})
    metadata = {}
    
    processed_metadata = metadata_processor.get_metadata(metadata, test_symlink)
    
    assert processed_metadata["is_symlink"] is True

def test_file_info_metadata_no_path():
    """
    Test that metadata without path returns unchanged.
    
    パスがないメタデータが変更されずに返されることをテスト。
    """
    metadata_processor = FileInfoMetadata()
    metadata = {"other": "value"}
    
    processed_metadata = metadata_processor.get_metadata(metadata)
    
    assert processed_metadata == metadata

def test_file_info_metadata_nonexistent_file(metadata_processor):
    """
    Test handling of nonexistent file.
    
    存在しないファイルの処理をテスト。
    """
    nonexistent_file = Path("nonexistent.txt")
    metadata = {"path": "nonexistent.txt"}
    
    processed_metadata = metadata_processor.get_metadata(metadata, nonexistent_file)
    
    # Check that original metadata is returned unchanged
    assert processed_metadata == metadata

def test_file_info_metadata_no_file_no_path(metadata_processor):
    """
    Test handling when no file and no path are provided.
    
    ファイルとパスの両方が提供されない場合の処理をテスト。
    """
    metadata = {}
    processed_metadata = metadata_processor.get_metadata(metadata)
    
    # Check that original metadata is returned unchanged
    assert processed_metadata == metadata

def test_file_info_metadata_timestamps(test_file):
    """
    Test that timestamps are correctly recorded.
    
    タイムスタンプが正しく記録されることをテスト。
    """
    metadata_processor = FileInfoMetadata({
        FileAttribute.CREATED_AT,
        FileAttribute.MODIFIED_AT,
        FileAttribute.IMPORTED_AT
    })
    
    # Get initial metadata
    metadata = {}
    initial_metadata = metadata_processor.get_metadata(metadata, test_file)
    initial_import_time = initial_metadata["imported_at"]
    
    # Wait a moment
    time.sleep(0.1)
    
    # Get metadata again
    second_metadata = metadata_processor.get_metadata(metadata, test_file)
    second_import_time = second_metadata["imported_at"]
    
    # Check that import time is different
    assert second_import_time > initial_import_time
    
    # Check that created and modified times are the same
    assert initial_metadata["created_at"] == second_metadata["created_at"]
    assert initial_metadata["modified_at"] == second_metadata["modified_at"] 