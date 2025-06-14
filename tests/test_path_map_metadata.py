import pytest
from pathlib import Path
from refinire_rag.metadata.path_map_metadata import PathMapMetadata
from refinire_rag.models.document import Document
from collections import OrderedDict

@pytest.fixture
def path_map():
    """
    Create a path map for testing.
    
    テスト用のパスマップを作成。
    """
    return OrderedDict([
        ("hr/*", {"department": "HR", "category": "personnel"}),
        ("finance/*", {"department": "Finance", "category": "financial"}),
        ("draft/*", {"status": "draft"}),
        ("**/policy.txt", {"type": "policy", "importance": "high"}),
        (r".*[/\\]contract.*\.pdf$", {"type": "contract"}),
        ("*report*", {"document_type": "report"}),
    ])

@pytest.fixture
def regex_path_map():
    # 正規表現用パターン
    return {
        r"policy\.txt$": {"type": "policy", "importance": "high"},
        r".*contract.*\.pdf$": {"type": "contract", "importance": "high", "department": "Finance", "category": "financial"},
        r"^hr/.*$": {"department": "HR", "category": "personnel"},
        r"^finance/.*$": {"department": "Finance", "category": "financial"},
        r"^draft/.*$": {"status": "draft"},
        r"^final/.*$": {"status": "final"},
        r"^confidential/.*$": {"security_level": "high"},
        r"report": {"document_type": "report"}
    }

@pytest.fixture
def metadata_processor(path_map):
    """
    Create a PathMapMetadata processor instance.
    
    PathMapMetadataプロセッサーのインスタンスを作成。
    """
    return PathMapMetadata(path_map)

@pytest.fixture
def regex_processor(regex_path_map):
    """
    Create a PathMapMetadata processor instance with regex support.
    
    正規表現サポート付きのPathMapMetadataプロセッサーのインスタンスを作成。
    """
    return PathMapMetadata(regex_path_map, use_regex=True)

@pytest.fixture
def priority_processor(path_map):
    """
    Create a PathMapMetadata processor instance with priority-based matching.
    
    優先順位ベースのマッチングを使用するPathMapMetadataプロセッサーのインスタンスを作成。
    """
    return PathMapMetadata(path_map, priority_based=True)

@pytest.fixture
def merge_processor(regex_path_map):
    """
    Create a PathMapMetadata processor instance with merge strategy.
    
    マージ戦略を使用するPathMapMetadataプロセッサーのインスタンスを作成。
    """
    return PathMapMetadata(regex_path_map, use_regex=True, merge_strategy="merge")

def test_path_map_metadata_single_match(metadata_processor):
    """
    Test that metadata is correctly added for a single path match.
    
    単一のパスマッチに対してメタデータが正しく追加されることをテスト。
    """
    metadata = {"path": "hr/employee_list.txt"}
    
    processed_metadata = metadata_processor.get_metadata(metadata["path"], metadata)
    assert processed_metadata["department"] == "HR"
    assert processed_metadata["category"] == "personnel"

def test_path_map_metadata_multiple_matches(metadata_processor):
    """
    Test that metadata is correctly merged for multiple path matches.
    
    複数のパスマッチに対してメタデータが正しくマージされることをテスト。
    """
    metadata = {"path": "hr/policy.txt"}
    
    processed_metadata = metadata_processor.get_metadata(metadata["path"], metadata)
    assert processed_metadata["type"] == "policy"
    assert processed_metadata["importance"] == "high"
    assert processed_metadata["department"] == "HR"
    assert processed_metadata["category"] == "personnel"

def test_path_map_metadata_no_match(metadata_processor):
    """
    Test that metadata is unchanged when no path matches.
    
    パスマッチがない場合にメタデータが変更されないことをテスト。
    """
    metadata = {"path": "other/file.txt"}
    
    processed_metadata = metadata_processor.get_metadata(metadata["path"], metadata)
    assert processed_metadata == metadata

def test_path_map_metadata_no_path(metadata_processor):
    """
    Test that metadata is unchanged when no path is present.
    
    パスが存在しない場合にメタデータが変更されないことをテスト。
    """
    metadata = {"other": "value"}
    
    processed_metadata = metadata_processor.get_metadata("", metadata)
    assert processed_metadata == metadata

def test_path_map_metadata_priority(priority_processor):
    """
    Test that metadata is correctly merged based on priority.
    
    優先順位に基づいてメタデータが正しくマージされることをテスト。
    """
    metadata = {"path": "draft/policy.txt"}
    
    processed_metadata = priority_processor.get_metadata(metadata["path"], metadata)
    assert processed_metadata["status"] == "draft"
    assert processed_metadata["type"] == "policy"
    assert processed_metadata["importance"] == "high"

def test_path_map_metadata_regex_patterns(regex_processor):
    """
    Test that regex patterns in path maps work correctly.
    
    パスマップの正規表現パターンが正しく機能することをテスト。
    """
    metadata = {"path": "finance/contract_2024.pdf"}
    
    processed_metadata = regex_processor.get_metadata(metadata["path"], metadata)
    assert processed_metadata["type"] == "contract"
    assert processed_metadata["importance"] == "high"
    assert processed_metadata["department"] == "Finance"
    assert processed_metadata["category"] == "financial"

def test_path_map_metadata_empty_map():
    """
    Test that empty path map is handled correctly.
    
    空のパスマップが正しく処理されることをテスト。
    """
    metadata_processor = PathMapMetadata({})
    metadata = {"path": "test/path.txt"}
    
    processed_metadata = metadata_processor.get_metadata(metadata["path"], metadata)
    assert processed_metadata == metadata

def test_path_map_metadata_matches_hr_document(metadata_processor):
    """
    Test that HR document gets correct metadata.
    
    人事部ドキュメントに正しいメタデータが付与されることをテスト。
    """
    doc = Document(
        id="test1",
        content="HR document content",
        metadata={"path": "hr/policy.txt"}
    )
    
    new_metadata = metadata_processor.get_metadata(doc.metadata["path"], doc.metadata)
    assert new_metadata["type"] == "policy"
    assert new_metadata["importance"] == "high"
    assert new_metadata["department"] == "HR"
    assert new_metadata["category"] == "personnel"

def test_path_map_metadata_matches_finance_document(metadata_processor):
    """
    Test that Finance document gets correct metadata.
    
    経理部ドキュメントに正しいメタデータが付与されることをテスト。
    """
    doc = Document(
        id="test2",
        content="Finance document content",
        metadata={"path": "finance/report.pdf"}
    )
    
    new_metadata = metadata_processor.get_metadata(doc.metadata["path"], doc.metadata)
    assert new_metadata["department"] == "Finance"
    assert new_metadata["category"] == "financial"
    assert new_metadata["document_type"] == "report"

def test_path_map_metadata_case_insensitive(metadata_processor):
    """
    Test that path matching is case insensitive.
    
    パスマッチングが大文字小文字を区別しないことをテスト。
    """
    doc = Document(
        id="test4",
        content="HR document content",
        metadata={"path": "HR/policy.txt"}
    )
    
    new_metadata = metadata_processor.get_metadata(doc.metadata["path"], doc.metadata)
    assert new_metadata["type"] == "policy"
    assert new_metadata["importance"] == "high"
    assert new_metadata["department"] == "HR"
    assert new_metadata["category"] == "personnel"

def test_regex_pattern_matching(regex_processor):
    """
    Test that regex patterns are correctly matched.
    
    正規表現パターンが正しくマッチすることをテスト。
    """
    metadata = {"path": "confidential/secret.txt"}
    
    processed_metadata = regex_processor.get_metadata(metadata["path"], metadata)
    assert processed_metadata["security_level"] == "high"

def test_priority_based_matching(priority_processor):
    """
    Test that priority-based matching works correctly.
    
    優先順位ベースのマッチングが正しく機能することをテスト。
    """
    metadata = {"path": "hr/report.txt"}
    
    processed_metadata = priority_processor.get_metadata(metadata["path"], metadata)
    assert processed_metadata["department"] == "HR"
    assert processed_metadata["category"] == "personnel"
    assert processed_metadata["document_type"] == "report"

def test_merge_strategy(merge_processor):
    """
    Test that merge strategy correctly combines metadata from multiple matches.
    
    マージ戦略が複数のマッチからメタデータを正しく結合することをテスト。
    """
    metadata = {"path": "confidential/report.txt"}
    
    processed_metadata = merge_processor.get_metadata(metadata["path"], metadata)
    assert processed_metadata["security_level"] == "high"
    assert processed_metadata["document_type"] == "report" 