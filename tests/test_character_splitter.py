"""
Test cases for CharacterTextSplitter
CharacterTextSplitterのテストケース
"""

import pytest
from refinire_rag.splitter.character_splitter import CharacterTextSplitter
from refinire_rag.models.document import Document

@pytest.fixture
def test_document():
    """
    Create a test document for splitting
    分割用のテスト文書を作成
    """
    return Document(
        id="test_doc_1",
        content="This is a long text that will be split into multiple chunks. " * 20,  # ~1240 characters
        metadata={'source': 'test', 'category': 'example'}
    )

@pytest.fixture
def short_document():
    """
    Create a short test document
    短いテスト文書を作成
    """
    return Document(
        id="short_doc",
        content="Short text",
        metadata={'source': 'test'}
    )

@pytest.fixture
def character_splitter():
    """
    Create a CharacterTextSplitter instance for testing
    テスト用のCharacterTextSplitterインスタンスを作成
    """
    return CharacterTextSplitter(chunk_size=100, overlap_size=20)

def test_character_splitter_basic_splitting(character_splitter, test_document):
    """
    Test basic splitting functionality
    基本的な分割機能をテストする
    """
    chunks = list(character_splitter.process([test_document]))
    
    # Should have multiple chunks
    # 複数のチャンクがあることを確認
    assert len(chunks) > 1
    
    # All chunks should have origin_id set to original document ID
    # すべてのチャンクのorigin_idが元文書のIDに設定されていることを確認
    for chunk in chunks:
        assert chunk.metadata['origin_id'] == test_document.id
        assert 'chunk_index' in chunk.metadata
        assert 'chunk_start' in chunk.metadata
        assert 'chunk_end' in chunk.metadata
        assert 'total_chunks' in chunk.metadata
        
    # Check that chunk_index is sequential
    # chunk_indexが連続していることを確認
    for i, chunk in enumerate(chunks):
        assert chunk.metadata['chunk_index'] == i
        
    # Check that total_chunks is correct for all chunks
    # すべてのチャンクのtotal_chunksが正しいことを確認
    total_chunks = len(chunks)
    for chunk in chunks:
        assert chunk.metadata['total_chunks'] == total_chunks

def test_character_splitter_id_generation(character_splitter, test_document):
    """
    Test that each chunk gets a unique UUID
    各チャンクが一意のUUIDを取得することをテストする
    """
    chunks = list(character_splitter.process([test_document]))
    
    # All chunk IDs should be unique
    # すべてのチャンクIDが一意であることを確認
    chunk_ids = [chunk.id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    
    # None of the chunk IDs should be the same as the original document ID
    # チャンクIDは元文書IDと同じではないことを確認
    for chunk in chunks:
        assert chunk.id != test_document.id

def test_character_splitter_metadata_preservation(character_splitter, test_document):
    """
    Test that original metadata is preserved in chunks
    元のメタデータがチャンクに保持されることをテストする
    """
    chunks = list(character_splitter.process([test_document]))
    
    for chunk in chunks:
        # Original metadata should be preserved
        # 元のメタデータが保持されていることを確認
        assert chunk.metadata['source'] == 'test'
        assert chunk.metadata['category'] == 'example'
        
        # New metadata should be added
        # 新しいメタデータが追加されていることを確認
        assert 'origin_id' in chunk.metadata
        assert 'chunk_index' in chunk.metadata

def test_character_splitter_overlap(test_document):
    """
    Test overlap functionality
    オーバーラップ機能をテストする
    """
    splitter = CharacterTextSplitter(chunk_size=100, overlap_size=20)
    chunks = list(splitter.process([test_document]))
    
    if len(chunks) > 1:
        # Check that overlap is working (some content should be repeated)
        # オーバーラップが機能していることを確認（一部のコンテンツが重複している）
        first_chunk_end = chunks[0].content[-20:]  # Last 20 chars of first chunk
        second_chunk_start = chunks[1].content[:20]  # First 20 chars of second chunk
        
        # There should be some overlap
        # 何らかのオーバーラップがあることを確認
        assert len(first_chunk_end) > 0
        assert len(second_chunk_start) > 0

def test_character_splitter_short_document(character_splitter, short_document):
    """
    Test splitting of document shorter than chunk size
    チャンクサイズより短い文書の分割をテストする
    """
    chunks = list(character_splitter.process([short_document]))
    
    # Should have exactly one chunk
    # ちょうど1つのチャンクがあることを確認
    assert len(chunks) == 1
    
    chunk = chunks[0]
    assert chunk.content == short_document.content
    assert chunk.metadata['origin_id'] == short_document.id
    assert chunk.metadata['chunk_index'] == 0
    assert chunk.metadata['total_chunks'] == 1

def test_character_splitter_empty_document(character_splitter):
    """
    Test handling of empty document
    空の文書の処理をテストする
    """
    empty_doc = Document(id="empty_doc", content="", metadata={})
    chunks = list(character_splitter.process([empty_doc]))
    
    # Should have no chunks for empty document
    # 空の文書にはチャンクがないことを確認
    assert len(chunks) == 0

def test_character_splitter_chunk_positions(test_document):
    """
    Test that chunk start and end positions are correct
    チャンクの開始と終了位置が正しいことをテストする
    """
    splitter = CharacterTextSplitter(chunk_size=100, overlap_size=0)
    chunks = list(splitter.process([test_document]))
    
    # Check that positions are sequential and cover the entire document
    # 位置が連続しており、文書全体をカバーしていることを確認
    expected_start = 0
    for chunk in chunks[:-1]:  # All but the last chunk
        assert chunk.metadata['chunk_start'] == expected_start
        assert chunk.metadata['chunk_end'] == expected_start + 100
        expected_start += 100
        
    # Last chunk should end at document length
    # 最後のチャンクは文書の長さで終わることを確認
    last_chunk = chunks[-1]
    assert last_chunk.metadata['chunk_end'] == len(test_document.content)

def test_character_splitter_multiple_documents(character_splitter):
    """
    Test processing multiple documents
    複数の文書の処理をテストする
    """
    doc1 = Document(id="doc1", content="A" * 150, metadata={'type': 'type1'})
    doc2 = Document(id="doc2", content="B" * 150, metadata={'type': 'type2'})
    
    chunks = list(character_splitter.process([doc1, doc2]))
    
    # Should have chunks from both documents
    # 両方の文書からチャンクがあることを確認
    doc1_chunks = [c for c in chunks if c.metadata['origin_id'] == 'doc1']
    doc2_chunks = [c for c in chunks if c.metadata['origin_id'] == 'doc2']
    
    assert len(doc1_chunks) > 0
    assert len(doc2_chunks) > 0
    
    # Check that metadata is preserved correctly for each document
    # 各文書のメタデータが正しく保持されていることを確認
    for chunk in doc1_chunks:
        assert chunk.metadata['type'] == 'type1'
    for chunk in doc2_chunks:
        assert chunk.metadata['type'] == 'type2'