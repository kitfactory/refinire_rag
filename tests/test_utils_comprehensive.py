"""
Comprehensive tests for utils functionality
utils機能の包括的テスト

This module provides comprehensive coverage for the utils module,
testing UUID generation functions, format validation, and uniqueness guarantees.
このモジュールは、utilsモジュールの包括的カバレッジを提供し、
UUID生成関数、フォーマット検証、一意性保証をテストします。
"""

import pytest
import uuid
import re
from typing import Set
from unittest.mock import patch, Mock

from refinire_rag.utils import generate_document_id, generate_chunk_id


class TestGenerateDocumentId:
    """
    Test generate_document_id function
    generate_document_id関数のテスト
    """
    
    def test_generate_document_id_returns_string(self):
        """
        Test that generate_document_id returns a string
        generate_document_idが文字列を返すことのテスト
        """
        result = generate_document_id()
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_document_id_uuid_format(self):
        """
        Test that generate_document_id returns valid UUID format
        generate_document_idが有効なUUID形式を返すことのテスト
        """
        result = generate_document_id()
        
        # Check UUID format (8-4-4-4-12 hexadecimal characters)
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        assert re.match(uuid_pattern, result), f"Result '{result}' is not a valid UUID format"
    
    def test_generate_document_id_is_valid_uuid(self):
        """
        Test that generate_document_id returns valid UUID that can be parsed
        generate_document_idが解析可能な有効なUUIDを返すことのテスト
        """
        result = generate_document_id()
        
        # Should be able to parse as UUID without raising exception
        parsed_uuid = uuid.UUID(result)
        assert str(parsed_uuid) == result
    
    def test_generate_document_id_uniqueness(self):
        """
        Test that generate_document_id generates unique IDs
        generate_document_idが一意のIDを生成することのテスト
        """
        ids = [generate_document_id() for _ in range(1000)]
        
        # All IDs should be unique
        assert len(set(ids)) == len(ids), "Generated IDs are not unique"
    
    def test_generate_document_id_multiple_calls(self):
        """
        Test multiple calls to generate_document_id return different values
        generate_document_idの複数回呼び出しが異なる値を返すことのテスト
        """
        id1 = generate_document_id()
        id2 = generate_document_id()
        id3 = generate_document_id()
        
        assert id1 != id2
        assert id1 != id3
        assert id2 != id3
    
    def test_generate_document_id_uuid_version(self):
        """
        Test that generate_document_id returns UUID version 4
        generate_document_idがUUIDバージョン4を返すことのテスト
        """
        result = generate_document_id()
        parsed_uuid = uuid.UUID(result)
        
        # UUID4 has version 4
        assert parsed_uuid.version == 4


class TestGenerateChunkId:
    """
    Test generate_chunk_id function
    generate_chunk_id関数のテスト
    """
    
    def test_generate_chunk_id_returns_string(self):
        """
        Test that generate_chunk_id returns a string
        generate_chunk_idが文字列を返すことのテスト
        """
        result = generate_chunk_id()
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_chunk_id_uuid_format(self):
        """
        Test that generate_chunk_id returns valid UUID format
        generate_chunk_idが有効なUUID形式を返すことのテスト
        """
        result = generate_chunk_id()
        
        # Check UUID format (8-4-4-4-12 hexadecimal characters)
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        assert re.match(uuid_pattern, result), f"Result '{result}' is not a valid UUID format"
    
    def test_generate_chunk_id_is_valid_uuid(self):
        """
        Test that generate_chunk_id returns valid UUID that can be parsed
        generate_chunk_idが解析可能な有効なUUIDを返すことのテスト
        """
        result = generate_chunk_id()
        
        # Should be able to parse as UUID without raising exception
        parsed_uuid = uuid.UUID(result)
        assert str(parsed_uuid) == result
    
    def test_generate_chunk_id_uniqueness(self):
        """
        Test that generate_chunk_id generates unique IDs
        generate_chunk_idが一意のIDを生成することのテスト
        """
        ids = [generate_chunk_id() for _ in range(1000)]
        
        # All IDs should be unique
        assert len(set(ids)) == len(ids), "Generated IDs are not unique"
    
    def test_generate_chunk_id_multiple_calls(self):
        """
        Test multiple calls to generate_chunk_id return different values
        generate_chunk_idの複数回呼び出しが異なる値を返すことのテスト
        """
        id1 = generate_chunk_id()
        id2 = generate_chunk_id()
        id3 = generate_chunk_id()
        
        assert id1 != id2
        assert id1 != id3
        assert id2 != id3
    
    def test_generate_chunk_id_uuid_version(self):
        """
        Test that generate_chunk_id returns UUID version 4
        generate_chunk_idがUUIDバージョン4を返すことのテスト
        """
        result = generate_chunk_id()
        parsed_uuid = uuid.UUID(result)
        
        # UUID4 has version 4
        assert parsed_uuid.version == 4


class TestDocumentIdVsChunkId:
    """
    Test differences and similarities between document and chunk ID generation
    文書IDとチャンクID生成の違いと類似点のテスト
    """
    
    def test_document_and_chunk_ids_are_different_functions(self):
        """
        Test that document and chunk IDs use the same generation logic but are independent
        文書IDとチャンクIDが同じ生成ロジックを使用するが独立していることのテスト
        """
        doc_id = generate_document_id()
        chunk_id = generate_chunk_id()
        
        # Both should be valid UUIDs
        assert uuid.UUID(doc_id)
        assert uuid.UUID(chunk_id)
        
        # They should be different (extremely high probability)
        assert doc_id != chunk_id
    
    def test_document_and_chunk_ids_same_format(self):
        """
        Test that document and chunk IDs have the same format
        文書IDとチャンクIDが同じ形式を持つことのテスト
        """
        doc_id = generate_document_id()
        chunk_id = generate_chunk_id()
        
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        
        assert re.match(uuid_pattern, doc_id)
        assert re.match(uuid_pattern, chunk_id)
        assert len(doc_id) == len(chunk_id)
    
    def test_mixed_document_and_chunk_ids_uniqueness(self):
        """
        Test uniqueness across both document and chunk ID generation
        文書IDとチャンクID生成全体での一意性テスト
        """
        all_ids = []
        
        # Generate mix of document and chunk IDs
        for _ in range(500):
            all_ids.append(generate_document_id())
            all_ids.append(generate_chunk_id())
        
        # All should be unique
        assert len(set(all_ids)) == len(all_ids), "Mixed IDs are not unique"


class TestUuidGenerationBehavior:
    """
    Test UUID generation behavior and edge cases
    UUID生成の動作とエッジケースのテスト
    """
    
    def test_uuid_generation_speed(self):
        """
        Test UUID generation performance
        UUID生成パフォーマンステスト
        """
        import time
        
        start_time = time.time()
        
        # Generate many IDs quickly
        for _ in range(10000):
            generate_document_id()
            generate_chunk_id()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time (less than 1 second for 20k IDs)
        assert duration < 1.0, f"ID generation too slow: {duration} seconds for 20k IDs"
    
    def test_uuid_randomness_distribution(self):
        """
        Test that UUID generation has good randomness distribution
        UUID生成が良好なランダム性分布を持つことのテスト
        """
        ids = [generate_document_id() for _ in range(100)]
        
        # Extract first character of each ID to check distribution
        first_chars = [id_[0] for id_ in ids]
        unique_first_chars = set(first_chars)
        
        # Should have reasonable distribution (at least 5 different first chars in 100 IDs)
        assert len(unique_first_chars) >= 5, f"Poor randomness distribution: only {len(unique_first_chars)} different first characters"
    
    def test_uuid_consistency_across_calls(self):
        """
        Test that UUID format is consistent across many calls
        多数の呼び出しでUUID形式が一貫していることのテスト
        """
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        
        # Test many IDs for consistent format
        for _ in range(100):
            doc_id = generate_document_id()
            chunk_id = generate_chunk_id()
            
            assert re.match(uuid_pattern, doc_id), f"Inconsistent format: {doc_id}"
            assert re.match(uuid_pattern, chunk_id), f"Inconsistent format: {chunk_id}"


class TestUuidMocking:
    """
    Test UUID generation with mocking for deterministic behavior
    決定論的動作のためのモックを使用したUUID生成テスト
    """
    
    @patch('refinire_rag.utils.uuid.uuid4')
    def test_generate_document_id_with_mock(self, mock_uuid4):
        """
        Test generate_document_id behavior with mocked UUID
        モックされたUUIDでのgenerate_document_id動作テスト
        """
        # Mock UUID to return predictable value
        mock_uuid_obj = Mock()
        mock_uuid_obj.__str__ = Mock(return_value='test-uuid-string')
        mock_uuid4.return_value = mock_uuid_obj
        
        result = generate_document_id()
        
        assert result == 'test-uuid-string'
        mock_uuid4.assert_called_once()
    
    @patch('refinire_rag.utils.uuid.uuid4')
    def test_generate_chunk_id_with_mock(self, mock_uuid4):
        """
        Test generate_chunk_id behavior with mocked UUID
        モックされたUUIDでのgenerate_chunk_id動作テスト
        """
        # Mock UUID to return predictable value
        mock_uuid_obj = Mock()
        mock_uuid_obj.__str__ = Mock(return_value='test-chunk-uuid')
        mock_uuid4.return_value = mock_uuid_obj
        
        result = generate_chunk_id()
        
        assert result == 'test-chunk-uuid'
        mock_uuid4.assert_called_once()
    
    @patch('refinire_rag.utils.uuid.uuid4')
    def test_multiple_calls_with_mock(self, mock_uuid4):
        """
        Test multiple calls with mocked UUID
        モックされたUUIDでの複数回呼び出しテスト
        """
        # Mock UUID to return different values for each call
        mock_uuid_obj1 = Mock()
        mock_uuid_obj1.__str__ = Mock(return_value='uuid-1')
        mock_uuid_obj2 = Mock()
        mock_uuid_obj2.__str__ = Mock(return_value='uuid-2')
        
        mock_uuid4.side_effect = [mock_uuid_obj1, mock_uuid_obj2]
        
        result1 = generate_document_id()
        result2 = generate_chunk_id()
        
        assert result1 == 'uuid-1'
        assert result2 == 'uuid-2'
        assert mock_uuid4.call_count == 2


class TestUuidIntegration:
    """
    Test UUID generation integration scenarios
    UUID生成統合シナリオのテスト
    """
    
    def test_document_id_as_parent_reference(self):
        """
        Test using document ID as parent reference for chunk IDs
        チャンクIDの親参照として文書IDを使用するテスト
        """
        doc_id = generate_document_id()
        
        # Generate several chunk IDs that could reference this document
        chunk_ids = [generate_chunk_id() for _ in range(5)]
        
        # Document ID should be different from all chunk IDs
        for chunk_id in chunk_ids:
            assert doc_id != chunk_id
        
        # All chunk IDs should be unique
        assert len(set(chunk_ids)) == len(chunk_ids)
    
    def test_id_generation_for_batch_processing(self):
        """
        Test ID generation for batch processing scenarios
        バッチ処理シナリオでのID生成テスト
        """
        batch_size = 100
        
        # Simulate batch processing: 1 document with many chunks
        document_ids = [generate_document_id() for _ in range(batch_size)]
        chunk_ids = [generate_chunk_id() for _ in range(batch_size * 10)]  # 10 chunks per doc
        
        # All document IDs should be unique
        assert len(set(document_ids)) == len(document_ids)
        
        # All chunk IDs should be unique
        assert len(set(chunk_ids)) == len(chunk_ids)
        
        # No overlap between document and chunk IDs
        all_ids = set(document_ids + chunk_ids)
        assert len(all_ids) == len(document_ids) + len(chunk_ids)
    
    def test_concurrent_id_generation_simulation(self):
        """
        Test simulated concurrent ID generation
        並行ID生成のシミュレーションテスト
        """
        # Simulate multiple "threads" generating IDs
        thread1_ids = [generate_document_id() for _ in range(50)]
        thread2_ids = [generate_chunk_id() for _ in range(50)]
        thread3_ids = [generate_document_id() for _ in range(30)]
        thread4_ids = [generate_chunk_id() for _ in range(30)]
        
        all_ids = thread1_ids + thread2_ids + thread3_ids + thread4_ids
        
        # All IDs should be unique across "threads"
        assert len(set(all_ids)) == len(all_ids), "Concurrent ID generation produced duplicates"


class TestUuidStringBehavior:
    """
    Test UUID string behavior and properties
    UUID文字列の動作と特性のテスト
    """
    
    def test_uuid_string_properties(self):
        """
        Test properties of generated UUID strings
        生成されたUUID文字列の特性テスト
        """
        doc_id = generate_document_id()
        chunk_id = generate_chunk_id()
        
        for id_value in [doc_id, chunk_id]:
            # Should be lowercase
            assert id_value.islower(), f"UUID should be lowercase: {id_value}"
            
            # Should contain exactly 4 hyphens
            assert id_value.count('-') == 4, f"UUID should have 4 hyphens: {id_value}"
            
            # Should be exactly 36 characters long
            assert len(id_value) == 36, f"UUID should be 36 characters: {id_value}"
            
            # Should not start or end with hyphen
            assert not id_value.startswith('-')
            assert not id_value.endswith('-')
    
    def test_uuid_string_immutability(self):
        """
        Test that UUID strings are immutable
        UUID文字列が不変であることのテスト
        """
        doc_id = generate_document_id()
        original_id = doc_id
        
        # String operations should not modify original
        modified = doc_id.upper()
        assert doc_id == original_id
        assert modified != doc_id
    
    def test_uuid_string_comparison(self):
        """
        Test UUID string comparison behavior
        UUID文字列比較動作のテスト
        """
        id1 = generate_document_id()
        id2 = generate_document_id()
        id1_copy = str(id1)  # Create copy
        
        # Same UUID should be equal
        assert id1 == id1_copy
        
        # Different UUIDs should not be equal
        assert id1 != id2
        
        # String comparison should work
        assert (id1 < id2) or (id1 > id2)  # One should be lexicographically less/greater