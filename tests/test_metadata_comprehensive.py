"""
Comprehensive tests for metadata module
metadata モジュールの包括的テスト

This module provides comprehensive coverage for the metadata module,
testing abstract base class interface, inheritance patterns, and implementation scenarios.
このモジュールは、metadataモジュールの包括的カバレッジを提供し、
抽象基底クラスインターフェース、継承パターン、実装シナリオをテストします。
"""

import pytest
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
from unittest.mock import Mock, patch

from refinire_rag.metadata.metadata import Metadata


class TestMetadataAbstractBase:
    """
    Test Metadata abstract base class
    Metadata抽象基底クラスのテスト
    """
    
    def test_metadata_is_abstract_base_class(self):
        """
        Test that Metadata is an abstract base class
        MetadataがABCを継承することのテスト
        """
        assert issubclass(Metadata, ABC)
    
    def test_metadata_cannot_be_instantiated_directly(self):
        """
        Test that Metadata cannot be instantiated directly
        Metadataが直接インスタンス化できないことのテスト
        """
        with pytest.raises(TypeError) as exc_info:
            Metadata()
        
        # Should mention abstract method
        assert "abstract" in str(exc_info.value).lower()
    
    def test_metadata_has_abstract_get_metadata_method(self):
        """
        Test that Metadata has abstract get_metadata method
        Metadataが抽象get_metadataメソッドを持つことのテスト
        """
        # Check that get_metadata is marked as abstract
        assert hasattr(Metadata, 'get_metadata')
        assert getattr(Metadata.get_metadata, '__isabstractmethod__', False)
    
    def test_metadata_get_metadata_signature(self):
        """
        Test get_metadata method signature
        get_metadataメソッドのシグネチャテスト
        """
        import inspect
        signature = inspect.signature(Metadata.get_metadata)
        params = list(signature.parameters.keys())
        
        # Should have self, metadata, and file parameters
        assert 'self' in params
        assert 'metadata' in params
        assert 'file' in params
        
        # Check parameter annotations
        param_metadata = signature.parameters['metadata']
        param_file = signature.parameters['file']
        
        assert param_metadata.annotation == Dict[str, Any]
        assert param_file.annotation == Optional[Path]
        assert param_file.default is None
        
        # Check return annotation
        assert signature.return_annotation == Dict[str, Any]


class ConcreteMetadata(Metadata):
    """
    Concrete implementation of Metadata for testing
    テスト用のMetadataの具象実装
    """
    
    def __init__(self, add_timestamp: bool = True, add_source: bool = True):
        """
        Initialize concrete metadata processor
        具象メタデータプロセッサーを初期化
        """
        self.add_timestamp = add_timestamp
        self.add_source = add_source
        self.call_count = 0
    
    def get_metadata(self, metadata: Dict[str, Any], file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Implementation of get_metadata for testing
        テスト用のget_metadata実装
        """
        self.call_count += 1
        result = metadata.copy()
        
        if self.add_timestamp:
            result['processing_timestamp'] = '2024-01-01T00:00:00Z'
        
        if self.add_source and file:
            result['source_file'] = str(file)
        
        result['processor_type'] = 'ConcreteMetadata'
        return result


class TestConcreteMetadataImplementation:
    """
    Test concrete implementation of Metadata
    Metadataの具象実装のテスト
    """
    
    def test_concrete_metadata_can_be_instantiated(self):
        """
        Test that concrete implementation can be instantiated
        具象実装がインスタンス化できることのテスト
        """
        processor = ConcreteMetadata()
        assert isinstance(processor, Metadata)
        assert isinstance(processor, ConcreteMetadata)
    
    def test_concrete_metadata_default_parameters(self):
        """
        Test concrete metadata with default parameters
        デフォルトパラメータでの具象メタデータテスト
        """
        processor = ConcreteMetadata()
        assert processor.add_timestamp is True
        assert processor.add_source is True
        assert processor.call_count == 0
    
    def test_concrete_metadata_custom_parameters(self):
        """
        Test concrete metadata with custom parameters
        カスタムパラメータでの具象メタデータテスト
        """
        processor = ConcreteMetadata(add_timestamp=False, add_source=False)
        assert processor.add_timestamp is False
        assert processor.add_source is False
        assert processor.call_count == 0
    
    def test_get_metadata_with_empty_metadata(self):
        """
        Test get_metadata with empty metadata
        空のメタデータでのget_metadataテスト
        """
        processor = ConcreteMetadata()
        metadata = {}
        
        result = processor.get_metadata(metadata)
        
        assert isinstance(result, dict)
        assert result['processing_timestamp'] == '2024-01-01T00:00:00Z'
        assert result['processor_type'] == 'ConcreteMetadata'
        assert processor.call_count == 1
    
    def test_get_metadata_with_existing_metadata(self):
        """
        Test get_metadata with existing metadata
        既存のメタデータでのget_metadataテスト
        """
        processor = ConcreteMetadata()
        metadata = {
            'title': 'Test Document',
            'author': 'Test Author',
            'version': 1
        }
        
        result = processor.get_metadata(metadata)
        
        # Original metadata should be preserved
        assert result['title'] == 'Test Document'
        assert result['author'] == 'Test Author'
        assert result['version'] == 1
        
        # New metadata should be added
        assert result['processing_timestamp'] == '2024-01-01T00:00:00Z'
        assert result['processor_type'] == 'ConcreteMetadata'
        assert processor.call_count == 1
    
    def test_get_metadata_with_file_parameter(self):
        """
        Test get_metadata with file parameter
        fileパラメータでのget_metadataテスト
        """
        processor = ConcreteMetadata()
        metadata = {'title': 'Test'}
        file_path = Path('/test/document.txt')
        
        result = processor.get_metadata(metadata, file_path)
        
        assert result['source_file'] == '/test/document.txt'
        assert result['processor_type'] == 'ConcreteMetadata'
        assert processor.call_count == 1
    
    def test_get_metadata_without_file_parameter(self):
        """
        Test get_metadata without file parameter
        fileパラメータなしでのget_metadataテスト
        """
        processor = ConcreteMetadata()
        metadata = {'title': 'Test'}
        
        result = processor.get_metadata(metadata)
        
        # source_file should not be added when file is None
        assert 'source_file' not in result
        assert result['processor_type'] == 'ConcreteMetadata'
        assert processor.call_count == 1
    
    def test_get_metadata_preserves_original_metadata(self):
        """
        Test that get_metadata preserves original metadata
        get_metadataが元のメタデータを保持することのテスト
        """
        processor = ConcreteMetadata()
        original_metadata = {'title': 'Original', 'id': 123}
        
        result = processor.get_metadata(original_metadata)
        
        # Original should be unchanged
        assert original_metadata == {'title': 'Original', 'id': 123}
        
        # Result should have original + new data
        assert result['title'] == 'Original'
        assert result['id'] == 123
        assert 'processing_timestamp' in result
    
    def test_get_metadata_configuration_disabled(self):
        """
        Test get_metadata with configuration disabled
        設定無効での get_metadataテスト
        """
        processor = ConcreteMetadata(add_timestamp=False, add_source=False)
        metadata = {'title': 'Test'}
        file_path = Path('/test/file.txt')
        
        result = processor.get_metadata(metadata, file_path)
        
        # Only processor_type should be added
        assert result['title'] == 'Test'
        assert result['processor_type'] == 'ConcreteMetadata'
        assert 'processing_timestamp' not in result
        assert 'source_file' not in result
        assert processor.call_count == 1
    
    def test_get_metadata_multiple_calls(self):
        """
        Test multiple calls to get_metadata
        get_metadataの複数回呼び出しテスト
        """
        processor = ConcreteMetadata()
        
        result1 = processor.get_metadata({'doc': 1})
        result2 = processor.get_metadata({'doc': 2})
        result3 = processor.get_metadata({'doc': 3})
        
        assert processor.call_count == 3
        assert result1['doc'] == 1
        assert result2['doc'] == 2
        assert result3['doc'] == 3


class AlternativeMetadata(Metadata):
    """
    Alternative concrete implementation for testing
    テスト用の代替具象実装
    """
    
    def get_metadata(self, metadata: Dict[str, Any], file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Alternative implementation that adds different metadata
        異なるメタデータを追加する代替実装
        """
        result = metadata.copy()
        result['processor'] = 'AlternativeMetadata'
        result['processed'] = True
        
        if file:
            result['filename'] = file.name
            result['file_suffix'] = file.suffix
        
        return result


class TestMultipleMetadataImplementations:
    """
    Test multiple implementations of Metadata
    Metadataの複数実装のテスト
    """
    
    def test_multiple_implementations_inheritance(self):
        """
        Test that multiple implementations inherit from Metadata
        複数の実装がMetadataを継承することのテスト
        """
        concrete = ConcreteMetadata()
        alternative = AlternativeMetadata()
        
        assert isinstance(concrete, Metadata)
        assert isinstance(alternative, Metadata)
        assert type(concrete) != type(alternative)
    
    def test_different_implementations_different_behavior(self):
        """
        Test that different implementations have different behavior
        異なる実装が異なる動作を持つことのテスト
        """
        concrete = ConcreteMetadata()
        alternative = AlternativeMetadata()
        
        metadata = {'title': 'Test'}
        file_path = Path('/test/document.pdf')
        
        result1 = concrete.get_metadata(metadata, file_path)
        result2 = alternative.get_metadata(metadata, file_path)
        
        # Both should preserve original metadata
        assert result1['title'] == 'Test'
        assert result2['title'] == 'Test'
        
        # But should add different fields
        assert result1['processor_type'] == 'ConcreteMetadata'
        assert result2['processor'] == 'AlternativeMetadata'
        
        assert 'processing_timestamp' in result1
        assert 'processed' in result2
        
        assert result1['source_file'] == '/test/document.pdf'
        assert result2['filename'] == 'document.pdf'
        assert result2['file_suffix'] == '.pdf'
    
    def test_polymorphic_usage(self):
        """
        Test polymorphic usage of metadata processors
        メタデータプロセッサーのポリモーフィック使用テスト
        """
        processors = [
            ConcreteMetadata(),
            AlternativeMetadata()
        ]
        
        metadata = {'data': 'test'}
        
        for processor in processors:
            # All should be callable with same interface
            result = processor.get_metadata(metadata)
            assert isinstance(result, dict)
            assert result['data'] == 'test'
            
            # All should be instances of Metadata
            assert isinstance(processor, Metadata)


class TestMetadataEdgeCases:
    """
    Test edge cases for metadata processing
    メタデータ処理のエッジケースのテスト
    """
    
    def test_get_metadata_with_none_metadata(self):
        """
        Test get_metadata with None metadata
        Noneメタデータでのget_metadataテスト
        """
        processor = ConcreteMetadata()
        
        # This should fail at runtime since None doesn't have .copy()
        with pytest.raises(AttributeError):
            processor.get_metadata(None)
    
    def test_get_metadata_with_complex_metadata(self):
        """
        Test get_metadata with complex metadata structures
        複雑なメタデータ構造でのget_metadataテスト
        """
        processor = ConcreteMetadata()
        complex_metadata = {
            'nested': {
                'level1': {
                    'level2': 'deep_value'
                }
            },
            'list_data': [1, 2, 3],
            'unicode': 'テスト文書',
            'special_chars': '!@#$%^&*()'
        }
        
        result = processor.get_metadata(complex_metadata)
        
        # Complex structure should be preserved
        assert result['nested']['level1']['level2'] == 'deep_value'
        assert result['list_data'] == [1, 2, 3]
        assert result['unicode'] == 'テスト文書'
        assert result['special_chars'] == '!@#$%^&*()'
        
        # New metadata should be added
        assert 'processing_timestamp' in result
        assert result['processor_type'] == 'ConcreteMetadata'
    
    def test_get_metadata_with_large_metadata(self):
        """
        Test get_metadata with large metadata
        大きなメタデータでのget_metadataテスト
        """
        processor = ConcreteMetadata()
        large_metadata = {f'key_{i}': f'value_{i}' for i in range(1000)}
        
        result = processor.get_metadata(large_metadata)
        
        # All original keys should be preserved
        assert len(result) >= 1000
        assert result['key_0'] == 'value_0'
        assert result['key_999'] == 'value_999'
        
        # New metadata should be added
        assert 'processing_timestamp' in result
        assert result['processor_type'] == 'ConcreteMetadata'
    
    def test_get_metadata_with_pathlib_paths(self):
        """
        Test get_metadata with various Path objects
        様々なPathオブジェクトでのget_metadataテスト
        """
        processor = ConcreteMetadata()
        metadata = {'title': 'Test'}
        
        test_paths = [
            Path('/absolute/path/file.txt'),
            Path('relative/path/file.txt'),
            Path('./current/file.txt'),
            Path('../parent/file.txt'),
            Path('file.txt'),
            Path('/path/with spaces/file name.txt'),
            Path('/path/with/unicode/ファイル.txt')
        ]
        
        for path in test_paths:
            result = processor.get_metadata(metadata, path)
            assert result['source_file'] == str(path)
            assert result['processor_type'] == 'ConcreteMetadata'
    
    def test_metadata_inheritance_hierarchy(self):
        """
        Test metadata inheritance hierarchy
        メタデータ継承階層のテスト
        """
        # Test method resolution order
        assert Metadata in ConcreteMetadata.__mro__
        assert ABC in ConcreteMetadata.__mro__
        
        # Test that both implementations share the same base
        assert issubclass(ConcreteMetadata, Metadata)
        assert issubclass(AlternativeMetadata, Metadata)
        
        # Test abstract method requirements
        class IncompleteMetadata(Metadata):
            pass
        
        with pytest.raises(TypeError):
            IncompleteMetadata()
    
    def test_get_metadata_interface_consistency(self):
        """
        Test that all implementations follow the same interface
        すべての実装が同じインターフェースに従うことのテスト
        """
        implementations = [ConcreteMetadata(), AlternativeMetadata()]
        
        for impl in implementations:
            # All should have the same method signature
            method = getattr(impl, 'get_metadata')
            assert callable(method)
            
            # All should accept the same parameters
            import inspect
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            assert 'metadata' in params
            assert 'file' in params
            
            # All should return Dict[str, Any]
            result = impl.get_metadata({'test': 'data'})
            assert isinstance(result, dict)