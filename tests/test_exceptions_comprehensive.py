"""
Comprehensive tests for exceptions module
exceptions モジュールの包括的テスト

This module provides comprehensive coverage for the exceptions module,
testing exception hierarchy, inheritance, utility functions, and edge cases.
このモジュールは、exceptionsモジュールの包括的カバレッジを提供し、
例外階層、継承、ユーティリティ関数、エッジケースをテストします。
"""

import pytest
from typing import Type, Any
from unittest.mock import Mock, patch

from refinire_rag.exceptions import (
    # Base exception
    RefinireRAGError,
    
    # Core processing errors
    ProcessingError,
    LoaderError,
    SplitterError,
    EmbeddingError,
    MetadataError,
    
    # Storage and retrieval errors
    StorageError,
    DocumentStoreError,
    VectorStoreError,
    RetrievalError,
    
    # Configuration and validation errors
    ConfigurationError,
    ValidationError,
    FilterError,
    
    # Use case and integration errors
    CorpusManagerError,
    QueryEngineError,
    EvaluationError,
    
    # External integration errors
    LLMError,
    PluginError,
    
    # Specific operation errors
    FileError,
    NetworkError,
    SerializationError,
    PermissionError,
    
    # Utility function
    wrap_exception
)


class TestRefinireRAGError:
    """
    Test base RefinireRAGError exception
    基底RefinireRAGError例外のテスト
    """
    
    def test_refinire_rag_error_is_exception(self):
        """
        Test that RefinireRAGError inherits from Exception
        RefinireRAGErrorがExceptionを継承することのテスト
        """
        assert issubclass(RefinireRAGError, Exception)
    
    def test_refinire_rag_error_instantiation(self):
        """
        Test RefinireRAGError instantiation
        RefinireRAGErrorのインスタンス化テスト
        """
        error = RefinireRAGError("test message")
        assert isinstance(error, RefinireRAGError)
        assert str(error) == "test message"
    
    def test_refinire_rag_error_without_message(self):
        """
        Test RefinireRAGError without message
        メッセージなしのRefinireRAGErrorテスト
        """
        error = RefinireRAGError()
        assert isinstance(error, RefinireRAGError)
        assert str(error) == ""
    
    def test_refinire_rag_error_with_multiple_args(self):
        """
        Test RefinireRAGError with multiple arguments
        複数引数のRefinireRAGErrorテスト
        """
        error = RefinireRAGError("message", "arg2", "arg3")
        assert isinstance(error, RefinireRAGError)
        # Exception class behavior with multiple args
        assert error.args == ("message", "arg2", "arg3")
    
    def test_refinire_rag_error_raising(self):
        """
        Test raising RefinireRAGError
        RefinireRAGErrorの発生テスト
        """
        with pytest.raises(RefinireRAGError) as exc_info:
            raise RefinireRAGError("test error")
        
        assert str(exc_info.value) == "test error"
        assert isinstance(exc_info.value, RefinireRAGError)


class TestCoreProcessingErrors:
    """
    Test core processing error classes
    コア処理エラークラスのテスト
    """
    
    def test_processing_error_inheritance(self):
        """
        Test ProcessingError inheritance
        ProcessingErrorの継承テスト
        """
        assert issubclass(ProcessingError, RefinireRAGError)
        assert issubclass(ProcessingError, Exception)
    
    def test_loader_error_inheritance(self):
        """
        Test LoaderError inheritance
        LoaderErrorの継承テスト
        """
        assert issubclass(LoaderError, RefinireRAGError)
        assert issubclass(LoaderError, Exception)
    
    def test_splitter_error_inheritance(self):
        """
        Test SplitterError inheritance
        SplitterErrorの継承テスト
        """
        assert issubclass(SplitterError, RefinireRAGError)
        assert issubclass(SplitterError, Exception)
    
    def test_embedding_error_inheritance(self):
        """
        Test EmbeddingError inheritance
        EmbeddingErrorの継承テスト
        """
        assert issubclass(EmbeddingError, RefinireRAGError)
        assert issubclass(EmbeddingError, Exception)
    
    def test_metadata_error_inheritance(self):
        """
        Test MetadataError inheritance
        MetadataErrorの継承テスト
        """
        assert issubclass(MetadataError, RefinireRAGError)
        assert issubclass(MetadataError, Exception)
    
    def test_core_errors_instantiation(self):
        """
        Test instantiation of all core processing errors
        全コア処理エラーのインスタンス化テスト
        """
        errors = [
            (ProcessingError, "processing failed"),
            (LoaderError, "loading failed"),
            (SplitterError, "splitting failed"),
            (EmbeddingError, "embedding failed"),
            (MetadataError, "metadata failed")
        ]
        
        for error_class, message in errors:
            error = error_class(message)
            assert isinstance(error, error_class)
            assert isinstance(error, RefinireRAGError)
            assert str(error) == message
    
    def test_core_errors_raising(self):
        """
        Test raising core processing errors
        コア処理エラーの発生テスト
        """
        with pytest.raises(ProcessingError):
            raise ProcessingError("processing error")
        
        with pytest.raises(LoaderError):
            raise LoaderError("loader error")
        
        with pytest.raises(SplitterError):
            raise SplitterError("splitter error")
        
        with pytest.raises(EmbeddingError):
            raise EmbeddingError("embedding error")
        
        with pytest.raises(MetadataError):
            raise MetadataError("metadata error")


class TestStorageAndRetrievalErrors:
    """
    Test storage and retrieval error classes
    ストレージ・検索エラークラスのテスト
    """
    
    def test_storage_error_inheritance(self):
        """
        Test StorageError inheritance
        StorageErrorの継承テスト
        """
        assert issubclass(StorageError, RefinireRAGError)
        assert issubclass(StorageError, Exception)
    
    def test_document_store_error_inheritance(self):
        """
        Test DocumentStoreError inheritance hierarchy
        DocumentStoreErrorの継承階層テスト
        """
        assert issubclass(DocumentStoreError, StorageError)
        assert issubclass(DocumentStoreError, RefinireRAGError)
        assert issubclass(DocumentStoreError, Exception)
    
    def test_vector_store_error_inheritance(self):
        """
        Test VectorStoreError inheritance hierarchy
        VectorStoreErrorの継承階層テスト
        """
        assert issubclass(VectorStoreError, StorageError)
        assert issubclass(VectorStoreError, RefinireRAGError)
        assert issubclass(VectorStoreError, Exception)
    
    def test_retrieval_error_inheritance(self):
        """
        Test RetrievalError inheritance
        RetrievalErrorの継承テスト
        """
        assert issubclass(RetrievalError, RefinireRAGError)
        assert issubclass(RetrievalError, Exception)
    
    def test_storage_errors_instantiation(self):
        """
        Test instantiation of storage and retrieval errors
        ストレージ・検索エラーのインスタンス化テスト
        """
        errors = [
            (StorageError, "storage failed"),
            (DocumentStoreError, "document store failed"),
            (VectorStoreError, "vector store failed"),
            (RetrievalError, "retrieval failed")
        ]
        
        for error_class, message in errors:
            error = error_class(message)
            assert isinstance(error, error_class)
            assert isinstance(error, RefinireRAGError)
            assert str(error) == message
    
    def test_storage_error_hierarchy_polymorphism(self):
        """
        Test polymorphism in storage error hierarchy
        ストレージエラー階層のポリモーフィズムテスト
        """
        doc_error = DocumentStoreError("document error")
        vec_error = VectorStoreError("vector error")
        
        # Both should be instances of StorageError
        assert isinstance(doc_error, StorageError)
        assert isinstance(vec_error, StorageError)
        
        # Should be catchable as StorageError
        with pytest.raises(StorageError):
            raise DocumentStoreError("test")
        
        with pytest.raises(StorageError):
            raise VectorStoreError("test")


class TestConfigurationAndValidationErrors:
    """
    Test configuration and validation error classes
    設定・検証エラークラスのテスト
    """
    
    def test_configuration_error_inheritance(self):
        """
        Test ConfigurationError inheritance
        ConfigurationErrorの継承テスト
        """
        assert issubclass(ConfigurationError, RefinireRAGError)
        assert issubclass(ConfigurationError, Exception)
    
    def test_validation_error_inheritance(self):
        """
        Test ValidationError inheritance
        ValidationErrorの継承テスト
        """
        assert issubclass(ValidationError, RefinireRAGError)
        assert issubclass(ValidationError, Exception)
    
    def test_filter_error_inheritance(self):
        """
        Test FilterError inheritance
        FilterErrorの継承テスト
        """
        assert issubclass(FilterError, RefinireRAGError)
        assert issubclass(FilterError, Exception)
    
    def test_config_validation_errors_instantiation(self):
        """
        Test instantiation of configuration and validation errors
        設定・検証エラーのインスタンス化テスト
        """
        errors = [
            (ConfigurationError, "configuration invalid"),
            (ValidationError, "validation failed"),
            (FilterError, "filter failed")
        ]
        
        for error_class, message in errors:
            error = error_class(message)
            assert isinstance(error, error_class)
            assert isinstance(error, RefinireRAGError)
            assert str(error) == message


class TestUseCaseAndIntegrationErrors:
    """
    Test use case and integration error classes
    ユースケース・統合エラークラスのテスト
    """
    
    def test_corpus_manager_error_inheritance(self):
        """
        Test CorpusManagerError inheritance
        CorpusManagerErrorの継承テスト
        """
        assert issubclass(CorpusManagerError, RefinireRAGError)
        assert issubclass(CorpusManagerError, Exception)
    
    def test_query_engine_error_inheritance(self):
        """
        Test QueryEngineError inheritance
        QueryEngineErrorの継承テスト
        """
        assert issubclass(QueryEngineError, RefinireRAGError)
        assert issubclass(QueryEngineError, Exception)
    
    def test_evaluation_error_inheritance(self):
        """
        Test EvaluationError inheritance
        EvaluationErrorの継承テスト
        """
        assert issubclass(EvaluationError, RefinireRAGError)
        assert issubclass(EvaluationError, Exception)
    
    def test_use_case_errors_instantiation(self):
        """
        Test instantiation of use case and integration errors
        ユースケース・統合エラーのインスタンス化テスト
        """
        errors = [
            (CorpusManagerError, "corpus management failed"),
            (QueryEngineError, "query engine failed"),
            (EvaluationError, "evaluation failed")
        ]
        
        for error_class, message in errors:
            error = error_class(message)
            assert isinstance(error, error_class)
            assert isinstance(error, RefinireRAGError)
            assert str(error) == message


class TestExternalIntegrationErrors:
    """
    Test external integration error classes
    外部統合エラークラスのテスト
    """
    
    def test_llm_error_inheritance(self):
        """
        Test LLMError inheritance
        LLMErrorの継承テスト
        """
        assert issubclass(LLMError, RefinireRAGError)
        assert issubclass(LLMError, Exception)
    
    def test_plugin_error_inheritance(self):
        """
        Test PluginError inheritance
        PluginErrorの継承テスト
        """
        assert issubclass(PluginError, RefinireRAGError)
        assert issubclass(PluginError, Exception)
    
    def test_external_integration_errors_instantiation(self):
        """
        Test instantiation of external integration errors
        外部統合エラーのインスタンス化テスト
        """
        errors = [
            (LLMError, "LLM communication failed"),
            (PluginError, "plugin operation failed")
        ]
        
        for error_class, message in errors:
            error = error_class(message)
            assert isinstance(error, error_class)
            assert isinstance(error, RefinireRAGError)
            assert str(error) == message


class TestSpecificOperationErrors:
    """
    Test specific operation error classes
    特定操作エラークラスのテスト
    """
    
    def test_file_error_inheritance(self):
        """
        Test FileError inheritance hierarchy
        FileErrorの継承階層テスト
        """
        assert issubclass(FileError, LoaderError)
        assert issubclass(FileError, RefinireRAGError)
        assert issubclass(FileError, Exception)
    
    def test_network_error_inheritance(self):
        """
        Test NetworkError inheritance
        NetworkErrorの継承テスト
        """
        assert issubclass(NetworkError, RefinireRAGError)
        assert issubclass(NetworkError, Exception)
    
    def test_serialization_error_inheritance(self):
        """
        Test SerializationError inheritance
        SerializationErrorの継承テスト
        """
        assert issubclass(SerializationError, RefinireRAGError)
        assert issubclass(SerializationError, Exception)
    
    def test_permission_error_inheritance(self):
        """
        Test PermissionError inheritance
        PermissionErrorの継承テスト
        """
        assert issubclass(PermissionError, RefinireRAGError)
        assert issubclass(PermissionError, Exception)
    
    def test_specific_operation_errors_instantiation(self):
        """
        Test instantiation of specific operation errors
        特定操作エラーのインスタンス化テスト
        """
        errors = [
            (FileError, "file operation failed"),
            (NetworkError, "network operation failed"),
            (SerializationError, "serialization failed"),
            (PermissionError, "permission denied")
        ]
        
        for error_class, message in errors:
            error = error_class(message)
            assert isinstance(error, error_class)
            assert isinstance(error, RefinireRAGError)
            assert str(error) == message
    
    def test_file_error_hierarchy_polymorphism(self):
        """
        Test FileError hierarchy polymorphism
        FileError階層のポリモーフィズムテスト
        """
        file_error = FileError("file error")
        
        # FileError should be catchable as LoaderError
        assert isinstance(file_error, LoaderError)
        
        with pytest.raises(LoaderError):
            raise FileError("test")


class TestWrapExceptionFunction:
    """
    Test wrap_exception utility function
    wrap_exception ユーティリティ関数のテスト
    """
    
    def test_wrap_exception_with_refinire_rag_error(self):
        """
        Test wrap_exception with RefinireRAGError returns original
        RefinireRAGErrorでのwrap_exceptionが元の例外を返すことのテスト
        """
        original_error = ProcessingError("original error")
        wrapped = wrap_exception(original_error)
        
        assert wrapped is original_error
        assert isinstance(wrapped, ProcessingError)
        assert str(wrapped) == "original error"
    
    def test_wrap_exception_with_file_not_found_error(self):
        """
        Test wrap_exception with FileNotFoundError
        FileNotFoundErrorでのwrap_exceptionテスト
        """
        original_error = FileNotFoundError("file not found")
        wrapped = wrap_exception(original_error)
        
        assert isinstance(wrapped, FileError)
        assert str(wrapped) == "file not found"
    
    def test_wrap_exception_with_io_error(self):
        """
        Test wrap_exception with IOError
        IOErrorでのwrap_exceptionテスト
        """
        original_error = IOError("I/O operation failed")
        wrapped = wrap_exception(original_error)
        
        assert isinstance(wrapped, FileError)
        assert str(wrapped) == "I/O operation failed"
    
    def test_wrap_exception_with_os_error(self):
        """
        Test wrap_exception with OSError
        OSErrorでのwrap_exceptionテスト
        """
        original_error = OSError("OS operation failed")
        wrapped = wrap_exception(original_error)
        
        assert isinstance(wrapped, FileError)
        assert str(wrapped) == "OS operation failed"
    
    def test_wrap_exception_with_value_error(self):
        """
        Test wrap_exception with ValueError
        ValueErrorでのwrap_exceptionテスト
        """
        original_error = ValueError("invalid value")
        wrapped = wrap_exception(original_error)
        
        assert isinstance(wrapped, ValidationError)
        assert str(wrapped) == "invalid value"
    
    def test_wrap_exception_with_type_error(self):
        """
        Test wrap_exception with TypeError
        TypeErrorでのwrap_exceptionテスト
        """
        original_error = TypeError("type mismatch")
        wrapped = wrap_exception(original_error)
        
        assert isinstance(wrapped, ValidationError)
        assert str(wrapped) == "type mismatch"
    
    def test_wrap_exception_with_permission_error(self):
        """
        Test wrap_exception with PermissionError
        PermissionErrorでのwrap_exceptionテスト
        """
        original_error = PermissionError("access denied")
        wrapped = wrap_exception(original_error)
        
        assert isinstance(wrapped, PermissionError)
        assert str(wrapped) == "access denied"
    
    def test_wrap_exception_with_generic_exception(self):
        """
        Test wrap_exception with generic Exception
        汎用Exceptionでのwrap_exceptionテスト
        """
        original_error = Exception("generic error")
        wrapped = wrap_exception(original_error)
        
        assert isinstance(wrapped, ProcessingError)
        assert str(wrapped) == "generic error"
    
    def test_wrap_exception_with_custom_message(self):
        """
        Test wrap_exception with custom message
        カスタムメッセージでのwrap_exceptionテスト
        """
        original_error = ValueError("original message")
        wrapped = wrap_exception(original_error, "Custom prefix")
        
        assert isinstance(wrapped, ValidationError)
        assert str(wrapped) == "Custom prefix: original message"
    
    def test_wrap_exception_with_empty_custom_message(self):
        """
        Test wrap_exception with empty custom message
        空のカスタムメッセージでのwrap_exceptionテスト
        """
        original_error = ValueError("original message")
        wrapped = wrap_exception(original_error, "")
        
        assert isinstance(wrapped, ValidationError)
        assert str(wrapped) == "original message"
    
    def test_wrap_exception_with_none_message(self):
        """
        Test wrap_exception with None message
        Noneメッセージでのwrap_exceptionテスト
        """
        original_error = ValueError("original message")
        wrapped = wrap_exception(original_error, None)
        
        assert isinstance(wrapped, ValidationError)
        assert str(wrapped) == "original message"
    
    def test_wrap_exception_with_complex_exception(self):
        """
        Test wrap_exception with complex exception
        複雑な例外でのwrap_exceptionテスト
        """
        try:
            raise ValueError("complex error with context")
        except Exception as e:
            wrapped = wrap_exception(e, "Context information")
            
            assert isinstance(wrapped, ValidationError)
            assert "Context information: complex error with context" in str(wrapped)


class TestExceptionHierarchyIntegration:
    """
    Test exception hierarchy integration scenarios
    例外階層統合シナリオのテスト
    """
    
    def test_catch_refinire_rag_error_base(self):
        """
        Test catching all RefinireRAG errors with base exception
        基底例外ですべてのRefinireRAGエラーをキャッチするテスト
        """
        exceptions_to_test = [
            ProcessingError("processing"),
            LoaderError("loading"),
            StorageError("storage"),
            DocumentStoreError("doc store"),
            ConfigurationError("config"),
            CorpusManagerError("corpus"),
            LLMError("llm"),
            FileError("file"),
            NetworkError("network")
        ]
        
        for exception in exceptions_to_test:
            with pytest.raises(RefinireRAGError):
                raise exception
    
    def test_exception_hierarchy_specificity(self):
        """
        Test exception hierarchy specificity
        例外階層の特異性テスト
        """
        # Most specific exception should be caught first
        try:
            raise DocumentStoreError("document store error")
        except DocumentStoreError as e:
            assert isinstance(e, DocumentStoreError)
            assert isinstance(e, StorageError)
            assert isinstance(e, RefinireRAGError)
        except StorageError:
            pytest.fail("Should catch DocumentStoreError specifically")
        except RefinireRAGError:
            pytest.fail("Should catch DocumentStoreError specifically")
    
    def test_mixed_exception_handling(self):
        """
        Test handling mixed exception types
        混合例外タイプの処理テスト
        """
        def process_with_mixed_errors(error_type: str):
            if error_type == "file":
                raise FileError("file error")
            elif error_type == "validation":
                raise ValidationError("validation error")
            elif error_type == "generic":
                raise RefinireRAGError("generic error")
        
        # Test handling different error types
        for error_type in ["file", "validation", "generic"]:
            with pytest.raises(RefinireRAGError):
                process_with_mixed_errors(error_type)
    
    def test_exception_chaining(self):
        """
        Test exception chaining scenarios
        例外チェーンシナリオのテスト
        """
        try:
            try:
                raise ValueError("original error")
            except ValueError as e:
                wrapped = wrap_exception(e, "Processing failed")
                raise wrapped
        except ValidationError as e:
            assert "Processing failed: original error" in str(e)
            assert isinstance(e, ValidationError)
            assert isinstance(e, RefinireRAGError)
    
    def test_error_categorization(self):
        """
        Test error categorization for different domains
        異なるドメインのエラー分類テスト
        """
        # Processing domain errors
        processing_errors = [ProcessingError, LoaderError, SplitterError, EmbeddingError, MetadataError]
        for error_class in processing_errors:
            error = error_class("test")
            assert isinstance(error, RefinireRAGError)
        
        # Storage domain errors
        storage_errors = [StorageError, DocumentStoreError, VectorStoreError]
        for error_class in storage_errors:
            error = error_class("test")
            assert isinstance(error, RefinireRAGError)
        
        # Use case domain errors
        usecase_errors = [CorpusManagerError, QueryEngineError, EvaluationError]
        for error_class in usecase_errors:
            error = error_class("test")
            assert isinstance(error, RefinireRAGError)


class TestExceptionEdgeCases:
    """
    Test exception edge cases and corner scenarios
    例外エッジケースとコーナーシナリオのテスト
    """
    
    def test_exception_with_unicode_message(self):
        """
        Test exceptions with unicode messages
        Unicode メッセージでの例外テスト
        """
        unicode_message = "エラーメッセージ with unicode 🚨"
        error = ProcessingError(unicode_message)
        
        assert str(error) == unicode_message
        assert isinstance(error, ProcessingError)
    
    def test_exception_with_very_long_message(self):
        """
        Test exceptions with very long messages
        非常に長いメッセージでの例外テスト
        """
        long_message = "a" * 10000
        error = ValidationError(long_message)
        
        assert str(error) == long_message
        assert len(str(error)) == 10000
    
    def test_exception_with_special_characters(self):
        """
        Test exceptions with special characters
        特殊文字での例外テスト
        """
        special_message = "Error: \n\t\r\"'\\&<>{}[]()@#$%^*+=|`~"
        error = ConfigurationError(special_message)
        
        assert str(error) == special_message
        assert isinstance(error, ConfigurationError)
    
    def test_wrap_exception_with_nested_exceptions(self):
        """
        Test wrap_exception with nested exceptions
        ネストした例外でのwrap_exceptionテスト
        """
        # Create nested exception scenario
        try:
            try:
                raise ValueError("inner error")
            except ValueError as e:
                raise IOError("outer error") from e
        except IOError as nested_error:
            wrapped = wrap_exception(nested_error, "Wrapped")
            
            assert isinstance(wrapped, FileError)
            assert "Wrapped: outer error" in str(wrapped)
    
    def test_exception_repr_and_str(self):
        """
        Test exception string representation methods
        例外文字列表現メソッドのテスト
        """
        error = ProcessingError("test error")
        
        # Test string representation
        str_repr = str(error)
        assert str_repr == "test error"
        
        # Test that repr includes class name
        repr_str = repr(error)
        assert "ProcessingError" in repr_str
        assert "test error" in repr_str
    
    def test_exception_with_none_message(self):
        """
        Test exceptions with None message
        Noneメッセージでの例外テスト
        """
        error = ValidationError(None)
        assert str(error) == "None"
        assert isinstance(error, ValidationError)
    
    def test_wrap_exception_with_exception_subclasses(self):
        """
        Test wrap_exception with various exception subclasses
        様々な例外サブクラスでのwrap_exceptionテスト
        """
        # Test with specific subclasses
        test_cases = [
            (FileNotFoundError("not found"), FileError),
            (IsADirectoryError("is directory"), FileError),
            (PermissionError("no permission"), PermissionError),
            (ValueError("bad value"), ValidationError),
            (TypeError("bad type"), ValidationError),
            (RuntimeError("runtime error"), ProcessingError),
            (KeyError("key missing"), ProcessingError),
            (IndexError("index out of range"), ProcessingError)
        ]
        
        for original_error, expected_type in test_cases:
            wrapped = wrap_exception(original_error)
            assert isinstance(wrapped, expected_type)
            assert isinstance(wrapped, RefinireRAGError)