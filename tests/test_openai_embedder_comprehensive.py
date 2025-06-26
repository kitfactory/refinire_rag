"""
Comprehensive tests for OpenAI Embedder
OpenAI埋め込みの包括的テスト
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from refinire_rag.embedding.openai_embedder import OpenAIEmbedder, OpenAIEmbeddingConfig
from refinire_rag.models.document import Document
from refinire_rag.exceptions import EmbeddingError


class TestOpenAIEmbeddingConfig:
    """Test OpenAI embedding configuration
    OpenAI埋め込み設定のテスト"""
    
    def test_default_configuration(self):
        """Test default configuration values
        デフォルト設定値のテスト"""
        config = OpenAIEmbeddingConfig()
        
        assert config.model_name == "text-embedding-3-small"
        assert config.api_key is None
        assert config.api_base is None
        assert config.organization is None
        assert config.embedding_dimension == 1536
        assert config.batch_size == 100
        assert config.max_tokens == 8191
        assert config.requests_per_minute == 3000
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.strip_newlines is True
        assert config.user_identifier is None
    
    def test_custom_configuration(self):
        """Test custom configuration values
        カスタム設定値のテスト"""
        config = OpenAIEmbeddingConfig(
            model_name="text-embedding-3-large",
            api_key="test-key",
            api_base="https://custom.openai.com",
            organization="org-123",
            embedding_dimension=3072,
            batch_size=50,
            max_tokens=4096,
            requests_per_minute=1000,
            max_retries=5,
            retry_delay_seconds=2.0,
            strip_newlines=False,
            user_identifier="test-user"
        )
        
        assert config.model_name == "text-embedding-3-large"
        assert config.api_key == "test-key"
        assert config.api_base == "https://custom.openai.com"
        assert config.organization == "org-123"
        assert config.embedding_dimension == 3072
        assert config.batch_size == 50
        assert config.max_tokens == 4096
        assert config.requests_per_minute == 1000
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 2.0
        assert config.strip_newlines is False
        assert config.user_identifier == "test-user"


class TestOpenAIEmbedderInitialization:
    """Test OpenAI embedder initialization
    OpenAI埋め込み初期化のテスト"""
    
    def test_initialization_with_default_config(self):
        """Test initialization with default configuration
        デフォルト設定での初期化テスト"""
        with patch('refinire_rag.embedding.openai_embedder.OpenAIEmbedder._init_client'):
            embedder = OpenAIEmbedder()
            assert isinstance(embedder.config, OpenAIEmbeddingConfig)
            assert embedder.config.model_name == "text-embedding-3-small"
    
    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration
        カスタム設定での初期化テスト"""
        config = OpenAIEmbeddingConfig(
            model_name="text-embedding-3-large",
            api_key="test-key"
        )
        
        with patch('refinire_rag.embedding.openai_embedder.OpenAIEmbedder._init_client'):
            embedder = OpenAIEmbedder(config)
            assert embedder.config.model_name == "text-embedding-3-large"
            assert embedder.config.api_key == "test-key"
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-env-key'})
    @patch('refinire_rag.embedding.openai_embedder.openai.OpenAI')
    def test_client_initialization_with_env_key(self, mock_openai):
        """Test client initialization with environment API key
        環境変数APIキーでのクライアント初期化テスト"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        embedder = OpenAIEmbedder()
        
        mock_openai.assert_called_once_with(api_key='test-env-key')
        assert embedder._client == mock_client
    
    @patch('refinire_rag.embedding.openai_embedder.openai.OpenAI')
    def test_client_initialization_with_config_key(self, mock_openai):
        """Test client initialization with config API key
        設定APIキーでのクライアント初期化テスト"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        config = OpenAIEmbeddingConfig(api_key="config-key")
        embedder = OpenAIEmbedder(config)
        
        mock_openai.assert_called_once_with(api_key='config-key')
        assert embedder._client == mock_client
    
    @patch('refinire_rag.embedding.openai_embedder.openai.OpenAI')
    def test_client_initialization_with_custom_base_url(self, mock_openai):
        """Test client initialization with custom base URL
        カスタムベースURLでのクライアント初期化テスト"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        config = OpenAIEmbeddingConfig(
            api_key="test-key",
            api_base="https://custom.openai.com",
            organization="org-123"
        )
        embedder = OpenAIEmbedder(config)
        
        mock_openai.assert_called_once_with(
            api_key='test-key',
            base_url='https://custom.openai.com',
            organization='org-123'
        )
    
    @patch.dict(os.environ, {}, clear=True)
    def test_client_initialization_missing_api_key(self):
        """Test client initialization with missing API key
        APIキー不足でのクライアント初期化テスト"""
        with pytest.raises(EmbeddingError, match="OpenAI API key not found"):
            OpenAIEmbedder()
    
    @patch('refinire_rag.embedding.openai_embedder.openai', None)
    def test_client_initialization_missing_openai_library(self):
        """Test client initialization with missing OpenAI library
        OpenAIライブラリ不足でのクライアント初期化テスト"""
        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(EmbeddingError, match="OpenAI library not found"):
                OpenAIEmbedder()


class TestOpenAIEmbedderEmbedding:
    """Test OpenAI embedder embedding functionality
    OpenAI埋め込み機能のテスト"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = OpenAIEmbeddingConfig(api_key="test-key")
        
        with patch('refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            self.embedder = OpenAIEmbedder(self.config)
            self.embedder._client = Mock()
    
    def test_embed_text_success(self):
        """Test successful text embedding
        テキスト埋め込み成功テスト"""
        # Mock OpenAI API response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage = Mock(prompt_tokens=10, total_tokens=10)
        
        self.embedder._client.embeddings.create.return_value = mock_response
        
        result = self.embedder.embed_text("test text")
        
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([0.1, 0.2, 0.3]))
        
        self.embedder._client.embeddings.create.assert_called_once()
    
    def test_embed_text_with_newlines_stripped(self):
        """Test text embedding with newlines stripped
        改行除去でのテキスト埋め込みテスト"""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage = Mock(prompt_tokens=10, total_tokens=10)
        
        self.embedder._client.embeddings.create.return_value = mock_response
        
        result = self.embedder.embed_text("test\ntext\nwith\nnewlines")
        
        # Check that newlines were stripped in the API call
        call_args = self.embedder._client.embeddings.create.call_args
        assert "\n" not in call_args[1]["input"][0]
    
    def test_embed_text_without_newline_stripping(self):
        """Test text embedding without newline stripping
        改行除去なしでのテキスト埋め込みテスト"""
        config = OpenAIEmbeddingConfig(api_key="test-key", strip_newlines=False)
        with patch('refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            embedder = OpenAIEmbedder(config)
            embedder._client = Mock()
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage = Mock(prompt_tokens=10, total_tokens=10)
        
        embedder._client.embeddings.create.return_value = mock_response
        
        text_with_newlines = "test\ntext\nwith\nnewlines"
        result = embedder.embed_text(text_with_newlines)
        
        # Check that newlines were preserved in the API call
        call_args = embedder._client.embeddings.create.call_args
        assert call_args[1]["input"][0] == text_with_newlines
    
    def test_embed_texts_batch_success(self):
        """Test successful batch text embedding
        バッチテキスト埋め込み成功テスト"""
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_response.usage = Mock(prompt_tokens=20, total_tokens=20)
        
        self.embedder._client.embeddings.create.return_value = mock_response
        
        texts = ["text 1", "text 2"]
        results = self.embedder.embed_texts(texts)
        
        assert len(results) == 2
        assert np.array_equal(results[0], np.array([0.1, 0.2, 0.3]))
        assert np.array_equal(results[1], np.array([0.4, 0.5, 0.6]))
        
        self.embedder._client.embeddings.create.assert_called_once()
    
    def test_embed_texts_large_batch(self):
        """Test embedding large batch of texts
        大きなバッチのテキスト埋め込みテスト"""
        # Create batch larger than batch_size
        texts = [f"text {i}" for i in range(150)]
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3]) for _ in range(100)]
        mock_response.usage = Mock(prompt_tokens=100, total_tokens=100)
        
        self.embedder._client.embeddings.create.return_value = mock_response
        
        results = self.embedder.embed_texts(texts)
        
        # Should make 2 API calls (100 + 50)
        assert self.embedder._client.embeddings.create.call_count == 2
        assert len(results) == 150
    
    def test_embed_text_api_error(self):
        """Test text embedding with API error
        APIエラーでのテキスト埋め込みテスト"""
        self.embedder._client.embeddings.create.side_effect = Exception("API Error")
        
        with pytest.raises(EmbeddingError, match="Failed to get embeddings from OpenAI"):
            self.embedder.embed_text("test text")
    
    def test_embed_text_with_retries(self):
        """Test text embedding with retries
        リトライでのテキスト埋め込みテスト"""
        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage = Mock(prompt_tokens=10, total_tokens=10)
        
        self.embedder._client.embeddings.create.side_effect = [
            Exception("Temporary error"),
            mock_response
        ]
        
        with patch('time.sleep'):  # Speed up test
            result = self.embedder.embed_text("test text")
        
        assert isinstance(result, np.ndarray)
        assert self.embedder._client.embeddings.create.call_count == 2


class TestOpenAIEmbedderDocumentIntegration:
    """Test OpenAI embedder document integration
    OpenAI埋め込みドキュメント統合のテスト"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = OpenAIEmbeddingConfig(api_key="test-key")
        
        with patch('refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            self.embedder = OpenAIEmbedder(self.config)
            self.embedder._client = Mock()
    
    def test_embed_document(self):
        """Test document embedding
        ドキュメント埋め込みテスト"""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage = Mock(prompt_tokens=10, total_tokens=10)
        
        self.embedder._client.embeddings.create.return_value = mock_response
        
        document = Document(
            id="doc1",
            content="Test document content",
            metadata={"source": "test"}
        )
        
        result = self.embedder.embed_document(document)
        
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([0.1, 0.2, 0.3]))
    
    def test_embed_documents_batch(self):
        """Test batch document embedding
        バッチドキュメント埋め込みテスト"""
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_response.usage = Mock(prompt_tokens=20, total_tokens=20)
        
        self.embedder._client.embeddings.create.return_value = mock_response
        
        documents = [
            Document(id="doc1", content="Content 1"),
            Document(id="doc2", content="Content 2")
        ]
        
        results = self.embedder.embed_documents(documents)
        
        assert len(results) == 2
        assert np.array_equal(results[0], np.array([0.1, 0.2, 0.3]))
        assert np.array_equal(results[1], np.array([0.4, 0.5, 0.6]))


class TestOpenAIEmbedderUtilityMethods:
    """Test OpenAI embedder utility methods
    OpenAI埋め込みユーティリティメソッドのテスト"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = OpenAIEmbeddingConfig(api_key="test-key")
        
        with patch('refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            self.embedder = OpenAIEmbedder(self.config)
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension
        埋め込み次元取得テスト"""
        dimension = self.embedder.get_embedding_dimension()
        assert dimension == 1536  # Default for text-embedding-3-small
    
    def test_is_available_with_client(self):
        """Test availability check with client
        クライアント付き利用可能性チェックテスト"""
        self.embedder._client = Mock()
        assert self.embedder.is_available() is True
    
    def test_is_available_without_client(self):
        """Test availability check without client
        クライアントなし利用可能性チェックテスト"""
        self.embedder._client = None
        assert self.embedder.is_available() is False
    
    def test_get_model_info(self):
        """Test getting model information
        モデル情報取得テスト"""
        info = self.embedder.get_model_info()
        
        assert info["model_name"] == "text-embedding-3-small"
        assert info["embedding_dimension"] == 1536
        assert info["provider"] == "OpenAI"
        assert "api_key_set" in info
        assert "rate_limit" in info


class TestOpenAIEmbedderRateLimiting:
    """Test OpenAI embedder rate limiting
    OpenAI埋め込みレート制限のテスト"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = OpenAIEmbeddingConfig(
            api_key="test-key",
            requests_per_minute=60  # 1 request per second
        )
        
        with patch('refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            self.embedder = OpenAIEmbedder(self.config)
            self.embedder._client = Mock()
    
    def test_rate_limiting_enforcement(self):
        """Test rate limiting enforcement
        レート制限実行テスト"""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage = Mock(prompt_tokens=10, total_tokens=10)
        
        self.embedder._client.embeddings.create.return_value = mock_response
        
        with patch('time.time') as mock_time, patch('time.sleep') as mock_sleep:
            # Simulate multiple requests in quick succession
            mock_time.side_effect = [0, 0.5, 1.0, 1.5]  # Fast requests
            
            self.embedder.embed_text("text 1")
            self.embedder.embed_text("text 2")
            
            # Should have called sleep to enforce rate limit
            mock_sleep.assert_called()


class TestOpenAIEmbedderModelDimensions:
    """Test OpenAI embedder model dimensions
    OpenAI埋め込みモデル次元のテスト"""
    
    def test_text_embedding_3_small_dimensions(self):
        """Test text-embedding-3-small dimensions
        text-embedding-3-smallの次元テスト"""
        config = OpenAIEmbeddingConfig(
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        with patch('refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            embedder = OpenAIEmbedder(config)
            assert embedder.config.embedding_dimension == 1536
    
    def test_text_embedding_3_large_dimensions(self):
        """Test text-embedding-3-large dimensions
        text-embedding-3-largeの次元テスト"""
        config = OpenAIEmbeddingConfig(
            api_key="test-key",
            model_name="text-embedding-3-large"
        )
        
        with patch('refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            embedder = OpenAIEmbedder(config)
            assert embedder.config.embedding_dimension == 3072
    
    def test_ada_002_dimensions(self):
        """Test text-embedding-ada-002 dimensions
        text-embedding-ada-002の次元テスト"""
        config = OpenAIEmbeddingConfig(
            api_key="test-key",
            model_name="text-embedding-ada-002"
        )
        
        with patch('refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            embedder = OpenAIEmbedder(config)
            assert embedder.config.embedding_dimension == 1536


class TestOpenAIEmbedderErrorHandling:
    """Test OpenAI embedder error handling
    OpenAI埋め込みエラーハンドリングのテスト"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = OpenAIEmbeddingConfig(api_key="test-key")
        
        with patch('refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            self.embedder = OpenAIEmbedder(self.config)
            self.embedder._client = Mock()
    
    def test_empty_text_handling(self):
        """Test empty text handling
        空テキストハンドリングテスト"""
        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            self.embedder.embed_text("")
    
    def test_none_text_handling(self):
        """Test None text handling
        Noneテキストハンドリングテスト"""
        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            self.embedder.embed_text(None)
    
    def test_max_retries_exceeded(self):
        """Test maximum retries exceeded
        最大リトライ数超過テスト"""
        self.embedder._client.embeddings.create.side_effect = Exception("Persistent error")
        
        with patch('time.sleep'):  # Speed up test
            with pytest.raises(EmbeddingError, match="Failed to get embeddings from OpenAI after 3 retries"):
                self.embedder.embed_text("test text")
        
        assert self.embedder._client.embeddings.create.call_count == 3