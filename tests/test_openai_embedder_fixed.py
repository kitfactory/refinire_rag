"""
Simplified tests for OpenAI Embedder
OpenAI埋め込みの簡単なテスト
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
        """Test default configuration values"""
        config = OpenAIEmbeddingConfig()
        
        assert config.model_name == "text-embedding-3-small"
        assert config.api_key is None
        assert config.embedding_dimension == 1536
        assert config.batch_size == 100
    
    def test_custom_configuration(self):
        """Test custom configuration values"""
        config = OpenAIEmbeddingConfig(
            model_name="text-embedding-3-large",
            api_key="test-key",
            embedding_dimension=3072
        )
        
        assert config.model_name == "text-embedding-3-large"
        assert config.api_key == "test-key"
        assert config.embedding_dimension == 3072


class TestOpenAIEmbedderBasics:
    """Test OpenAI embedder basic functionality"""
    
    def test_initialization_with_config(self):
        """Test initialization with configuration"""
        config = OpenAIEmbeddingConfig(api_key="test-key")
        
        with patch('src.refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            embedder = OpenAIEmbedder(config=config)
            assert embedder.config.api_key == "test-key"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        """Test initialization with missing API key"""
        with pytest.raises(EmbeddingError, match="OpenAI API key not found"):
            OpenAIEmbedder()
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension"""
        config = OpenAIEmbeddingConfig(api_key="test-key")
        
        with patch('src.refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            embedder = OpenAIEmbedder(config=config)
            assert embedder.get_embedding_dimension() == 1536


class TestOpenAIEmbedderEmbedding:
    """Test OpenAI embedder embedding functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = OpenAIEmbeddingConfig(api_key="test-key")
        
        with patch('src.refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            self.embedder = OpenAIEmbedder(config=self.config)
            self.embedder._client = Mock()
    
    def test_embed_text_success(self):
        """Test successful text embedding"""
        # Mock OpenAI API response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage = Mock(prompt_tokens=10, total_tokens=10)
        
        self.embedder._client.embeddings.create.return_value = mock_response
        
        result = self.embedder.embed_text("test text")
        
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([0.1, 0.2, 0.3]))
        self.embedder._client.embeddings.create.assert_called_once()
    
    def test_embed_texts_batch(self):
        """Test batch text embedding"""
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
    
    def test_embed_document(self):
        """Test document embedding"""
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
    
    def test_empty_text_handling(self):
        """Test empty text handling"""
        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            self.embedder.embed_text("")
    
    def test_none_text_handling(self):
        """Test None text handling"""
        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            self.embedder.embed_text(None)
    
    def test_api_error_handling(self):
        """Test API error handling"""
        self.embedder._client.embeddings.create.side_effect = Exception("API Error")
        
        # Should return zero vector when fail_on_error is False
        self.embedder.config.fail_on_error = False
        result = self.embedder.embed_text("test text")
        
        assert isinstance(result, np.ndarray)
        assert len(result) == self.embedder.config.embedding_dimension
        assert np.all(result == 0)


class TestOpenAIEmbedderUtilities:
    """Test OpenAI embedder utility methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = OpenAIEmbeddingConfig(api_key="test-key")
        
        with patch('src.refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            self.embedder = OpenAIEmbedder(config=self.config)
    
    def test_is_available_with_client(self):
        """Test availability check with client"""
        self.embedder._client = Mock()
        assert self.embedder.is_available() is True
    
    def test_is_available_without_client(self):
        """Test availability check without client"""
        self.embedder._client = None
        assert self.embedder.is_available() is False
    
    def test_get_model_info(self):
        """Test getting model information"""
        info = self.embedder.get_model_info()
        
        assert info["model_name"] == "text-embedding-3-small"
        assert info["embedding_dimension"] == 1536
        assert info["provider"] == "OpenAI"
        assert "api_key_set" in info
        assert "rate_limit" in info


class TestOpenAIEmbedderModelDimensions:
    """Test OpenAI embedder model dimensions"""
    
    def test_text_embedding_3_small_dimensions(self):
        """Test text-embedding-3-small dimensions"""
        config = OpenAIEmbeddingConfig(
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        with patch('src.refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            embedder = OpenAIEmbedder(config=config)
            assert embedder.config.embedding_dimension == 1536
    
    def test_text_embedding_3_large_dimensions(self):
        """Test text-embedding-3-large dimensions"""
        config = OpenAIEmbeddingConfig(
            api_key="test-key",
            model_name="text-embedding-3-large"
        )
        
        with patch('src.refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            embedder = OpenAIEmbedder(config=config)
            assert embedder.config.embedding_dimension == 3072
    
    def test_ada_002_dimensions(self):
        """Test text-embedding-ada-002 dimensions"""
        config = OpenAIEmbeddingConfig(
            api_key="test-key",
            model_name="text-embedding-ada-002"
        )
        
        with patch('src.refinire_rag.embedding.openai_embedder.openai.OpenAI'):
            embedder = OpenAIEmbedder(config=config)
            assert embedder.config.embedding_dimension == 1536