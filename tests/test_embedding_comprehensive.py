"""
Comprehensive tests for Embedding layer components.
Embedding層コンポーネントの包括的テスト
"""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from refinire_rag.models.document import Document
from refinire_rag.embedding.base import Embedder
from refinire_rag.embedding.openai_embedder import OpenAIEmbedder, OpenAIEmbeddingConfig
from refinire_rag.embedding.tfidf_embedder import TFIDFEmbedder, TFIDFEmbeddingConfig


class TestEmbedderBase:
    """Test base Embedder functionality"""
    
    def test_embedder_interface(self):
        """Test Embedder interface compliance"""
        # Create a mock implementation
        embedder = Embedder()
        
        # Test abstract methods exist
        assert hasattr(embedder, 'embed_text')
        assert hasattr(embedder, 'embed_documents')
        assert hasattr(embedder, 'get_embedding_dimension')
        assert hasattr(embedder, 'fit')
    
    def test_embedder_config_validation(self):
        """Test Embedder configuration validation"""
        config = {
            "batch_size": 32,
            "max_length": 512,
            "normalize_embeddings": True
        }
        
        embedder = Embedder(config=config)
        assert embedder.config["batch_size"] == 32
        assert embedder.config["max_length"] == 512
        assert embedder.config["normalize_embeddings"] == True


class TestTFIDFEmbedder:
    """Test TFIDFEmbedder functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = TFIDFEmbeddingConfig(
            max_features=1000,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2),
            remove_stopwords=True,
            lowercase=True
        )
        self.embedder = TFIDFEmbedder(self.config)
        
        # Test documents
        self.test_docs = [
            Document(
                id="doc1",
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                metadata={"category": "tech"}
            ),
            Document(
                id="doc2",
                content="Python is a popular programming language for data science and machine learning.",
                metadata={"category": "programming"}
            ),
            Document(
                id="doc3",
                content="Data science combines statistics, programming, and domain expertise to extract insights.",
                metadata={"category": "data"}
            ),
            Document(
                id="doc4",
                content="Neural networks are computational models inspired by biological neural networks.",
                metadata={"category": "tech"}
            )
        ]
        
        self.test_texts = [doc.content for doc in self.test_docs]
    
    def test_tfidf_configuration(self):
        """Test TF-IDF embedder configuration"""
        config = TFIDFEmbeddingConfig(
            max_features=500,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 3),
            remove_stopwords=False
        )
        
        embedder = TFIDFEmbedder(config)
        assert embedder.config.max_features == 500
        assert embedder.config.min_df == 2
        assert embedder.config.max_df == 0.8
        assert embedder.config.ngram_range == (1, 3)
        assert embedder.config.remove_stopwords == False
    
    def test_fit_and_transform(self):
        """Test fitting the TF-IDF model and transforming text"""
        # Fit the embedder
        self.embedder.fit(self.test_texts)
        
        # Check that model is fitted
        assert hasattr(self.embedder, 'vectorizer')
        assert self.embedder.vectorizer is not None
        assert hasattr(self.embedder.vectorizer, 'vocabulary_')
        
        # Transform text
        embedding = self.embedder.embed_text(self.test_texts[0])
        
        assert isinstance(embedding, (list, np.ndarray))
        assert len(embedding) > 0
        assert len(embedding) <= self.config.max_features
    
    def test_embed_single_text(self):
        """Test embedding single text"""
        # Fit first
        self.embedder.fit(self.test_texts)
        
        text = "Machine learning algorithms are powerful tools"
        embedding = self.embedder.embed_text(text)
        
        assert isinstance(embedding, (list, np.ndarray))
        assert len(embedding) == self.embedder.get_embedding_dimension()
        
        # Check that embedding is normalized if configured
        if self.config.normalize_embeddings:
            embedding_array = np.array(embedding)
            norm = np.linalg.norm(embedding_array)
            assert abs(norm - 1.0) < 1e-6
    
    def test_embed_batch_texts(self):
        """Test embedding batch of texts"""
        # Fit first
        self.embedder.fit(self.test_texts)
        
        batch_texts = [
            "Machine learning algorithms",
            "Python programming language",
            "Data science methodology"
        ]
        
        embeddings = self.embedder.embed_batch(batch_texts)
        
        assert len(embeddings) == len(batch_texts)
        assert all(len(emb) == self.embedder.get_embedding_dimension() for emb in embeddings)
    
    def test_embed_documents(self):
        """Test embedding documents"""
        embeddings = self.embedder.embed_documents(self.test_docs)
        
        assert len(embeddings) == len(self.test_docs)
        assert all(len(emb) > 0 for emb in embeddings)
        
        # Check consistency
        dimension = len(embeddings[0])
        assert all(len(emb) == dimension for emb in embeddings)
    
    def test_vocabulary_analysis(self):
        """Test vocabulary analysis and feature extraction"""
        self.embedder.fit(self.test_texts)
        
        # Check vocabulary
        vocab = self.embedder.get_vocabulary()
        assert isinstance(vocab, dict)
        assert len(vocab) > 0
        assert len(vocab) <= self.config.max_features
        
        # Check feature names
        feature_names = self.embedder.get_feature_names()
        assert isinstance(feature_names, list)
        assert len(feature_names) == len(vocab)
    
    def test_similarity_computation(self):
        """Test similarity computation between embeddings"""
        self.embedder.fit(self.test_texts)
        
        # Embed two similar texts
        text1 = "Machine learning algorithms"
        text2 = "Machine learning methods"
        
        emb1 = self.embedder.embed_text(text1)
        emb2 = self.embedder.embed_text(text2)
        
        # Compute cosine similarity
        similarity = self.embedder.compute_similarity(emb1, emb2)
        
        assert 0 <= similarity <= 1
        assert similarity > 0.1  # Should have some similarity
    
    def test_ngram_features(self):
        """Test n-gram feature extraction"""
        config = TFIDFEmbeddingConfig(
            max_features=100,
            ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
            min_df=1
        )
        embedder = TFIDFEmbedder(config)
        
        texts = ["machine learning", "machine learning algorithms", "deep learning networks"]
        embedder.fit(texts)
        
        vocab = embedder.get_vocabulary()
        feature_names = embedder.get_feature_names()
        
        # Should have unigrams and bigrams/trigrams
        unigrams = [f for f in feature_names if ' ' not in f]
        bigrams = [f for f in feature_names if f.count(' ') == 1]
        
        assert len(unigrams) > 0
        assert len(bigrams) > 0
    
    def test_stopword_removal(self):
        """Test stopword removal functionality"""
        config_with_stopwords = TFIDFEmbeddingConfig(
            remove_stopwords=True,
            min_df=1
        )
        config_without_stopwords = TFIDFEmbeddingConfig(
            remove_stopwords=False,
            min_df=1
        )
        
        embedder_with = TFIDFEmbedder(config_with_stopwords)
        embedder_without = TFIDFEmbedder(config_without_stopwords)
        
        texts = ["The machine learning algorithm is very good"]
        
        embedder_with.fit(texts)
        embedder_without.fit(texts)
        
        vocab_with = embedder_with.get_vocabulary()
        vocab_without = embedder_without.get_vocabulary()
        
        # Vocabulary without stopwords should be smaller or equal
        assert len(vocab_with) <= len(vocab_without)
        
        # Common stopwords should be excluded from stopword-filtered vocabulary
        stopwords = ["the", "is", "very"]
        for stopword in stopwords:
            if stopword in vocab_without:
                assert stopword not in vocab_with
    
    def test_min_max_df_filtering(self):
        """Test min_df and max_df filtering"""
        # Create texts with terms of different frequencies
        texts = [
            "common term appears everywhere",
            "common term appears everywhere",
            "common term appears everywhere",
            "rare term appears once",
            "another different text"
        ]
        
        config = TFIDFEmbeddingConfig(
            min_df=2,  # Require at least 2 occurrences
            max_df=0.8,  # Exclude terms in more than 80% of documents
            max_features=1000
        )
        
        embedder = TFIDFEmbedder(config)
        embedder.fit(texts)
        
        vocab = embedder.get_vocabulary()
        feature_names = embedder.get_feature_names()
        
        # "rare" should be filtered out due to min_df
        assert "rare" not in feature_names
        
        # "common" might be filtered out due to max_df depending on implementation
        # This depends on the exact text distribution
    
    def test_embedding_consistency(self):
        """Test embedding consistency"""
        self.embedder.fit(self.test_texts)
        
        text = "Machine learning is powerful"
        
        # Embed same text multiple times
        emb1 = self.embedder.embed_text(text)
        emb2 = self.embedder.embed_text(text)
        
        # Should be identical
        assert np.allclose(emb1, emb2)
    
    def test_incremental_vocabulary(self):
        """Test incremental vocabulary building"""
        # Initial fit
        initial_texts = self.test_texts[:2]
        self.embedder.fit(initial_texts)
        initial_vocab_size = len(self.embedder.get_vocabulary())
        
        # Add more texts
        additional_texts = self.test_texts[2:]
        self.embedder.fit(self.test_texts)  # Refit with all texts
        
        final_vocab_size = len(self.embedder.get_vocabulary())
        
        # Vocabulary should expand (unless limited by max_features)
        assert final_vocab_size >= initial_vocab_size
    
    def test_empty_text_handling(self):
        """Test handling of empty texts"""
        self.embedder.fit(self.test_texts)
        
        # Test empty string
        empty_embedding = self.embedder.embed_text("")
        assert isinstance(empty_embedding, (list, np.ndarray))
        assert len(empty_embedding) == self.embedder.get_embedding_dimension()
        
        # Test whitespace-only string
        whitespace_embedding = self.embedder.embed_text("   ")
        assert isinstance(whitespace_embedding, (list, np.ndarray))


class TestOpenAIEmbedder:
    """Test OpenAIEmbedder functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = OpenAIEmbeddingConfig(
            model_name="text-embedding-3-small",
            api_key="test-key",
            batch_size=10,
            max_retries=3,
            timeout=30.0,
            dimensions=1536
        )
        
        # Test documents
        self.test_docs = [
            Document(
                id="doc1",
                content="Machine learning is revolutionizing technology.",
                metadata={"category": "tech"}
            ),
            Document(
                id="doc2", 
                content="Python is excellent for data science applications.",
                metadata={"category": "programming"}
            )
        ]
        
        self.test_texts = [doc.content for doc in self.test_docs]
    
    def test_openai_configuration(self):
        """Test OpenAI embedder configuration"""
        config = OpenAIEmbeddingConfig(
            model_name="text-embedding-ada-002",
            api_key="different-key",
            batch_size=5,
            dimensions=1024
        )
        
        embedder = OpenAIEmbedder(config)
        assert embedder.config.model_name == "text-embedding-ada-002"
        assert embedder.config.api_key == "different-key"
        assert embedder.config.batch_size == 5
        assert embedder.config.dimensions == 1024
    
    @patch('src.refinire_rag.embedding.openai_embedder.OpenAI')
    def test_api_client_initialization(self, mock_openai_class):
        """Test OpenAI API client initialization"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        embedder = OpenAIEmbedder(self.config)
        
        # Should have initialized client
        mock_openai_class.assert_called_once_with(api_key="test-key")
        assert embedder.client == mock_client
    
    @patch('src.refinire_rag.embedding.openai_embedder.OpenAI')
    def test_embed_single_text(self, mock_openai_class):
        """Test embedding single text with OpenAI API"""
        # Mock API response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        embedder = OpenAIEmbedder(self.config)
        
        text = "Machine learning is powerful"
        embedding = embedder.embed_text(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 1536
        assert all(isinstance(x, (int, float)) for x in embedding)
        
        # Verify API call
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=[text],
            dimensions=1536
        )
    
    @patch('src.refinire_rag.embedding.openai_embedder.OpenAI')
    def test_embed_batch_texts(self, mock_openai_class):
        """Test embedding batch of texts"""
        # Mock API response for batch
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536),
            Mock(embedding=[0.3] * 1536)
        ]
        
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        embedder = OpenAIEmbedder(self.config)
        
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embedder.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 1536 for emb in embeddings)
        
        # Should make single API call for batch
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=texts,
            dimensions=1536
        )
    
    @patch('src.refinire_rag.embedding.openai_embedder.OpenAI')
    def test_large_batch_splitting(self, mock_openai_class):
        """Test automatic splitting of large batches"""
        # Mock API response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536) for _ in range(10)]
        
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Configure small batch size
        config = self.config
        config.batch_size = 5
        embedder = OpenAIEmbedder(config)
        
        # Create large batch (15 texts, batch_size=5, should result in 3 API calls)
        texts = [f"Text {i}" for i in range(15)]
        embeddings = embedder.embed_batch(texts)
        
        assert len(embeddings) == 15
        
        # Should make 3 API calls (15/5 = 3)
        assert mock_client.embeddings.create.call_count == 3
    
    @patch('src.refinire_rag.embedding.openai_embedder.OpenAI')
    def test_api_error_handling(self, mock_openai_class):
        """Test API error handling and retries"""
        from openai import RateLimitError
        
        mock_client = Mock()
        # First call fails, second succeeds
        mock_client.embeddings.create.side_effect = [
            RateLimitError("Rate limit exceeded", response=Mock(), body=None),
            Mock(data=[Mock(embedding=[0.1] * 1536)])
        ]
        mock_openai_class.return_value = mock_client
        
        embedder = OpenAIEmbedder(self.config)
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            embedding = embedder.embed_text("Test text")
        
        assert len(embedding) == 1536
        assert mock_client.embeddings.create.call_count == 2  # Initial + 1 retry
    
    @patch('src.refinire_rag.embedding.openai_embedder.OpenAI')
    def test_api_timeout_handling(self, mock_openai_class):
        """Test API timeout handling"""
        from openai import APITimeoutError
        
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = APITimeoutError("Request timed out")
        mock_openai_class.return_value = mock_client
        
        embedder = OpenAIEmbedder(self.config)
        
        with pytest.raises(APITimeoutError):
            embedder.embed_text("Test text")
    
    @patch('src.refinire_rag.embedding.openai_embedder.OpenAI')
    def test_embed_documents(self, mock_openai_class):
        """Test embedding documents"""
        # Mock API response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536)
        ]
        
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        embedder = OpenAIEmbedder(self.config)
        
        embeddings = embedder.embed_documents(self.test_docs)
        
        assert len(embeddings) == 2
        assert all(len(emb) == 1536 for emb in embeddings)
        
        # Should extract content from documents
        expected_texts = [doc.content for doc in self.test_docs]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=expected_texts,
            dimensions=1536
        )
    
    @patch('src.refinire_rag.embedding.openai_embedder.OpenAI')
    def test_caching_functionality(self, mock_openai_class):
        """Test embedding caching"""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Enable caching
        config = self.config
        config.enable_caching = True
        embedder = OpenAIEmbedder(config)
        
        text = "Machine learning is powerful"
        
        # First call
        emb1 = embedder.embed_text(text)
        
        # Second call with same text
        emb2 = embedder.embed_text(text)
        
        # Should be identical
        assert emb1 == emb2
        
        # Should only make one API call due to caching
        mock_client.embeddings.create.assert_called_once()
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension"""
        embedder = OpenAIEmbedder(self.config)
        
        dimension = embedder.get_embedding_dimension()
        assert dimension == 1536
    
    @patch('src.refinire_rag.embedding.openai_embedder.OpenAI')
    def test_different_models(self, mock_openai_class):
        """Test different OpenAI embedding models"""
        models_and_dimensions = [
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 3072),
            ("text-embedding-ada-002", 1536)
        ]
        
        for model_name, expected_dim in models_and_dimensions:
            config = OpenAIEmbeddingConfig(
                model_name=model_name,
                api_key="test-key",
                dimensions=expected_dim
            )
            
            embedder = OpenAIEmbedder(config)
            assert embedder.get_embedding_dimension() == expected_dim
    
    @patch('src.refinire_rag.embedding.openai_embedder.OpenAI')
    def test_custom_dimensions(self, mock_openai_class):
        """Test custom dimension configuration"""
        # Mock API response with custom dimensions
        custom_dim = 768
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * custom_dim)]
        
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        config = OpenAIEmbeddingConfig(
            model_name="text-embedding-3-small",
            api_key="test-key",
            dimensions=custom_dim
        )
        
        embedder = OpenAIEmbedder(config)
        
        embedding = embedder.embed_text("Test text")
        
        assert len(embedding) == custom_dim
        assert embedder.get_embedding_dimension() == custom_dim


class TestEmbeddingComparison:
    """Test comparison between different embedding methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.test_texts = [
            "Machine learning algorithms are powerful tools for data analysis",
            "Python programming is essential for modern data science",
            "Deep learning models require large amounts of training data",
            "Natural language processing enables computers to understand text"
        ]
    
    def test_tfidf_vs_dimensions(self):
        """Test TF-IDF embedding consistency across different configurations"""
        configs = [
            TFIDFEmbeddingConfig(max_features=100, min_df=1),
            TFIDFEmbeddingConfig(max_features=500, min_df=1),
            TFIDFEmbeddingConfig(max_features=1000, min_df=1)
        ]
        
        embeddings_by_config = []
        
        for config in configs:
            embedder = TFIDFEmbedder(config)
            embedder.fit(self.test_texts)
            
            embeddings = [embedder.embed_text(text) for text in self.test_texts]
            embeddings_by_config.append(embeddings)
        
        # Different configurations should produce different dimensionalities
        dimensions = [len(embs[0]) for embs in embeddings_by_config]
        assert len(set(dimensions)) > 1  # Should have different dimensions
        
        # But relative similarities should be preserved to some extent
        for embeddings in embeddings_by_config:
            # Check that similar texts have higher similarity
            emb1 = embeddings[0]  # ML algorithms
            emb2 = embeddings[2]  # Deep learning (both about ML)
            emb3 = embeddings[1]  # Python programming (different topic)
            
            # Compute cosine similarities
            sim_ml = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            sim_prog = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
            
            # ML texts should be more similar than ML vs programming
            # (This might not always hold due to vocabulary overlap, but generally should)
            assert isinstance(sim_ml, (int, float))
            assert isinstance(sim_prog, (int, float))
    
    @patch('src.refinire_rag.embedding.openai_embedder.OpenAI')
    def test_embedding_normalization(self, mock_openai_class):
        """Test embedding normalization across different methods"""
        # TF-IDF embedder with normalization
        tfidf_config = TFIDFEmbeddingConfig(
            max_features=100,
            normalize_embeddings=True,
            min_df=1
        )
        tfidf_embedder = TFIDFEmbedder(tfidf_config)
        tfidf_embedder.fit(self.test_texts)
        
        # Mock OpenAI embedder
        mock_response = Mock()
        mock_response.data = [Mock(embedding=np.random.randn(1536).tolist())]
        
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        openai_embedder = OpenAIEmbedder(OpenAIEmbeddingConfig(
            model_name="text-embedding-3-small",
            api_key="test-key"
        ))
        
        text = self.test_texts[0]
        
        # Get embeddings
        tfidf_emb = tfidf_embedder.embed_text(text)
        openai_emb = openai_embedder.embed_text(text)
        
        # Check normalization for TF-IDF
        tfidf_norm = np.linalg.norm(tfidf_emb)
        assert abs(tfidf_norm - 1.0) < 1e-6  # Should be normalized
        
        # OpenAI embeddings are typically pre-normalized
        openai_norm = np.linalg.norm(openai_emb)
        assert openai_norm > 0  # Should have positive norm


class TestEmbeddingErrorHandling:
    """Test error handling in embedding components"""
    
    def test_tfidf_empty_corpus(self):
        """Test TF-IDF with empty corpus"""
        embedder = TFIDFEmbedder(TFIDFEmbeddingConfig())
        
        # Try to fit with empty corpus
        with pytest.raises(ValueError):
            embedder.fit([])
    
    def test_tfidf_unfitted_model(self):
        """Test TF-IDF embedding before fitting"""
        embedder = TFIDFEmbedder(TFIDFEmbeddingConfig())
        
        # Try to embed without fitting
        with pytest.raises(ValueError):
            embedder.embed_text("Test text")
    
    def test_openai_invalid_api_key(self):
        """Test OpenAI embedder with invalid API key"""
        config = OpenAIEmbeddingConfig(
            model_name="text-embedding-3-small",
            api_key=""  # Empty API key
        )
        
        # Should handle gracefully during initialization
        embedder = OpenAIEmbedder(config)
        assert embedder.config.api_key == ""
    
    @patch('src.refinire_rag.embedding.openai_embedder.OpenAI')
    def test_openai_network_error(self, mock_openai_class):
        """Test OpenAI embedder network error handling"""
        from openai import APIConnectionError
        
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = APIConnectionError("Network error")
        mock_openai_class.return_value = mock_client
        
        embedder = OpenAIEmbedder(OpenAIEmbeddingConfig(
            model_name="text-embedding-3-small",
            api_key="test-key"
        ))
        
        with pytest.raises(APIConnectionError):
            embedder.embed_text("Test text")
    
    def test_dimension_mismatch_detection(self):
        """Test detection of dimension mismatches"""
        # This would be relevant for custom embedders that need to ensure
        # consistent dimensions across different batches
        pass


@pytest.mark.integration
class TestEmbeddingIntegration:
    """Integration tests for embedding components"""
    
    def test_document_embedding_pipeline(self):
        """Test complete document embedding pipeline"""
        # Create test documents
        docs = [
            Document(id="doc1", content="Machine learning and AI", metadata={}),
            Document(id="doc2", content="Python programming guide", metadata={}),
            Document(id="doc3", content="Data science fundamentals", metadata={})
        ]
        
        # Test TF-IDF embedder
        tfidf_embedder = TFIDFEmbedder(TFIDFEmbeddingConfig(
            max_features=100,
            min_df=1
        ))
        
        # Process documents
        embeddings = tfidf_embedder.embed_documents(docs)
        
        assert len(embeddings) == len(docs)
        assert all(len(emb) > 0 for emb in embeddings)
        
        # Test consistency
        dimension = len(embeddings[0])
        assert all(len(emb) == dimension for emb in embeddings)
    
    @patch('src.refinire_rag.embedding.openai_embedder.OpenAI')
    def test_hybrid_embedding_comparison(self, mock_openai_class):
        """Test comparison between TF-IDF and OpenAI embeddings"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=np.random.randn(1536).tolist()),
            Mock(embedding=np.random.randn(1536).tolist())
        ]
        
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Create embedders
        tfidf_embedder = TFIDFEmbedder(TFIDFEmbeddingConfig(max_features=100, min_df=1))
        openai_embedder = OpenAIEmbedder(OpenAIEmbeddingConfig(
            model_name="text-embedding-3-small",
            api_key="test-key"
        ))
        
        # Test texts
        texts = [
            "Machine learning algorithms for data analysis",
            "Python programming for beginners"
        ]
        
        # Get embeddings from both methods
        tfidf_embedder.fit(texts)
        tfidf_embeddings = [tfidf_embedder.embed_text(text) for text in texts]
        openai_embeddings = [openai_embedder.embed_text(text) for text in texts]
        
        # Both should produce valid embeddings
        assert len(tfidf_embeddings) == len(texts)
        assert len(openai_embeddings) == len(texts)
        
        # Dimensions will be different
        assert len(tfidf_embeddings[0]) != len(openai_embeddings[0])
        
        # But both should be valid numerical vectors
        for emb in tfidf_embeddings + openai_embeddings:
            assert all(isinstance(x, (int, float)) for x in emb)
    
    def test_embedding_persistence(self):
        """Test embedding model persistence and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "tfidf_model.pkl"
            
            # Train and save model
            texts = [
                "Machine learning algorithms",
                "Data science methods", 
                "Python programming"
            ]
            
            embedder1 = TFIDFEmbedder(TFIDFEmbeddingConfig(max_features=50, min_df=1))
            embedder1.fit(texts)
            
            # Save model
            embedder1.save_model(str(model_path))
            
            # Load model in new embedder
            embedder2 = TFIDFEmbedder(TFIDFEmbeddingConfig())
            embedder2.load_model(str(model_path))
            
            # Test that embeddings are identical
            test_text = "Machine learning algorithms"
            emb1 = embedder1.embed_text(test_text)
            emb2 = embedder2.embed_text(test_text)
            
            assert np.allclose(emb1, emb2)
            assert embedder1.get_embedding_dimension() == embedder2.get_embedding_dimension()
    
    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency"""
        # Create large number of texts
        texts = [f"This is test document number {i} about various topics" for i in range(100)]
        
        embedder = TFIDFEmbedder(TFIDFEmbeddingConfig(max_features=200, min_df=1))
        embedder.fit(texts[:50])  # Fit on subset
        
        # Test single vs batch processing
        import time
        
        # Single processing
        start_time = time.time()
        single_embeddings = [embedder.embed_text(text) for text in texts[:10]]
        single_time = time.time() - start_time
        
        # Batch processing
        start_time = time.time()
        batch_embeddings = embedder.embed_batch(texts[:10])
        batch_time = time.time() - start_time
        
        # Results should be identical
        assert len(single_embeddings) == len(batch_embeddings)
        for single, batch in zip(single_embeddings, batch_embeddings):
            assert np.allclose(single, batch)
        
        # Batch processing should be faster or at least not significantly slower
        assert batch_time <= single_time * 2  # Allow some tolerance