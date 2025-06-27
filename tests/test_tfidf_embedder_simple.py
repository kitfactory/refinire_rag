"""
Simple focused tests for TFIDFEmbedder to improve coverage
TFIDFEmbedderのカバレッジ向上のためのシンプルなテスト
"""

import pytest
import tempfile
from pathlib import Path

from refinire_rag.embedding.tfidf_embedder import TFIDFEmbedder, TFIDFEmbeddingConfig
from refinire_rag.models.document import Document


class TestTFIDFEmbedderSimple:
    """Simple TF-IDF embedder tests"""
    
    def test_basic_configuration(self):
        """Test basic configuration"""
        config = TFIDFEmbeddingConfig(
            max_features=100,
            min_df=1,
            ngram_range=(1, 2)
        )
        
        embedder = TFIDFEmbedder(config=config)
        assert embedder.config.max_features == 100
        assert embedder.config.min_df == 1
        assert embedder.config.ngram_range == (1, 2)
    
    def test_fit_and_embed(self):
        """Test fitting and embedding text"""
        config = TFIDFEmbeddingConfig(max_features=50, min_df=1)
        embedder = TFIDFEmbedder(config=config)
        
        # Test texts
        texts = [
            "machine learning algorithms",
            "python programming language", 
            "data science methods"
        ]
        
        # Fit the embedder
        embedder.fit(texts)
        
        # Test embedding
        embedding = embedder.embed_text("machine learning")
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert len(embedding) <= 50
        assert all(isinstance(x, (int, float)) for x in embedding)
    
    def test_embed_documents(self):
        """Test embedding documents"""
        config = TFIDFEmbeddingConfig(max_features=30, min_df=1)
        embedder = TFIDFEmbedder(config=config)
        
        docs = [
            Document(id="doc1", content="machine learning", metadata={}),
            Document(id="doc2", content="data science", metadata={})
        ]
        
        embeddings = embedder.embed_documents(docs)
        
        assert len(embeddings) == 2
        assert all(len(emb) > 0 for emb in embeddings)
    
    def test_get_dimension(self):
        """Test getting embedding dimension"""
        config = TFIDFEmbeddingConfig(max_features=20, min_df=1)
        embedder = TFIDFEmbedder(config=config)
        
        texts = ["test text", "another text"]
        embedder.fit(texts)
        
        dimension = embedder.get_embedding_dimension()
        assert isinstance(dimension, int)
        assert dimension > 0
        assert dimension <= 20
    
    def test_vocabulary_access(self):
        """Test vocabulary access"""
        config = TFIDFEmbeddingConfig(max_features=15, min_df=1)
        embedder = TFIDFEmbedder(config=config)
        
        texts = ["machine learning", "data science"]
        embedder.fit(texts)
        
        vocab = embedder.get_vocabulary()
        assert isinstance(vocab, dict)
        assert len(vocab) > 0
        
        feature_names = embedder.get_feature_names()
        assert isinstance(feature_names, list)
        assert len(feature_names) == len(vocab)
    
    def test_similarity_calculation(self):
        """Test similarity calculation"""
        config = TFIDFEmbeddingConfig(max_features=25, min_df=1)
        embedder = TFIDFEmbedder(config=config)
        
        texts = ["machine learning", "data science", "programming"]
        embedder.fit(texts)
        
        emb1 = embedder.embed_text("machine learning")
        emb2 = embedder.embed_text("machine")
        
        similarity = embedder.compute_similarity(emb1, emb2)
        
        assert isinstance(similarity, (int, float))
        assert 0 <= similarity <= 1
    
    def test_batch_embedding(self):
        """Test batch embedding"""
        config = TFIDFEmbeddingConfig(max_features=40, min_df=1)
        embedder = TFIDFEmbedder(config=config)
        
        texts = ["text one", "text two", "text three"]
        embedder.fit(texts)
        
        batch_texts = ["new text", "another text"]
        embeddings = embedder.embed_batch(batch_texts)
        
        assert len(embeddings) == 2
        assert all(len(emb) > 0 for emb in embeddings)
        
        # Check dimensions are consistent
        dim = len(embeddings[0])
        assert all(len(emb) == dim for emb in embeddings)
    
    def test_save_and_load(self):
        """Test model persistence"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "tfidf_model.pkl"
            
            # Train and save
            config = TFIDFEmbeddingConfig(max_features=10, min_df=1)
            embedder1 = TFIDFEmbedder(config)
            
            texts = ["test", "data"]
            embedder1.fit(texts)
            embedder1.save_model(str(model_path))
            
            # Load and test
            embedder2 = TFIDFEmbedder(config)
            embedder2.load_model(str(model_path))
            
            # Test consistency
            test_text = "test"
            emb1 = embedder1.embed_text(test_text)
            emb2 = embedder2.embed_text(test_text)
            
            assert emb1 == emb2
    
    def test_empty_text_handling(self):
        """Test empty text handling"""
        config = TFIDFEmbeddingConfig(max_features=5, min_df=1)
        embedder = TFIDFEmbedder(config=config)
        
        texts = ["some text"]
        embedder.fit(texts)
        
        # Test empty string
        empty_emb = embedder.embed_text("")
        assert isinstance(empty_emb, list)
        assert len(empty_emb) == embedder.get_embedding_dimension()
    
    def test_normalization(self):
        """Test embedding normalization"""
        config = TFIDFEmbeddingConfig(
            max_features=15,
            min_df=1,
            normalize_embeddings=True
        )
        embedder = TFIDFEmbedder(config=config)
        
        texts = ["machine learning", "data science"]
        embedder.fit(texts)
        
        embedding = embedder.embed_text("machine learning")
        
        # Check normalization
        import numpy as np
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6


class TestTFIDFEmbedderAdvanced:
    """Advanced TF-IDF embedder tests"""
    
    def test_ngram_features(self):
        """Test n-gram feature extraction"""
        config = TFIDFEmbeddingConfig(
            max_features=50,
            ngram_range=(1, 2),
            min_df=1
        )
        embedder = TFIDFEmbedder(config=config)
        
        texts = ["machine learning", "deep learning"]
        embedder.fit(texts)
        
        feature_names = embedder.get_feature_names()
        
        # Should have unigrams and bigrams
        unigrams = [f for f in feature_names if ' ' not in f]
        bigrams = [f for f in feature_names if f.count(' ') == 1]
        
        assert len(unigrams) > 0
        assert len(bigrams) > 0
    
    def test_min_max_df_filtering(self):
        """Test document frequency filtering"""
        # Create texts with different term frequencies
        texts = [
            "common word appears",
            "common word appears", 
            "common word appears",
            "rare term once"
        ]
        
        config = TFIDFEmbeddingConfig(
            min_df=2,  # Require at least 2 documents
            max_df=0.8,  # Exclude terms in >80% of docs
            max_features=100
        )
        
        embedder = TFIDFEmbedder(config=config)
        embedder.fit(texts)
        
        feature_names = embedder.get_feature_names()
        
        # "rare" should be filtered out due to min_df
        assert "rare" not in feature_names
    
    def test_stopword_removal(self):
        """Test stopword removal"""
        config_with_stopwords = TFIDFEmbeddingConfig(
            remove_stopwords=True,
            min_df=1,
            max_features=20
        )
        config_without_stopwords = TFIDFEmbeddingConfig(
            remove_stopwords=False,
            min_df=1,
            max_features=20
        )
        
        texts = ["the machine learning algorithm is good"]
        
        embedder_with = TFIDFEmbedder(config_with_stopwords)
        embedder_with.fit(texts)
        
        embedder_without = TFIDFEmbedder(config_without_stopwords)
        embedder_without.fit(texts)
        
        vocab_with = embedder_with.get_vocabulary()
        vocab_without = embedder_without.get_vocabulary()
        
        # With stopword removal should have fewer or equal terms
        assert len(vocab_with) <= len(vocab_without)


class TestTFIDFEmbedderError:
    """Error handling tests"""
    
    def test_embed_before_fit(self):
        """Test embedding before fitting"""
        embedder = TFIDFEmbedder(config=TFIDFEmbeddingConfig())
        
        with pytest.raises(ValueError):
            embedder.embed_text("test")
    
    def test_fit_empty_corpus(self):
        """Test fitting with empty corpus"""
        embedder = TFIDFEmbedder(config=TFIDFEmbeddingConfig())
        
        with pytest.raises(ValueError):
            embedder.fit([])
    
    def test_fit_with_empty_strings(self):
        """Test fitting with all empty strings"""
        embedder = TFIDFEmbedder(config=TFIDFEmbeddingConfig(min_df=1))
        
        # This might raise an error or handle gracefully
        try:
            embedder.fit(["", " ", ""])
        except ValueError:
            # Expected behavior for all-empty corpus
            pass