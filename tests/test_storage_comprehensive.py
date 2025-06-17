"""
Comprehensive tests for Storage layer components.
Storage層コンポーネントの包括的テスト
"""

import os
import tempfile
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.refinire_rag.models.document import Document
from src.refinire_rag.storage.vector_store import VectorStore
from src.refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from src.refinire_rag.storage.evaluation_store import SQLiteEvaluationStore
from src.refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from src.refinire_rag.storage.pickle_vector_store import PickleVectorStore


class TestVectorStore:
    """Test VectorStore base functionality"""
    
    def test_vector_store_interface(self):
        """Test VectorStore interface compliance"""
        # Create a mock implementation
        vector_store = VectorStore()
        
        # Test abstract methods exist
        assert hasattr(vector_store, 'store_embedding')
        assert hasattr(vector_store, 'search_similar')
        assert hasattr(vector_store, 'delete_embedding')
        assert hasattr(vector_store, 'clear_all')
        assert hasattr(vector_store, 'get_embedding_count')
    
    def test_vector_store_config_validation(self):
        """Test VectorStore configuration validation"""
        config = {
            "similarity_threshold": 0.8,
            "max_results": 10,
            "distance_metric": "cosine"
        }
        
        vector_store = VectorStore(config=config)
        assert vector_store.config["similarity_threshold"] == 0.8
        assert vector_store.config["max_results"] == 10
        assert vector_store.config["distance_metric"] == "cosine"


class TestInMemoryVectorStore:
    """Test InMemoryVectorStore functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.vector_store = InMemoryVectorStore()
        
        # Create test documents and embeddings
        self.test_docs = [
            Document(
                id="doc1",
                content="Machine learning is a subset of AI",
                metadata={"category": "tech"}
            ),
            Document(
                id="doc2", 
                content="Python is a programming language",
                metadata={"category": "programming"}
            ),
            Document(
                id="doc3",
                content="Data science involves statistics and programming",
                metadata={"category": "data"}
            )
        ]
        
        # Mock embeddings (simple vectors for testing)
        self.test_embeddings = [
            [0.1, 0.2, 0.3, 0.4],  # doc1
            [0.2, 0.3, 0.4, 0.5],  # doc2
            [0.15, 0.25, 0.35, 0.45]  # doc3
        ]
    
    def test_store_and_retrieve_embeddings(self):
        """Test storing and retrieving embeddings"""
        # Store embeddings
        for doc, embedding in zip(self.test_docs, self.test_embeddings):
            self.vector_store.store_embedding(doc.id, embedding, doc.metadata)
        
        # Verify storage
        assert self.vector_store.get_embedding_count() == 3
        
        # Test retrieval
        stored_embedding = self.vector_store.get_embedding("doc1")
        assert stored_embedding is not None
        assert len(stored_embedding) == 4
    
    def test_similarity_search(self):
        """Test similarity search functionality"""
        # Store test embeddings
        for doc, embedding in zip(self.test_docs, self.test_embeddings):
            self.vector_store.store_embedding(doc.id, embedding, doc.metadata)
        
        # Search with query embedding similar to doc1
        query_embedding = [0.11, 0.21, 0.31, 0.41]
        results = self.vector_store.search_similar(query_embedding, limit=2)
        
        assert len(results) <= 2
        assert len(results) > 0
        
        # Results should be sorted by similarity
        if len(results) > 1:
            assert results[0].score >= results[1].score
    
    def test_delete_embedding(self):
        """Test embedding deletion"""
        # Store embeddings
        for doc, embedding in zip(self.test_docs, self.test_embeddings):
            self.vector_store.store_embedding(doc.id, embedding, doc.metadata)
        
        assert self.vector_store.get_embedding_count() == 3
        
        # Delete one embedding
        self.vector_store.delete_embedding("doc1")
        assert self.vector_store.get_embedding_count() == 2
        
        # Verify it's gone
        assert self.vector_store.get_embedding("doc1") is None
    
    def test_clear_all_embeddings(self):
        """Test clearing all embeddings"""
        # Store embeddings
        for doc, embedding in zip(self.test_docs, self.test_embeddings):
            self.vector_store.store_embedding(doc.id, embedding, doc.metadata)
        
        assert self.vector_store.get_embedding_count() == 3
        
        # Clear all
        self.vector_store.clear_all()
        assert self.vector_store.get_embedding_count() == 0
    
    def test_metadata_filtering(self):
        """Test metadata-based filtering"""
        # Store embeddings with metadata
        for doc, embedding in zip(self.test_docs, self.test_embeddings):
            self.vector_store.store_embedding(doc.id, embedding, doc.metadata)
        
        # Search with metadata filter
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = self.vector_store.search_similar(
            query_embedding,
            limit=10,
            filters={"category": "tech"}
        )
        
        # Should only return tech category results
        for result in results:
            if hasattr(result, 'metadata'):
                assert result.metadata.get("category") == "tech"


class TestPickleVectorStore:
    """Test PickleVectorStore functionality"""
    
    def test_pickle_persistence(self):
        """Test pickle file persistence"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "test_vectors.pkl"
            
            # Create vector store with pickle file
            vector_store = PickleVectorStore(file_path=str(pickle_path))
            
            # Store some embeddings
            vector_store.store_embedding("doc1", [0.1, 0.2, 0.3], {"type": "test"})
            vector_store.store_embedding("doc2", [0.2, 0.3, 0.4], {"type": "test"})
            
            # Save to file
            vector_store.save()
            
            # Verify file exists
            assert pickle_path.exists()
            
            # Create new instance and load
            new_vector_store = PickleVectorStore(file_path=str(pickle_path))
            new_vector_store.load()
            
            # Verify data persisted
            assert new_vector_store.get_embedding_count() == 2
            assert new_vector_store.get_embedding("doc1") is not None
    
    def test_automatic_persistence(self):
        """Test automatic save/load on operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "auto_test.pkl"
            
            # Create vector store with auto-save
            vector_store = PickleVectorStore(
                file_path=str(pickle_path),
                auto_save=True
            )
            
            # Store embedding (should auto-save)
            vector_store.store_embedding("doc1", [0.1, 0.2, 0.3], {})
            
            # File should exist
            assert pickle_path.exists()


class TestSQLiteDocumentStore:
    """Test SQLiteDocumentStore functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.doc_store = SQLiteDocumentStore(str(self.db_path))
        
        # Test documents
        self.test_docs = [
            Document(
                id="doc1",
                content="Test content 1",
                metadata={"source": "file1.txt", "type": "text"}
            ),
            Document(
                id="doc2",
                content="Test content 2", 
                metadata={"source": "file2.txt", "type": "text"}
            )
        ]
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if hasattr(self, 'doc_store'):
            self.doc_store.close()
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_store_and_retrieve_document(self):
        """Test document storage and retrieval"""
        doc = self.test_docs[0]
        
        # Store document
        self.doc_store.store_document(doc)
        
        # Retrieve document
        retrieved = self.doc_store.get_document(doc.id)
        
        assert retrieved is not None
        assert retrieved.id == doc.id
        assert retrieved.content == doc.content
        assert retrieved.metadata == doc.metadata
    
    def test_document_exists(self):
        """Test document existence check"""
        doc = self.test_docs[0]
        
        # Initially should not exist
        assert not self.doc_store.document_exists(doc.id)
        
        # Store document
        self.doc_store.store_document(doc)
        
        # Now should exist
        assert self.doc_store.document_exists(doc.id)
    
    def test_list_documents(self):
        """Test document listing"""
        # Store multiple documents
        for doc in self.test_docs:
            self.doc_store.store_document(doc)
        
        # List all documents
        doc_list = self.doc_store.list_documents()
        
        assert len(doc_list) >= 2
        doc_ids = [doc.id for doc in doc_list]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
    
    def test_search_documents(self):
        """Test document search functionality"""
        # Store documents
        for doc in self.test_docs:
            self.doc_store.store_document(doc)
        
        # Search by content
        results = self.doc_store.search_documents("Test content")
        assert len(results) >= 1
        
        # Search by metadata
        results = self.doc_store.search_documents_by_metadata({"type": "text"})
        assert len(results) >= 2
    
    def test_update_document(self):
        """Test document updating"""
        doc = self.test_docs[0]
        
        # Store original
        self.doc_store.store_document(doc)
        
        # Update content
        updated_doc = Document(
            id=doc.id,
            content="Updated content",
            metadata=doc.metadata
        )
        
        self.doc_store.update_document(updated_doc)
        
        # Retrieve and verify
        retrieved = self.doc_store.get_document(doc.id)
        assert retrieved.content == "Updated content"
    
    def test_delete_document(self):
        """Test document deletion"""
        doc = self.test_docs[0]
        
        # Store document
        self.doc_store.store_document(doc)
        assert self.doc_store.document_exists(doc.id)
        
        # Delete document
        self.doc_store.delete_document(doc.id)
        assert not self.doc_store.document_exists(doc.id)
    
    def test_get_document_count(self):
        """Test document count functionality"""
        # Initially empty
        assert self.doc_store.get_document_count() == 0
        
        # Store documents
        for doc in self.test_docs:
            self.doc_store.store_document(doc)
        
        # Count should match
        assert self.doc_store.get_document_count() == 2
    
    def test_clear_all_documents(self):
        """Test clearing all documents"""
        # Store documents
        for doc in self.test_docs:
            self.doc_store.store_document(doc)
        
        assert self.doc_store.get_document_count() > 0
        
        # Clear all
        self.doc_store.clear_all_documents()
        assert self.doc_store.get_document_count() == 0
    
    def test_batch_operations(self):
        """Test batch document operations"""
        # Batch store
        self.doc_store.store_documents(self.test_docs)
        assert self.doc_store.get_document_count() == len(self.test_docs)
        
        # Batch retrieve
        doc_ids = [doc.id for doc in self.test_docs]
        retrieved_docs = self.doc_store.get_documents(doc_ids)
        assert len(retrieved_docs) == len(self.test_docs)
    
    def test_database_constraints(self):
        """Test database constraint handling"""
        doc = self.test_docs[0]
        
        # Store document
        self.doc_store.store_document(doc)
        
        # Try to store same ID again (should handle gracefully)
        # This tests the REPLACE functionality
        duplicate_doc = Document(
            id=doc.id,
            content="Different content", 
            metadata={"new": "metadata"}
        )
        
        self.doc_store.store_document(duplicate_doc)
        
        # Should have updated, not created duplicate
        assert self.doc_store.get_document_count() == 1
        retrieved = self.doc_store.get_document(doc.id)
        assert retrieved.content == "Different content"


class TestSQLiteEvaluationStore:
    """Test SQLiteEvaluationStore functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "eval_test.db"
        self.eval_store = SQLiteEvaluationStore(str(self.db_path))
        
        # Test evaluation data
        self.test_evaluation = {
            "evaluation_id": "eval_001",
            "qa_set_name": "test_set",
            "corpus_name": "test_corpus",
            "timestamp": 1234567890,
            "results": {
                "total_tests": 10,
                "passed_tests": 8,
                "success_rate": 0.8,
                "average_response_time": 2.5
            },
            "metadata": {"model": "gpt-4", "version": "1.0"}
        }
        
        self.test_qa_pair = {
            "qa_pair_id": "qa_001",
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of AI",
            "document_id": "doc1",
            "qa_set_name": "test_set",
            "metadata": {"type": "factual", "difficulty": "easy"}
        }
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if hasattr(self, 'eval_store'):
            self.eval_store.close()
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_store_and_retrieve_evaluation(self):
        """Test evaluation storage and retrieval"""
        # Store evaluation
        self.eval_store.store_evaluation_result(self.test_evaluation)
        
        # Retrieve evaluation
        retrieved = self.eval_store.get_evaluation_result(
            self.test_evaluation["evaluation_id"]
        )
        
        assert retrieved is not None
        assert retrieved["evaluation_id"] == self.test_evaluation["evaluation_id"]
        assert retrieved["qa_set_name"] == self.test_evaluation["qa_set_name"]
        assert retrieved["results"]["success_rate"] == 0.8
    
    def test_store_and_retrieve_qa_pair(self):
        """Test QA pair storage and retrieval"""
        # Store QA pair
        self.eval_store.store_qa_pair(self.test_qa_pair)
        
        # Retrieve QA pair
        retrieved = self.eval_store.get_qa_pair(self.test_qa_pair["qa_pair_id"])
        
        assert retrieved is not None
        assert retrieved["qa_pair_id"] == self.test_qa_pair["qa_pair_id"]
        assert retrieved["question"] == self.test_qa_pair["question"]
        assert retrieved["answer"] == self.test_qa_pair["answer"]
    
    def test_list_evaluations(self):
        """Test evaluation listing"""
        # Store multiple evaluations
        eval_data = []
        for i in range(3):
            eval_item = self.test_evaluation.copy()
            eval_item["evaluation_id"] = f"eval_{i:03d}"
            eval_item["qa_set_name"] = f"test_set_{i}"
            eval_data.append(eval_item)
            self.eval_store.store_evaluation_result(eval_item)
        
        # List all evaluations
        evaluations = self.eval_store.list_evaluations()
        assert len(evaluations) >= 3
        
        # List by QA set
        qa_set_evals = self.eval_store.list_evaluations(qa_set_name="test_set_1")
        assert len(qa_set_evals) >= 1
        assert qa_set_evals[0]["qa_set_name"] == "test_set_1"
    
    def test_list_qa_pairs(self):
        """Test QA pair listing"""
        # Store multiple QA pairs
        qa_pairs = []
        for i in range(5):
            qa_pair = self.test_qa_pair.copy()
            qa_pair["qa_pair_id"] = f"qa_{i:03d}"
            qa_pair["question"] = f"Question {i}?"
            qa_pairs.append(qa_pair)
            self.eval_store.store_qa_pair(qa_pair)
        
        # List all QA pairs
        all_pairs = self.eval_store.list_qa_pairs()
        assert len(all_pairs) >= 5
        
        # List by QA set
        set_pairs = self.eval_store.list_qa_pairs(qa_set_name="test_set")
        assert len(set_pairs) >= 5
    
    def test_delete_evaluation(self):
        """Test evaluation deletion"""
        # Store evaluation
        self.eval_store.store_evaluation_result(self.test_evaluation)
        
        # Verify it exists
        assert self.eval_store.get_evaluation_result(
            self.test_evaluation["evaluation_id"]
        ) is not None
        
        # Delete evaluation
        self.eval_store.delete_evaluation(self.test_evaluation["evaluation_id"])
        
        # Verify it's gone
        assert self.eval_store.get_evaluation_result(
            self.test_evaluation["evaluation_id"]
        ) is None
    
    def test_get_evaluation_statistics(self):
        """Test evaluation statistics"""
        # Store multiple evaluations with different success rates
        eval_data = []
        success_rates = [0.6, 0.7, 0.8, 0.9, 0.95]
        
        for i, rate in enumerate(success_rates):
            eval_item = self.test_evaluation.copy()
            eval_item["evaluation_id"] = f"eval_{i:03d}"
            eval_item["results"]["success_rate"] = rate
            eval_item["results"]["passed_tests"] = int(10 * rate)
            eval_data.append(eval_item)
            self.eval_store.store_evaluation_result(eval_item)
        
        # Get statistics
        stats = self.eval_store.get_evaluation_statistics()
        
        assert "total_evaluations" in stats
        assert "average_success_rate" in stats
        assert "best_success_rate" in stats
        assert "worst_success_rate" in stats
        
        assert stats["total_evaluations"] >= 5
        assert stats["average_success_rate"] > 0
        assert stats["best_success_rate"] >= stats["average_success_rate"]
        assert stats["worst_success_rate"] <= stats["average_success_rate"]
    
    def test_clear_all_data(self):
        """Test clearing all evaluation data"""
        # Store some data
        self.eval_store.store_evaluation_result(self.test_evaluation)
        self.eval_store.store_qa_pair(self.test_qa_pair)
        
        # Verify data exists
        assert len(self.eval_store.list_evaluations()) > 0
        assert len(self.eval_store.list_qa_pairs()) > 0
        
        # Clear all
        self.eval_store.clear_all_evaluations()
        self.eval_store.clear_all_qa_pairs()
        
        # Verify data is gone
        assert len(self.eval_store.list_evaluations()) == 0
        assert len(self.eval_store.list_qa_pairs()) == 0


class TestStorageErrorHandling:
    """Test error handling across storage components"""
    
    def test_sqlite_connection_error(self):
        """Test SQLite connection error handling"""
        # Try to create store with invalid path
        invalid_path = "/invalid/path/that/does/not/exist/test.db"
        
        with pytest.raises(Exception):
            SQLiteDocumentStore(invalid_path)
    
    def test_document_store_constraint_errors(self):
        """Test document store constraint error handling"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "constraint_test.db"
            doc_store = SQLiteDocumentStore(str(db_path))
            
            # Test with invalid document data
            invalid_doc = Document(
                id="",  # Empty ID should be handled gracefully
                content="Test content",
                metadata={}
            )
            
            # Should handle gracefully without crashing
            try:
                doc_store.store_document(invalid_doc)
            except Exception as e:
                # Should be a specific, handled exception
                assert isinstance(e, (ValueError, sqlite3.Error))
            
            doc_store.close()
    
    def test_vector_store_dimension_mismatch(self):
        """Test vector store dimension mismatch handling"""
        vector_store = InMemoryVectorStore()
        
        # Store embedding with 4 dimensions
        vector_store.store_embedding("doc1", [0.1, 0.2, 0.3, 0.4], {})
        
        # Try to search with different dimension
        with pytest.raises(ValueError):
            vector_store.search_similar([0.1, 0.2, 0.3])  # Only 3 dimensions
    
    def test_pickle_file_corruption(self):
        """Test pickle file corruption handling"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "corrupted.pkl"
            
            # Create corrupted file
            pickle_path.write_text("This is not valid pickle data")
            
            # Should handle corruption gracefully
            vector_store = PickleVectorStore(file_path=str(pickle_path))
            
            try:
                vector_store.load()
            except Exception as e:
                # Should be specific exception, not generic crash
                assert "pickle" in str(e).lower() or "corrupt" in str(e).lower()


@pytest.mark.integration
class TestStorageIntegration:
    """Integration tests for storage components"""
    
    def test_document_and_vector_integration(self):
        """Test integration between document store and vector store"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create stores
            doc_store = SQLiteDocumentStore(str(Path(temp_dir) / "docs.db"))
            vector_store = InMemoryVectorStore()
            
            # Create test document
            doc = Document(
                id="integration_doc",
                content="Integration test document",
                metadata={"test": "integration"}
            )
            
            # Store document and embedding
            doc_store.store_document(doc)
            vector_store.store_embedding(doc.id, [0.1, 0.2, 0.3, 0.4], doc.metadata)
            
            # Verify both stores have the data
            retrieved_doc = doc_store.get_document(doc.id)
            retrieved_embedding = vector_store.get_embedding(doc.id)
            
            assert retrieved_doc is not None
            assert retrieved_embedding is not None
            assert retrieved_doc.id == doc.id
            
            # Test coordinated deletion
            doc_store.delete_document(doc.id)
            vector_store.delete_embedding(doc.id)
            
            assert doc_store.get_document(doc.id) is None
            assert vector_store.get_embedding(doc.id) is None
            
            doc_store.close()
    
    def test_evaluation_workflow_integration(self):
        """Test complete evaluation workflow with all stores"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create all stores
            doc_store = SQLiteDocumentStore(str(Path(temp_dir) / "docs.db"))
            vector_store = InMemoryVectorStore()
            eval_store = SQLiteEvaluationStore(str(Path(temp_dir) / "eval.db"))
            
            # Create test documents
            docs = [
                Document(id="doc1", content="AI and machine learning", metadata={}),
                Document(id="doc2", content="Python programming", metadata={})
            ]
            
            # Store documents and embeddings
            for i, doc in enumerate(docs):
                doc_store.store_document(doc)
                vector_store.store_embedding(doc.id, [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i], {})
            
            # Create QA pairs
            qa_pairs = [
                {
                    "qa_pair_id": "qa1",
                    "question": "What is AI?",
                    "answer": "Artificial Intelligence",
                    "document_id": "doc1",
                    "qa_set_name": "integration_test",
                    "metadata": {}
                },
                {
                    "qa_pair_id": "qa2", 
                    "question": "What is Python?",
                    "answer": "A programming language",
                    "document_id": "doc2",
                    "qa_set_name": "integration_test",
                    "metadata": {}
                }
            ]
            
            # Store QA pairs
            for qa_pair in qa_pairs:
                eval_store.store_qa_pair(qa_pair)
            
            # Store evaluation result
            evaluation = {
                "evaluation_id": "integration_eval",
                "qa_set_name": "integration_test",
                "corpus_name": "integration_corpus",
                "timestamp": 1234567890,
                "results": {
                    "total_tests": 2,
                    "passed_tests": 2,
                    "success_rate": 1.0
                },
                "metadata": {"integration": True}
            }
            
            eval_store.store_evaluation_result(evaluation)
            
            # Verify complete workflow
            assert doc_store.get_document_count() == 2
            assert vector_store.get_embedding_count() == 2
            assert len(eval_store.list_qa_pairs()) >= 2
            assert len(eval_store.list_evaluations()) >= 1
            
            # Clean up
            doc_store.close()
            eval_store.close()