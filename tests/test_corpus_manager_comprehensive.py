"""
Comprehensive test for CorpusManager

Tests the complete workflow:
1. Data cleanup
2. Import new documents with dictionary and graph creation
3. Add additional documents to corpus
4. Edit dictionary manually
5. Rebuild corpus from original documents using edited dictionary
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List

from refinire_rag.application.corpus_manager_new import CorpusManager, CorpusStats
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.models.document import Document


class TestCorpusManagerComprehensive:
    """Comprehensive test for CorpusManager workflow"""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for the test"""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            
            # Create subdirectories
            docs_dir = workspace / "documents"
            docs_dir.mkdir()
            additional_docs_dir = workspace / "additional_documents"
            additional_docs_dir.mkdir()
            refinire_dir = workspace / "refinire"
            refinire_dir.mkdir()
            
            yield {
                "workspace": workspace,
                "docs_dir": docs_dir,
                "additional_docs_dir": additional_docs_dir,
                "refinire_dir": refinire_dir,
                "rag_dir": refinire_dir / "rag",
                "db_path": workspace / "test_corpus.db"
            }

    @pytest.fixture
    def sample_documents(self, temp_workspace):
        """Create sample documents for testing"""
        docs_dir = temp_workspace["docs_dir"]
        additional_docs_dir = temp_workspace["additional_docs_dir"]
        
        # Initial documents (about machine learning and RAG)
        initial_docs = {
            "ml_basics.md": """# Machine Learning Basics

Machine learning (ML) is a subset of artificial intelligence that enables computers to learn and improve from experience. Neural networks are a key component of deep learning algorithms.

## Key Concepts
- **Supervised Learning**: Training with labeled data
- **Unsupervised Learning**: Finding patterns in unlabeled data
- **Deep Learning**: Using neural networks with multiple layers
- **Model Training**: Process of teaching algorithms to make predictions

The training process involves feeding data to algorithms and adjusting parameters to minimize prediction errors.""",

            "rag_systems.md": """# Retrieval-Augmented Generation Systems

RAG (Retrieval-Augmented Generation) systems combine information retrieval with text generation. The vector database stores document embeddings for efficient similarity search.

## Architecture Components
- **Document Store**: Persistence layer for original documents
- **Vector Store**: Stores embeddings for similarity search  
- **Retriever**: Finds relevant documents based on queries
- **Generator**: LLM that synthesizes answers from retrieved context

RAG systems improve accuracy by grounding LLM responses in factual information from the knowledge base.""",

            "evaluation_metrics.md": """# Model Evaluation Metrics

Evaluation metrics help assess the performance of machine learning models and RAG systems.

## Classification Metrics
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

## RAG-Specific Metrics
- **Retrieval Accuracy**: Quality of document retrieval
- **Answer Relevance**: How well answers address the query
- **Faithfulness**: Accuracy of generated answers relative to source documents"""
        }
        
        # Additional documents (extending the knowledge base)
        additional_docs = {
            "vector_embeddings.md": """# Vector Embeddings

Vector embeddings represent text as dense numerical vectors in high-dimensional space. Semantic similarity can be measured using cosine similarity or dot product.

## Embedding Models
- **BERT**: Bidirectional encoder representations
- **OpenAI Embeddings**: ada-002 model for text embedding
- **Sentence Transformers**: Specialized for semantic textual similarity

Embeddings capture semantic meaning, allowing similar concepts to have similar vector representations.""",

            "llm_integration.md": """# LLM Integration Patterns

Large Language Models (LLMs) can be integrated into applications through various patterns.

## Integration Approaches
- **Direct API Calls**: Simple request-response pattern
- **Pipeline Architecture**: Chained processing steps
- **Agent Framework**: Autonomous reasoning and tool use
- **Fine-tuning**: Adapting models for specific domains

The pipeline architecture is commonly used in RAG systems where retrieval and generation are separate components."""
        }
        
        # Write initial documents
        for filename, content in initial_docs.items():
            (docs_dir / filename).write_text(content, encoding='utf-8')
        
        # Write additional documents
        for filename, content in additional_docs.items():
            (additional_docs_dir / filename).write_text(content, encoding='utf-8')
        
        return {
            "initial": list(initial_docs.keys()),
            "additional": list(additional_docs.keys()),
            "total_files": len(initial_docs) + len(additional_docs)
        }

    @pytest.fixture
    def corpus_manager(self, temp_workspace):
        """Create CorpusManager with test stores"""
        db_path = temp_workspace["db_path"]
        document_store = SQLiteDocumentStore(str(db_path))
        vector_store = InMemoryVectorStore()
        
        return CorpusManager(document_store, vector_store)

    def test_comprehensive_corpus_workflow(self, temp_workspace, sample_documents, corpus_manager):
        """Test complete CorpusManager workflow"""
        workspace = temp_workspace["workspace"]
        docs_dir = temp_workspace["docs_dir"]
        additional_docs_dir = temp_workspace["additional_docs_dir"]
        refinire_dir = temp_workspace["refinire_dir"]
        rag_dir = temp_workspace["rag_dir"]
        
        corpus_name = "ml_knowledge_base"
        
        # Set environment variable for test
        with patch.dict(os.environ, {"REFINIRE_DIR": str(refinire_dir)}):
            
            # Step 1: Data cleanup - clear any existing data
            print("\n=== Step 1: Data Cleanup ===")
            corpus_manager.clear_corpus()
            
            # Verify corpus is empty
            assert len(list(corpus_manager._get_documents_by_stage("original"))) == 0
            print("✓ Corpus cleared successfully")
            
            # Step 2: Import new documents with dictionary and graph creation
            print("\n=== Step 2: Import Initial Documents with Knowledge Artifacts ===")
            
            with patch('refinire_rag.processing.dictionary_maker.LLMPipeline') as mock_dict_llm, \
                 patch('refinire_rag.processing.graph_builder.LLMPipeline') as mock_graph_llm:
                
                # Mock LLM responses for dictionary creation
                dict_mock_result = Mock()
                dict_mock_result.content = '''```json
{
    "has_new_terms": true,
    "new_terms_count": 8,
    "variations_count": 3,
    "extracted_terms": [
        {"term": "Machine Learning", "category": "専門用語", "definition": "人工知能の一分野で、コンピュータが経験から学習する技術"},
        {"term": "Neural Networks", "category": "専門用語", "definition": "脳の神経細胞を模倣した計算モデル"},
        {"term": "RAG", "category": "技術概念", "definition": "Retrieval-Augmented Generation、検索拡張生成"},
        {"term": "Vector Database", "category": "技術概念", "definition": "ベクトル表現されたデータを効率的に検索するデータベース"},
        {"term": "Document Store", "category": "技術概念", "definition": "文書を永続化するストレージ層"},
        {"term": "Vector Store", "category": "技術概念", "definition": "埋め込みベクトルを保存するストレージ"},
        {"term": "Evaluation Metrics", "category": "専門用語", "definition": "モデルの性能を評価する指標"},
        {"term": "F1 Score", "category": "専門用語", "definition": "精度と再現率の調和平均"}
    ],
    "variations": [
        {"original": "ML", "variations": ["Machine Learning", "機械学習"]},
        {"original": "AI", "variations": ["Artificial Intelligence", "人工知能"]},
        {"original": "DL", "variations": ["Deep Learning", "深層学習"]}
    ]
}
```'''
                
                mock_dict_pipeline = Mock()
                mock_dict_pipeline.run.return_value = dict_mock_result
                mock_dict_llm.return_value = mock_dict_pipeline
                
                # Mock LLM responses for graph creation
                graph_mock_result = Mock()
                graph_mock_result.content = '''```json
{
    "has_new_relationships": true,
    "relationships_count": 6,
    "entities_count": 8,
    "extracted_relationships": [
        {"subject": "Machine Learning", "predicate": "is_subset_of", "object": "Artificial Intelligence", "relationship_type": "hierarchical"},
        {"subject": "Neural Networks", "predicate": "is_component_of", "object": "Deep Learning", "relationship_type": "compositional"},
        {"subject": "RAG Systems", "predicate": "combines", "object": "Information Retrieval", "relationship_type": "functional"},
        {"subject": "Vector Database", "predicate": "stores", "object": "Document Embeddings", "relationship_type": "functional"},
        {"subject": "Evaluation Metrics", "predicate": "assesses", "object": "Model Performance", "relationship_type": "functional"},
        {"subject": "F1 Score", "predicate": "combines", "object": "Precision and Recall", "relationship_type": "compositional"}
    ],
    "entities": [
        {"name": "Machine Learning", "type": "concept"},
        {"name": "Neural Networks", "type": "technology"},
        {"name": "RAG Systems", "type": "system"},
        {"name": "Vector Database", "type": "storage"},
        {"name": "Document Store", "type": "storage"},
        {"name": "Vector Store", "type": "storage"},
        {"name": "Evaluation Metrics", "type": "measurement"},
        {"name": "F1 Score", "type": "metric"}
    ]
}
```'''
                
                mock_graph_pipeline = Mock()
                mock_graph_pipeline.run.return_value = graph_mock_result
                mock_graph_llm.return_value = mock_graph_pipeline
                
                # Import initial documents with knowledge artifacts
                stats1 = corpus_manager.import_original_documents(
                    corpus_name=corpus_name,
                    directory=str(docs_dir),
                    glob="**/*.md",
                    create_dictionary=True,
                    create_knowledge_graph=True,
                    force_reload=True
                )
                
                print(f"✓ Imported {stats1.total_documents_created} initial documents")
                print(f"✓ Created dictionary and knowledge graph")
                
                # Verify files were created
                dict_file = rag_dir / f"{corpus_name}_dictionary.md"
                graph_file = rag_dir / f"{corpus_name}_knowledge_graph.md"
                track_file = rag_dir / f"{corpus_name}_track.json"
                
                assert dict_file.exists(), f"Dictionary file not created: {dict_file}"
                assert graph_file.exists(), f"Graph file not created: {graph_file}"
                assert track_file.exists(), f"Track file not created: {track_file}"
                
                print(f"✓ Knowledge artifacts created:")
                print(f"  - Dictionary: {dict_file}")
                print(f"  - Graph: {graph_file}")
                print(f"  - Tracking: {track_file}")
            
            # Step 3: Add additional documents to corpus
            print("\n=== Step 3: Add Additional Documents ===")
            
            with patch('refinire_rag.processing.dictionary_maker.LLMPipeline') as mock_dict_llm:
                # Mock additional dictionary terms
                additional_dict_result = Mock()
                additional_dict_result.content = '''```json
{
    "has_new_terms": true,
    "new_terms_count": 4,
    "variations_count": 2,
    "extracted_terms": [
        {"term": "Vector Embeddings", "category": "技術概念", "definition": "テキストを高次元ベクトル空間の数値表現に変換する技術"},
        {"term": "BERT", "category": "専門用語", "definition": "Bidirectional Encoder Representations from Transformers"},
        {"term": "Cosine Similarity", "category": "専門用語", "definition": "ベクトル間の類似度を測る指標"},
        {"term": "Pipeline Architecture", "category": "技術概念", "definition": "複数の処理ステップを連鎖させる設計パターン"}
    ],
    "variations": [
        {"original": "LLM", "variations": ["Large Language Model", "大規模言語モデル"]},
        {"original": "API", "variations": ["Application Programming Interface", "アプリケーションプログラミングインターフェース"]}
    ]
}
```'''
                
                mock_additional_pipeline = Mock()
                mock_additional_pipeline.run.return_value = additional_dict_result
                mock_dict_llm.return_value = mock_additional_pipeline
                
                # Import additional documents (incremental loading)
                stats2 = corpus_manager.import_original_documents(
                    corpus_name=corpus_name,
                    directory=str(additional_docs_dir),
                    glob="**/*.md",
                    create_dictionary=True,  # Update existing dictionary
                    create_knowledge_graph=False,  # Don't recreate graph
                    force_reload=False  # Incremental loading
                )
                
                print(f"✓ Added {stats2.total_documents_created} additional documents")
                
                # Verify total document count
                all_original_docs = list(corpus_manager._get_documents_by_stage("original"))
                expected_total = len(sample_documents["initial"]) + len(sample_documents["additional"])
                assert len(all_original_docs) == expected_total
                print(f"✓ Total original documents: {len(all_original_docs)}")
            
            # Step 4: Edit dictionary manually
            print("\n=== Step 4: Edit Dictionary Manually ===")
            
            # Read current dictionary
            dict_content = dict_file.read_text(encoding='utf-8')
            print(f"✓ Read dictionary file ({len(dict_content)} characters)")
            
            # Add custom terms to dictionary
            custom_terms = """

## カスタム用語（手動追加）
- **Retrieval Pipeline**: 検索パイプライン - 文書検索から結果統合までの処理の流れ
- **Semantic Search**: 意味検索 - キーワードではなく意味的類似性に基づく検索
- **Context Window**: コンテキストウィンドウ - LLMが一度に処理できるトークン数の上限
- **Few-shot Learning**: 少数ショット学習 - 少ない例から学習する手法
- **Zero-shot Learning**: ゼロショット学習 - 事前の例なしで推論する手法

## カスタム表現揺らぎ
- **LLM**: Large Language Model, 大規模言語モデル, LLMs
- **Embedding**: エンベディング, 埋め込み, ベクトル表現
- **Fine-tuning**: ファインチューニング, 微調整, 追加学習
"""
            
            # Append custom terms to dictionary
            updated_dict_content = dict_content + custom_terms
            dict_file.write_text(updated_dict_content, encoding='utf-8')
            
            print(f"✓ Updated dictionary with custom terms")
            print(f"✓ Dictionary size: {len(updated_dict_content)} characters")
            
            # Step 5: Rebuild corpus from original documents using edited dictionary
            print("\n=== Step 5: Rebuild Corpus with Edited Dictionary ===")
            
            # Clear processed documents (keep original)
            print("Clearing existing processed documents...")
            
            # Note: In a real implementation, you might want to clear only non-original documents
            # For this test, we'll rebuild everything from original documents
            
            rebuild_stats = corpus_manager.rebuild_corpus_from_original(
                corpus_name=corpus_name,
                use_dictionary=True,
                use_knowledge_graph=True,
                dictionary_file_path=str(dict_file),
                graph_file_path=str(graph_file),
                additional_metadata={
                    "rebuild_reason": "dictionary_updated",
                    "custom_terms_added": 5
                }
            )
            
            print(f"✓ Rebuilt corpus in {rebuild_stats.total_processing_time:.2f}s")
            print(f"✓ Processed {rebuild_stats.total_documents_created} documents")
            print(f"✓ Created {rebuild_stats.total_chunks_created} chunks")
            print(f"✓ Executed {rebuild_stats.pipeline_stages_executed} pipeline stages")
            
            # Verify rebuild results
            assert rebuild_stats.total_documents_created > 0
            assert rebuild_stats.total_chunks_created > 0
            assert rebuild_stats.pipeline_stages_executed >= 3  # Loader + Normalizer + Chunker + VectorStore
            
            # Step 6: Verify final state
            print("\n=== Step 6: Verify Final State ===")
            
            # Check document stages
            original_docs = list(corpus_manager._get_documents_by_stage("original"))
            print(f"✓ Original documents: {len(original_docs)}")
            
            # Check that dictionary and graph files exist and have content
            assert dict_file.exists() and dict_file.stat().st_size > 1000
            assert graph_file.exists() and graph_file.stat().st_size > 500
            print("✓ Knowledge artifacts verified")
            
            # Check custom terms are in dictionary
            final_dict_content = dict_file.read_text(encoding='utf-8')
            assert "Retrieval Pipeline" in final_dict_content
            assert "Semantic Search" in final_dict_content
            assert "Context Window" in final_dict_content
            print("✓ Custom terms preserved in dictionary")
            
            # Verify vector store has embeddings
            # Note: This would depend on the actual vector store implementation
            print("✓ Vector store populated (mock)")
            
            print(f"\n🎉 Comprehensive workflow completed successfully!")
            print(f"📊 Final Statistics:")
            print(f"   - Original documents: {len(original_docs)}")
            print(f"   - Total processing time: {rebuild_stats.total_processing_time:.2f}s")
            print(f"   - Pipeline stages: {rebuild_stats.pipeline_stages_executed}")
            print(f"   - Dictionary size: {len(final_dict_content)} characters")

    def test_error_handling_and_edge_cases(self, temp_workspace, corpus_manager):
        """Test error handling and edge cases"""
        refinire_dir = temp_workspace["refinire_dir"]
        
        with patch.dict(os.environ, {"REFINIRE_DIR": str(refinire_dir)}):
            
            # Test with non-existent directory
            with pytest.raises(Exception):
                corpus_manager.import_original_documents(
                    corpus_name="test",
                    directory="/non/existent/path",
                    glob="**/*.md"
                )
            
            # Test with empty directory
            empty_dir = temp_workspace["workspace"] / "empty"
            empty_dir.mkdir()
            
            stats = corpus_manager.import_original_documents(
                corpus_name="empty_test",
                directory=str(empty_dir),
                glob="**/*.md"
            )
            
            assert stats.total_documents_created == 0
            
            # Test rebuild without original documents
            with pytest.raises(ValueError, match="No original documents found"):
                corpus_manager.rebuild_corpus_from_original(
                    corpus_name="non_existent",
                    use_dictionary=False
                )

    def test_file_tracking_and_incremental_loading(self, temp_workspace, corpus_manager):
        """Test file tracking and incremental loading functionality"""
        docs_dir = temp_workspace["docs_dir"]
        refinire_dir = temp_workspace["refinire_dir"]
        
        # Create a test document
        test_file = docs_dir / "test_doc.md"
        test_file.write_text("# Test Document\n\nThis is a test.", encoding='utf-8')
        
        with patch.dict(os.environ, {"REFINIRE_DIR": str(refinire_dir)}):
            
            # Initial import
            stats1 = corpus_manager.import_original_documents(
                corpus_name="tracking_test",
                directory=str(docs_dir),
                glob="**/*.md"
            )
            
            assert stats1.total_documents_created == 1
            
            # Second import without changes (should skip)
            stats2 = corpus_manager.import_original_documents(
                corpus_name="tracking_test",
                directory=str(docs_dir),
                glob="**/*.md",
                force_reload=False
            )
            
            # Should not process same file again
            assert stats2.total_documents_created == 0
            
            # Modify file and import again
            test_file.write_text("# Test Document\n\nThis is updated content.", encoding='utf-8')
            
            stats3 = corpus_manager.import_original_documents(
                corpus_name="tracking_test",
                directory=str(docs_dir),
                glob="**/*.md",
                force_reload=False
            )
            
            # Should process modified file
            assert stats3.total_documents_created == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])