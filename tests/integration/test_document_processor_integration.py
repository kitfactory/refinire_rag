"""
Integration tests for DocumentProcessor classes

Tests the unified DocumentProcessor architecture and pipeline functionality
across all processor implementations.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List
import sys
import os

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from refinire_rag.models.document import Document
from refinire_rag.processing.document_processor import DocumentProcessor, DocumentProcessorConfig
from refinire_rag.processing.document_pipeline import DocumentPipeline
from refinire_rag.processing.dictionary_maker import DictionaryMaker, DictionaryMakerConfig
from refinire_rag.processing.normalizer import Normalizer, NormalizerConfig
from refinire_rag.processing.graph_builder import GraphBuilder, GraphBuilderConfig
from refinire_rag.processing.chunker import Chunker, ChunkingConfig
from refinire_rag.loaders.base import Loader, LoaderConfig


class TestDocumentProcessorBase:
    """Test base DocumentProcessor functionality"""
    
    def test_config_class_method(self):
        """Test that all processors implement get_config_class method"""
        processors = [
            DictionaryMaker,
            Normalizer, 
            GraphBuilder,
            Chunker
        ]
        
        for processor_class in processors:
            # Test get_config_class method exists and returns correct type
            config_class = processor_class.get_config_class()
            assert issubclass(config_class, DocumentProcessorConfig)
            
            # Test config can be instantiated
            config = config_class()
            assert isinstance(config, DocumentProcessorConfig)
    
    def test_processor_initialization(self):
        """Test processor initialization with and without config"""
        # Test with default config
        dict_maker = DictionaryMaker()
        assert isinstance(dict_maker.config, DictionaryMakerConfig)
        
        # Test with custom config
        custom_config = DictionaryMakerConfig(
            dictionary_file_path="./custom_dict.md",
            llm_temperature=0.5
        )
        dict_maker_custom = DictionaryMaker(custom_config)
        assert dict_maker_custom.config.dictionary_file_path == "./custom_dict.md"
        assert dict_maker_custom.config.llm_temperature == 0.5
    
    def test_processing_stats(self):
        """Test processing statistics functionality"""
        dict_maker = DictionaryMaker()
        
        # Initial stats should be available
        stats = dict_maker.get_processing_stats()
        assert isinstance(stats, dict)
        assert "documents_processed" in stats
        assert stats["documents_processed"] == 0


class TestDocumentProcessorIntegration:
    """Test integration between different DocumentProcessor implementations"""
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing"""
        return Document(
            id="test_doc_001",
            content="""
            RAGシステム（Retrieval-Augmented Generation）は、検索拡張生成の手法です。
            この技術は、ベクトル検索とLLMを組み合わせて、より正確な回答を生成します。
            主要な構成要素として、文書埋め込み、ベクトルデータベース、検索機能があります。
            評価では精度（Precision）と再現率（Recall）が重要な指標となります。
            """,
            metadata={
                "title": "RAGシステムの概要",
                "source": "test_document",
                "domain": "AI技術"
            }
        )
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_dictionary_maker_integration(self, sample_document, temp_dir):
        """Test DictionaryMaker integration"""
        dict_path = Path(temp_dir) / "test_dictionary.md"
        
        config = DictionaryMakerConfig(
            dictionary_file_path=str(dict_path),
            focus_on_technical_terms=True,
            extract_abbreviations=True
        )
        
        dict_maker = DictionaryMaker(config)
        
        # Process document
        result_docs = dict_maker.process(sample_document)
        
        # Verify results
        assert len(result_docs) == 1
        result_doc = result_docs[0]
        
        # Check metadata enrichment
        assert "dictionary_metadata" in result_doc.metadata
        dict_metadata = result_doc.metadata["dictionary_metadata"]
        assert "dictionary_extraction_applied" in dict_metadata
        assert dict_metadata["dictionary_extraction_applied"] is True
        
        # Check dictionary file creation
        assert dict_path.exists()
        content = dict_path.read_text(encoding='utf-8')
        assert "# ドメイン用語辞書" in content
        
        # Check processing stats
        stats = dict_maker.get_extraction_stats()
        assert stats["documents_processed"] == 1
    
    def test_normalizer_integration(self, sample_document, temp_dir):
        """Test Normalizer integration with dictionary"""
        dict_path = Path(temp_dir) / "test_dictionary.md"
        
        # Create a sample dictionary
        dict_content = """# ドメイン用語辞書

## 専門用語
- **RAG** (Retrieval-Augmented Generation): 検索拡張生成
  - 表現揺らぎ: 検索拡張生成, 検索強化生成, RAGシステム

## 技術概念  
- **ベクトル検索**: 類似度に基づく文書検索手法
  - 表現揺らぎ: セマンティック検索, 埋め込み検索, 意味検索
"""
        dict_path.write_text(dict_content, encoding='utf-8')
        
        # Test normalizer
        config = NormalizerConfig(
            dictionary_file_path=str(dict_path),
            normalize_variations=True,
            expand_abbreviations=True
        )
        
        normalizer = Normalizer(config)
        result_docs = normalizer.process(sample_document)
        
        # Verify results
        assert len(result_docs) == 1
        result_doc = result_docs[0]
        
        # Check normalization metadata
        norm_stats = result_doc.metadata.get("normalization_stats", {})
        assert "total_replacements" in norm_stats
        
        # Check processing stats  
        stats = normalizer.get_normalization_stats()
        assert stats["documents_processed"] == 1
    
    def test_graph_builder_integration(self, sample_document, temp_dir):
        """Test GraphBuilder integration"""
        graph_path = Path(temp_dir) / "test_graph.md"
        dict_path = Path(temp_dir) / "test_dictionary.md"
        
        # Create sample dictionary
        dict_content = """# ドメイン用語辞書
## 専門用語
- **RAG** (Retrieval-Augmented Generation): 検索拡張生成
"""
        dict_path.write_text(dict_content, encoding='utf-8')
        
        config = GraphBuilderConfig(
            graph_file_path=str(graph_path),
            dictionary_file_path=str(dict_path),
            focus_on_important_relationships=True
        )
        
        graph_builder = GraphBuilder(config)
        result_docs = graph_builder.process(sample_document)
        
        # Verify results
        assert len(result_docs) == 1
        result_doc = result_docs[0]
        
        # Check graph metadata
        assert "graph_metadata" in result_doc.metadata
        graph_metadata = result_doc.metadata["graph_metadata"]
        assert "graph_extraction_applied" in graph_metadata
        
        # Check graph file creation
        assert graph_path.exists()
        content = graph_path.read_text(encoding='utf-8')
        assert "# ドメイン知識グラフ" in content
        
        # Check processing stats
        stats = graph_builder.get_graph_stats()
        assert stats["documents_processed"] == 1
    
    def test_chunker_integration(self, sample_document):
        """Test Chunker integration"""
        config = ChunkingConfig(
            chunk_size=100,  # Small size for testing
            overlap=20
        )
        
        chunker = Chunker(config)
        result_docs = chunker.process(sample_document)
        
        # Verify chunking
        assert len(result_docs) >= 1  # Should produce at least one chunk
        
        for i, chunk_doc in enumerate(result_docs):
            # Check chunk metadata
            assert chunk_doc.metadata["processing_stage"] == "chunked"
            assert chunk_doc.metadata["parent_document_id"] == sample_document.id
            assert chunk_doc.metadata["original_document_id"] == sample_document.id
            assert chunk_doc.metadata["chunk_position"] == i
            assert chunk_doc.metadata["chunk_total"] == len(result_docs)
        
        # Check processing stats
        stats = chunker.get_processing_stats()
        assert stats["documents_processed"] == 1
        assert stats["chunks_created"] == len(result_docs)


class TestDocumentPipelineIntegration:
    """Test DocumentPipeline with multiple processors"""
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample document for pipeline testing"""
        return Document(
            id="pipeline_test_001",
            content="RAGシステムはAI技術の一つで、検索と生成を組み合わせます。",
            metadata={"title": "パイプラインテスト", "source": "test"}
        )
    
    @pytest.fixture  
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_sequential_processing_pipeline(self, sample_document, temp_dir):
        """Test sequential processing through multiple processors"""
        dict_path = Path(temp_dir) / "pipeline_dict.md"
        graph_path = Path(temp_dir) / "pipeline_graph.md"
        
        # Create pipeline with multiple processors
        pipeline = DocumentPipeline([
            DictionaryMaker(DictionaryMakerConfig(
                dictionary_file_path=str(dict_path),
                focus_on_technical_terms=True
            )),
            Normalizer(NormalizerConfig(
                dictionary_file_path=str(dict_path),
                normalize_variations=True
            )),
            GraphBuilder(GraphBuilderConfig(
                graph_file_path=str(graph_path),
                dictionary_file_path=str(dict_path)
            )),
            Chunker(ChunkingConfig(
                chunk_size=50,
                overlap=10
            ))
        ])
        
        # Process document through pipeline
        result_docs = pipeline.process_document(sample_document)
        
        # Verify pipeline results
        assert len(result_docs) >= 1  # Should have chunks
        
        # Check that files were created
        assert dict_path.exists()
        assert graph_path.exists()
        
        # Verify final documents have all processing stages
        for chunk_doc in result_docs:
            metadata = chunk_doc.metadata
            assert metadata["processing_stage"] == "chunked"
            assert metadata["original_document_id"] == sample_document.id
        
        # Check pipeline statistics
        stats = pipeline.get_pipeline_stats()
        assert "total_documents_processed" in stats
        assert "processors_executed" in stats
        assert len(stats["processors_executed"]) == 4
    
    def test_pipeline_error_handling(self, sample_document):
        """Test pipeline error handling with invalid config"""
        # Create pipeline with invalid path (should handle gracefully)
        pipeline = DocumentPipeline([
            DictionaryMaker(DictionaryMakerConfig(
                dictionary_file_path="/invalid/path/dict.md"
            )),
            Chunker(ChunkingConfig(chunk_size=50))
        ])
        
        # Process should not fail completely
        result_docs = pipeline.process_document(sample_document)
        
        # Should still get chunked results even if dictionary fails
        assert len(result_docs) >= 1
        
        # Check that some processing occurred
        stats = pipeline.get_pipeline_stats()
        assert stats["total_documents_processed"] == 1


@pytest.mark.integration
class TestEndToEndProcessing:
    """End-to-end integration tests"""
    
    def test_complete_document_processing_workflow(self, tmp_path):
        """Test complete workflow from document to chunks"""
        # Create test document
        test_doc = Document(
            id="e2e_test_001",
            content="""
            RAG（Retrieval-Augmented Generation）システムは、検索拡張生成技術です。
            この技術では、ベクトル検索を使用して関連文書を検索し、
            その情報を基にLLMで回答を生成します。
            評価指標として精度（Precision）、再現率（Recall）、F1スコアが使用されます。
            NLI（Natural Language Inference）による矛盾検出も重要な機能です。
            """,
            metadata={
                "title": "RAGシステム詳細",
                "author": "テスト作成者",
                "domain": "機械学習"
            }
        )
        
        dict_path = tmp_path / "e2e_dictionary.md"
        graph_path = tmp_path / "e2e_graph.md"
        
        # Create complete processing pipeline
        processors = [
            DictionaryMaker(DictionaryMakerConfig(
                dictionary_file_path=str(dict_path),
                focus_on_technical_terms=True,
                extract_abbreviations=True
            )),
            Normalizer(NormalizerConfig(
                dictionary_file_path=str(dict_path),
                normalize_variations=True,
                expand_abbreviations=True
            )),
            GraphBuilder(GraphBuilderConfig(
                graph_file_path=str(graph_path),
                dictionary_file_path=str(dict_path),
                focus_on_important_relationships=True
            )),
            Chunker(ChunkingConfig(
                chunk_size=128,
                overlap=32
            ))
        ]
        
        pipeline = DocumentPipeline(processors)
        
        # Process document
        final_chunks = pipeline.process_document(test_doc)
        
        # Verify end-to-end results
        assert len(final_chunks) >= 1
        
        # Check file creation
        assert dict_path.exists()
        assert graph_path.exists()
        
        # Verify dictionary content
        dict_content = dict_path.read_text(encoding='utf-8')
        assert "RAG" in dict_content or "ドメイン用語辞書" in dict_content
        
        # Verify graph content
        graph_content = graph_path.read_text(encoding='utf-8')
        assert "知識グラフ" in graph_content or "関係" in graph_content
        
        # Verify final chunks have complete lineage
        for chunk in final_chunks:
            metadata = chunk.metadata
            assert metadata["original_document_id"] == test_doc.id
            assert metadata["processing_stage"] == "chunked"
            assert "chunk_position" in metadata
            assert "chunk_total" in metadata
        
        # Verify pipeline statistics
        stats = pipeline.get_pipeline_stats()
        assert stats["total_documents_processed"] == 1
        assert len(stats["processors_executed"]) == 4
        assert stats["total_processing_time"] > 0
        
        print(f"✅ End-to-end test completed successfully!")
        print(f"   - Input document: {test_doc.id}")
        print(f"   - Final chunks: {len(final_chunks)}")
        print(f"   - Dictionary file: {dict_path.exists()}")
        print(f"   - Graph file: {graph_path.exists()}")
        print(f"   - Processing time: {stats['total_processing_time']:.3f}s")


if __name__ == "__main__":
    # Run tests directly for development
    pytest.main([__file__, "-v", "--tb=short"])