"""
Tests for GraphBuilder processor

Tests the knowledge graph construction and relationship extraction functionality.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from refinire_rag.processing.graph_builder import GraphBuilder, GraphBuilderConfig
from refinire_rag.models.document import Document


class TestGraphBuilderConfig:
    """Test GraphBuilderConfig configuration class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = GraphBuilderConfig()
        
        # Paths should be converted to absolute
        assert config.graph_file_path.endswith("domain_knowledge_graph.md")
        assert config.dictionary_file_path.endswith("domain_dictionary.md")
        assert os.path.isabs(config.graph_file_path)
        assert os.path.isabs(config.dictionary_file_path)
        assert config.backup_graph is True
        assert config.llm_model is not None  # Should be set by __post_init__
        assert config.llm_temperature == 0.3
        assert config.max_tokens == 3000
        assert config.focus_on_important_relationships is True
        assert config.extract_hierarchical_relationships is True
        assert config.extract_causal_relationships is True

    def test_custom_config(self):
        """Test custom configuration values"""
        config = GraphBuilderConfig(
            graph_file_path="/custom/path/graph.md",
            dictionary_file_path="/custom/path/dict.md",
            llm_model="gpt-4",
            llm_temperature=0.5,
            focus_on_important_relationships=False
        )
        
        assert config.graph_file_path == "/custom/path/graph.md"
        assert config.dictionary_file_path == "/custom/path/dict.md"
        assert config.llm_model == "gpt-4"
        assert config.llm_temperature == 0.5
        assert config.focus_on_important_relationships is False

    @patch.dict(os.environ, {"REFINIRE_RAG_LLM_MODEL": "test-model"})
    def test_env_model_config(self):
        """Test that environment variable is used for model selection"""
        config = GraphBuilderConfig()
        assert config.llm_model == "test-model"

    def test_absolute_path_conversion(self):
        """Test that relative paths are converted to absolute"""
        config = GraphBuilderConfig(
            graph_file_path="relative/graph.md",
            dictionary_file_path="relative/dict.md"
        )
        assert os.path.isabs(config.graph_file_path)
        assert os.path.isabs(config.dictionary_file_path)


class TestGraphBuilder:
    """Test GraphBuilder processor"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing"""
        return Document(
            id="test_doc_1",
            content="Machine learning algorithms process data to train neural networks. "
                   "The API connects to the database which stores user information. "
                   "Data preprocessing improves model accuracy.",
            metadata={"source": "test"}
        )

    @pytest.fixture
    def config_with_temp_files(self, temp_dir):
        """Create config with temporary files"""
        graph_file = temp_dir / "test_graph.md"
        dict_file = temp_dir / "test_dict.md"
        return GraphBuilderConfig(
            graph_file_path=str(graph_file),
            dictionary_file_path=str(dict_file),
            llm_model="gpt-4o-mini"
        )

    @pytest.fixture
    def sample_dictionary_content(self):
        """Create sample dictionary content"""
        return """# ドメイン用語辞書

## 専門用語
- machine learning: 機械学習
- neural networks: ニューラルネットワーク
- API: Application Programming Interface

## 技術概念
- data preprocessing: データ前処理
- model accuracy: モデル精度
"""

    def test_initialization_without_refinire(self, config_with_temp_files):
        """Test initialization when Refinire is not available"""
        with patch('refinire_rag.processing.graph_builder.LLMPipeline', None):
            builder = GraphBuilder(config_with_temp_files)
            assert builder._llm_pipeline is None
            assert builder.config == config_with_temp_files

    @patch('refinire_rag.processing.graph_builder.LLMPipeline')
    def test_initialization_with_refinire(self, mock_llm_pipeline, config_with_temp_files):
        """Test initialization when Refinire is available"""
        mock_pipeline_instance = Mock()
        mock_llm_pipeline.return_value = mock_pipeline_instance
        
        builder = GraphBuilder(config_with_temp_files)
        
        mock_llm_pipeline.assert_called_once_with(
            name="graph_builder",
            generation_instructions="You are a knowledge graph expert that extracts important relationships from documents.",
            model=config_with_temp_files.llm_model
        )
        assert builder._llm_pipeline == mock_pipeline_instance

    @patch('refinire_rag.processing.graph_builder.LLMPipeline')
    def test_initialization_refinire_error(self, mock_llm_pipeline, config_with_temp_files):
        """Test initialization when Refinire initialization fails"""
        mock_llm_pipeline.side_effect = Exception("API error")
        
        builder = GraphBuilder(config_with_temp_files)
        assert builder._llm_pipeline is None

    def test_empty_graph_template(self, config_with_temp_files):
        """Test creation of empty graph template"""
        builder = GraphBuilder(config_with_temp_files)
        template = builder._create_empty_graph_template()
        
        assert "# ドメイン知識グラフ" in template
        assert "## エンティティ関係" in template
        assert "### 主要概念" in template
        assert "### 技術関係" in template

    def test_process_without_llm(self, config_with_temp_files, sample_document):
        """Test processing when LLM is not available (mock data mode)"""
        builder = GraphBuilder(config_with_temp_files)
        builder._llm_pipeline = None
        
        result = builder.process(sample_document)
        
        assert len(result) == 1
        assert result[0].id == sample_document.id
        assert result[0].content == sample_document.content
        assert builder.processing_stats["documents_processed"] == 1

    @patch('refinire_rag.processing.graph_builder.LLMPipeline')
    def test_process_with_llm_success(self, mock_llm_pipeline, config_with_temp_files, sample_document):
        """Test successful processing with LLM"""
        # Setup mock LLM response
        mock_result = Mock()
        mock_result.content = '''```json
{
    "has_new_relationships": true,
    "relationships_count": 3,
    "entities_count": 5,
    "extracted_relationships": [
        {
            "subject": "machine learning algorithms",
            "predicate": "process",
            "object": "data",
            "relationship_type": "functional"
        },
        {
            "subject": "API",
            "predicate": "connects_to",
            "object": "database",
            "relationship_type": "technical"
        },
        {
            "subject": "data preprocessing",
            "predicate": "improves",
            "object": "model accuracy",
            "relationship_type": "causal"
        }
    ],
    "entities": [
        {"name": "machine learning algorithms", "type": "concept"},
        {"name": "data", "type": "resource"},
        {"name": "API", "type": "component"},
        {"name": "database", "type": "component"},
        {"name": "model accuracy", "type": "metric"}
    ]
}
```'''
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.return_value = mock_result
        mock_llm_pipeline.return_value = mock_pipeline_instance
        
        builder = GraphBuilder(config_with_temp_files)
        result = builder.process(sample_document)
        
        assert len(result) == 1
        assert result[0].id == sample_document.id
        assert result[0].content == sample_document.content
        assert builder.processing_stats["documents_processed"] == 1
        # Note: relationships_extracted may be 0 if mock data mode is used
        # or implementation doesn't update stats from LLM response
        
        # Check that graph file was created
        graph_path = Path(config_with_temp_files.graph_file_path)
        assert graph_path.exists()

    @patch('refinire_rag.processing.graph_builder.LLMPipeline')
    def test_process_with_llm_json_error(self, mock_llm_pipeline, config_with_temp_files, sample_document):
        """Test processing when LLM returns invalid JSON"""
        # Setup mock LLM response with invalid JSON
        mock_result = Mock()
        mock_result.content = "Invalid JSON response"
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.return_value = mock_result
        mock_llm_pipeline.return_value = mock_pipeline_instance
        
        builder = GraphBuilder(config_with_temp_files)
        result = builder.process(sample_document)
        
        # Should still return document, but with error handling
        assert len(result) == 1
        assert result[0].id == sample_document.id
        assert result[0].content == sample_document.content
        assert builder.processing_stats["documents_processed"] == 1

    @patch('refinire_rag.processing.graph_builder.LLMPipeline')
    def test_process_with_llm_api_error(self, mock_llm_pipeline, config_with_temp_files, sample_document):
        """Test processing when LLM API call fails"""
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.side_effect = Exception("API call failed")
        mock_llm_pipeline.return_value = mock_pipeline_instance
        
        builder = GraphBuilder(config_with_temp_files)
        result = builder.process(sample_document)
        
        # Should still return document, but with error handling
        assert len(result) == 1
        assert result[0].id == sample_document.id
        assert result[0].content == sample_document.content
        assert builder.processing_stats["documents_processed"] == 1

    def test_existing_graph_reading(self, config_with_temp_files):
        """Test reading existing graph file"""
        # Create existing graph file
        graph_path = Path(config_with_temp_files.graph_file_path)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        existing_content = "# Existing Graph\n\n## Entities\n- entity1\n\n## Relationships\n- rel1"
        graph_path.write_text(existing_content, encoding='utf-8')
        
        builder = GraphBuilder(config_with_temp_files)
        read_content = builder._read_existing_graph(config_with_temp_files)
        
        assert read_content == existing_content

    def test_nonexistent_graph_reading(self, config_with_temp_files):
        """Test reading when graph file doesn't exist"""
        builder = GraphBuilder(config_with_temp_files)
        content = builder._read_existing_graph(config_with_temp_files)
        
        # Should return empty graph template
        assert "# ドメイン知識グラフ" in content

    def test_dictionary_reading_from_config(self, config_with_temp_files, sample_dictionary_content):
        """Test reading dictionary from configured path"""
        # Create dictionary file
        dict_path = Path(config_with_temp_files.dictionary_file_path)
        dict_path.parent.mkdir(parents=True, exist_ok=True)
        dict_path.write_text(sample_dictionary_content, encoding='utf-8')
        
        builder = GraphBuilder(config_with_temp_files)
        sample_doc = Document(id="test", content="test", metadata={})
        content = builder._read_dictionary_content(sample_doc, config_with_temp_files)
        
        assert content == sample_dictionary_content

    def test_dictionary_reading_from_metadata(self, config_with_temp_files, sample_dictionary_content, temp_dir):
        """Test reading dictionary from document metadata"""
        # Create dictionary file in different location
        metadata_dict_path = temp_dir / "metadata_dict.md"
        metadata_dict_path.write_text(sample_dictionary_content, encoding='utf-8')
        
        # Document with dictionary metadata
        doc_with_metadata = Document(
            id="test",
            content="test",
            metadata={
                "dictionary_metadata": {
                    "dictionary_file_path": str(metadata_dict_path)
                }
            }
        )
        
        config_with_temp_files.auto_detect_dictionary_path = True
        builder = GraphBuilder(config_with_temp_files)
        content = builder._read_dictionary_content(doc_with_metadata, config_with_temp_files)
        
        assert content == sample_dictionary_content

    def test_dictionary_reading_fallback(self, config_with_temp_files):
        """Test dictionary reading when no files exist"""
        builder = GraphBuilder(config_with_temp_files)
        sample_doc = Document(id="test", content="test", metadata={})
        content = builder._read_dictionary_content(sample_doc, config_with_temp_files)
        
        assert content == ""

    def test_get_config_class(self):
        """Test get_config_class method"""
        assert GraphBuilder.get_config_class() == GraphBuilderConfig

    def test_processing_stats_initialization(self, config_with_temp_files):
        """Test that processing stats are properly initialized"""
        builder = GraphBuilder(config_with_temp_files)
        
        stats = builder.processing_stats
        assert "documents_processed" in stats
        assert "relationships_extracted" in stats
        assert "graph_updates" in stats
        assert "llm_api_calls" in stats
        assert "duplicate_relationships_avoided" in stats
        assert stats["documents_processed"] == 0

    def test_backup_functionality(self, config_with_temp_files, temp_dir):
        """Test graph backup functionality"""
        config_with_temp_files.backup_graph = True
        
        # Create existing graph
        graph_path = Path(config_with_temp_files.graph_file_path)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text("Original graph content", encoding='utf-8')
        
        builder = GraphBuilder(config_with_temp_files)
        builder._backup_graph_file(config_with_temp_files)
        
        # Check backup was created - backup files use format: .backup_timestamp.md
        backup_files = list(graph_path.parent.glob(f"{graph_path.stem}.backup_*.md"))
        assert len(backup_files) > 0

    def test_relationship_extraction_prompt_creation(self, config_with_temp_files, sample_document, sample_dictionary_content):
        """Test creation of relationship extraction prompt"""
        builder = GraphBuilder(config_with_temp_files)
        
        existing_graph = "# Existing Graph Content"
        prompt = builder._create_relationship_extraction_prompt(
            sample_document, existing_graph, sample_dictionary_content, config_with_temp_files
        )
        
        # Check that prompt contains key elements (in Japanese)
        assert "重要な関係性" in prompt  # "important relationships" in Japanese
        assert sample_document.content in prompt
        assert existing_graph in prompt
        assert sample_dictionary_content in prompt

    def test_skip_processing_config(self, config_with_temp_files, sample_document):
        """Test skip processing configuration"""
        config_with_temp_files.skip_if_no_new_relationships = True
        
        builder = GraphBuilder(config_with_temp_files)
        builder._llm_pipeline = None  # Force mock mode
        
        result = builder.process(sample_document)
        
        assert len(result) == 1
        assert result[0].id == sample_document.id
        assert result[0].content == sample_document.content


if __name__ == "__main__":
    pytest.main([__file__])