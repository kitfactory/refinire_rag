"""
Tests for DictionaryMaker processor

Tests the domain-specific term extraction and dictionary update functionality.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from refinire_rag.processing.dictionary_maker import DictionaryMaker, DictionaryMakerConfig
from refinire_rag.models.document import Document


class TestDictionaryMakerConfig:
    """Test DictionaryMakerConfig configuration class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = DictionaryMakerConfig()
        
        # Path should be converted to absolute
        assert config.dictionary_file_path.endswith("domain_dictionary.md")
        assert os.path.isabs(config.dictionary_file_path)
        assert config.backup_dictionary is True
        assert config.llm_model is not None  # Should be set by __post_init__
        assert config.llm_temperature == 0.3
        assert config.max_tokens == 2000
        assert config.focus_on_technical_terms is True
        assert config.extract_abbreviations is True
        assert config.detect_expression_variations is True

    def test_custom_config(self):
        """Test custom configuration values"""
        config = DictionaryMakerConfig(
            dictionary_file_path="/custom/path/dict.md",
            llm_model="gpt-4",
            llm_temperature=0.5,
            focus_on_technical_terms=False
        )
        
        assert config.dictionary_file_path == "/custom/path/dict.md"
        assert config.llm_model == "gpt-4"
        assert config.llm_temperature == 0.5
        assert config.focus_on_technical_terms is False

    @patch.dict(os.environ, {"REFINIRE_RAG_LLM_MODEL": "test-model"})
    def test_env_model_config(self):
        """Test that environment variable is used for model selection"""
        config = DictionaryMakerConfig()
        assert config.llm_model == "test-model"

    def test_absolute_path_conversion(self):
        """Test that relative paths are converted to absolute"""
        config = DictionaryMakerConfig(dictionary_file_path="relative/path.md")
        assert os.path.isabs(config.dictionary_file_path)


class TestDictionaryMaker:
    """Test DictionaryMaker processor"""

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
            content="This document discusses machine learning algorithms and neural networks. "
                   "The API provides REST endpoints for data processing.",
            metadata={"source": "test"}
        )

    @pytest.fixture
    def config_with_temp_file(self, temp_dir):
        """Create config with temporary dictionary file"""
        dict_file = temp_dir / "test_dictionary.md"
        return DictionaryMakerConfig(
            dictionary_file_path=str(dict_file),
            llm_model="gpt-4o-mini"
        )

    def test_initialization_without_refinire(self, config_with_temp_file):
        """Test initialization when Refinire is not available"""
        with patch('refinire_rag.processing.dictionary_maker.LLMPipeline', None):
            maker = DictionaryMaker(config_with_temp_file)
            assert maker._llm_pipeline is None
            assert maker.config == config_with_temp_file

    @patch('refinire_rag.processing.dictionary_maker.LLMPipeline')
    def test_initialization_with_refinire(self, mock_llm_pipeline, config_with_temp_file):
        """Test initialization when Refinire is available"""
        mock_pipeline_instance = Mock()
        mock_llm_pipeline.return_value = mock_pipeline_instance
        
        maker = DictionaryMaker(config_with_temp_file)
        
        mock_llm_pipeline.assert_called_once_with(
            name="dictionary_maker",
            generation_instructions="You are a domain expert that extracts technical terms and their variations from documents.",
            model=config_with_temp_file.llm_model
        )
        assert maker._llm_pipeline == mock_pipeline_instance

    @patch('refinire_rag.processing.dictionary_maker.LLMPipeline')
    def test_initialization_refinire_error(self, mock_llm_pipeline, config_with_temp_file):
        """Test initialization when Refinire initialization fails"""
        mock_llm_pipeline.side_effect = Exception("API error")
        
        maker = DictionaryMaker(config_with_temp_file)
        assert maker._llm_pipeline is None

    def test_empty_dictionary_template(self, config_with_temp_file):
        """Test creation of empty dictionary template"""
        maker = DictionaryMaker(config_with_temp_file)
        template = maker._create_empty_dictionary_template()
        
        assert "# ドメイン用語辞書" in template
        assert "## 専門用語" in template
        assert "## 技術概念" in template
        assert "## 表現揺らぎ" in template

    def test_process_without_llm(self, config_with_temp_file, sample_document):
        """Test processing when LLM is not available (mock data mode)"""
        maker = DictionaryMaker(config_with_temp_file)
        maker._llm_pipeline = None
        
        result = maker.process(sample_document)
        
        assert len(result) == 1
        assert result[0].id == sample_document.id
        assert result[0].content == sample_document.content
        assert maker.processing_stats["documents_processed"] == 1

    @patch('refinire_rag.processing.dictionary_maker.LLMPipeline')
    def test_process_with_llm_success(self, mock_llm_pipeline, config_with_temp_file, sample_document):
        """Test successful processing with LLM"""
        # Setup mock LLM response
        mock_result = Mock()
        mock_result.content = '''```json
{
    "has_new_terms": true,
    "new_terms_count": 3,
    "variations_count": 1,
    "extracted_terms": [
        {"term": "machine learning", "category": "専門用語", "importance": "high"},
        {"term": "neural networks", "category": "専門用語", "importance": "high"},
        {"term": "API", "category": "技術概念", "importance": "medium"}
    ],
    "variations": [
        {"original": "REST endpoints", "variations": ["RESTful endpoints", "REST API endpoints"]}
    ]
}
```'''
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.return_value = mock_result
        mock_llm_pipeline.return_value = mock_pipeline_instance
        
        maker = DictionaryMaker(config_with_temp_file)
        result = maker.process(sample_document)
        
        assert len(result) == 1
        assert result[0].id == sample_document.id
        assert result[0].content == sample_document.content
        assert maker.processing_stats["documents_processed"] == 1
        assert maker.processing_stats["terms_extracted"] == 3
        assert maker.processing_stats["variations_detected"] == 1
        
        # Check that dictionary file was created
        dict_path = Path(config_with_temp_file.dictionary_file_path)
        assert dict_path.exists()

    @patch('refinire_rag.processing.dictionary_maker.LLMPipeline')
    def test_process_with_llm_json_error(self, mock_llm_pipeline, config_with_temp_file, sample_document):
        """Test processing when LLM returns invalid JSON"""
        # Setup mock LLM response with invalid JSON
        mock_result = Mock()
        mock_result.content = "Invalid JSON response"
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.return_value = mock_result
        mock_llm_pipeline.return_value = mock_pipeline_instance
        
        maker = DictionaryMaker(config_with_temp_file)
        result = maker.process(sample_document)
        
        # Should still return document, but with error handling
        assert len(result) == 1
        assert result[0].id == sample_document.id
        assert result[0].content == sample_document.content
        assert maker.processing_stats["documents_processed"] == 1

    @patch('refinire_rag.processing.dictionary_maker.LLMPipeline')
    def test_process_with_llm_api_error(self, mock_llm_pipeline, config_with_temp_file, sample_document):
        """Test processing when LLM API call fails"""
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.side_effect = Exception("API call failed")
        mock_llm_pipeline.return_value = mock_pipeline_instance
        
        maker = DictionaryMaker(config_with_temp_file)
        result = maker.process(sample_document)
        
        # Should still return document, but with error handling
        assert len(result) == 1
        assert result[0].id == sample_document.id
        assert result[0].content == sample_document.content
        assert maker.processing_stats["documents_processed"] == 1

    def test_existing_dictionary_reading(self, config_with_temp_file):
        """Test reading existing dictionary file"""
        # Create existing dictionary file
        dict_path = Path(config_with_temp_file.dictionary_file_path)
        dict_path.parent.mkdir(parents=True, exist_ok=True)
        existing_content = "# Existing Dictionary\n\n## Terms\n- existing term"
        dict_path.write_text(existing_content, encoding='utf-8')
        
        maker = DictionaryMaker(config_with_temp_file)
        read_content = maker._read_existing_dictionary(config_with_temp_file)
        
        assert read_content == existing_content

    def test_nonexistent_dictionary_reading(self, config_with_temp_file):
        """Test reading when dictionary file doesn't exist"""
        maker = DictionaryMaker(config_with_temp_file)
        content = maker._read_existing_dictionary(config_with_temp_file)
        
        # Should return empty dictionary template
        assert "# ドメイン用語辞書" in content

    def test_get_config_class(self):
        """Test get_config_class method"""
        assert DictionaryMaker.get_config_class() == DictionaryMakerConfig

    def test_processing_stats_initialization(self, config_with_temp_file):
        """Test that processing stats are properly initialized"""
        maker = DictionaryMaker(config_with_temp_file)
        
        stats = maker.processing_stats
        assert "documents_processed" in stats
        assert "terms_extracted" in stats
        assert "variations_detected" in stats
        assert "dictionary_updates" in stats
        assert "llm_api_calls" in stats
        assert stats["documents_processed"] == 0

    def test_backup_functionality(self, config_with_temp_file, temp_dir):
        """Test dictionary backup functionality"""
        config_with_temp_file.backup_dictionary = True
        
        # Create existing dictionary
        dict_path = Path(config_with_temp_file.dictionary_file_path)
        dict_path.parent.mkdir(parents=True, exist_ok=True)
        dict_path.write_text("Original content", encoding='utf-8')
        
        maker = DictionaryMaker(config_with_temp_file)
        maker._backup_dictionary_file(config_with_temp_file)
        
        # Check backup was created - backup files use format: .backup_timestamp.md
        backup_files = list(dict_path.parent.glob(f"{dict_path.stem}.backup_*.md"))
        assert len(backup_files) > 0


if __name__ == "__main__":
    pytest.main([__file__])