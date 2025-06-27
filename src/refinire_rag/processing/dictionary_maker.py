"""
DictionaryMaker - Domain-specific term extraction and dictionary update processor

A DocumentProcessor that uses LLM to extract domain-specific terms and expression variations
from documents, maintaining a cumulative Markdown dictionary across all processed documents.
"""

import logging
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Type, Any
from pathlib import Path

from ..document_processor import DocumentProcessor, DocumentProcessorConfig
from ..models.document import Document
from ..utils.model_config import get_default_llm_model

try:
    from refinire import LLMPipeline
except ImportError:
    LLMPipeline = None

logger = logging.getLogger(__name__)


@dataclass
class DictionaryMakerConfig(DocumentProcessorConfig):
    """Configuration for DictionaryMaker processor"""
    
    # Dictionary file settings
    dictionary_file_path: str = "./domain_dictionary.md"
    backup_dictionary: bool = True
    
    # LLM settings
    llm_model: str = None  # Will be set to default in __post_init__
    llm_temperature: float = 0.3
    max_tokens: int = 2000
    
    # Term extraction settings
    focus_on_technical_terms: bool = True
    extract_abbreviations: bool = True
    detect_expression_variations: bool = True
    min_term_importance: str = "medium"  # "low", "medium", "high"
    
    # Processing settings
    skip_if_no_new_terms: bool = False
    validate_extracted_terms: bool = True
    
    # Output settings
    update_document_metadata: bool = True
    preserve_original_document: bool = True

    def __post_init__(self):
        """Initialize default values"""
        # Set default LLM model from environment variables if not specified
        if self.llm_model is None:
            self.llm_model = get_default_llm_model()
        
        # Ensure dictionary file path is absolute
        if not os.path.isabs(self.dictionary_file_path):
            self.dictionary_file_path = os.path.abspath(self.dictionary_file_path)


class DictionaryMaker(DocumentProcessor):
    """Processor that extracts domain-specific terms using LLM and maintains cumulative dictionary
    
    This processor:
    1. Reads existing Markdown dictionary file
    2. Sends document + existing dictionary + extraction prompt to LLM
    3. Extracts domain-specific terms and expression variations
    4. Updates the cumulative Markdown dictionary file
    5. Maintains one dictionary across all processed documents
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize DictionaryMaker processor
        
        Args:
            config: Optional DictionaryMakerConfig object (for backward compatibility)
            **kwargs: Configuration parameters, supports both individual parameters
                     and config dict, with environment variable fallback
        """
        # Handle backward compatibility with config object
        if config is not None and hasattr(config, 'dictionary_file_path'):
            # Traditional config object passed
            super().__init__(config)
            self.dictionary_file_path = config.dictionary_file_path
            self.backup_dictionary = config.backup_dictionary
            self.llm_model = config.llm_model
            self.llm_temperature = config.llm_temperature
            self.max_tokens = config.max_tokens
            self.focus_on_technical_terms = config.focus_on_technical_terms
            self.extract_abbreviations = config.extract_abbreviations
            self.detect_expression_variations = config.detect_expression_variations
            self.min_term_importance = config.min_term_importance
            self.skip_if_no_new_terms = config.skip_if_no_new_terms
            self.validate_extracted_terms = config.validate_extracted_terms
            self.update_document_metadata = config.update_document_metadata
            self.preserve_original_document = config.preserve_original_document
        else:
            # Extract config dict if provided
            config_dict = kwargs.get('config', {})
            
            # Environment variable fallback with priority: kwargs > config dict > env vars > defaults
            self.dictionary_file_path = kwargs.get('dictionary_file_path', 
                                                 config_dict.get('dictionary_file_path', 
                                                                os.getenv('REFINIRE_RAG_DICT_FILE_PATH', './domain_dictionary.md')))
            self.backup_dictionary = kwargs.get('backup_dictionary', 
                                               config_dict.get('backup_dictionary', 
                                                              os.getenv('REFINIRE_RAG_DICT_BACKUP', 'true').lower() == 'true'))
            self.llm_model = kwargs.get('llm_model', 
                                       config_dict.get('llm_model', 
                                                      os.getenv('REFINIRE_RAG_DICT_LLM_MODEL', get_default_llm_model())))
            self.llm_temperature = kwargs.get('llm_temperature', 
                                             config_dict.get('llm_temperature', 
                                                            float(os.getenv('REFINIRE_RAG_DICT_TEMPERATURE', '0.3'))))
            self.max_tokens = kwargs.get('max_tokens', 
                                        config_dict.get('max_tokens', 
                                                       int(os.getenv('REFINIRE_RAG_DICT_MAX_TOKENS', '2000'))))
            self.focus_on_technical_terms = kwargs.get('focus_on_technical_terms', 
                                                      config_dict.get('focus_on_technical_terms', 
                                                                     os.getenv('REFINIRE_RAG_DICT_TECHNICAL_TERMS', 'true').lower() == 'true'))
            self.extract_abbreviations = kwargs.get('extract_abbreviations', 
                                                   config_dict.get('extract_abbreviations', 
                                                                  os.getenv('REFINIRE_RAG_DICT_ABBREVIATIONS', 'true').lower() == 'true'))
            self.detect_expression_variations = kwargs.get('detect_expression_variations', 
                                                          config_dict.get('detect_expression_variations', 
                                                                         os.getenv('REFINIRE_RAG_DICT_VARIATIONS', 'true').lower() == 'true'))
            self.min_term_importance = kwargs.get('min_term_importance', 
                                                 config_dict.get('min_term_importance', 
                                                                os.getenv('REFINIRE_RAG_DICT_MIN_IMPORTANCE', 'medium')))
            self.skip_if_no_new_terms = kwargs.get('skip_if_no_new_terms', 
                                                  config_dict.get('skip_if_no_new_terms', 
                                                                 os.getenv('REFINIRE_RAG_DICT_SKIP_NO_NEW', 'false').lower() == 'true'))
            self.validate_extracted_terms = kwargs.get('validate_extracted_terms', 
                                                      config_dict.get('validate_extracted_terms', 
                                                                     os.getenv('REFINIRE_RAG_DICT_VALIDATE', 'true').lower() == 'true'))
            self.update_document_metadata = kwargs.get('update_document_metadata', 
                                                      config_dict.get('update_document_metadata', 
                                                                     os.getenv('REFINIRE_RAG_DICT_UPDATE_METADATA', 'true').lower() == 'true'))
            self.preserve_original_document = kwargs.get('preserve_original_document', 
                                                        config_dict.get('preserve_original_document', 
                                                                       os.getenv('REFINIRE_RAG_DICT_PRESERVE_ORIGINAL', 'true').lower() == 'true'))
            
            # Create config object for backward compatibility
            config = DictionaryMakerConfig(
                dictionary_file_path=self.dictionary_file_path,
                backup_dictionary=self.backup_dictionary,
                llm_model=self.llm_model,
                llm_temperature=self.llm_temperature,
                max_tokens=self.max_tokens,
                focus_on_technical_terms=self.focus_on_technical_terms,
                extract_abbreviations=self.extract_abbreviations,
                detect_expression_variations=self.detect_expression_variations,
                min_term_importance=self.min_term_importance,
                skip_if_no_new_terms=self.skip_if_no_new_terms,
                validate_extracted_terms=self.validate_extracted_terms,
                update_document_metadata=self.update_document_metadata,
                preserve_original_document=self.preserve_original_document
            )
            
            super().__init__(config)
        
        # Initialize Refinire LLM Pipeline
        if LLMPipeline is not None:
            try:
                self._llm_pipeline = LLMPipeline(
                    name="dictionary_maker",
                    generation_instructions="You are a domain expert that extracts technical terms and their variations from documents.",
                    model=self.llm_model
                )
                logger.info(f"Initialized Refinire LLMPipeline with model: {self.llm_model}")
            except Exception as e:
                self._llm_pipeline = None
                logger.warning(f"Failed to initialize Refinire LLMPipeline: {e}. DictionaryMaker will use mock data.")
        else:
            self._llm_pipeline = None
            logger.warning("Refinire not available. DictionaryMaker will use mock data.")
        
        # Processing statistics
        self.processing_stats.update({
            "documents_processed": 0,
            "terms_extracted": 0,
            "variations_detected": 0,
            "dictionary_updates": 0,
            "llm_api_calls": 0
        })
        
        logger.info(f"Initialized DictionaryMaker with dictionary: {self.dictionary_file_path}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary
        現在の設定を辞書として取得
        
        Returns:
            Dict[str, Any]: Current configuration dictionary
        """
        return {
            'dictionary_file_path': self.dictionary_file_path,
            'backup_dictionary': self.backup_dictionary,
            'llm_model': self.llm_model,
            'llm_temperature': self.llm_temperature,
            'max_tokens': self.max_tokens,
            'focus_on_technical_terms': self.focus_on_technical_terms,
            'extract_abbreviations': self.extract_abbreviations,
            'detect_expression_variations': self.detect_expression_variations,
            'min_term_importance': self.min_term_importance,
            'skip_if_no_new_terms': self.skip_if_no_new_terms,
            'validate_extracted_terms': self.validate_extracted_terms,
            'update_document_metadata': self.update_document_metadata,
            'preserve_original_document': self.preserve_original_document
        }
    
    @classmethod
    def get_config_class(cls) -> Type[DictionaryMakerConfig]:
        """Get the configuration class for this processor (backward compatibility)
        このプロセッサーの設定クラスを取得（下位互換性）
        """
        return DictionaryMakerConfig
    
    def process(self, document: Document, config: Optional[DictionaryMakerConfig] = None) -> List[Document]:
        """Process document to extract terms and update dictionary
        
        Args:
            document: Input document to analyze for domain terms
            config: Optional configuration override
            
        Returns:
            List containing the original document (optionally with dictionary metadata)
        """
        try:
            # Use provided config or fall back to instance config
            dict_config = config or self.config
            
            logger.debug(f"Processing document {document.id} with DictionaryMaker")
            
            # Read existing dictionary
            existing_dictionary = self._read_existing_dictionary(dict_config)
            
            # Extract terms using LLM
            extracted_data = self._extract_terms_with_llm(document, existing_dictionary, dict_config)
            
            if not extracted_data or not extracted_data.get("has_new_terms", False):
                if dict_config.skip_if_no_new_terms:
                    logger.debug(f"No new terms found for document {document.id}, skipping")
                    self.processing_stats["documents_processed"] += 1
                    return [document]
            
            # Update dictionary file
            updated_dictionary = self._update_dictionary_file(existing_dictionary, extracted_data, dict_config)
            
            # Update statistics
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["terms_extracted"] += extracted_data.get("new_terms_count", 0)
            self.processing_stats["variations_detected"] += extracted_data.get("variations_count", 0)
            self.processing_stats["dictionary_updates"] += 1
            self.processing_stats["llm_api_calls"] += 1
            
            # Create output document
            if dict_config.update_document_metadata:
                enriched_document = self._add_dictionary_metadata(document, extracted_data, dict_config)
                return [enriched_document]
            else:
                return [document]
            
        except Exception as e:
            logger.error(f"Error in DictionaryMaker for document {document.id}: {e}")
            return [document]  # Return original on error
    
    def _read_existing_dictionary(self, config: DictionaryMakerConfig) -> str:
        """Read existing Markdown dictionary file
        
        Args:
            config: Configuration containing dictionary file path
            
        Returns:
            Content of existing dictionary file, or empty template if file doesn't exist
        """
        try:
            dictionary_path = Path(config.dictionary_file_path)
            
            if dictionary_path.exists():
                with open(dictionary_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.debug(f"Read existing dictionary: {len(content)} characters")
                return content
            else:
                # Create empty dictionary template
                template = self._create_empty_dictionary_template()
                logger.info(f"Creating new dictionary file at: {dictionary_path}")
                
                # Ensure directory exists
                dictionary_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create initial file
                with open(dictionary_path, 'w', encoding='utf-8') as f:
                    f.write(template)
                
                return template
                
        except Exception as e:
            logger.error(f"Error reading dictionary file: {e}")
            return self._create_empty_dictionary_template()
    
    def _create_empty_dictionary_template(self) -> str:
        """Create empty dictionary template"""
        return """# ドメイン用語辞書

## 専門用語

## 技術概念

## 表現揺らぎ

---
*この辞書は自動生成され、処理された文書から継続的に更新されます。*
"""
    
    def _extract_terms_with_llm(self, document: Document, existing_dictionary: str, 
                               config: DictionaryMakerConfig) -> Dict:
        """Extract terms using LLM
        
        Args:
            document: Document to analyze
            existing_dictionary: Current dictionary content
            config: Configuration
            
        Returns:
            Dictionary containing extracted terms and metadata
        """
        # Create prompt for LLM
        prompt = self._create_extraction_prompt(document, existing_dictionary, config)
        
        # Use Refinire LLM Pipeline for term extraction
        if self._llm_pipeline is not None:
            try:
                result = self._llm_pipeline.run(prompt)
                response = result.content
                
                # Parse JSON response
                try:
                    # Clean response - remove markdown code block markers if present
                    clean_response = response.strip()
                    if clean_response.startswith("```json"):
                        clean_response = clean_response[7:]
                    if clean_response.endswith("```"):
                        clean_response = clean_response[:-3]
                    clean_response = clean_response.strip()
                    
                    extracted_data = json.loads(clean_response)
                    logger.info(f"Extracted {extracted_data.get('new_terms_count', 0)} terms from document {document.id}")
                    return extracted_data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response as JSON: {e}")
                    logger.debug(f"Raw LLM response: {response}")
                    return {"has_new_terms": False, "new_terms_count": 0, "variations_count": 0, "extracted_terms": []}
                    
            except Exception as e:
                logger.error(f"LLM call failed for document {document.id}: {e}")
                return {"has_new_terms": False, "new_terms_count": 0, "variations_count": 0, "extracted_terms": []}
        else:
            # Fallback to mock data when Refinire is not available
            logger.info(f"Using mock data for document {document.id} (Refinire LLM not available)")
            mock_data = {
                "has_new_terms": True,
                "new_terms_count": 3,
                "variations_count": 2,
                "extracted_terms": [
                    {
                        "term": "RAG",
                        "full_form": "Retrieval-Augmented Generation", 
                        "definition": "検索拡張生成",
                        "variations": ["検索拡張生成", "検索強化生成", "RAGシステム"]
                    },
                    {
                        "term": "ベクトル検索",
                        "definition": "類似度に基づく文書検索手法",
                        "variations": ["セマンティック検索", "埋め込み検索", "意味検索"]
                    }
                ],
                "reasoning": "Document contains technical terms related to RAG systems and vector search that should be included in domain dictionary."
            }
            return mock_data
    
    def _create_extraction_prompt(self, document: Document, existing_dictionary: str,
                                config: DictionaryMakerConfig) -> str:
        """Create prompt for LLM term extraction"""
        
        prompt = f"""以下のドメイン文書から、重要な専門用語と表現揺らぎを抽出してください。

## 既存の用語辞書
{existing_dictionary}

## 分析対象文書
タイトル: {document.metadata.get('title', 'Unknown')}
内容:
{document.content[:2000]}  # Limit content to avoid token limits

## 抽出指示
1. 文書のドメインに固有の重要な専門用語を特定
2. 略語とその展開形を特定
3. 同じ概念を表す表現の揺らぎを検出
4. 既存辞書にない新しい用語のみを抽出

## 出力形式
以下のJSON形式で回答してください:
{{
    "has_new_terms": true/false,
    "new_terms_count": 数値,
    "variations_count": 数値,
    "extracted_terms": [
        {{
            "term": "用語名",
            "full_form": "完全形（略語の場合）",
            "definition": "定義や説明",
            "variations": ["表現揺らぎ1", "表現揺らぎ2"]
        }}
    ],
    "reasoning": "抽出理由の説明"
}}

重要度レベル: {config.min_term_importance}
技術用語重視: {config.focus_on_technical_terms}
略語抽出: {config.extract_abbreviations}
"""
        
        return prompt
    
    def _update_dictionary_file(self, existing_dictionary: str, extracted_data: Dict,
                               config: DictionaryMakerConfig) -> str:
        """Update dictionary file with extracted terms
        
        Args:
            existing_dictionary: Current dictionary content
            extracted_data: Data extracted by LLM
            config: Configuration
            
        Returns:
            Updated dictionary content
        """
        try:
            # Backup existing dictionary if configured
            if config.backup_dictionary:
                self._backup_dictionary_file(config)
            
            # Parse and update dictionary
            updated_dictionary = self._merge_terms_into_dictionary(existing_dictionary, extracted_data)
            
            # Write updated dictionary
            dictionary_path = Path(config.dictionary_file_path)
            with open(dictionary_path, 'w', encoding='utf-8') as f:
                f.write(updated_dictionary)
            
            logger.info(f"Updated dictionary with {extracted_data.get('new_terms_count', 0)} new terms")
            return updated_dictionary
            
        except Exception as e:
            logger.error(f"Error updating dictionary file: {e}")
            return existing_dictionary
    
    def _backup_dictionary_file(self, config: DictionaryMakerConfig):
        """Create backup of dictionary file"""
        try:
            dictionary_path = Path(config.dictionary_file_path)
            if dictionary_path.exists():
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = dictionary_path.with_suffix(f".backup_{timestamp}.md")
                
                import shutil
                shutil.copy2(dictionary_path, backup_path)
                logger.debug(f"Created dictionary backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create dictionary backup: {e}")
    
    def _merge_terms_into_dictionary(self, existing_dictionary: str, extracted_data: Dict) -> str:
        """Merge extracted terms into existing dictionary
        
        Args:
            existing_dictionary: Current dictionary content
            extracted_data: Extracted terms data
            
        Returns:
            Updated dictionary content
        """
        # Simple implementation - in practice would need sophisticated parsing and merging
        lines = existing_dictionary.split('\n')
        
        # Find insertion points for different sections
        tech_section_idx = None
        concept_section_idx = None
        
        for i, line in enumerate(lines):
            if line.strip() == "## 専門用語":
                tech_section_idx = i
            elif line.strip() == "## 技術概念":
                concept_section_idx = i
        
        # Add new terms
        for term_data in extracted_data.get("extracted_terms", []):
            term_entry = self._format_term_entry(term_data)
            
            # Add to appropriate section
            if tech_section_idx is not None:
                lines.insert(tech_section_idx + 1, term_entry)
                tech_section_idx += 1  # Adjust index for next insertion
        
        return '\n'.join(lines)
    
    def _format_term_entry(self, term_data: Dict) -> str:
        """Format term data as Markdown entry"""
        term = term_data.get("term", "")
        full_form = term_data.get("full_form", "")
        definition = term_data.get("definition", "")
        variations = term_data.get("variations", [])
        
        entry_lines = []
        
        if full_form:
            entry_lines.append(f"- **{term}** ({full_form}): {definition}")
        else:
            entry_lines.append(f"- **{term}**: {definition}")
        
        if variations:
            entry_lines.append(f"  - 表現揺らぎ: {', '.join(variations)}")
        
        return '\n'.join(entry_lines)
    
    def _add_dictionary_metadata(self, document: Document, extracted_data: Dict,
                                config: DictionaryMakerConfig) -> Document:
        """Add dictionary extraction metadata to document"""
        
        dictionary_metadata = {
            "dictionary_extraction_applied": True,
            "new_terms_extracted": extracted_data.get("new_terms_count", 0),
            "variations_detected": extracted_data.get("variations_count", 0),
            "extraction_reasoning": extracted_data.get("reasoning", ""),
            "dictionary_file_path": config.dictionary_file_path,
            "extracted_by": "DictionaryMaker"
        }
        
        enriched_metadata = {
            **document.metadata,
            "dictionary_metadata": dictionary_metadata,
            "processing_stage": "dictionary_analyzed"
        }
        
        enriched_document = Document(
            id=document.id,
            content=document.content,
            metadata=enriched_metadata
        )
        
        return enriched_document
    
    def get_dictionary_path(self) -> str:
        """Get current dictionary file path"""
        return self.config.dictionary_file_path
    
    def get_dictionary_content(self) -> str:
        """Get current dictionary content"""
        return self._read_existing_dictionary(self.config)
    
    def get_extraction_stats(self) -> Dict:
        """Get dictionary extraction statistics"""
        return {
            **self.get_processing_stats(),
            "dictionary_file": self.config.dictionary_file_path,
            "llm_model": self.config.llm_model,
            "focus_technical_terms": self.config.focus_on_technical_terms
        }