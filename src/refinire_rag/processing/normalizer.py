"""
Normalizer - Expression variation normalization processor

A DocumentProcessor that normalizes text by correcting expression variations
using a Markdown dictionary file created by DictionaryMaker.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Type, Tuple
from pathlib import Path

from ..document_processor import DocumentProcessor, DocumentProcessorConfig
from ..models.document import Document

logger = logging.getLogger(__name__)


@dataclass
class NormalizerConfig(DocumentProcessorConfig):
    """Configuration for Normalizer processor"""
    
    # Dictionary file settings
    dictionary_file_path: str = "./domain_dictionary.md"
    auto_detect_dictionary_path: bool = True  # Try to find dictionary from DictionaryMaker metadata
    
    # Normalization settings
    normalize_variations: bool = True
    expand_abbreviations: bool = True
    standardize_technical_terms: bool = True
    
    # Replacement behavior
    case_sensitive_replacement: bool = False
    whole_word_only: bool = True
    preserve_original_in_parentheses: bool = True  # "RAG (Retrieval-Augmented Generation)"
    
    # Processing settings
    skip_if_no_dictionary: bool = False
    validate_replacements: bool = True
    max_replacements_per_term: int = 1000
    
    # Output settings
    add_normalization_metadata: bool = True
    preserve_original_document: bool = True

    def __post_init__(self):
        """Initialize default values"""
        # Ensure dictionary file path is absolute
        if self.dictionary_file_path and not Path(self.dictionary_file_path).is_absolute():
            self.dictionary_file_path = str(Path(self.dictionary_file_path).absolute())


class Normalizer(DocumentProcessor):
    """Processor that normalizes text using Markdown dictionary from DictionaryMaker
    
    This processor:
    1. Reads Markdown dictionary file (from DictionaryMaker)
    2. Parses expression variations and term mappings
    3. Applies normalization to document content
    4. Corrects expression variations to standard forms
    5. Expands abbreviations with full forms
    """
    
    def __init__(self, config: Optional[NormalizerConfig] = None):
        """Initialize Normalizer processor
        
        Args:
            config: Configuration for the processor
        """
        super().__init__(config or NormalizerConfig())
        
        # Cached normalization mappings
        self._normalization_mappings = {}
        self._dictionary_last_modified = None
        
        # Processing statistics
        self.processing_stats.update({
            "documents_processed": 0,
            "total_replacements": 0,
            "variations_normalized": 0,
            "abbreviations_expanded": 0,
            "dictionary_load_errors": 0
        })
        
        logger.info(f"Initialized Normalizer with dictionary: {self.config.dictionary_file_path}")
    
    @classmethod
    def get_config_class(cls) -> Type[NormalizerConfig]:
        """Get the configuration class for this processor"""
        return NormalizerConfig
    
    def process(self, document: Document, config: Optional[NormalizerConfig] = None) -> List[Document]:
        """Process document to normalize expression variations
        
        Args:
            document: Input document to normalize
            config: Optional configuration override
            
        Returns:
            List containing the normalized document
        """
        try:
            # Use provided config or fall back to instance config
            norm_config = config or self.config
            
            logger.debug(f"Processing document {document.id} with Normalizer")
            
            # Determine dictionary path
            dictionary_path = self._determine_dictionary_path(document, norm_config)
            
            if not dictionary_path:
                if norm_config.skip_if_no_dictionary:
                    logger.debug(f"No dictionary available for document {document.id}, skipping")
                    return [document]
                else:
                    logger.warning(f"No dictionary found for document {document.id}")
                    return [document]
            
            # Load normalization mappings
            mappings = self._load_normalization_mappings(dictionary_path)
            
            if not mappings:
                logger.warning(f"No normalization mappings found in dictionary: {dictionary_path}")
                return [document]
            
            # Apply normalization
            normalized_content, normalization_info = self._normalize_content(
                document.content, mappings, norm_config
            )
            
            # Create normalized document
            normalized_document = self._create_normalized_document(
                document, normalized_content, normalization_info, norm_config
            )
            
            # Update statistics
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["total_replacements"] += normalization_info["total_replacements"]
            self.processing_stats["variations_normalized"] += normalization_info["variations_count"]
            self.processing_stats["abbreviations_expanded"] += normalization_info["abbreviations_count"]
            
            logger.info(f"Normalizer: Applied {normalization_info['total_replacements']} normalizations to {document.id}")
            return [normalized_document]
            
        except Exception as e:
            logger.error(f"Error in Normalizer for document {document.id}: {e}")
            return [document]  # Return original on error
    
    def _determine_dictionary_path(self, document: Document, config: NormalizerConfig) -> Optional[str]:
        """Determine dictionary file path from config or document metadata
        
        Args:
            document: Document being processed
            config: Configuration
            
        Returns:
            Path to dictionary file, or None if not found
        """
        # Try to get dictionary path from document metadata (from DictionaryMaker)
        if config.auto_detect_dictionary_path:
            dict_metadata = document.metadata.get("dictionary_metadata", {})
            if dict_metadata and "dictionary_file_path" in dict_metadata:
                auto_path = dict_metadata["dictionary_file_path"]
                if Path(auto_path).exists():
                    logger.debug(f"Using dictionary path from document metadata: {auto_path}")
                    return auto_path
        
        # Fall back to configured path
        if config.dictionary_file_path and Path(config.dictionary_file_path).exists():
            return config.dictionary_file_path
        
        return None
    
    def _load_normalization_mappings(self, dictionary_path: str) -> Dict[str, str]:
        """Load normalization mappings from Markdown dictionary
        
        Args:
            dictionary_path: Path to dictionary file
            
        Returns:
            Dictionary mapping variations to standard forms
        """
        try:
            dictionary_file = Path(dictionary_path)
            
            # Check if we need to reload mappings
            current_modified = dictionary_file.stat().st_mtime
            if (self._dictionary_last_modified == current_modified and 
                self._normalization_mappings):
                logger.debug("Using cached normalization mappings")
                return self._normalization_mappings
            
            # Read and parse dictionary
            with open(dictionary_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            mappings = self._parse_dictionary_content(content)
            
            # Cache the mappings
            self._normalization_mappings = mappings
            self._dictionary_last_modified = current_modified
            
            logger.info(f"Loaded {len(mappings)} normalization mappings from dictionary")
            if mappings:
                logger.debug(f"Sample mappings: {dict(list(mappings.items())[:3])}")
            else:
                logger.warning("No normalization mappings found in dictionary: " + dictionary_path)
            return mappings
            
        except Exception as e:
            logger.error(f"Error loading dictionary {dictionary_path}: {e}")
            self.processing_stats["dictionary_load_errors"] += 1
            return {}
    
    def _parse_dictionary_content(self, content: str) -> Dict[str, str]:
        """Parse Markdown dictionary content to extract normalization mappings
        
        Args:
            content: Dictionary content in Markdown format
            
        Returns:
            Dictionary mapping variations to standard terms
        """
        mappings = {}
        lines = content.split('\n')
        
        current_term = None
        current_standard_form = None
        
        for line in lines:
            line = line.strip()
            
            # Parse term definitions: - **Term** (Full Form): Definition
            term_match = re.match(r'- \*\*([^*]+)\*\*(?:\s*\(([^)]+)\))?: (.+)', line)
            if term_match:
                term = term_match.group(1).strip()
                full_form = term_match.group(2).strip() if term_match.group(2) else None
                definition = term_match.group(3).strip()
                
                current_term = term
                
                # Determine standard form
                if full_form:
                    # If there's a full form, use "Term (Full Form)" as standard
                    current_standard_form = f"{term} ({full_form})"
                    # Map abbreviation to full form
                    mappings[term] = current_standard_form
                else:
                    current_standard_form = term
                
                continue
            
            # Parse variations: - 表現揺らぎ: variation1, variation2, variation3
            variations_match = re.match(r'- 表現揺らぎ: (.+)', line)
            if variations_match and current_standard_form:
                variations_text = variations_match.group(1).strip()
                variations = [v.strip() for v in variations_text.split(',')]
                
                # Map each variation to the standard form
                for variation in variations:
                    if variation and variation != current_standard_form:
                        mappings[variation] = current_standard_form
                
                continue
        
        return mappings
    
    def _normalize_content(self, content: str, mappings: Dict[str, str], 
                          config: NormalizerConfig) -> Tuple[str, Dict]:
        """Apply normalization to content using mappings
        
        Args:
            content: Original content
            mappings: Normalization mappings
            config: Configuration
            
        Returns:
            Tuple of (normalized_content, normalization_info)
        """
        normalized_content = content
        normalization_info = {
            "total_replacements": 0,
            "variations_count": 0,
            "abbreviations_count": 0,
            "replacements": []
        }
        
        # Sort mappings by length (longest first) to avoid partial replacements
        sorted_mappings = sorted(mappings.items(), key=lambda x: len(x[0]), reverse=True)
        
        for variation, standard_form in sorted_mappings:
            if not variation or not standard_form:
                continue
            
            # Create replacement pattern
            if config.whole_word_only:
                if config.case_sensitive_replacement:
                    pattern = re.compile(r'\b' + re.escape(variation) + r'\b')
                else:
                    pattern = re.compile(r'\b' + re.escape(variation) + r'\b', re.IGNORECASE)
            else:
                if config.case_sensitive_replacement:
                    pattern = re.compile(re.escape(variation))
                else:
                    pattern = re.compile(re.escape(variation), re.IGNORECASE)
            
            # Find matches
            matches = list(pattern.finditer(normalized_content))
            logger.debug(f"Pattern '{pattern.pattern}' for variation '{variation}': {len(matches)} matches")
            
            if matches and len(matches) <= config.max_replacements_per_term:
                # Apply replacement
                replaced_content = pattern.sub(standard_form, normalized_content)
                
                if replaced_content != normalized_content:
                    # Track replacement
                    replacement_info = {
                        "original": variation,
                        "replacement": standard_form,
                        "count": len(matches),
                        "type": self._classify_replacement_type(variation, standard_form)
                    }
                    
                    normalization_info["replacements"].append(replacement_info)
                    normalization_info["total_replacements"] += len(matches)
                    
                    if replacement_info["type"] == "variation":
                        normalization_info["variations_count"] += len(matches)
                    elif replacement_info["type"] == "abbreviation":
                        normalization_info["abbreviations_count"] += len(matches)
                    
                    normalized_content = replaced_content
                    logger.debug(f"Replaced '{variation}' → '{standard_form}' ({len(matches)} times)")
            
            elif len(matches) > config.max_replacements_per_term:
                logger.warning(f"Too many matches ({len(matches)}) for term '{variation}', skipping")
        
        return normalized_content, normalization_info
    
    def _classify_replacement_type(self, original: str, replacement: str) -> str:
        """Classify the type of replacement
        
        Args:
            original: Original term
            replacement: Replacement term
            
        Returns:
            Type of replacement: "abbreviation", "variation", or "other"
        """
        # Check if it's an abbreviation expansion
        if "(" in replacement and ")" in replacement:
            # Pattern like "RAG (Retrieval-Augmented Generation)"
            return "abbreviation"
        elif len(original) < len(replacement):
            # Shorter to longer might be abbreviation
            return "abbreviation"
        else:
            # Likely a variation normalization
            return "variation"
    
    def _create_normalized_document(self, original_document: Document, 
                                   normalized_content: str, normalization_info: Dict,
                                   config: NormalizerConfig) -> Document:
        """Create normalized document with metadata
        
        Args:
            original_document: Original document
            normalized_content: Normalized content
            normalization_info: Information about applied normalizations
            config: Configuration
            
        Returns:
            Normalized document with metadata
        """
        # Prepare metadata
        normalized_metadata = {
            **original_document.metadata,
            "processing_stage": "normalized",
            "original_document_id": original_document.metadata.get("original_document_id", original_document.id),
            "parent_document_id": original_document.id
        }
        
        # Add normalization metadata if configured
        if config.add_normalization_metadata:
            normalized_metadata.update({
                "normalization_applied": True,
                "normalization_stats": {
                    "total_replacements": normalization_info["total_replacements"],
                    "variations_normalized": normalization_info["variations_count"],
                    "abbreviations_expanded": normalization_info["abbreviations_count"],
                    "replacement_details": normalization_info["replacements"]
                },
                "dictionary_file_used": config.dictionary_file_path,
                "normalized_by": "Normalizer"
            })
        
        # Create normalized document
        normalized_document = Document(
            id=original_document.id,  # Keep same ID as this is content modification
            content=normalized_content,
            metadata=normalized_metadata
        )
        
        return normalized_document
    
    def get_normalization_mappings(self) -> Dict[str, str]:
        """Get current normalization mappings"""
        return self._normalization_mappings.copy()
    
    def reload_dictionary(self, dictionary_path: Optional[str] = None) -> int:
        """Reload dictionary and return number of mappings loaded
        
        Args:
            dictionary_path: Optional path override
            
        Returns:
            Number of normalization mappings loaded
        """
        path = dictionary_path or self.config.dictionary_file_path
        if path:
            # Force reload by clearing cache
            self._dictionary_last_modified = None
            mappings = self._load_normalization_mappings(path)
            return len(mappings)
        return 0
    
    def get_normalization_stats(self) -> Dict:
        """Get normalization statistics"""
        return {
            **self.get_processing_stats(),
            "dictionary_file": self.config.dictionary_file_path,
            "mappings_loaded": len(self._normalization_mappings),
            "auto_detect_dictionary": self.config.auto_detect_dictionary_path
        }