"""
Custom DocumentProcessor Example

This example shows how to create your own DocumentProcessor implementations
with custom configuration classes following the new design pattern.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Type, List, Optional
import re

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag import (
    Document,
    DocumentProcessor,
    DocumentProcessorConfig,
    DocumentPipeline,
    SQLiteDocumentStore
)


# Example 1: Text Normalization Processor
@dataclass
class TextNormalizationConfig(DocumentProcessorConfig):
    """Configuration for text normalization processor
    テキスト正規化プロセッサの設定"""
    
    # Text transformation options
    lowercase: bool = True
    remove_extra_whitespace: bool = True
    expand_contractions: bool = True
    remove_punctuation: bool = False
    
    # Language-specific options
    language: str = "en"
    
    # Formatting options
    preserve_line_breaks: bool = False
    normalize_unicode: bool = True


class TextNormalizationProcessor(DocumentProcessor):
    """Normalizes text content according to configuration
    設定に従ってテキストコンテンツを正規化"""
    
    @classmethod
    def get_config_class(cls) -> Type[TextNormalizationConfig]:
        return TextNormalizationConfig
    
    def process(self, document: Document, config: Optional[TextNormalizationConfig] = None) -> List[Document]:
        """Normalize text based on configuration"""
        norm_config = config or self.config
        
        content = document.content
        
        # Apply normalization steps
        if norm_config.normalize_unicode:
            import unicodedata
            content = unicodedata.normalize('NFKC', content)
        
        if norm_config.lowercase:
            content = content.lower()
        
        if norm_config.remove_extra_whitespace:
            content = re.sub(r'\s+', ' ', content).strip()
        
        if norm_config.expand_contractions and norm_config.language == "en":
            contractions = {
                "don't": "do not", "won't": "will not", "can't": "cannot",
                "it's": "it is", "you're": "you are", "we're": "we are",
                "they're": "they are", "i'm": "i am", "he's": "he is",
                "she's": "she is", "we've": "we have", "you've": "you have"
            }
            for contraction, expansion in contractions.items():
                content = content.replace(contraction, expansion)
        
        if norm_config.remove_punctuation:
            content = re.sub(r'[^\w\s]', '', content)
        
        if not norm_config.preserve_line_breaks:
            content = content.replace('\n', ' ').replace('\r', ' ')
        
        # Create normalized document
        normalized_doc = Document(
            id=f"{document.id}_normalized",
            content=content,
            metadata={
                **document.metadata,
                "original_document_id": document.id,
                "parent_document_id": document.id,
                "processing_stage": "normalized",
                "normalization_applied": {
                    "lowercase": norm_config.lowercase,
                    "expand_contractions": norm_config.expand_contractions,
                    "remove_punctuation": norm_config.remove_punctuation
                }
            }
        )
        
        return [normalized_doc]


# Example 2: Document Enrichment Processor
@dataclass
class DocumentEnrichmentConfig(DocumentProcessorConfig):
    """Configuration for document enrichment processor
    文書エンリッチメントプロセッサの設定"""
    
    # Analysis options
    add_statistics: bool = True
    extract_keywords: bool = True
    detect_language: bool = True
    analyze_readability: bool = False
    
    # Keyword extraction settings
    max_keywords: int = 10
    min_keyword_length: int = 3
    
    # Language detection settings
    language_detection_method: str = "simple"  # "simple" or "advanced"


class DocumentEnrichmentProcessor(DocumentProcessor):
    """Enriches documents with additional metadata and analysis
    追加のメタデータと分析で文書を豊富にする"""
    
    @classmethod
    def get_config_class(cls) -> Type[DocumentEnrichmentConfig]:
        return DocumentEnrichmentConfig
    
    def process(self, document: Document, config: Optional[DocumentEnrichmentConfig] = None) -> List[Document]:
        """Enrich document with metadata"""
        enrich_config = config or self.config
        
        enriched_doc = Document(
            id=f"{document.id}_enriched",
            content=document.content,
            metadata=document.metadata.copy()
        )
        
        # Add document statistics
        if enrich_config.add_statistics:
            stats = self._calculate_statistics(document.content)
            enriched_doc.metadata.update(stats)
        
        # Extract keywords
        if enrich_config.extract_keywords:
            keywords = self._extract_keywords(
                document.content, 
                enrich_config.max_keywords,
                enrich_config.min_keyword_length
            )
            enriched_doc.metadata["keywords"] = keywords
        
        # Detect language
        if enrich_config.detect_language:
            language = self._detect_language(document.content)
            enriched_doc.metadata["detected_language"] = language
        
        # Analyze readability
        if enrich_config.analyze_readability:
            readability = self._analyze_readability(document.content)
            enriched_doc.metadata["readability_score"] = readability
        
        # Update processing metadata
        enriched_doc.metadata.update({
            "parent_document_id": document.id,
            "processing_stage": "enriched",
            "enrichment_timestamp": "2024-01-15T10:00:00Z"
        })
        
        return [enriched_doc]
    
    def _calculate_statistics(self, content: str) -> dict:
        """Calculate basic text statistics"""
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        paragraphs = content.split('\n\n')
        
        return {
            "char_count": len(content),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len([p for p in paragraphs if p.strip()]),
            "avg_words_per_sentence": len(words) / max(len(sentences), 1)
        }
    
    def _extract_keywords(self, content: str, max_keywords: int, min_length: int) -> List[str]:
        """Simple keyword extraction based on frequency"""
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Filter by length and common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        filtered_words = [w for w in words if len(w) >= min_length and w not in stop_words]
        
        # Count frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def _detect_language(self, content: str) -> str:
        """Simple language detection"""
        # Very basic detection - in real implementation, use proper library
        japanese_chars = re.findall(r'[ひらがなカタカナ漢字]', content)
        if len(japanese_chars) > len(content) * 0.1:
            return "ja"
        return "en"
    
    def _analyze_readability(self, content: str) -> float:
        """Simple readability analysis (Flesch Reading Ease approximation)"""
        words = len(content.split())
        sentences = len(re.split(r'[.!?]+', content))
        syllables = len(re.findall(r'[aeiouAEIOU]', content))  # Very rough syllable count
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0.0, min(100.0, score))


# Example 3: Document Splitter Processor
@dataclass
class DocumentSplitterConfig(DocumentProcessorConfig):
    """Configuration for document splitting processor
    文書分割プロセッサの設定"""
    
    # Splitting options
    split_by: str = "paragraph"  # "paragraph", "section", "custom"
    custom_delimiter: str = "---"
    
    # Size constraints
    min_split_size: int = 50
    max_split_size: int = 2000
    
    # Content preservation
    preserve_headers: bool = True
    add_split_context: bool = True


class DocumentSplitterProcessor(DocumentProcessor):
    """Splits documents into smaller parts based on content structure
    コンテンツ構造に基づいて文書を小さな部分に分割"""
    
    @classmethod
    def get_config_class(cls) -> Type[DocumentSplitterConfig]:
        return DocumentSplitterConfig
    
    def process(self, document: Document, config: Optional[DocumentSplitterConfig] = None) -> List[Document]:
        """Split document based on configuration"""
        split_config = config or self.config
        
        if split_config.split_by == "paragraph":
            splits = self._split_by_paragraph(document.content)
        elif split_config.split_by == "section":
            splits = self._split_by_section(document.content)
        elif split_config.split_by == "custom":
            splits = self._split_by_custom_delimiter(document.content, split_config.custom_delimiter)
        else:
            splits = [document.content]  # No splitting
        
        # Filter by size constraints
        valid_splits = []
        for split in splits:
            if split_config.min_split_size <= len(split) <= split_config.max_split_size:
                valid_splits.append(split)
        
        # Create split documents
        split_docs = []
        for i, split_content in enumerate(valid_splits):
            split_doc = Document(
                id=f"{document.id}_split_{i:03d}",
                content=split_content,
                metadata={
                    **document.metadata,
                    "original_document_id": document.id,
                    "parent_document_id": document.id,
                    "processing_stage": "split",
                    "split_index": i,
                    "split_total": len(valid_splits),
                    "split_method": split_config.split_by
                }
            )
            split_docs.append(split_doc)
        
        return split_docs
    
    def _split_by_paragraph(self, content: str) -> List[str]:
        """Split by paragraphs (double newlines)"""
        return [p.strip() for p in content.split('\n\n') if p.strip()]
    
    def _split_by_section(self, content: str) -> List[str]:
        """Split by sections (lines starting with #)"""
        lines = content.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            if line.startswith('#') and current_section:
                sections.append('\n'.join(current_section).strip())
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section).strip())
        
        return [s for s in sections if s]
    
    def _split_by_custom_delimiter(self, content: str, delimiter: str) -> List[str]:
        """Split by custom delimiter"""
        return [part.strip() for part in content.split(delimiter) if part.strip()]


def example_text_normalization():
    """Demonstrate text normalization processor"""
    print("=== Text Normalization Example ===\n")
    
    # Create test document with various text issues
    document = Document(
        id="norm_test_001",
        content="""  This   is a    MESSY   document!  
        It has   extra    whitespace, contractions like don't and won't,
        UPPERCASE text,     and  various   formatting issues.
        
        We'll   normalize   this    text   to   make   it   cleaner!""",
        metadata={
            "path": "/examples/messy.txt",
            "created_at": "2024-01-15T10:00:00Z",
            "file_type": ".txt",
            "size_bytes": 200
        }
    )
    
    print(f"Original document:")
    print(f"'{document.content}'")
    
    # Test different normalization configurations
    configs = [
        TextNormalizationConfig(
            lowercase=True,
            remove_extra_whitespace=True,
            expand_contractions=True,
            remove_punctuation=False
        ),
        TextNormalizationConfig(
            lowercase=True,
            remove_extra_whitespace=True,
            expand_contractions=True,
            remove_punctuation=True
        )
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1} ---")
        print(f"Config: {config}")
        
        processor = TextNormalizationProcessor(config)
        results = processor.process(document)
        
        print(f"Normalized: '{results[0].content}'")
        print(f"Metadata: {results[0].metadata.get('normalization_applied')}")
    
    return results


def example_document_enrichment():
    """Demonstrate document enrichment processor"""
    print("\n=== Document Enrichment Example ===\n")
    
    # Create test document
    document = Document(
        id="enrich_test_001",
        content="""Machine learning is a fascinating field of artificial intelligence. 
        It involves training algorithms to learn patterns from data without explicit programming.
        Deep learning, neural networks, and data science are related concepts.
        The applications include natural language processing, computer vision, and robotics.
        Many companies use machine learning for recommendation systems and predictive analytics.""",
        metadata={
            "path": "/examples/ml_article.txt",
            "created_at": "2024-01-15T11:00:00Z",
            "file_type": ".txt",
            "size_bytes": 500
        }
    )
    
    print(f"Original document: {document.id}")
    print(f"Content: {document.content[:100]}...")
    
    # Test enrichment
    config = DocumentEnrichmentConfig(
        add_statistics=True,
        extract_keywords=True,
        detect_language=True,
        analyze_readability=True,
        max_keywords=8
    )
    
    processor = DocumentEnrichmentProcessor(config)
    results = processor.process(document)
    
    enriched_doc = results[0]
    print(f"\nEnriched document: {enriched_doc.id}")
    print(f"Added metadata:")
    
    # Show statistics
    stats_keys = ['char_count', 'word_count', 'sentence_count', 'avg_words_per_sentence']
    for key in stats_keys:
        if key in enriched_doc.metadata:
            print(f"  {key}: {enriched_doc.metadata[key]}")
    
    # Show keywords
    if 'keywords' in enriched_doc.metadata:
        print(f"  keywords: {enriched_doc.metadata['keywords']}")
    
    # Show language and readability
    if 'detected_language' in enriched_doc.metadata:
        print(f"  detected_language: {enriched_doc.metadata['detected_language']}")
    
    if 'readability_score' in enriched_doc.metadata:
        print(f"  readability_score: {enriched_doc.metadata['readability_score']:.1f}")
    
    return results


def example_document_splitting():
    """Demonstrate document splitting processor"""
    print("\n=== Document Splitting Example ===\n")
    
    # Create test document with multiple sections
    document = Document(
        id="split_test_001",
        content="""# Introduction
This is the introduction section of our document.
It provides an overview of the topics we will cover.

# Main Content
This section contains the main content of the document.
Here we dive deep into the technical details and explanations.

The content continues with more paragraphs and information.
Each paragraph adds value to the overall understanding.

# Conclusion
This is the conclusion section where we summarize everything.
We provide final thoughts and recommendations for the reader.""",
        metadata={
            "path": "/examples/structured_doc.txt",
            "created_at": "2024-01-15T12:00:00Z",
            "file_type": ".txt",
            "size_bytes": 600
        }
    )
    
    print(f"Original document: {document.id}")
    print(f"Content length: {len(document.content)} characters")
    
    # Test splitting by sections
    config = DocumentSplitterConfig(
        split_by="section",
        min_split_size=30,
        max_split_size=1000
    )
    
    processor = DocumentSplitterProcessor(config)
    results = processor.process(document)
    
    print(f"\nSplit into {len(results)} sections:")
    for split_doc in results:
        split_index = split_doc.metadata.get('split_index')
        split_method = split_doc.metadata.get('split_method')
        print(f"\n  Split {split_index} ({split_method}):")
        print(f"    ID: {split_doc.id}")
        print(f"    Content: {split_doc.content[:100]}...")
        print(f"    Length: {len(split_doc.content)} characters")
    
    return results


def example_custom_pipeline():
    """Demonstrate a pipeline with custom processors"""
    print("\n=== Custom Pipeline Example ===\n")
    
    # Create test document
    document = Document(
        id="pipeline_test_001",
        content="""# Technical Documentation

This   is   a   TECHNICAL   document   that   needs   processing!
It contains   various   formatting   issues   and   we'll   normalize   it.

The document discusses machine learning algorithms and their applications.
Deep learning and neural networks are important topics in artificial intelligence.

## Implementation Details

Here we provide detailed implementation guidelines.
The code examples show best practices for the field.""",
        metadata={
            "path": "/examples/tech_doc.txt",
            "created_at": "2024-01-15T13:00:00Z",
            "file_type": ".txt",
            "size_bytes": 400
        }
    )
    
    print(f"Original document: {document.id}")
    print(f"Content: {document.content[:100]}...")
    
    # Create pipeline with custom processors
    pipeline = DocumentPipeline([
        TextNormalizationProcessor(TextNormalizationConfig(
            lowercase=True,
            remove_extra_whitespace=True,
            expand_contractions=True
        )),
        DocumentEnrichmentProcessor(DocumentEnrichmentConfig(
            add_statistics=True,
            extract_keywords=True,
            max_keywords=5
        )),
        DocumentSplitterProcessor(DocumentSplitterConfig(
            split_by="section",
            min_split_size=50
        ))
    ])
    
    # Process document through pipeline
    results = pipeline.process_document(document)
    
    print(f"\nPipeline processed document into {len(results)} final documents:")
    
    for doc in results:
        stage = doc.metadata.get('processing_stage', 'original')
        print(f"\n  Document: {doc.id} ({stage})")
        
        if stage == "normalized":
            print(f"    Normalized content: {doc.content[:80]}...")
        elif stage == "enriched":
            keywords = doc.metadata.get('keywords', [])
            word_count = doc.metadata.get('word_count', 'unknown')
            print(f"    Word count: {word_count}")
            print(f"    Keywords: {keywords}")
        elif stage == "split":
            split_idx = doc.metadata.get('split_index', '?')
            split_total = doc.metadata.get('split_total', '?')
            print(f"    Split {split_idx}/{split_total-1 if isinstance(split_total, int) else split_total}")
            print(f"    Content: {doc.content[:60]}...")
    
    # Show pipeline statistics
    stats = pipeline.get_pipeline_stats()
    print(f"\nPipeline Statistics:")
    print(f"  Documents processed: {stats['documents_processed']}")
    print(f"  Total time: {stats['total_pipeline_time']:.3f}s")
    print(f"  Processors:")
    for name, proc_stats in stats['processor_stats'].items():
        print(f"    {name}: {proc_stats['total_time']:.3f}s")
    
    return results


def main():
    """Run all custom processor examples"""
    print("Custom DocumentProcessor Examples")
    print("=" * 50)
    
    try:
        # Run examples
        norm_results = example_text_normalization()
        enrich_results = example_document_enrichment()
        split_results = example_document_splitting()
        pipeline_results = example_custom_pipeline()
        
        print("\n" + "=" * 50)
        print("✅ All custom processor examples completed successfully!")
        
        print(f"\nExample Summary:")
        print(f"  ✅ Text normalization: {len(norm_results)} documents")
        print(f"  ✅ Document enrichment: {len(enrich_results)} documents")
        print(f"  ✅ Document splitting: {len(split_results)} documents")
        print(f"  ✅ Custom pipeline: {len(pipeline_results)} documents")
        
        print(f"\nNext steps:")
        print(f"  - Create your own processor by inheriting from DocumentProcessor")
        print(f"  - Define a custom config class inheriting from DocumentProcessorConfig")
        print(f"  - Implement get_config_class() and process() methods")
        print(f"  - Try pipeline_example.py for complex processing workflows")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())