"""
Test for improved DocumentProcessor design where each processor defines its own config
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Type, List, Optional

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag import (
    Document,
    DocumentProcessor,
    DocumentProcessorConfig,
    DocumentPipeline,
    TokenBasedChunker,
    ChunkingConfig,
    SQLiteDocumentStore
)


# Example custom processor with its own config
@dataclass
class NormalizationConfig(DocumentProcessorConfig):
    """Custom configuration for text normalization
    テキスト正規化のカスタム設定"""
    
    lowercase: bool = True
    remove_punctuation: bool = False
    expand_contractions: bool = True
    language: str = "en"


class TextNormalizer(DocumentProcessor):
    """Example text normalizer with custom config
    カスタム設定を持つテキスト正規化の例"""
    
    @classmethod
    def get_config_class(cls) -> Type[NormalizationConfig]:
        """Get configuration class for this processor"""
        return NormalizationConfig
    
    def process(self, document: Document, config: Optional[NormalizationConfig] = None) -> List[Document]:
        """Normalize document text based on configuration"""
        # Use provided config or instance config
        norm_config = config or self.config
        
        content = document.content
        
        # Apply normalization based on config
        if norm_config.lowercase:
            content = content.lower()
        
        if norm_config.expand_contractions and norm_config.language == "en":
            # Simple contraction expansion
            content = content.replace("don't", "do not")
            content = content.replace("won't", "will not")
            content = content.replace("it's", "it is")
        
        # Create normalized document
        normalized_doc = Document(
            id=f"{document.id}_normalized",
            content=content,
            metadata={
                **document.metadata,
                "original_document_id": document.id,
                "parent_document_id": document.id,
                "processing_stage": "normalized",
                "normalization_config": norm_config.to_dict()
            }
        )
        
        return [normalized_doc]


# Another example processor
@dataclass
class EnrichmentConfig(DocumentProcessorConfig):
    """Configuration for document enrichment
    文書エンリッチメントの設定"""
    
    add_word_count: bool = True
    add_language_detection: bool = True
    add_readability_score: bool = False
    extract_keywords: bool = True
    max_keywords: int = 10


class DocumentEnricher(DocumentProcessor):
    """Example document enricher with metadata addition
    メタデータ追加を行う文書エンリッチャーの例"""
    
    @classmethod
    def get_config_class(cls) -> Type[EnrichmentConfig]:
        """Get configuration class for this processor"""
        return EnrichmentConfig
    
    def process(self, document: Document, config: Optional[EnrichmentConfig] = None) -> List[Document]:
        """Enrich document with additional metadata"""
        enrich_config = config or self.config
        
        # Create enriched document (same content, enhanced metadata)
        enriched_doc = Document(
            id=f"{document.id}_enriched",
            content=document.content,
            metadata=document.metadata.copy()
        )
        
        # Add metadata based on config
        if enrich_config.add_word_count:
            word_count = len(document.content.split())
            enriched_doc.metadata["word_count"] = word_count
        
        if enrich_config.add_language_detection:
            # Simple language detection (in real implementation, use proper library)
            if any(char in document.content for char in "あいうえお"):
                language = "ja"
            else:
                language = "en"
            enriched_doc.metadata["detected_language"] = language
        
        if enrich_config.extract_keywords:
            # Simple keyword extraction (in real implementation, use NLP)
            words = document.content.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Simple filter
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            enriched_doc.metadata["keywords"] = [k[0] for k in keywords[:enrich_config.max_keywords]]
        
        # Update metadata
        enriched_doc.metadata.update({
            "parent_document_id": document.id,
            "processing_stage": "enriched",
            "enrichment_config": enrich_config.to_dict()
        })
        
        return [enriched_doc]


def test_processor_config_system():
    """Test the improved processor configuration system"""
    print("=== Testing Improved Processor Configuration System ===\n")
    
    # 1. Test getting config class from processor
    print("1. Testing config class retrieval:")
    print(f"   TextNormalizer config class: {TextNormalizer.get_config_class().__name__}")
    print(f"   DocumentEnricher config class: {DocumentEnricher.get_config_class().__name__}")
    print(f"   TokenBasedChunker config class: {TokenBasedChunker.get_config_class().__name__}")
    
    # 2. Test default config creation
    print("\n2. Testing default config creation:")
    normalizer = TextNormalizer()
    print(f"   Normalizer default config: {normalizer.config}")
    
    enricher = DocumentEnricher()
    print(f"   Enricher default config: {enricher.config}")
    
    # 3. Test custom config
    print("\n3. Testing custom config:")
    custom_norm_config = NormalizationConfig(
        lowercase=False,
        expand_contractions=True,
        language="en"
    )
    custom_normalizer = TextNormalizer(custom_norm_config)
    print(f"   Custom normalizer config: {custom_normalizer.config}")
    
    # 4. Test processing with different configs
    print("\n4. Testing processing with different configs:")
    test_doc = Document(
        id="test_001",
        content="Don't worry, it's going to be fine! This TEST demonstrates CONFIG flexibility.",
        metadata={
            "path": "/test/doc.txt",
            "created_at": "2024-01-15T10:00:00Z",
            "file_type": ".txt",
            "size_bytes": 100
        }
    )
    
    # Process with default config (lowercase=True)
    default_result = normalizer.process(test_doc)
    print(f"   Default normalization: '{default_result[0].content[:50]}...'")
    
    # Process with custom config (lowercase=False)
    custom_result = custom_normalizer.process(test_doc)
    print(f"   Custom normalization: '{custom_result[0].content[:50]}...'")
    
    # 5. Test config validation
    print("\n5. Testing config validation:")
    try:
        # Try to use wrong config type
        wrong_config = ChunkingConfig(chunk_size=100)
        normalizer.process_with_stats(test_doc, wrong_config)
    except Exception as e:
        print(f"   ✓ Config validation caught wrong type (this is expected)")
    
    # 6. Test pipeline with mixed processors
    print("\n6. Testing pipeline with mixed processors:")
    
    pipeline = DocumentPipeline([
        TextNormalizer(NormalizationConfig(lowercase=True, expand_contractions=True)),
        DocumentEnricher(EnrichmentConfig(add_word_count=True, extract_keywords=True)),
        TokenBasedChunker(ChunkingConfig(chunk_size=30, overlap=5))
    ])
    
    results = pipeline.process_document(test_doc)
    print(f"   Pipeline produced {len(results)} documents:")
    for doc in results:
        stage = doc.metadata.get("processing_stage", "original")
        print(f"     - {doc.id} ({stage})")
    
    # 7. Show processor info
    print("\n7. Processor information:")
    for processor in pipeline.processors:
        info = processor.get_processor_info()
        print(f"   {info['processor_class']}:")
        print(f"     - Config class: {info['config_class']}")
        print(f"     - Config: {info['config']}")


def test_config_serialization():
    """Test config serialization and deserialization"""
    print("\n\n=== Testing Config Serialization ===\n")
    
    # Create a config
    original_config = ChunkingConfig(
        chunk_size=256,
        overlap=32,
        split_by_sentence=True,
        min_chunk_size=50
    )
    
    print(f"1. Original config: {original_config}")
    
    # Serialize to dict
    config_dict = original_config.to_dict()
    print(f"\n2. Serialized to dict: {config_dict}")
    
    # Deserialize from dict
    restored_config = ChunkingConfig.from_dict(config_dict)
    print(f"\n3. Restored config: {restored_config}")
    
    # Verify they match
    print(f"\n4. Configs match: {original_config == restored_config}")
    
    # Test with extra fields (should be filtered)
    dict_with_extra = {
        **config_dict,
        "extra_field": "should be ignored",
        "another_extra": 123
    }
    
    config_from_extra = ChunkingConfig.from_dict(dict_with_extra)
    print(f"\n5. Config from dict with extra fields: {config_from_extra}")
    print(f"   Still matches original: {config_from_extra == original_config}")


def test_dynamic_config_discovery():
    """Test dynamic discovery of processor configs"""
    print("\n\n=== Testing Dynamic Config Discovery ===\n")
    
    # Collect all processor classes
    processor_classes = [
        TextNormalizer,
        DocumentEnricher,
        TokenBasedChunker
    ]
    
    print("Discovered processor configurations:")
    for processor_class in processor_classes:
        config_class = processor_class.get_config_class()
        default_config = processor_class.get_default_config()
        
        print(f"\n{processor_class.__name__}:")
        print(f"  Config class: {config_class.__name__}")
        print(f"  Fields:")
        for field_name, field_info in config_class.__dataclass_fields__.items():
            default_value = getattr(default_config, field_name)
            print(f"    - {field_name}: {field_info.type.__name__} = {default_value}")


def main():
    """Run all improved processor tests"""
    print("Improved DocumentProcessor Configuration System Test")
    print("=" * 60)
    
    try:
        test_processor_config_system()
        test_config_serialization()
        test_dynamic_config_discovery()
        
        print("\n\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())