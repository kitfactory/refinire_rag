"""
Comprehensive DocumentProcessor test suite
"""

import sys
import tempfile
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


# Test processors with custom configs
@dataclass
class NormalizationConfig(DocumentProcessorConfig):
    """Configuration for text normalization"""
    lowercase: bool = True
    remove_punctuation: bool = False
    expand_contractions: bool = True
    language: str = "en"


class TextNormalizer(DocumentProcessor):
    """Text normalizer with custom config"""
    
    @classmethod
    def get_config_class(cls) -> Type[NormalizationConfig]:
        return NormalizationConfig
    
    def process(self, document: Document, config: Optional[NormalizationConfig] = None) -> List[Document]:
        norm_config = config or self.config
        
        content = document.content
        
        if norm_config.lowercase:
            content = content.lower()
        
        if norm_config.expand_contractions and norm_config.language == "en":
            content = content.replace("don't", "do not")
            content = content.replace("won't", "will not")
            content = content.replace("it's", "it is")
        
        normalized_doc = Document(
            id=f"{document.id}_normalized",
            content=content,
            metadata={
                **document.metadata,
                "original_document_id": document.id,
                "parent_document_id": document.id,
                "processing_stage": "normalized"
            }
        )
        
        return [normalized_doc]


@dataclass
class EnrichmentConfig(DocumentProcessorConfig):
    """Configuration for document enrichment"""
    add_word_count: bool = True
    add_language_detection: bool = True
    extract_keywords: bool = True
    max_keywords: int = 5


class DocumentEnricher(DocumentProcessor):
    """Document enricher with metadata addition"""
    
    @classmethod
    def get_config_class(cls) -> Type[EnrichmentConfig]:
        return EnrichmentConfig
    
    def process(self, document: Document, config: Optional[EnrichmentConfig] = None) -> List[Document]:
        enrich_config = config or self.config
        
        enriched_doc = Document(
            id=f"{document.id}_enriched",
            content=document.content,
            metadata=document.metadata.copy()
        )
        
        if enrich_config.add_word_count:
            word_count = len(document.content.split())
            enriched_doc.metadata["word_count"] = word_count
        
        if enrich_config.add_language_detection:
            if any(char in document.content for char in "あいうえお"):
                language = "ja"
            else:
                language = "en"
            enriched_doc.metadata["detected_language"] = language
        
        if enrich_config.extract_keywords:
            words = document.content.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            enriched_doc.metadata["keywords"] = [k[0] for k in keywords[:enrich_config.max_keywords]]
        
        enriched_doc.metadata.update({
            "parent_document_id": document.id,
            "processing_stage": "enriched"
        })
        
        return [enriched_doc]


def test_basic_processor_functionality():
    """Test basic DocumentProcessor functionality"""
    print("=== Test 1: Basic Processor Functionality ===")
    
    # Test document
    test_doc = Document(
        id="test_001",
        content="Don't worry, it's going to be fine! This test demonstrates processor functionality.",
        metadata={
            "path": "/test/doc.txt",
            "created_at": "2024-01-15T10:00:00Z",
            "file_type": ".txt",
            "size_bytes": 100
        }
    )
    
    # Test TextNormalizer
    print("\n1.1 Testing TextNormalizer:")
    normalizer = TextNormalizer()
    print(f"   Config class: {normalizer.get_config_class().__name__}")
    print(f"   Default config: {normalizer.config}")
    
    normalized_docs = normalizer.process_with_stats(test_doc)
    print(f"   Original: '{test_doc.content}'")
    print(f"   Normalized: '{normalized_docs[0].content}'")
    print(f"   Processing stats: {normalizer.get_processing_stats()}")
    
    # Test DocumentEnricher
    print("\n1.2 Testing DocumentEnricher:")
    enricher = DocumentEnricher()
    enriched_docs = enricher.process_with_stats(test_doc)
    
    print(f"   Added metadata:")
    for key, value in enriched_docs[0].metadata.items():
        if key not in test_doc.metadata:
            print(f"     {key}: {value}")
    
    return normalized_docs[0], enriched_docs[0]


def test_config_system():
    """Test the configuration system"""
    print("\n=== Test 2: Configuration System ===")
    
    # Test custom configurations
    print("\n2.1 Testing custom configurations:")
    
    custom_norm_config = NormalizationConfig(
        lowercase=False,
        expand_contractions=True
    )
    
    custom_enrich_config = EnrichmentConfig(
        add_word_count=True,
        extract_keywords=True,
        max_keywords=3
    )
    
    normalizer = TextNormalizer(custom_norm_config)
    enricher = DocumentEnricher(custom_enrich_config)
    
    print(f"   Custom normalizer config: lowercase={normalizer.config.lowercase}")
    print(f"   Custom enricher config: max_keywords={enricher.config.max_keywords}")
    
    # Test config validation
    print("\n2.2 Testing config validation:")
    test_doc = Document(
        id="test_002",
        content="Test document for config validation.",
        metadata={"path": "/test/config.txt", "created_at": "2024-01-15T11:00:00Z", "file_type": ".txt", "size_bytes": 50}
    )
    
    # Try wrong config type (should trigger validation)
    wrong_config = ChunkingConfig(chunk_size=100)
    print(f"   Trying wrong config type: {type(wrong_config).__name__}")
    
    result = normalizer.process_with_stats(test_doc, wrong_config)
    print(f"   Result: {len(result)} documents (should use default config due to validation)")
    
    # Test serialization
    print("\n2.3 Testing config serialization:")
    config_dict = custom_norm_config.to_dict()
    print(f"   Serialized: {config_dict}")
    
    restored_config = NormalizationConfig.from_dict(config_dict)
    print(f"   Restored: {restored_config}")
    print(f"   Configs match: {custom_norm_config == restored_config}")


def test_pipeline_functionality():
    """Test pipeline functionality without database"""
    print("\n=== Test 3: Pipeline Functionality ===")
    
    # Create processors with specific configs
    normalizer = TextNormalizer(NormalizationConfig(lowercase=True, expand_contractions=True))
    enricher = DocumentEnricher(EnrichmentConfig(add_word_count=True, extract_keywords=True, max_keywords=3))
    chunker = TokenBasedChunker(ChunkingConfig(chunk_size=20, overlap=5))
    
    # Create pipeline without document store
    pipeline = DocumentPipeline(
        processors=[normalizer, enricher, chunker],
        document_store=None,  # No database for this test
        store_intermediate_results=False
    )
    
    test_doc = Document(
        id="pipeline_test_001",
        content="Don't worry, it's going to work! This pipeline test demonstrates multiple processors working together. Each processor adds value to the document processing chain.",
        metadata={
            "path": "/test/pipeline.txt",
            "created_at": "2024-01-15T12:00:00Z",
            "file_type": ".txt",
            "size_bytes": 200
        }
    )
    
    print(f"\n3.1 Processing document through pipeline:")
    print(f"   Original: '{test_doc.content[:50]}...'")
    print(f"   Pipeline processors: {[p.__class__.__name__ for p in pipeline.processors]}")
    
    results = pipeline.process_document(test_doc)
    
    print(f"\n3.2 Pipeline results:")
    print(f"   Total documents produced: {len(results)}")
    
    for doc in results:
        stage = doc.metadata.get("processing_stage", "original")
        print(f"   - {doc.id} ({stage})")
        if stage == "enriched":
            print(f"     Word count: {doc.metadata.get('word_count', 'N/A')}")
            print(f"     Keywords: {doc.metadata.get('keywords', 'N/A')}")
        elif stage == "chunked":
            pos = doc.metadata.get("chunk_position", "?")
            total = doc.metadata.get("chunk_total", "?")
            print(f"     Chunk {pos}/{total-1 if isinstance(total, int) else total}: '{doc.content[:30]}...'")
    
    # Get pipeline stats
    stats = pipeline.get_pipeline_stats()
    print(f"\n3.3 Pipeline statistics:")
    print(f"   Documents processed: {stats['documents_processed']}")
    print(f"   Total time: {stats['total_pipeline_time']:.3f}s")
    print(f"   Errors: {stats['errors']}")
    
    for processor_name, processor_stats in stats['processor_stats'].items():
        print(f"   {processor_name}: {processor_stats['total_time']:.3f}s")
    
    return results


def test_pipeline_with_database():
    """Test pipeline with database storage"""
    print("\n=== Test 4: Pipeline with Database ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_pipeline.db"
        doc_store = SQLiteDocumentStore(str(db_path))
        
        try:
            # Create pipeline with database
            pipeline = DocumentPipeline(
                processors=[
                    TextNormalizer(NormalizationConfig(lowercase=True)),
                    DocumentEnricher(EnrichmentConfig(add_word_count=True, extract_keywords=True)),
                    TokenBasedChunker(ChunkingConfig(chunk_size=15, overlap=3))
                ],
                document_store=doc_store,
                store_intermediate_results=True
            )
            
            test_doc = Document(
                id="db_test_001",
                content="This is a database test. We will store all intermediate results. Each processing stage will be preserved for later analysis.",
                metadata={
                    "path": "/test/db_test.txt",
                    "created_at": "2024-01-15T13:00:00Z",
                    "file_type": ".txt",
                    "size_bytes": 150
                }
            )
            
            print(f"\n4.1 Processing with database storage:")
            results = pipeline.process_document(test_doc)
            
            print(f"   Pipeline results: {len(results)} documents")
            print(f"   Documents in database: {doc_store.count_documents()}")
            
            # Test lineage tracking
            print(f"\n4.2 Testing lineage tracking:")
            lineage_docs = doc_store.get_documents_by_lineage(test_doc.id)
            print(f"   Documents in lineage: {len(lineage_docs)}")
            
            for doc in lineage_docs:
                stage = doc.metadata.get("processing_stage", "original")
                print(f"   - {doc.id} ({stage})")
            
            # Test metadata search
            print(f"\n4.3 Testing metadata search:")
            enriched_docs = doc_store.search_by_metadata({"processing_stage": "enriched"})
            print(f"   Enriched documents: {len(enriched_docs)}")
            
            if enriched_docs:
                doc = enriched_docs[0].document
                print(f"   Word count: {doc.metadata.get('word_count')}")
                print(f"   Keywords: {doc.metadata.get('keywords')}")
            
            return results
            
        finally:
            doc_store.close()


def test_error_handling():
    """Test error handling in processors"""
    print("\n=== Test 5: Error Handling ===")
    
    class FailingProcessor(DocumentProcessor):
        """Processor that always fails for testing"""
        
        @classmethod
        def get_config_class(cls) -> Type[DocumentProcessorConfig]:
            return DocumentProcessorConfig
        
        def process(self, document: Document, config=None) -> List[Document]:
            raise ValueError("Intentional failure for testing")
    
    # Test with fail_on_error=False (default)
    print("\n5.1 Testing with fail_on_error=False:")
    failing_processor = FailingProcessor()
    
    test_doc = Document(
        id="error_test_001",
        content="This will cause an error.",
        metadata={"path": "/test/error.txt", "created_at": "2024-01-15T14:00:00Z", "file_type": ".txt", "size_bytes": 25}
    )
    
    results = failing_processor.process_with_stats(test_doc)
    print(f"   Results: {len(results)} documents (should be 0)")
    print(f"   Errors in stats: {failing_processor.get_processing_stats()['errors']}")
    
    # Test with fail_on_error=True
    print("\n5.2 Testing with fail_on_error=True:")
    strict_config = DocumentProcessorConfig(fail_on_error=True)
    strict_failing_processor = FailingProcessor(strict_config)
    
    try:
        strict_failing_processor.process_with_stats(test_doc)
        print("   ERROR: Should have raised an exception!")
    except ValueError as e:
        print(f"   ✓ Correctly raised exception: {e}")


def test_processor_info():
    """Test processor information gathering"""
    print("\n=== Test 6: Processor Information ===")
    
    processors = [
        TextNormalizer(NormalizationConfig(lowercase=False)),
        DocumentEnricher(EnrichmentConfig(max_keywords=10)),
        TokenBasedChunker(ChunkingConfig(chunk_size=100))
    ]
    
    print("\n6.1 Processor information:")
    for processor in processors:
        info = processor.get_processor_info()
        print(f"\n   {info['processor_class']}:")
        print(f"     Config class: {info['config_class']}")
        print(f"     Config summary:")
        for key, value in info['config'].items():
            if not key.startswith('preserve_') and not key.startswith('add_') and not key.startswith('fail_'):
                print(f"       {key}: {value}")


def main():
    """Run comprehensive DocumentProcessor tests"""
    print("Comprehensive DocumentProcessor Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        normalized_doc, enriched_doc = test_basic_processor_functionality()
        test_config_system()
        pipeline_results = test_pipeline_functionality()
        db_results = test_pipeline_with_database()
        test_error_handling()
        test_processor_info()
        
        print("\n" + "=" * 60)
        print("✅ All DocumentProcessor tests completed successfully!")
        
        print(f"\nTest Summary:")
        print(f"  ✅ Basic processor functionality")
        print(f"  ✅ Configuration system")
        print(f"  ✅ Pipeline functionality")
        print(f"  ✅ Database integration")
        print(f"  ✅ Error handling")
        print(f"  ✅ Processor information")
        
        print(f"\nResults:")
        print(f"  - Pipeline without DB: {len(pipeline_results)} documents")
        print(f"  - Pipeline with DB: {len(db_results)} documents")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())