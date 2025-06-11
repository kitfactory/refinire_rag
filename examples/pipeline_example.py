"""
Advanced DocumentPipeline Example

This example demonstrates complex pipeline configurations, database integration,
and advanced features of the DocumentProcessor system.
"""

import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Type, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag import (
    Document,
    DocumentProcessor,
    DocumentProcessorConfig,
    DocumentPipeline,
    TokenBasedChunker,
    SentenceAwareChunker,
    ChunkingConfig,
    SQLiteDocumentStore
)


# Import custom processors from custom_processor_example
sys.path.insert(0, str(Path(__file__).parent))
try:
    from custom_processor_example import (
        TextNormalizationProcessor,
        TextNormalizationConfig,
        DocumentEnrichmentProcessor, 
        DocumentEnrichmentConfig,
        DocumentSplitterProcessor,
        DocumentSplitterConfig
    )
except ImportError:
    print("Custom processors not found. Please ensure custom_processor_example.py exists.")
    sys.exit(1)


def example_simple_pipeline():
    """Simple pipeline without database storage"""
    print("=== Simple Pipeline Example ===\n")
    
    # Create test document
    document = Document(
        id="simple_pipeline_001",
        content="""This is a simple pipeline test document. It contains multiple sentences
        that will be processed through our pipeline. The pipeline will normalize the text,
        add enrichment metadata, and then split it into manageable chunks.
        
        Each step of the pipeline adds value to the document processing workflow.
        The final result should be a set of processed documents ready for further analysis.""",
        metadata={
            "path": "/examples/simple_test.txt",
            "created_at": "2024-01-15T10:00:00Z",
            "file_type": ".txt",
            "size_bytes": 400,
            "category": "test"
        }
    )
    
    print(f"Original document: {document.id}")
    print(f"Content length: {len(document.content)} characters")
    print(f"Content preview: {document.content[:100]}...")
    
    # Create simple pipeline (no database)
    pipeline = DocumentPipeline(
        processors=[
            TextNormalizationProcessor(TextNormalizationConfig(
                lowercase=True,
                remove_extra_whitespace=True,
                expand_contractions=True
            )),
            TokenBasedChunker(ChunkingConfig(
                chunk_size=30,
                overlap=5,
                split_by_sentence=False
            ))
        ],
        document_store=None,  # No database
        store_intermediate_results=False
    )
    
    print(f"\nPipeline configuration:")
    for i, processor in enumerate(pipeline.processors):
        info = processor.get_processor_info()
        print(f"  {i+1}. {info['processor_class']}")
        print(f"     Config: {info['config']}")
    
    # Process document
    results = pipeline.process_document(document)
    
    print(f"\nPipeline Results:")
    print(f"  Total documents created: {len(results)}")
    
    for doc in results:
        stage = doc.metadata.get('processing_stage', 'original')
        print(f"    {doc.id} ({stage})")
        if stage == 'chunked':
            pos = doc.metadata.get('chunk_position', '?')
            total = doc.metadata.get('chunk_total', '?')
            tokens = doc.metadata.get('token_count', '?')
            print(f"      Position: {pos}/{total-1 if isinstance(total, int) else total}, Tokens: {tokens}")
    
    # Show pipeline statistics
    stats = pipeline.get_pipeline_stats()
    print(f"\nPipeline Statistics:")
    print(f"  Documents processed: {stats['documents_processed']}")
    print(f"  Total time: {stats['total_pipeline_time']:.3f}s")
    print(f"  Errors: {stats['errors']}")
    
    return results


def example_comprehensive_pipeline():
    """Comprehensive pipeline with all processing stages"""
    print("\n=== Comprehensive Pipeline Example ===\n")
    
    # Create a more complex test document
    document = Document(
        id="comprehensive_001",
        content="""# AI and Machine Learning Overview

Artificial Intelligence (AI) and Machine Learning (ML) are transformative technologies.
They're revolutionizing industries and changing how we approach problem-solving.

## Core Concepts

Machine learning involves training algorithms on data. The algorithms learn patterns
and can make predictions on new, unseen data. It's a subset of artificial intelligence
that focuses on the ability of machines to receive data and learn for themselves.

## Applications

Common applications include:
- Natural language processing
- Computer vision  
- Recommendation systems
- Predictive analytics
- Autonomous vehicles

## Future Directions

The field continues to evolve rapidly. New architectures like transformers have
enabled breakthrough capabilities in language understanding. We can expect continued
innovation in the coming years.""",
        metadata={
            "path": "/examples/ai_overview.md",
            "created_at": "2024-01-15T11:00:00Z",
            "file_type": ".md",
            "size_bytes": 1200,
            "category": "technical",
            "author": "AI Researcher"
        }
    )
    
    print(f"Document: {document.id}")
    print(f"Content length: {len(document.content)} characters")
    print(f"Metadata: {document.metadata}")
    
    # Create comprehensive pipeline
    pipeline = DocumentPipeline(
        processors=[
            TextNormalizationProcessor(TextNormalizationConfig(
                lowercase=True,
                remove_extra_whitespace=True,
                expand_contractions=True,
                preserve_line_breaks=False
            )),
            DocumentEnrichmentProcessor(DocumentEnrichmentConfig(
                add_statistics=True,
                extract_keywords=True,
                detect_language=True,
                analyze_readability=True,
                max_keywords=8
            )),
            DocumentSplitterProcessor(DocumentSplitterConfig(
                split_by="section",
                min_split_size=100,
                max_split_size=800
            )),
            SentenceAwareChunker(ChunkingConfig(
                chunk_size=40,
                overlap=8,
                split_by_sentence=True,
                min_chunk_size=15
            ))
        ],
        document_store=None,
        store_intermediate_results=False
    )
    
    print(f"\nComprehensive Pipeline (4 stages):")
    for i, processor in enumerate(pipeline.processors):
        print(f"  Stage {i+1}: {processor.__class__.__name__}")
    
    # Process document
    results = pipeline.process_document(document)
    
    print(f"\nProcessing Results:")
    print(f"  Final documents created: {len(results)}")
    
    # Group results by processing stage
    stages = {}
    for doc in results:
        stage = doc.metadata.get('processing_stage', 'original')
        if stage not in stages:
            stages[stage] = []
        stages[stage].append(doc)
    
    for stage, docs in stages.items():
        print(f"\n  {stage.title()} Stage: {len(docs)} documents")
        for doc in docs[:3]:  # Show first 3 documents
            if stage == 'enriched':
                keywords = doc.metadata.get('keywords', [])
                word_count = doc.metadata.get('word_count', '?')
                print(f"    {doc.id}: {word_count} words, keywords: {keywords[:3]}...")
            elif stage == 'split':
                split_idx = doc.metadata.get('split_index', '?')
                print(f"    {doc.id}: Split {split_idx}, content: {doc.content[:50]}...")
            elif stage == 'chunked':
                pos = doc.metadata.get('chunk_position', '?')
                tokens = doc.metadata.get('token_count', '?')
                print(f"    {doc.id}: Chunk {pos}, {tokens} tokens")
        
        if len(docs) > 3:
            print(f"    ... and {len(docs) - 3} more documents")
    
    # Show detailed statistics
    stats = pipeline.get_pipeline_stats()
    print(f"\nDetailed Pipeline Statistics:")
    print(f"  Documents processed: {stats['documents_processed']}")
    print(f"  Total pipeline time: {stats['total_pipeline_time']:.3f}s")
    print(f"  Errors: {stats['errors']}")
    print(f"  Processor performance:")
    
    for processor_name, proc_stats in stats['processor_stats'].items():
        print(f"    {processor_name}:")
        print(f"      Time: {proc_stats['total_time']:.3f}s")
        print(f"      Documents processed: {proc_stats['documents_processed']}")
        print(f"      Documents created: {proc_stats['documents_created']}")
        print(f"      Errors: {proc_stats['errors']}")
    
    return results


def example_database_pipeline():
    """Pipeline with database storage and lineage tracking"""
    print("\n=== Database Pipeline Example ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "pipeline_example.db"
        doc_store = SQLiteDocumentStore(str(db_path))
        
        try:
            # Create documents to process
            documents = [
                Document(
                    id="db_doc_001",
                    content="""Deep learning is a subset of machine learning that uses neural networks.
                    These networks have multiple layers that can learn complex patterns in data.
                    Applications include image recognition, natural language processing, and game playing.""",
                    metadata={
                        "path": "/docs/deep_learning.txt",
                        "created_at": "2024-01-15T09:00:00Z",
                        "file_type": ".txt",
                        "size_bytes": 300,
                        "topic": "deep_learning"
                    }
                ),
                Document(
                    id="db_doc_002", 
                    content="""Natural language processing (NLP) enables computers to understand human language.
                    Modern NLP uses transformer architectures and large language models.
                    Applications include translation, sentiment analysis, and text generation.""",
                    metadata={
                        "path": "/docs/nlp.txt",
                        "created_at": "2024-01-15T09:30:00Z",
                        "file_type": ".txt",
                        "size_bytes": 280,
                        "topic": "nlp"
                    }
                )
            ]
            
            print(f"Processing {len(documents)} documents with database storage:")
            for doc in documents:
                print(f"  {doc.id}: {doc.metadata.get('topic')} ({len(doc.content)} chars)")
            
            # Create pipeline with database storage
            pipeline = DocumentPipeline(
                processors=[
                    TextNormalizationProcessor(TextNormalizationConfig(
                        lowercase=True,
                        remove_extra_whitespace=True
                    )),
                    DocumentEnrichmentProcessor(DocumentEnrichmentConfig(
                        add_statistics=True,
                        extract_keywords=True,
                        max_keywords=5
                    )),
                    TokenBasedChunker(ChunkingConfig(
                        chunk_size=25,
                        overlap=5,
                        split_by_sentence=True
                    ))
                ],
                document_store=doc_store,
                store_intermediate_results=True  # Store all intermediate results
            )
            
            print(f"\nDatabase Pipeline Configuration:")
            print(f"  Database: {db_path}")
            print(f"  Store intermediate results: Yes")
            print(f"  Processors: {len(pipeline.processors)}")
            
            # Process all documents
            all_results = []
            for document in documents:
                print(f"\nProcessing {document.id}...")
                results = pipeline.process_document(document)
                all_results.extend(results)
                print(f"  Created {len(results)} final documents")
            
            print(f"\nDatabase Contents:")
            print(f"  Total documents in database: {doc_store.count_documents()}")
            
            # Test lineage tracking
            print(f"\nLineage Tracking:")
            for original_doc in documents:
                lineage_docs = doc_store.get_documents_by_lineage(original_doc.id)
                print(f"  {original_doc.id} lineage: {len(lineage_docs)} documents")
                
                stages = {}
                for doc in lineage_docs:
                    stage = doc.metadata.get('processing_stage', 'original')
                    stages[stage] = stages.get(stage, 0) + 1
                
                for stage, count in stages.items():
                    print(f"    {stage}: {count} documents")
            
            # Test metadata search
            print(f"\nMetadata Search Examples:")
            
            # Find enriched documents
            enriched_results = doc_store.search_by_metadata({"processing_stage": "enriched"})
            print(f"  Enriched documents: {len(enriched_results)}")
            
            # Find documents by topic
            nlp_results = doc_store.search_by_metadata({"topic": "nlp"})
            print(f"  NLP topic documents: {len(nlp_results)}")
            
            # Find chunked documents with high token count
            chunked_results = doc_store.search_by_metadata({"processing_stage": "chunked"})
            high_token_chunks = [r for r in chunked_results 
                               if r.document.metadata.get('token_count', 0) > 20]
            print(f"  High-token chunks (>20): {len(high_token_chunks)}")
            
            # Show some example enriched metadata
            if enriched_results:
                example_doc = enriched_results[0].document
                print(f"\nExample enriched metadata ({example_doc.id}):")
                enrichment_keys = ['word_count', 'keywords', 'detected_language', 'readability_score']
                for key in enrichment_keys:
                    if key in example_doc.metadata:
                        value = example_doc.metadata[key]
                        print(f"    {key}: {value}")
            
            # Pipeline statistics
            stats = pipeline.get_pipeline_stats()
            print(f"\nFinal Pipeline Statistics:")
            print(f"  Total documents processed: {stats['documents_processed']}")
            print(f"  Total processing time: {stats['total_pipeline_time']:.3f}s")
            print(f"  Average time per document: {stats['total_pipeline_time'] / len(documents):.3f}s")
            
            return all_results
            
        finally:
            doc_store.close()


def example_error_recovery_pipeline():
    """Pipeline with error handling and recovery"""
    print("\n=== Error Recovery Pipeline Example ===\n")
    
    # Create a custom processor that might fail
    @dataclass
    class UnreliableProcessorConfig(DocumentProcessorConfig):
        fail_probability: float = 0.5
        simulate_errors: bool = True
    
    class UnreliableProcessor(DocumentProcessor):
        @classmethod
        def get_config_class(cls) -> Type[UnreliableProcessorConfig]:
            return UnreliableProcessorConfig
        
        def process(self, document: Document, config: Optional[UnreliableProcessorConfig] = None) -> List[Document]:
            proc_config = config or self.config
            
            if proc_config.simulate_errors and proc_config.fail_probability > 0.3:
                # Simulate occasional failures
                if "error" in document.content.lower():
                    raise ValueError(f"Simulated processing error for {document.id}")
            
            # Normal processing
            result_doc = Document(
                id=f"{document.id}_unreliable",
                content=f"Processed: {document.content}",
                metadata={
                    **document.metadata,
                    "parent_document_id": document.id,
                    "processing_stage": "unreliable_processed"
                }
            )
            return [result_doc]
    
    # Create test documents (some will trigger errors)
    documents = [
        Document(
            id="good_doc_001",
            content="This is a good document that will process successfully.",
            metadata={"type": "good"}
        ),
        Document(
            id="error_doc_001", 
            content="This document contains an error keyword that will trigger failure.",
            metadata={"type": "error"}
        ),
        Document(
            id="good_doc_002",
            content="Another good document for successful processing.",
            metadata={"type": "good"}
        )
    ]
    
    print(f"Test documents:")
    for doc in documents:
        doc_type = doc.metadata.get('type')
        print(f"  {doc.id} ({doc_type}): {doc.content[:50]}...")
    
    # Test with fail_on_error=False (graceful handling)
    print(f"\n--- Test 1: Graceful Error Handling (fail_on_error=False) ---")
    
    pipeline_graceful = DocumentPipeline(
        processors=[
            UnreliableProcessor(UnreliableProcessorConfig(
                fail_on_error=False,  # Don't stop pipeline on errors
                simulate_errors=True
            )),
            TokenBasedChunker(ChunkingConfig(
                chunk_size=15,
                overlap=3,
                fail_on_error=False
            ))
        ],
        document_store=None,
        store_intermediate_results=False
    )
    
    all_graceful_results = []
    for document in documents:
        print(f"\nProcessing {document.id}...")
        try:
            results = pipeline_graceful.process_document(document)
            all_graceful_results.extend(results)
            print(f"  ✅ Success: {len(results)} documents created")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Show results
    print(f"\nGraceful handling results:")
    print(f"  Total successful documents: {len(all_graceful_results)}")
    
    # Show error statistics
    stats = pipeline_graceful.get_pipeline_stats()
    print(f"  Pipeline errors: {stats['errors']}")
    for proc_name, proc_stats in stats['processor_stats'].items():
        if proc_stats['errors'] > 0:
            print(f"    {proc_name}: {proc_stats['errors']} errors")
    
    # Test with fail_on_error=True (strict handling)
    print(f"\n--- Test 2: Strict Error Handling (fail_on_error=True) ---")
    
    pipeline_strict = DocumentPipeline(
        processors=[
            UnreliableProcessor(UnreliableProcessorConfig(
                fail_on_error=True,  # Stop pipeline on errors
                simulate_errors=True
            )),
            TokenBasedChunker(ChunkingConfig(
                chunk_size=15,
                overlap=3,
                fail_on_error=True
            ))
        ],
        document_store=None,
        store_intermediate_results=False
    )
    
    successful_strict = 0
    failed_strict = 0
    
    for document in documents:
        print(f"\nProcessing {document.id}...")
        try:
            results = pipeline_strict.process_document(document)
            successful_strict += 1
            print(f"  ✅ Success: {len(results)} documents created")
        except Exception as e:
            failed_strict += 1
            print(f"  ❌ Failed with exception: {type(e).__name__}: {e}")
    
    print(f"\nStrict handling results:")
    print(f"  Successful documents: {successful_strict}")
    print(f"  Failed documents: {failed_strict}")
    
    print(f"\nComparison:")
    print(f"  Graceful handling: {len(all_graceful_results)} final documents, {stats['errors']} errors handled")
    print(f"  Strict handling: {successful_strict} successful, {failed_strict} failed")
    print(f"  Recommendation: Use graceful handling for production pipelines")


def main():
    """Run all pipeline examples"""
    print("Advanced DocumentPipeline Examples")
    print("=" * 60)
    
    try:
        # Run examples
        simple_results = example_simple_pipeline()
        comprehensive_results = example_comprehensive_pipeline()
        database_results = example_database_pipeline()
        example_error_recovery_pipeline()
        
        print("\n" + "=" * 60)
        print("✅ All pipeline examples completed successfully!")
        
        print(f"\nExample Summary:")
        print(f"  ✅ Simple pipeline: {len(simple_results)} final documents")
        print(f"  ✅ Comprehensive pipeline: {len(comprehensive_results)} final documents")
        print(f"  ✅ Database pipeline: {len(database_results)} final documents")
        print(f"  ✅ Error recovery pipeline: Demonstrated graceful vs strict handling")
        
        print(f"\nKey Features Demonstrated:")
        print(f"  📝 Multiple processor stages in sequence")
        print(f"  🗄️  Database storage with lineage tracking")
        print(f"  📊 Comprehensive pipeline statistics")
        print(f"  🔍 Metadata search and filtering")
        print(f"  ⚠️  Error handling and recovery strategies")
        print(f"  🔧 Dynamic processor configuration")
        
        print(f"\nNext Steps:")
        print(f"  - Experiment with different processor combinations")
        print(f"  - Try custom metadata search queries")
        print(f"  - Implement your own custom processors")
        print(f"  - Scale up to larger document collections")
        
    except Exception as e:
        print(f"\n❌ Pipeline example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())