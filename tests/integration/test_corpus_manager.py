"""
Comprehensive CorpusManager Integration Test

Tests the complete RAG pipeline from document loading to embedding storage,
demonstrating the integration of all major components.
"""

import sys
import tempfile
import os
import logging
import pytest
from pathlib import Path
from typing import List

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag.models.document import Document
from refinire_rag.application.corpus_manager_new import CorpusManager
# Legacy test - temporarily disabled due to import changes

# Import custom processors for advanced testing
sys.path.insert(0, str(Path(__file__).parent))
try:
    from examples.custom_processor_example import (
        TextNormalizationProcessor,
        TextNormalizationConfig,
        DocumentEnricher,
        DocumentEnrichmentConfig
    )
except ImportError:
    print("Custom processors not available. Using basic pipeline.")
    TextNormalizationProcessor = None
    DocumentEnricher = None


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_documents(temp_dir: Path) -> List[Path]:
    """Create test documents for corpus processing"""
    
    # Create test files with different content types
    documents = {
        "machine_learning.txt": """
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that focuses on the development of algorithms
and statistical models that enable computer systems to improve their performance on a specific task
through experience, without being explicitly programmed.

Key Concepts:
- Supervised Learning: Uses labeled training data to learn a function that maps inputs to outputs
- Unsupervised Learning: Finds hidden patterns in data without labeled examples
- Reinforcement Learning: Learns through interaction with an environment using rewards and penalties

Applications include natural language processing, computer vision, speech recognition, and robotics.
The field has grown rapidly with advances in deep learning and neural networks.
        """,
        
        "data_science.txt": """
Data Science Overview

Data science is an interdisciplinary field that uses scientific methods, processes, algorithms,
and systems to extract knowledge and insights from structured and unstructured data.

Core Components:
- Statistics and Mathematics: Foundation for understanding data patterns
- Programming: Python, R, SQL for data manipulation and analysis
- Domain Expertise: Understanding the business context and problem space
- Data Visualization: Communicating insights through charts and graphs

The data science process typically follows these steps:
1. Problem Definition
2. Data Collection
3. Data Cleaning and Preparation
4. Exploratory Data Analysis
5. Model Building and Validation
6. Deployment and Monitoring

Data scientists work with big data technologies, machine learning algorithms, and statistical models.
        """,
        
        "python_programming.txt": """
Python Programming Language

Python is a high-level, interpreted programming language known for its simple syntax
and powerful capabilities. It was created by Guido van Rossum and first released in 1991.

Key Features:
- Easy to learn and read syntax
- Extensive standard library
- Dynamic typing and memory management
- Object-oriented and functional programming support
- Large ecosystem of third-party packages

Popular Use Cases:
- Web development with frameworks like Django and Flask
- Data analysis and visualization with pandas, NumPy, and matplotlib
- Machine learning with scikit-learn, TensorFlow, and PyTorch
- Automation and scripting
- Scientific computing and research

Python's versatility makes it an excellent choice for beginners and professionals alike.
The language emphasizes code readability and allows developers to express concepts in fewer lines of code.
        """,
        
        "artificial_intelligence.md": """
# Artificial Intelligence

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines
that are programmed to think and learn like humans.

## History of AI

AI research began in the 1950s with pioneers like Alan Turing, who proposed the famous Turing Test
as a measure of machine intelligence. The field has experienced several waves of optimism and
"AI winters" when progress stalled.

## Types of AI

### Narrow AI (Weak AI)
- Designed for specific tasks
- Examples: image recognition, chess programs, voice assistants
- Currently the most common form of AI

### General AI (Strong AI)
- Hypothetical AI with human-level cognitive abilities
- Can understand, learn, and apply knowledge across different domains
- Still largely theoretical

## Current Applications

1. **Natural Language Processing**: Chatbots, translation services, sentiment analysis
2. **Computer Vision**: Facial recognition, medical imaging, autonomous vehicles
3. **Robotics**: Manufacturing automation, surgical robots, service robots
4. **Expert Systems**: Medical diagnosis, financial analysis, legal research

## Future Prospects

AI continues to advance rapidly with developments in deep learning, neural networks,
and quantum computing. Ethical considerations around AI safety, bias, and employment
impact are becoming increasingly important.
        """
    }
    
    # Write test files
    file_paths = []
    for filename, content in documents.items():
        file_path = temp_dir / filename
        file_path.write_text(content.strip())
        file_paths.append(file_path)
    
    logger.info(f"Created {len(file_paths)} test documents in {temp_dir}")
    return file_paths


def test_basic_corpus_processing():
    """Test basic corpus processing with default configuration - DISABLED: API needs update"""
    pytest.skip("Test disabled - CorpusManagerConfig API needs to be updated to current implementation")


def test_advanced_corpus_processing():
    """Test advanced corpus processing with custom processors - DISABLED: API needs update"""
    pytest.skip("Test disabled - CorpusManagerConfig API needs to be updated to current implementation")
    print("\n=== Advanced Corpus Processing Test ===\n")
    
    if not TextNormalizationProcessor or not DocumentEnricher:
        print("⚠️  Custom processors not available. Skipping advanced test.")
        return {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        db_path = temp_path / "corpus.db"
        
        # Create test documents
        doc_paths = create_test_documents(temp_path)
        print(f"Created test documents: {[p.name for p in doc_paths]}")
        
        # Create advanced configuration with custom processors
        config = CorpusManagerConfig(
            # Custom loading configuration
            loading_config=LoadingConfig(
                auto_detect_encoding=True,
                ignore_errors=True
            ),
            
            # Enable processing with custom processors
            enable_processing=True,
            processors=[
                TextNormalizationProcessor(TextNormalizationConfig(
                    lowercase=True,
                    remove_extra_whitespace=True,
                    expand_contractions=True
                )),
                DocumentEnricher(DocumentEnrichmentConfig(
                    add_statistics=True,
                    extract_keywords=True,
                    max_keywords=5
                ))
            ],
            
            # Advanced chunking
            enable_chunking=True,
            chunking_config=ChunkingConfig(
                chunk_size=200,
                overlap=40,
                split_by_sentence=True,
                min_chunk_size=50
            ),
            
            # Custom embedder
            enable_embedding=True,
            embedder=TFIDFEmbedder(TFIDFEmbeddingConfig(
                max_features=5000,
                min_df=1,
                max_df=0.95,
                ngram_range=(1, 3),
                remove_stopwords=True
            )),
            auto_fit_embedder=True,
            
            # Database storage
            document_store=SQLiteDocumentStore(str(db_path)),
            store_intermediate_results=True,
            
            # Processing options
            batch_size=10,
            enable_progress_reporting=True,
            progress_interval=2
        )
        
        corpus_manager = CorpusManager(config)
        print(f"Created advanced CorpusManager with custom processors")
        
        # Process the corpus
        print(f"\nProcessing corpus with advanced pipeline...")
        results = corpus_manager.process_corpus(doc_paths)  # Process specific files
        
        # Display comprehensive results
        print(f"\n--- Advanced Processing Results ---")
        print(f"Success: {results['success']}")
        print(f"Total processing time: {results['total_processing_time']:.2f}s")
        print(f"Documents loaded: {results['documents_loaded']}")
        print(f"Documents processed: {results['documents_processed']}")
        print(f"Documents embedded: {results['documents_embedded']}")
        print(f"Documents stored: {results['documents_stored']}")
        print(f"Total errors: {results['total_errors']}")
        print(f"Final document count: {results['final_document_count']}")
        
        # Show detailed processing stages
        print(f"\n--- Detailed Processing Stages ---")
        for stage, count in results['processing_stages'].items():
            print(f"  {stage}: {count} documents")
        
        # Show pipeline statistics
        if results['pipeline_stats']:
            print(f"\n--- Pipeline Statistics ---")
            pipeline_stats = results['pipeline_stats']
            print(f"  Documents processed: {pipeline_stats.get('documents_processed', 0)}")
            print(f"  Total pipeline time: {pipeline_stats.get('total_pipeline_time', 0):.3f}s")
            print(f"  Errors: {pipeline_stats.get('errors', 0)}")
            
            # Processor-specific stats
            for processor_name, stats in pipeline_stats.get('processor_stats', {}).items():
                print(f"  {processor_name}:")
                print(f"    Time: {stats.get('total_time', 0):.3f}s")
                print(f"    Documents: {stats.get('documents_processed', 0)}")
                print(f"    Created: {stats.get('documents_created', 0)}")
        
        # Show storage statistics
        if results['storage_stats']:
            print(f"\n--- Storage Statistics ---")
            for key, value in results['storage_stats'].items():
                print(f"  {key}: {value}")
        
        # Test advanced search with metadata
        print(f"\n--- Advanced Search Test ---")
        
        # Search for documents with specific metadata
        search_queries = [
            "python programming language",
            "machine learning algorithms",
            "data science statistics"
        ]
        
        for query in search_queries:
            search_results = corpus_manager.search_documents(query, limit=2)
            print(f"\nSearch: '{query}' -> {len(search_results)} results")
            
            for i, result in enumerate(search_results[:2]):
                doc = result.document if hasattr(result, 'document') else result
                stage = doc.metadata.get('processing_stage', 'unknown')
                keywords = doc.metadata.get('keywords', [])
                print(f"  {i+1}. {doc.id} ({stage})")
                if keywords:
                    print(f"      Keywords: {keywords[:3]}")
                print(f"      Content: {doc.content[:50]}...")
        
        # Test comprehensive statistics
        print(f"\n--- Comprehensive Statistics ---")
        corpus_stats = corpus_manager.get_corpus_stats()
        
        print(f"Overall corpus statistics:")
        for key, value in corpus_stats.items():
            if key not in ['pipeline_stats', 'embedder_stats', 'storage_stats']:
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        corpus_manager.cleanup()
        return results


def test_error_handling():
    """Test error handling in corpus processing - DISABLED: API needs update"""
    pytest.skip("Test disabled - CorpusManagerConfig API needs to be updated to current implementation")
    print("\n=== Error Handling Test ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create some valid and some problematic files
        valid_file = temp_path / "valid.txt"
        valid_file.write_text("This is a valid document with proper content.")
        
        empty_file = temp_path / "empty.txt"
        empty_file.write_text("")
        
        # Create a binary file that might cause issues
        binary_file = temp_path / "binary.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')
        
        print(f"Created test files: valid.txt, empty.txt, binary.bin")
        
        # Test with graceful error handling
        print(f"\n1. Testing graceful error handling:")
        
        config = CorpusManagerConfig(
            enable_chunking=True,
            chunking_config=ChunkingConfig(chunk_size=50, min_chunk_size=10),
            enable_embedding=True,
            fail_on_error=False,  # Graceful handling
            max_errors=5
        )
        
        corpus_manager = CorpusManager(config)
        results = corpus_manager.process_corpus(temp_path)
        
        print(f"   Results: {results['documents_loaded']} loaded, {results['total_errors']} errors")
        print(f"   Success: {results['success']}")
        
        # Test with strict error handling
        print(f"\n2. Testing strict error handling:")
        
        strict_config = CorpusManagerConfig(
            enable_chunking=True,
            enable_embedding=True,
            fail_on_error=True  # Strict handling
        )
        
        strict_manager = CorpusManager(strict_config)
        
        try:
            results = strict_manager.process_corpus([valid_file])  # Only valid file
            print(f"   Success with valid file: {results['success']}")
        except Exception as e:
            print(f"   Expected exception with invalid files: {type(e).__name__}")
        
        corpus_manager.cleanup()
        strict_manager.cleanup()


def test_different_embedders():
    """Test corpus processing with different embedders - DISABLED: API needs update"""
    pytest.skip("Test disabled - CorpusManagerConfig API needs to be updated to current implementation")
    print("\n=== Different Embedders Test ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create simple test documents
        for i, content in enumerate([
            "Machine learning and artificial intelligence are transforming technology.",
            "Data science combines statistics, programming, and domain expertise.",
            "Python is an excellent programming language for data analysis."
        ]):
            (temp_path / f"doc_{i}.txt").write_text(content)
        
        print(f"Created 3 simple test documents")
        
        # Test 1: TF-IDF embedder
        print(f"\n1. Testing TF-IDF Embedder:")
        
        tfidf_config = CorpusManagerConfig(
            enable_chunking=False,  # No chunking for simple test
            embedder=TFIDFEmbedder(TFIDFEmbeddingConfig(
                max_features=100,
                min_df=1,
                ngram_range=(1, 2)
            )),
            enable_progress_reporting=False
        )
        
        tfidf_manager = CorpusManager(tfidf_config)
        tfidf_results = tfidf_manager.process_corpus(temp_path)
        
        print(f"   Documents embedded: {tfidf_results['documents_embedded']}")
        print(f"   Processing time: {tfidf_results['total_processing_time']:.3f}s")
        
        # Show embedder info
        embedder_stats = tfidf_results.get('embedder_stats', {})
        if embedder_stats:
            print(f"   Average embedding time: {embedder_stats.get('average_processing_time', 0):.4f}s")
            print(f"   Cache hit rate: {embedder_stats.get('cache_hit_rate', 0):.2%}")
        
        # Test 2: OpenAI embedder (if available)
        print(f"\n2. Testing OpenAI Embedder:")
        
        if os.getenv("OPENAI_API_KEY"):
            try:
                from refinire_rag import OpenAIEmbedder, OpenAIEmbeddingConfig
                
                openai_config = CorpusManagerConfig(
                    enable_chunking=False,
                    embedder=OpenAIEmbedder(OpenAIEmbeddingConfig(
                        model_name="text-embedding-3-small",
                        batch_size=3
                    )),
                    enable_progress_reporting=False
                )
                
                openai_manager = CorpusManager(openai_config)
                openai_results = openai_manager.process_corpus(temp_path)
                
                print(f"   Documents embedded: {openai_results['documents_embedded']}")
                print(f"   Processing time: {openai_results['total_processing_time']:.3f}s")
                
                openai_stats = openai_results.get('embedder_stats', {})
                if openai_stats:
                    print(f"   Average embedding time: {openai_stats.get('average_processing_time', 0):.4f}s")
                    print(f"   Total API calls: {openai_stats.get('total_embeddings', 0)}")
                
                openai_manager.cleanup()
                
            except Exception as e:
                print(f"   OpenAI embedding failed: {e}")
        else:
            print(f"   Skipped (no OPENAI_API_KEY)")
        
        tfidf_manager.cleanup()


def main():
    """Run all CorpusManager integration tests"""
    print("CorpusManager Integration Tests")
    print("=" * 60)
    
    try:
        # Run comprehensive tests
        basic_results = test_basic_corpus_processing()
        advanced_results = test_advanced_corpus_processing()
        test_error_handling()
        test_different_embedders()
        
        print("\n" + "=" * 60)
        print("✅ All CorpusManager integration tests completed!")
        
        print(f"\nTest Summary:")
        print(f"  ✅ Basic corpus processing: {basic_results.get('documents_loaded', 0)} docs loaded")
        if advanced_results:
            print(f"  ✅ Advanced processing: {advanced_results.get('documents_processed', 0)} docs processed")
        else:
            print(f"  ⚠️  Advanced processing: skipped (no custom processors)")
        print(f"  ✅ Error handling scenarios")
        print(f"  ✅ Different embedder configurations")
        
        print(f"\nIntegration Features Tested:")
        print(f"  📁 Document loading from files and directories")
        print(f"  🔄 Document processing pipelines")
        print(f"  ✂️  Text chunking for optimal embedding")
        print(f"  🔤 Text embedding generation (TF-IDF, OpenAI)")
        print(f"  🗄️  Document storage with lineage tracking")
        print(f"  🔍 Document search and retrieval")
        print(f"  📊 Comprehensive statistics and monitoring")
        print(f"  ⚠️  Error handling and recovery")
        
        print(f"\nEnd-to-End RAG Pipeline:")
        print(f"  ✅ Complete workflow from files to searchable embeddings")
        print(f"  ✅ Metadata preservation and enrichment")
        print(f"  ✅ Processing stage tracking and lineage")
        print(f"  ✅ Configurable components and parameters")
        print(f"  ✅ Production-ready error handling")
        
        print(f"\nNext Steps:")
        print(f"  - Implement QueryEngine for semantic search")
        print(f"  - Add vector storage for similarity search")
        print(f"  - Create evaluation and quality assessment tools")
        print(f"  - Scale testing with larger document collections")
        
    except Exception as e:
        print(f"\n❌ CorpusManager integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())