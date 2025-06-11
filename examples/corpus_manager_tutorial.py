"""
CorpusManager Complete Tutorial

This comprehensive tutorial demonstrates how to build a complete document processing
pipeline from raw files to searchable embeddings using CorpusManager.

このチュートリアルでは、CorpusManagerを使用して生のファイルから検索可能な
埋め込みまでの完全な文書処理パイプラインを構築する方法を説明します。
"""

import sys
import tempfile
import os
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag import (
    Document,
    CorpusManager,
    CorpusManagerConfig,
    LoadingConfig,
    ChunkingConfig,
    TFIDFEmbedder,
    TFIDFEmbeddingConfig,
    OpenAIEmbedder,
    OpenAIEmbeddingConfig,
    SQLiteDocumentStore
)

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_sample_documents(base_dir: Path) -> Dict[str, Path]:
    """Create sample documents for the tutorial
    チュートリアル用のサンプル文書を作成"""
    
    documents = {
        "ai_overview.md": """
# Artificial Intelligence Overview

## Introduction
Artificial Intelligence (AI) represents one of the most transformative technologies of our time.
It encompasses machine learning, deep learning, natural language processing, and computer vision.

## Key Technologies

### Machine Learning
Machine learning algorithms enable computers to learn from data without explicit programming.
Popular techniques include:
- Supervised learning with labeled datasets
- Unsupervised learning for pattern discovery
- Reinforcement learning through trial and error

### Deep Learning
Deep learning uses neural networks with multiple layers to process complex data.
Applications include:
- Image recognition and classification
- Natural language understanding
- Speech synthesis and recognition

## Applications
AI is transforming industries including healthcare, finance, transportation, and education.
""",

        "python_guide.txt": """
Python Programming for Data Science

Python has become the de facto language for data science and machine learning.
Its simple syntax and extensive ecosystem make it ideal for beginners and experts alike.

Essential Libraries:
- NumPy: Numerical computing with arrays
- Pandas: Data manipulation and analysis
- Scikit-learn: Machine learning algorithms
- TensorFlow/PyTorch: Deep learning frameworks
- Matplotlib/Seaborn: Data visualization

Getting Started:
1. Install Python and pip
2. Set up a virtual environment
3. Install required packages
4. Start with Jupyter notebooks

Best Practices:
- Write clean, readable code
- Use virtual environments
- Document your code thoroughly
- Follow PEP 8 style guidelines
- Test your code regularly

Python's versatility extends beyond data science to web development,
automation, scientific computing, and artificial intelligence research.
""",

        "data_analysis.txt": """
Data Analysis Fundamentals

Data analysis is the process of examining datasets to draw conclusions about the information they contain.
It involves collecting, cleaning, transforming, and modeling data to discover useful insights.

The Data Analysis Process:

1. Problem Definition
   - Clearly define the business question
   - Identify success metrics
   - Understand stakeholder requirements

2. Data Collection
   - Gather data from various sources
   - Ensure data quality and completeness
   - Document data sources and collection methods

3. Data Cleaning
   - Handle missing values
   - Remove duplicates
   - Fix inconsistencies
   - Standardize formats

4. Exploratory Data Analysis (EDA)
   - Understand data distribution
   - Identify patterns and trends
   - Detect outliers and anomalies
   - Visualize relationships

5. Statistical Analysis
   - Apply appropriate statistical tests
   - Calculate correlations and dependencies
   - Perform hypothesis testing
   - Build predictive models

6. Interpretation and Communication
   - Draw actionable insights
   - Create compelling visualizations
   - Present findings to stakeholders
   - Recommend next steps

Tools commonly used include Excel, R, Python, SQL, and specialized platforms like Tableau.
""",

        "machine_learning.json": """
{
  "title": "Machine Learning Concepts",
  "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
  "categories": [
    {
      "name": "Supervised Learning",
      "description": "Learning with labeled examples",
      "algorithms": ["Linear Regression", "Decision Trees", "Random Forest", "Support Vector Machines"]
    },
    {
      "name": "Unsupervised Learning", 
      "description": "Finding patterns in unlabeled data",
      "algorithms": ["K-Means Clustering", "Hierarchical Clustering", "PCA", "t-SNE"]
    },
    {
      "name": "Reinforcement Learning",
      "description": "Learning through interaction and rewards",
      "algorithms": ["Q-Learning", "Policy Gradient", "Actor-Critic", "Deep Q-Networks"]
    }
  ],
  "applications": [
    "Image recognition",
    "Natural language processing", 
    "Recommendation systems",
    "Fraud detection",
    "Autonomous vehicles",
    "Medical diagnosis"
  ]
}
""",

        "tech_trends.csv": """
Technology,Year,Adoption_Rate,Market_Size_Billion
Artificial Intelligence,2024,45%,150.2
Machine Learning,2024,38%,96.7
Cloud Computing,2024,87%,484.0
Internet of Things,2024,31%,537.4
Blockchain,2024,18%,67.3
Quantum Computing,2024,5%,1.3
Edge Computing,2024,22%,43.8
5G Networks,2024,29%,31.1
Augmented Reality,2024,15%,31.7
Virtual Reality,2024,12%,28.1
"""
    }
    
    # Create documents directory
    docs_dir = base_dir / "sample_documents"
    docs_dir.mkdir(exist_ok=True)
    
    # Write sample documents
    file_paths = {}
    for filename, content in documents.items():
        file_path = docs_dir / filename
        file_path.write_text(content.strip())
        file_paths[filename] = file_path
        
    logger.info(f"Created {len(file_paths)} sample documents in {docs_dir}")
    return file_paths


def tutorial_step_1_basic_usage():
    """Step 1: Basic CorpusManager Usage
    ステップ1：CorpusManagerの基本的な使用法"""
    
    print("=" * 80)
    print("STEP 1: Basic CorpusManager Usage")
    print("ステップ1：CorpusManagerの基本的な使用法")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample documents
        doc_paths = setup_sample_documents(temp_path)
        docs_dir = temp_path / "sample_documents"
        
        print(f"\n📁 Created sample documents:")
        for filename in doc_paths.keys():
            print(f"   - {filename}")
        
        # Create basic CorpusManager with default settings
        print(f"\n🔧 Creating CorpusManager with default configuration...")
        corpus_manager = CorpusManager()
        
        print(f"   ✅ CorpusManager initialized")
        print(f"   - Chunking: Enabled (512 tokens, 50 overlap)")
        print(f"   - Embedding: TF-IDF (auto-fit)")
        print(f"   - Storage: In-memory SQLite")
        
        # Process the entire directory
        print(f"\n🔄 Processing document corpus...")
        results = corpus_manager.process_corpus(docs_dir)
        
        # Display results
        print(f"\n📊 Processing Results:")
        print(f"   Documents loaded: {results['documents_loaded']}")
        print(f"   Documents processed: {results['documents_processed']}")
        print(f"   Documents embedded: {results['documents_embedded']}")
        print(f"   Processing time: {results['total_processing_time']:.2f}s")
        print(f"   Success: {results['success']}")
        
        # Show processing stages
        print(f"\n📈 Processing Stages:")
        for stage, count in results['processing_stages'].items():
            print(f"   {stage}: {count} documents")
        
        # Show embedder statistics
        embedder_stats = results.get('embedder_stats', {})
        if embedder_stats:
            print(f"\n🔤 Embedding Statistics:")
            print(f"   Total embeddings: {embedder_stats.get('total_embeddings', 0)}")
            print(f"   Average time per embedding: {embedder_stats.get('average_processing_time', 0):.4f}s")
            print(f"   Cache hit rate: {embedder_stats.get('cache_hit_rate', 0):.1%}")
        
        # Test basic search
        print(f"\n🔍 Testing Basic Search:")
        search_queries = ["machine learning", "python programming", "data analysis"]
        
        for query in search_queries:
            results_search = corpus_manager.search_documents(query, limit=2)
            print(f"   Query: '{query}' -> {len(results_search)} results")
        
        corpus_manager.cleanup()
        print(f"\n✅ Step 1 completed successfully!")
        return results


def tutorial_step_2_custom_configuration():
    """Step 2: Custom Configuration
    ステップ2：カスタム設定"""
    
    print("\n" + "=" * 80)
    print("STEP 2: Custom Configuration")
    print("ステップ2：カスタム設定")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample documents
        doc_paths = setup_sample_documents(temp_path)
        docs_dir = temp_path / "sample_documents"
        
        print(f"\n🔧 Creating custom configuration...")
        
        # Create custom configuration
        config = CorpusManagerConfig(
            # Document loading configuration
            loading_config=LoadingConfig(
                auto_detect_encoding=True,
                ignore_errors=True
            ),
            
            # Chunking configuration  
            enable_chunking=True,
            chunking_config=ChunkingConfig(
                chunk_size=256,          # Smaller chunks for better granularity
                overlap=32,              # Moderate overlap
                split_by_sentence=True,  # Respect sentence boundaries
                min_chunk_size=50        # Minimum viable chunk size
            ),
            
            # Embedding configuration
            enable_embedding=True,
            embedder=TFIDFEmbedder(TFIDFEmbeddingConfig(
                max_features=5000,       # Larger vocabulary
                min_df=1,                # Include rare terms
                max_df=0.95,             # Exclude very common terms
                ngram_range=(1, 3),      # Unigrams, bigrams, and trigrams
                remove_stopwords=True
            )),
            auto_fit_embedder=True,
            
            # Processing options
            batch_size=10,
            enable_progress_reporting=True,
            progress_interval=2,
            
            # Error handling
            fail_on_error=False,
            max_errors=5
        )
        
        print(f"   ✅ Custom configuration created:")
        print(f"   - Chunk size: {config.chunking_config.chunk_size} tokens")
        print(f"   - Overlap: {config.chunking_config.overlap} tokens") 
        print(f"   - Max features: {config.embedder.config.max_features}")
        print(f"   - N-gram range: {config.embedder.config.ngram_range}")
        print(f"   - Batch size: {config.batch_size}")
        
        # Create CorpusManager with custom config
        corpus_manager = CorpusManager(config)
        
        print(f"\n🔄 Processing with custom configuration...")
        results = corpus_manager.process_corpus(docs_dir)
        
        # Display detailed results
        print(f"\n📊 Custom Processing Results:")
        print(f"   Documents loaded: {results['documents_loaded']}")
        print(f"   Documents processed: {results['documents_processed']}")
        print(f"   Documents embedded: {results['documents_embedded']}")
        print(f"   Processing time: {results['total_processing_time']:.2f}s")
        print(f"   Errors: {results['total_errors']}")
        
        # Show pipeline statistics if available
        pipeline_stats = results.get('pipeline_stats', {})
        if pipeline_stats:
            print(f"\n⚙️  Pipeline Statistics:")
            print(f"   Total pipeline time: {pipeline_stats.get('total_pipeline_time', 0):.3f}s")
            
            for processor_name, stats in pipeline_stats.get('processor_stats', {}).items():
                print(f"   {processor_name}:")
                print(f"     Documents processed: {stats.get('documents_processed', 0)}")
                print(f"     Documents created: {stats.get('documents_created', 0)}")
                print(f"     Processing time: {stats.get('total_time', 0):.3f}s")
        
        # Test enhanced search with better embeddings
        print(f"\n🔍 Testing Enhanced Search:")
        search_queries = [
            "artificial intelligence machine learning",
            "python data science programming",
            "statistical analysis visualization"
        ]
        
        for query in search_queries:
            results_search = corpus_manager.search_documents(query, limit=3)
            print(f"   Query: '{query}'")
            print(f"   Results: {len(results_search)} documents found")
        
        corpus_manager.cleanup()
        print(f"\n✅ Step 2 completed successfully!")
        return results


def tutorial_step_3_persistent_storage():
    """Step 3: Persistent Storage with Database
    ステップ3：データベースを使った永続ストレージ"""
    
    print("\n" + "=" * 80)
    print("STEP 3: Persistent Storage with Database")
    print("ステップ3：データベースを使った永続ストレージ")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample documents
        doc_paths = setup_sample_documents(temp_path)
        docs_dir = temp_path / "sample_documents"
        
        # Create database path
        db_path = temp_path / "corpus_database.db"
        
        print(f"\n🗄️  Setting up persistent database storage...")
        print(f"   Database path: {db_path}")
        
        # Create document store
        document_store = SQLiteDocumentStore(str(db_path))
        
        # Create configuration with persistent storage
        config = CorpusManagerConfig(
            # Chunking for better retrieval
            enable_chunking=True,
            chunking_config=ChunkingConfig(
                chunk_size=200,
                overlap=25,
                split_by_sentence=True
            ),
            
            # Custom embedder with model persistence
            enable_embedding=True,
            embedder=TFIDFEmbedder(TFIDFEmbeddingConfig(
                max_features=3000,
                min_df=1,
                ngram_range=(1, 2),
                model_path=str(temp_path / "tfidf_model.pkl"),
                auto_save_model=True
            )),
            
            # Persistent storage configuration
            document_store=document_store,
            store_intermediate_results=True,  # Store all processing stages
            
            # Progress tracking
            enable_progress_reporting=True,
            batch_size=5
        )
        
        corpus_manager = CorpusManager(config)
        
        print(f"   ✅ Configured persistent storage")
        print(f"   - Database: SQLite at {db_path}")
        print(f"   - Model persistence: Enabled")
        print(f"   - Intermediate results: Stored")
        
        print(f"\n🔄 Processing corpus with persistent storage...")
        results = corpus_manager.process_corpus(docs_dir)
        
        # Display results
        print(f"\n📊 Persistent Storage Results:")
        print(f"   Documents loaded: {results['documents_loaded']}")
        print(f"   Documents processed: {results['documents_processed']}")
        print(f"   Documents stored: {results['documents_stored']}")
        print(f"   Processing time: {results['total_processing_time']:.2f}s")
        
        # Check database contents
        print(f"\n🗄️  Database Contents:")
        total_docs = document_store.count_documents()
        print(f"   Total documents in database: {total_docs}")
        
        # Show documents by processing stage
        stages = ["original", "chunked", "embedded"]
        for stage in stages:
            try:
                stage_docs = document_store.search_by_metadata({"processing_stage": stage})
                print(f"   Documents in '{stage}' stage: {len(stage_docs)}")
            except Exception:
                print(f"   Documents in '{stage}' stage: Not available")
        
        # Test lineage tracking
        print(f"\n🔗 Testing Document Lineage:")
        if total_docs > 0:
            # Get first original document
            all_docs = document_store.list_documents(limit=1)
            if all_docs:
                first_doc = all_docs[0]
                lineage = corpus_manager.get_document_lineage(first_doc.id)
                print(f"   Original document: {first_doc.id}")
                print(f"   Lineage documents: {len(lineage)}")
                
                for doc in lineage[:3]:  # Show first 3
                    stage = doc.metadata.get('processing_stage', 'unknown')
                    print(f"     {doc.id} ({stage})")
        
        # Test model persistence
        print(f"\n💾 Testing Model Persistence:")
        model_path = temp_path / "tfidf_model.pkl"
        if model_path.exists():
            print(f"   ✅ TF-IDF model saved to: {model_path}")
            print(f"   Model size: {model_path.stat().st_size} bytes")
            
            # Test loading the model with a new embedder
            new_embedder = TFIDFEmbedder()
            new_embedder.load_model(str(model_path))
            print(f"   ✅ Model successfully loaded by new embedder")
            print(f"   Vocabulary size: {len(new_embedder.get_vocabulary())}")
        else:
            print(f"   ⚠️  Model file not found")
        
        # Test search on persistent data
        print(f"\n🔍 Testing Search on Persistent Data:")
        search_queries = ["machine learning algorithms", "python data analysis"]
        
        for query in search_queries:
            results_search = corpus_manager.search_documents(query, limit=2)
            print(f"   Query: '{query}' -> {len(results_search)} results")
            
            for i, result in enumerate(results_search[:1]):  # Show first result
                doc = result.document if hasattr(result, 'document') else result
                stage = doc.metadata.get('processing_stage', 'unknown')
                print(f"     Result {i+1}: {doc.id} ({stage})")
                print(f"       Content: {doc.content[:60]}...")
        
        corpus_manager.cleanup()
        print(f"\n✅ Step 3 completed successfully!")
        print(f"   Database file: {db_path} ({db_path.stat().st_size} bytes)")
        return results, str(db_path)


def tutorial_step_4_advanced_features():
    """Step 4: Advanced Features and Best Practices
    ステップ4：高度な機能とベストプラクティス"""
    
    print("\n" + "=" * 80)
    print("STEP 4: Advanced Features and Best Practices")
    print("ステップ4：高度な機能とベストプラクティス")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample documents
        doc_paths = setup_sample_documents(temp_path)
        docs_dir = temp_path / "sample_documents"
        
        print(f"\n🚀 Demonstrating advanced features...")
        
        # Test different embedders if available
        embedders_to_test = []
        
        # Always available: TF-IDF
        embedders_to_test.append(("TF-IDF", TFIDFEmbedder(TFIDFEmbeddingConfig(
            max_features=1000,
            min_df=1,
            ngram_range=(1, 2)
        ))))
        
        # Test OpenAI if available
        if os.getenv("OPENAI_API_KEY"):
            try:
                embedders_to_test.append(("OpenAI", OpenAIEmbedder(OpenAIEmbeddingConfig(
                    model_name="text-embedding-3-small",
                    batch_size=5
                ))))
                print(f"   ✅ OpenAI embedder available")
            except Exception as e:
                print(f"   ⚠️  OpenAI embedder not available: {e}")
        else:
            print(f"   ⚠️  OpenAI embedder skipped (no API key)")
        
        # Compare different embedders
        print(f"\n📊 Comparing Embedders:")
        
        for embedder_name, embedder in embedders_to_test:
            print(f"\n   Testing {embedder_name} Embedder:")
            
            try:
                config = CorpusManagerConfig(
                    enable_chunking=False,  # Skip chunking for clean comparison
                    embedder=embedder,
                    enable_progress_reporting=False
                )
                
                corpus_manager = CorpusManager(config)
                results = corpus_manager.process_corpus(docs_dir)
                
                embedder_stats = results.get('embedder_stats', {})
                
                print(f"     Documents embedded: {results['documents_embedded']}")
                print(f"     Total time: {results['total_processing_time']:.3f}s")
                print(f"     Avg time per doc: {embedder_stats.get('average_processing_time', 0):.4f}s")
                
                if embedder_name == "TF-IDF":
                    vocab_size = len(embedder.get_vocabulary()) if embedder.is_fitted() else 0
                    print(f"     Vocabulary size: {vocab_size}")
                elif embedder_name == "OpenAI":
                    print(f"     Embedding dimension: {embedder.get_embedding_dimension()}")
                
                corpus_manager.cleanup()
                
            except Exception as e:
                print(f"     ❌ Failed: {e}")
        
        # Demonstrate comprehensive statistics
        print(f"\n📈 Comprehensive Statistics Example:")
        
        config = CorpusManagerConfig(
            enable_chunking=True,
            chunking_config=ChunkingConfig(chunk_size=150, overlap=20),
            embedder=TFIDFEmbedder(TFIDFEmbeddingConfig(max_features=800)),
            enable_progress_reporting=True,
            batch_size=3
        )
        
        corpus_manager = CorpusManager(config)
        results = corpus_manager.process_corpus(docs_dir)
        
        # Get comprehensive statistics
        full_stats = corpus_manager.get_corpus_stats()
        
        print(f"   Processing Summary:")
        print(f"     Documents loaded: {full_stats.get('documents_loaded', 0)}")
        print(f"     Documents processed: {full_stats.get('documents_processed', 0)}")
        print(f"     Documents embedded: {full_stats.get('documents_embedded', 0)}")
        print(f"     Total errors: {full_stats.get('errors', 0)}")
        
        # Show processing efficiency
        total_time = full_stats.get('total_processing_time', 0)
        if total_time > 0:
            docs_per_second = full_stats.get('documents_loaded', 0) / total_time
            print(f"   Performance Metrics:")
            print(f"     Documents per second: {docs_per_second:.2f}")
            print(f"     Total processing time: {total_time:.3f}s")
        
        # Best practices summary
        print(f"\n💡 Best Practices Summary:")
        print(f"   ✅ Use appropriate chunk sizes (200-512 tokens)")
        print(f"   ✅ Enable intermediate result storage for debugging")
        print(f"   ✅ Configure batch processing for large datasets")
        print(f"   ✅ Enable progress reporting for long operations")
        print(f"   ✅ Use graceful error handling in production")
        print(f"   ✅ Save TF-IDF models for consistency")
        print(f"   ✅ Choose embedders based on use case:")
        print(f"       - TF-IDF: Fast, interpretable, good for keyword search")
        print(f"       - OpenAI: Semantic understanding, cross-domain similarity")
        
        corpus_manager.cleanup()
        print(f"\n✅ Step 4 completed successfully!")
        return results


def tutorial_step_5_production_example():
    """Step 5: Production-Ready Example
    ステップ5：本番環境での例"""
    
    print("\n" + "=" * 80)
    print("STEP 5: Production-Ready Example")
    print("ステップ5：本番環境での例")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create larger sample dataset
        doc_paths = setup_sample_documents(temp_path)
        docs_dir = temp_path / "sample_documents"
        
        # Production database and model paths
        db_path = temp_path / "production_corpus.db"
        model_path = temp_path / "production_tfidf.pkl"
        
        print(f"\n🏭 Setting up production-ready configuration...")
        
        # Production-ready configuration
        config = CorpusManagerConfig(
            # Robust loading
            loading_config=LoadingConfig(
                auto_detect_encoding=True,
                ignore_errors=True  # Don't fail on individual file errors
            ),
            
            # Optimized chunking for retrieval
            enable_chunking=True,
            chunking_config=ChunkingConfig(
                chunk_size=384,              # Good balance of context and granularity
                overlap=64,                  # Adequate overlap for continuity
                split_by_sentence=True,      # Preserve semantic boundaries
                min_chunk_size=100,          # Ensure meaningful content
                max_chunk_size=600           # Prevent overly large chunks
            ),
            
            # Production embedder with persistence
            enable_embedding=True,
            embedder=TFIDFEmbedder(TFIDFEmbeddingConfig(
                max_features=10000,          # Large vocabulary for comprehensive coverage
                min_df=2,                    # Filter out very rare terms
                max_df=0.85,                 # Filter out very common terms
                ngram_range=(1, 3),          # Include phrases up to 3 words
                remove_stopwords=True,
                model_path=str(model_path),
                auto_save_model=True
            )),
            auto_fit_embedder=True,
            
            # Persistent storage
            document_store=SQLiteDocumentStore(str(db_path)),
            store_intermediate_results=True,
            
            # Production processing settings
            batch_size=20,                   # Efficient batch processing
            enable_progress_reporting=True,
            progress_interval=5,
            
            # Robust error handling
            fail_on_error=False,             # Graceful degradation
            max_errors=100                   # Allow some failures
        )
        
        print(f"   ✅ Production configuration ready:")
        print(f"   - Database: {db_path}")
        print(f"   - Model: {model_path}")
        print(f"   - Chunk size: {config.chunking_config.chunk_size} tokens")
        print(f"   - Batch size: {config.batch_size}")
        print(f"   - Max features: {config.embedder.config.max_features}")
        
        # Initialize production corpus manager
        print(f"\n🚀 Initializing production corpus manager...")
        corpus_manager = CorpusManager(config)
        
        # Process corpus with detailed monitoring
        print(f"\n🔄 Processing corpus in production mode...")
        start_time = logger.info("Production processing started")
        
        results = corpus_manager.process_corpus(docs_dir)
        
        print(f"\n📊 Production Processing Results:")
        print(f"   Success: {results['success']}")
        print(f"   Documents loaded: {results['documents_loaded']}")
        print(f"   Documents processed: {results['documents_processed']}")
        print(f"   Documents embedded: {results['documents_embedded']}")
        print(f"   Documents stored: {results['documents_stored']}")
        print(f"   Total processing time: {results['total_processing_time']:.2f}s")
        print(f"   Errors encountered: {results['total_errors']}")
        
        # Production health checks
        print(f"\n🔍 Production Health Checks:")
        
        # Check database integrity
        document_store = corpus_manager._document_store
        total_docs = document_store.count_documents()
        print(f"   ✅ Database integrity: {total_docs} documents stored")
        
        # Check model persistence
        if Path(model_path).exists():
            model_size = Path(model_path).stat().st_size
            print(f"   ✅ Model persistence: {model_size} bytes saved")
        else:
            print(f"   ❌ Model persistence: Failed")
        
        # Performance metrics
        embedder_stats = results.get('embedder_stats', {})
        if embedder_stats:
            avg_time = embedder_stats.get('average_processing_time', 0)
            throughput = 1.0 / avg_time if avg_time > 0 else 0
            print(f"   📈 Performance: {throughput:.1f} embeddings/second")
            print(f"   💾 Cache efficiency: {embedder_stats.get('cache_hit_rate', 0):.1%}")
        
        # Test production search capabilities
        print(f"\n🔍 Production Search Testing:")
        
        search_scenarios = [
            ("Technical Query", "machine learning artificial intelligence"),
            ("Programming Query", "python data science programming"),
            ("Analysis Query", "statistical analysis data visualization"),
            ("Broad Query", "technology trends applications")
        ]
        
        for scenario_name, query in search_scenarios:
            try:
                search_results = corpus_manager.search_documents(query, limit=3)
                print(f"   {scenario_name}: '{query}'")
                print(f"     Results: {len(search_results)} documents found")
                
                # Show result quality indicators
                for i, result in enumerate(search_results[:1]):
                    doc = result.document if hasattr(result, 'document') else result
                    stage = doc.metadata.get('processing_stage', 'unknown')
                    content_preview = doc.content.replace('\n', ' ')[:50]
                    print(f"     Top result: {doc.id} ({stage})")
                    print(f"       Content: {content_preview}...")
                    
            except Exception as e:
                print(f"   ❌ {scenario_name}: Search failed - {e}")
        
        # Production monitoring summary
        print(f"\n📊 Production Monitoring Summary:")
        full_stats = corpus_manager.get_corpus_stats()
        
        print(f"   System Status: {'✅ Healthy' if results['success'] else '❌ Issues'}")
        print(f"   Total Documents: {full_stats.get('documents_loaded', 0)}")
        print(f"   Processing Success Rate: {((full_stats.get('documents_embedded', 0) / max(full_stats.get('documents_loaded', 1), 1)) * 100):.1f}%")
        print(f"   Error Rate: {((full_stats.get('errors', 0) / max(full_stats.get('documents_loaded', 1), 1)) * 100):.1f}%")
        
        # Cleanup
        corpus_manager.cleanup()
        
        print(f"\n✅ Step 5 completed successfully!")
        print(f"   Production assets:")
        print(f"     Database: {db_path} ({db_path.stat().st_size} bytes)")
        if Path(model_path).exists():
            print(f"     Model: {model_path} ({Path(model_path).stat().st_size} bytes)")
        
        return results, str(db_path), str(model_path)


def main():
    """Run the complete CorpusManager tutorial
    完全なCorpusManagerチュートリアルを実行"""
    
    print("🎓 CorpusManager Complete Tutorial")
    print("🎓 CorpusManager 完全チュートリアル")
    print("=" * 80)
    print()
    print("This tutorial demonstrates the complete document processing pipeline:")
    print("このチュートリアルでは、完全な文書処理パイプラインを説明します：")
    print()
    print("1. Basic Usage - 基本的な使用法")
    print("2. Custom Configuration - カスタム設定") 
    print("3. Persistent Storage - 永続ストレージ")
    print("4. Advanced Features - 高度な機能")
    print("5. Production Example - 本番環境の例")
    print()
    
    try:
        # Run tutorial steps
        step1_results = tutorial_step_1_basic_usage()
        step2_results = tutorial_step_2_custom_configuration()
        step3_results, db_path = tutorial_step_3_persistent_storage()
        step4_results = tutorial_step_4_advanced_features()
        step5_results, prod_db, prod_model = tutorial_step_5_production_example()
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 TUTORIAL COMPLETE!")
        print("🎉 チュートリアル完了！")
        print("=" * 80)
        
        print(f"\n📋 Summary of Steps Completed:")
        print(f"   ✅ Step 1: Basic usage with default settings")
        print(f"   ✅ Step 2: Custom configuration for optimization")
        print(f"   ✅ Step 3: Persistent storage with database")
        print(f"   ✅ Step 4: Advanced features and comparisons")
        print(f"   ✅ Step 5: Production-ready deployment")
        
        print(f"\n📊 Total Documents Processed Across All Steps:")
        total_processed = (
            step1_results.get('documents_loaded', 0) +
            step2_results.get('documents_loaded', 0) +
            step3_results.get('documents_loaded', 0) +
            step4_results.get('documents_loaded', 0) +
            step5_results.get('documents_loaded', 0)
        )
        print(f"   Documents: {total_processed}")
        
        print(f"\n🔧 Key Components Demonstrated:")
        print(f"   📁 Document Loading: Multiple file formats (txt, md, json, csv)")
        print(f"   🔄 Document Processing: Chunking and pipeline processing")
        print(f"   🔤 Text Embedding: TF-IDF and OpenAI embeddings")
        print(f"   🗄️  Storage: In-memory and persistent SQLite storage")
        print(f"   🔍 Search: Content-based document retrieval")
        print(f"   📊 Monitoring: Comprehensive statistics and health checks")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Implement QueryEngine for advanced semantic search")
        print(f"   2. Add vector storage for similarity-based retrieval")
        print(f"   3. Create evaluation metrics for RAG quality assessment")
        print(f"   4. Scale to larger document collections")
        print(f"   5. Deploy in production environment")
        
        print(f"\n📚 What You've Learned:")
        print(f"   • How to configure CorpusManager for different use cases")
        print(f"   • Best practices for document processing pipelines")
        print(f"   • Comparison of different embedding approaches")
        print(f"   • Production deployment considerations")
        print(f"   • Monitoring and health checking strategies")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Tutorial failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())