"""
CorpusManager Quick Start Guide

A quick introduction to building a document processing pipeline with CorpusManager.
Get up and running in just a few minutes!

CorpusManagerを使用した文書処理パイプラインの構築のクイック入門ガイドです。
わずか数分で始められます！
"""

import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag import (
    CorpusManager,
    CorpusManagerConfig,
    ChunkingConfig,
    TFIDFEmbedder,
    TFIDFEmbeddingConfig
)


def quick_example_1_minimal():
    """Quick Example 1: Minimal Setup (30 seconds)
    クイック例1：最小セットアップ（30秒）"""
    
    print("🚀 Quick Example 1: Minimal Setup")
    print("🚀 クイック例1：最小セットアップ")
    print("-" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a simple text file
        sample_file = temp_path / "sample.txt"
        sample_file.write_text("""
        Machine learning is transforming how we process and understand data.
        Python has become the leading language for data science and AI development.
        Document processing pipelines help organize and search large text collections.
        """)
        
        print(f"📄 Created sample document: {sample_file.name}")
        
        # Use CorpusManager with default settings
        corpus_manager = CorpusManager()
        
        print(f"⚙️  Processing with default settings...")
        results = corpus_manager.process_corpus(sample_file)
        
        print(f"✅ Results:")
        print(f"   Documents loaded: {results['documents_loaded']}")
        print(f"   Documents embedded: {results['documents_embedded']}")
        print(f"   Processing time: {results['total_processing_time']:.2f}s")
        
        # Quick search test
        search_results = corpus_manager.search_documents("machine learning", limit=1)
        print(f"   Search test: {len(search_results)} results for 'machine learning'")
        
        corpus_manager.cleanup()
        print(f"🎉 Done! Your first corpus is processed.")


def quick_example_2_custom():
    """Quick Example 2: Custom Configuration (2 minutes)
    クイック例2：カスタム設定（2分）"""
    
    print("\n🔧 Quick Example 2: Custom Configuration")
    print("🔧 クイック例2：カスタム設定")
    print("-" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create multiple documents
        documents = {
            "ai_basics.txt": "Artificial intelligence enables machines to perform tasks that typically require human intelligence.",
            "python_tutorial.txt": "Python programming language is versatile and widely used in data science, web development, and automation.",
            "data_analysis.txt": "Data analysis involves examining datasets to draw conclusions and support decision-making processes."
        }
        
        for filename, content in documents.items():
            (temp_path / filename).write_text(content)
        
        print(f"📁 Created {len(documents)} documents")
        
        # Custom configuration for better results
        config = CorpusManagerConfig(
            # Better chunking for small documents
            chunking_config=ChunkingConfig(
                chunk_size=100,      # Smaller chunks
                overlap=20,          # Some overlap
                split_by_sentence=True
            ),
            
            # Custom embedder settings
            embedder=TFIDFEmbedder(TFIDFEmbeddingConfig(
                max_features=1000,   # Moderate vocabulary
                ngram_range=(1, 2),  # Include bigrams
                min_df=1             # Keep all terms
            )),
            
            # Processing options
            enable_progress_reporting=True
        )
        
        corpus_manager = CorpusManager(config)
        
        print(f"⚙️  Processing with custom configuration...")
        results = corpus_manager.process_corpus(temp_path)
        
        print(f"✅ Custom Results:")
        print(f"   Documents loaded: {results['documents_loaded']}")
        print(f"   Documents processed: {results['documents_processed']}")
        print(f"   Documents embedded: {results['documents_embedded']}")
        print(f"   Processing time: {results['total_processing_time']:.2f}s")
        
        # Test multiple searches
        queries = ["artificial intelligence", "python programming", "data analysis"]
        print(f"🔍 Search Results:")
        for query in queries:
            results_search = corpus_manager.search_documents(query, limit=1)
            print(f"   '{query}': {len(results_search)} results")
        
        corpus_manager.cleanup()
        print(f"🎉 Custom configuration complete!")


def quick_example_3_production():
    """Quick Example 3: Production Setup (5 minutes)
    クイック例3：本番セットアップ（5分）"""
    
    print("\n🏭 Quick Example 3: Production Setup")
    print("🏭 クイック例3：本番セットアップ")
    print("-" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a more comprehensive document set
        documents = {
            "machine_learning.md": """
# Machine Learning Guide

## Introduction
Machine learning is a method of data analysis that automates analytical model building.

## Key Algorithms
- Linear Regression
- Decision Trees  
- Random Forest
- Neural Networks

## Applications
Machine learning is used in recommendation systems, fraud detection, and image recognition.
            """,
            
            "python_data_science.txt": """
Python Data Science Ecosystem

NumPy provides support for large arrays and matrices.
Pandas offers data structures and operations for manipulating numerical tables.
Scikit-learn includes simple and efficient tools for data mining and analysis.
Matplotlib creates static, animated, and interactive visualizations.

These libraries form the foundation of Python's data science capabilities.
            """,
            
            "ai_ethics.txt": """
Artificial Intelligence Ethics

As AI systems become more prevalent, ethical considerations become crucial.
Key areas include fairness, transparency, accountability, and privacy.

Bias in AI systems can perpetuate or amplify existing societal inequalities.
Explainable AI helps users understand how decisions are made.
Data privacy must be protected throughout the AI development lifecycle.
            """
        }
        
        # Write documents
        for filename, content in documents.items():
            (temp_path / filename).write_text(content.strip())
        
        print(f"📚 Created comprehensive document set: {len(documents)} files")
        
        # Production-ready configuration
        from refinire_rag import SQLiteDocumentStore
        
        db_path = temp_path / "production.db"
        model_path = temp_path / "model.pkl"
        
        config = CorpusManagerConfig(
            # Optimized chunking
            chunking_config=ChunkingConfig(
                chunk_size=200,
                overlap=30,
                split_by_sentence=True,
                min_chunk_size=50
            ),
            
            # Production embedder with persistence
            embedder=TFIDFEmbedder(TFIDFEmbeddingConfig(
                max_features=5000,
                min_df=1,
                max_df=0.95,
                ngram_range=(1, 3),
                model_path=str(model_path),
                auto_save_model=True
            )),
            
            # Database storage
            document_store=SQLiteDocumentStore(str(db_path)),
            store_intermediate_results=True,
            
            # Production settings
            batch_size=10,
            enable_progress_reporting=True,
            fail_on_error=False
        )
        
        corpus_manager = CorpusManager(config)
        
        print(f"⚙️  Processing for production deployment...")
        results = corpus_manager.process_corpus(temp_path)
        
        print(f"✅ Production Results:")
        print(f"   Documents loaded: {results['documents_loaded']}")
        print(f"   Documents processed: {results['documents_processed']}")
        print(f"   Documents embedded: {results['documents_embedded']}")
        print(f"   Documents stored: {results['documents_stored']}")
        print(f"   Processing time: {results['total_processing_time']:.2f}s")
        print(f"   Success rate: {(results['documents_embedded'] / max(results['documents_loaded'], 1) * 100):.1f}%")
        
        # Check production assets
        print(f"📁 Production Assets:")
        print(f"   Database: {db_path.name} ({db_path.stat().st_size} bytes)")
        if model_path.exists():
            print(f"   Model: {model_path.name} ({model_path.stat().st_size} bytes)")
        
        # Production search test
        print(f"🔍 Production Search Test:")
        search_scenarios = [
            "machine learning algorithms",
            "python data science libraries", 
            "AI ethics and fairness"
        ]
        
        for query in search_scenarios:
            results_search = corpus_manager.search_documents(query, limit=2)
            print(f"   '{query}': {len(results_search)} results")
        
        # Get comprehensive stats
        stats = corpus_manager.get_corpus_stats()
        print(f"📊 System Health:")
        print(f"   Total embeddings: {stats.get('total_embeddings', 0)}")
        print(f"   Average processing time: {stats.get('embedder_stats', {}).get('average_processing_time', 0):.4f}s")
        print(f"   Error count: {stats.get('errors', 0)}")
        
        corpus_manager.cleanup()
        print(f"🎉 Production setup complete!")
        print(f"💡 Tip: Save database and model files for reuse in production")


def main():
    """Run the quick start guide
    クイックスタートガイドを実行"""
    
    print("⚡ CorpusManager Quick Start Guide")
    print("⚡ CorpusManager クイックスタートガイド")
    print("=" * 60)
    print()
    print("Learn CorpusManager in 3 quick examples:")
    print("3つのクイック例でCorpusManagerを学びましょう：")
    print()
    print("1. 🚀 Minimal (30 seconds) - Get started immediately")
    print("2. 🔧 Custom (2 minutes) - Configure for your needs") 
    print("3. 🏭 Production (5 minutes) - Deploy-ready setup")
    print()
    
    try:
        # Run quick examples
        quick_example_1_minimal()
        quick_example_2_custom()
        quick_example_3_production()
        
        # Final guidance
        print("\n" + "=" * 60)
        print("🎓 Congratulations! You've completed the quick start.")
        print("🎓 おめでとうございます！クイックスタートが完了しました。")
        print("=" * 60)
        
        print(f"\n📋 What You Accomplished:")
        print(f"   ✅ Processed documents with minimal setup")
        print(f"   ✅ Configured custom chunking and embedding")
        print(f"   ✅ Set up production-ready pipeline with persistence")
        print(f"   ✅ Tested search functionality across all examples")
        
        print(f"\n🔧 Key Concepts Learned:")
        print(f"   📄 Document Loading: Automatic file format detection")
        print(f"   ✂️  Chunking: Breaking documents into searchable pieces")
        print(f"   🔤 Embedding: Converting text to numerical vectors")
        print(f"   🗄️  Storage: Persistent database and model storage")
        print(f"   🔍 Search: Content-based document retrieval")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Try the full tutorial: python corpus_manager_tutorial.py")
        print(f"   2. Process your own documents")
        print(f"   3. Experiment with different configurations")
        print(f"   4. Integrate with your application")
        
        print(f"\n💡 Pro Tips:")
        print(f"   • Use chunk_size=200-500 for most documents")
        print(f"   • Enable store_intermediate_results for debugging")
        print(f"   • Save TF-IDF models for consistent results")
        print(f"   • Monitor processing statistics in production")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Quick start failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())