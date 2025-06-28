#!/usr/bin/env python3
"""
Simple Hybrid RAG Example - 3 Clear Steps

この例は、refinire-ragライブラリを使った基本的なRAGワークフローを、
4つの明確なステップで示します：

1. 環境変数の設定（Environment Variable Setup）
2. コーパスの作成（Corpus Creation）  
3. クエリエンジンでの検索（Query Engine Search）
4. 品質評価（Quality Evaluation with QualityLab）

Requirements:
- Optional: refinire-rag[bm25,chroma] for hybrid search capabilities
- Environment variables for LLM integration (OpenAI recommended)

使用方法 / Usage:
```bash
# OpenAI API keyを設定（推奨）
export OPENAI_API_KEY="your-api-key-here"

# プログラム実行
python examples/hybrid_rag_example.py
```
"""

import os
import sys
import shutil
import glob
from pathlib import Path

# Add src to Python path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.application import CorpusManager, QueryEngine, QualityLab
from refinire_rag.registry import PluginRegistry

def cleanup_existing_data():
    """既存のデータファイルをクリーンアップ"""
    print("🧹 Cleaning up existing data...")
    
    cleanup_targets = [
        "./business_rag.db", "./data/documents.db", "./refinire/rag/",
        "./bm25s_index/", "./bm25s_data/", "./data/bm25s/", "./data/bm25s_index/", 
        "*.bm25s", "*.index", "./data/bm25s_keyword_store.db"
    ]
    
    cleaned_count = 0
    for target in cleanup_targets:
        if "*" in target:
            for file_path in glob.glob(target):
                try:
                    Path(file_path).unlink()
                    cleaned_count += 1
                except Exception:
                    pass
        else:
            target_path = Path(target)
            if target_path.exists():
                try:
                    if target_path.is_file():
                        target_path.unlink()
                    else:
                        shutil.rmtree(target_path)
                    cleaned_count += 1
                except Exception:
                    pass
    
    print(f"   ✅ Cleaned up {cleaned_count} items" if cleaned_count > 0 else "   ✨ Starting fresh")

def step1_setup_environment():
    """
    ステップ1: 環境変数の設定
    
    refinire-ragライブラリは環境変数を使って各コンポーネントを自動設定します。
    ここでは利用可能なプラグインを検出し、最適な設定を環境変数に設定します。
    """
    print("\n" + "="*60)
    print("🔧 STEP 1: Environment Variable Setup")
    print("="*60)
    
    # データクリーンアップ
    cleanup_existing_data()
    
    # 基本設定: Document Store（文書保存）
    print("\n📁 Setting up Document Store...")
    os.environ.setdefault("REFINIRE_RAG_DOCUMENT_STORES", "sqlite")
    os.environ.setdefault("REFINIRE_RAG_SQLITE_DB_PATH", "./business_rag.db")
    print("   ✅ SQLite document store configured")
    
    # Embedder設定: ベクター検索用の埋め込みモデル
    print("\n🧠 Setting up Embedder...")
    if os.environ.get("OPENAI_API_KEY"):
        os.environ.setdefault("REFINIRE_RAG_EMBEDDERS", "openai")
        print("   ✅ OpenAI embedder configured (high quality)")
    else:
        os.environ.setdefault("REFINIRE_RAG_EMBEDDERS", "tfidf")
        print("   ⚠️  TF-IDF embedder configured (no API key required)")
        print("      💡 For better results, set OPENAI_API_KEY environment variable")
    
    # プラグイン可用性チェック
    print("\n📦 Checking Plugin Availability...")
    available_plugins = {
        'chroma': PluginRegistry.get_plugin_class('vector_stores', 'chroma') is not None,
        'bm25s': PluginRegistry.get_plugin_class('keyword_stores', 'bm25s_keyword') is not None
    }
    
    # Vector Store設定: ベクター検索
    if available_plugins['chroma']:
        os.environ.setdefault("REFINIRE_RAG_VECTOR_STORES", "chroma")
        print("   ✅ Chroma vector store configured (external plugin)")
    else:
        os.environ.setdefault("REFINIRE_RAG_VECTOR_STORES", "inmemory_vector")
        print("   ✅ InMemory vector store configured (built-in)")
    
    # Keyword Store設定: キーワード検索
    if available_plugins['bm25s']:
        os.environ.setdefault("REFINIRE_RAG_KEYWORD_STORES", "bm25s_keyword")
        os.environ.setdefault("REFINIRE_RAG_BM25S_INDEX_PATH", "./data/bm25s_index")
        print("   ✅ BM25s keyword store configured (external plugin)")
    else:
        print("   ⚠️  No keyword store available (install refinire-rag[bm25] for hybrid search)")
    
    # Hybrid Search可能かチェック
    if available_plugins['chroma'] and available_plugins['bm25s']:
        print("\n🎯 Configuring Hybrid Search Components...")
        
        # Reranker設定: 検索結果の再ランキング
        available_rerankers = PluginRegistry.list_available_plugins('rerankers')
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        
        if "llm" in available_rerankers and has_openai:
            os.environ.setdefault("REFINIRE_RAG_RERANKERS", "llm")
            print("   ✅ LLM reranker configured (highest quality)")
        elif "rrf" in available_rerankers:
            os.environ.setdefault("REFINIRE_RAG_RERANKERS", "rrf")
            print("   ✅ RRF reranker configured (mathematical fusion)")
        elif "heuristic" in available_rerankers:
            os.environ.setdefault("REFINIRE_RAG_RERANKERS", "heuristic")
            print("   ✅ Heuristic reranker configured (keyword-based)")
        
        # Answer Synthesizer設定: 回答生成
        os.environ.setdefault("REFINIRE_RAG_SYNTHESIZERS", "answer")
        print("   ✅ Answer synthesizer configured")
        
        if has_openai:
            os.environ.setdefault("REFINIRE_RAG_LLM_MODEL", "gpt-4o-mini")
            print("   ✅ LLM model: gpt-4o-mini")
        
        print("   🚀 Hybrid search ready: Vector + Keyword + Reranking + LLM")
    else:
        # Simple retrieval設定
        os.environ.setdefault("REFINIRE_RAG_RETRIEVERS", "simple")
        print("   ✅ Simple retrieval configured")
    
    print(f"\n📋 Environment Setup Summary:")
    print(f"   • Document Store: {os.environ.get('REFINIRE_RAG_DOCUMENT_STORES', 'None')}")
    print(f"   • Vector Store: {os.environ.get('REFINIRE_RAG_VECTOR_STORES', 'None')}")
    print(f"   • Keyword Store: {os.environ.get('REFINIRE_RAG_KEYWORD_STORES', 'None')}")
    print(f"   • Reranker: {os.environ.get('REFINIRE_RAG_RERANKERS', 'None')}")
    print(f"   • Synthesizer: {os.environ.get('REFINIRE_RAG_SYNTHESIZERS', 'None')}")
    print(f"   • Embedder: {os.environ.get('REFINIRE_RAG_EMBEDDERS', 'None')}")
    
    return available_plugins

def step2_create_corpus():
    """
    ステップ2: CorpusManagerでコーパス作成
    
    CorpusManagerは環境変数の設定に基づいて自動的にコンポーネントを初期化し、
    ビジネスデータセットからコーパス（検索可能な文書集合）を作成します。
    """
    print("\n" + "="*60)
    print("📚 STEP 2: Corpus Creation with CorpusManager")
    print("="*60)
        
    # CorpusManager作成（環境変数から自動設定）
    corpus_manager = CorpusManager()
    data_path = Path(__file__).parent.parent / "tests" / "data" / "business_dataset"
    
    # 文書インポート
    import_stats = corpus_manager.import_original_documents(
        corpus_name="business_knowledge",
        directory=str(data_path),
        glob="*.txt"
    )
    print(f"   ✅ Documents imported: {import_stats.total_documents_created}")
    print(f"   ⏱️  Import time: {import_stats.total_processing_time:.2f}s")
    
    # コーパス構築（埋め込み生成・インデックス作成）
    print(f"\n🔨 Building corpus with embeddings and indexing...")
    build_stats = corpus_manager.rebuild_corpus_from_original(
        corpus_name="business_knowledge"
    )
    print(f"   ✅ Chunks created: {build_stats.total_chunks_created}")
    print(f"   ⏱️  Build time: {build_stats.total_processing_time:.2f}s")
    
    # コーパス情報表示
    print(f"\n📈 Corpus Information:")
    print(f"   🏷️  Name: business_knowledge")
    print(f"   📄 Documents: {import_stats.total_documents_created}")
    print(f"   📝 Chunks: {build_stats.total_chunks_created}")
    print(f"   🔍 Retrievers: {len(corpus_manager.retrievers)}")
    print(f"   🧠 Embedder: {os.environ.get('REFINIRE_RAG_EMBEDDERS')}")
    
    return corpus_manager

def step3_query_engine_search():
    """
    ステップ3: QueryEngineで検索・回答生成
    
    QueryEngineも環境変数の設定に基づいて自動的にコンポーネントを初期化し、
    クエリに対して検索・再ランキング・回答生成を行います。
    """
    print("\n" + "="*60)
    print("🔍 STEP 3: Query Engine Search & Answer Generation")
    print("="*60)
    
    # QueryEngine作成（環境変数から自動設定）
    print("🏗️  Creating QueryEngine from environment variables...")
    query_engine = QueryEngine()
    
    print(f"   ✅ QueryEngine initialized with:")
    print(f"      • Retrievers: {[type(r).__name__ for r in query_engine.retrievers]}")
    print(f"      • Reranker: {type(query_engine.reranker).__name__ if query_engine.reranker else 'None'}")
    print(f"      • Synthesizer: {type(query_engine.synthesizer).__name__ if query_engine.synthesizer else 'None'}")
    
    # サンプルクエリで検索テスト（日本語ビジネス関連）
    sample_queries = [
        "会社の主な事業内容は何ですか？",
        "AIソリューションの製品ラインナップを教えてください",
        "2023年度の売上高と営業利益はいくらですか？",
        "リモートワークの制度について教えてください",
        "情報セキュリティの取り組みはどのようなものがありますか？"
    ]
    
    print(f"\n🔍 Testing {len(sample_queries)} sample queries...")
    
    for i, query_text in enumerate(sample_queries, 1):
        print(f"\n📝 Query {i}: {query_text}")
        print("-" * 50)
        
        try:
            # クエリ実行
            result = query_engine.query(query_text, retriever_top_k=3)
            
            if result.sources:
                print(f"   📄 Found {len(result.sources)} relevant documents:")
                
                # 検索結果表示
                for j, source in enumerate(result.sources, 1):
                    print(f"      {j}. Score: {source.score:.3f}")
                    print(f"         Doc ID: {source.document_id}")
                    print(f"         Content: {source.document.content[:100]}...")
                    
                    # Reranker情報表示
                    if "reranked_by" in source.metadata:
                        reranker_type = source.metadata["reranked_by"]
                        original_score = source.metadata.get("original_score", "N/A")
                        llm_score = source.metadata.get("llm_score", "N/A")
                        print(f"         🎯 Reranked by: {reranker_type}")
                        print(f"         📊 Original → LLM: {original_score:.3f} → {llm_score:.3f}")
                    
                    # Use the retriever_type metadata to show actual storage technology
                    retriever_info = source.metadata.get("retriever_type", "Unknown")
                    retrieval_method = source.metadata.get("retrieval_method", "unknown")
                    
                    if retriever_info != "Unknown":
                        if retrieval_method == "vector_similarity":
                            print(f"         🔍 Source: {retriever_info} (vector search)")
                        elif retrieval_method == "keyword_search":
                            print(f"         🔍 Source: {retriever_info} (keyword search)")
                        else:
                            print(f"         🔍 Source: {retriever_info}")
                    else:
                        print(f"         🔍 Source: Unknown")
                
                # 生成された回答表示
                if result.answer:
                    print(f"\n   🤖 Generated Answer:")
                    print(f"      {result.answer[:150]}{'...' if len(result.answer) > 150 else ''}")
                else:
                    print(f"\n   ⚠️  No answer generated")
                    if not query_engine.synthesizer:
                        print(f"      💡 No synthesizer configured - only search results available")
                
                print(f"   ⏱️  Processing time: {result.processing_time:.3f}s")
            else:
                print("   ⚠️  No relevant documents found")
                
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
    
    return query_engine


def step4_quality_evaluation(query_engine, sample_queries):
    """
    ステップ4: QualityLabで品質評価
    
    QualityLabも環境変数の設定に基づいて自動的にコンポーネントを初期化し、
    RAGシステムの品質を包括的に評価します。
    """
    print("\n" + "="*60)
    print("🔬 STEP 4: Quality Evaluation with QualityLab")
    print("="*60)
    
    # QualityLab設定用の環境変数を設定
    print("🔧 Setting up QualityLab environment variables...")
    os.environ.setdefault("REFINIRE_RAG_TEST_SUITES", "llm")
    os.environ.setdefault("REFINIRE_RAG_EVALUATORS", "standard") 
    os.environ.setdefault("REFINIRE_RAG_CONTRADICTION_DETECTORS", "llm")
    os.environ.setdefault("REFINIRE_RAG_INSIGHT_REPORTERS", "standard")
    os.environ.setdefault("REFINIRE_RAG_QA_PAIRS_PER_DOCUMENT", "2")
    os.environ.setdefault("REFINIRE_RAG_EVALUATION_TIMEOUT", "30")
    os.environ.setdefault("REFINIRE_RAG_INCLUDE_CONTRADICTION_DETECTION", "true")
    
    print("   ✅ QualityLab evaluation environment configured")
    print(f"      • Test Suite: {os.environ.get('REFINIRE_RAG_TEST_SUITES')}")
    print(f"      • Evaluator: {os.environ.get('REFINIRE_RAG_EVALUATORS')}")
    print(f"      • Contradiction Detector: {os.environ.get('REFINIRE_RAG_CONTRADICTION_DETECTORS')}")
    print(f"      • Insight Reporter: {os.environ.get('REFINIRE_RAG_INSIGHT_REPORTERS')}")
    
    # QualityLab作成（環境変数から自動設定）
    print("\n🏗️  Creating QualityLab from environment variables...")
    try:
        quality_lab = QualityLab()
        print("   ✅ QualityLab initialized successfully")
        print(f"      • Test Suite: {type(quality_lab.test_suite).__name__}")
        print(f"      • Evaluator: {type(quality_lab.evaluator).__name__}")
        print(f"      • Contradiction Detector: {type(quality_lab.contradiction_detector).__name__}")
        print(f"      • Insight Reporter: {type(quality_lab.insight_reporter).__name__}")
    except Exception as e:
        print(f"   ⚠️  QualityLab initialization failed: {e}")
        print("   💡 Continuing without quality evaluation...")
        return None
    
    # サンプル評価の実行
    print(f"\n🧪 Running evaluation on business knowledge corpus...")
    try:
        # 簡単な評価実行（実際のテストケース生成と評価）
        evaluation_result = quality_lab.evaluate_rag_pipeline(
            corpus_name="business_knowledge",
            query_engine=query_engine,
            sample_size=min(3, len(sample_queries)),  # 小さなサンプルサイズで実行
            custom_queries=sample_queries[:3] if sample_queries else None
        )
        
        print(f"   ✅ Evaluation completed successfully")
        print(f"   📊 Overall Score: {evaluation_result.overall_score:.2f}")
        print(f"   ⏱️  Total evaluation time: {evaluation_result.processing_time:.2f}s")
        
        # 詳細結果表示
        if hasattr(evaluation_result, 'metrics') and evaluation_result.metrics:
            print(f"\n📈 Detailed Metrics:")
            for metric, value in evaluation_result.metrics.items():
                if isinstance(value, (int, float)):
                    print(f"      • {metric}: {value:.3f}")
                else:
                    print(f"      • {metric}: {value}")
        
        # インサイト表示
        if hasattr(evaluation_result, 'insights') and evaluation_result.insights:
            print(f"\n💡 Key Insights:")
            for i, insight in enumerate(evaluation_result.insights[:3], 1):  # 最初の3つのみ表示
                print(f"      {i}. {insight.get('title', 'Insight')}: {insight.get('description', 'No description')}")
        
        return evaluation_result
        
    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")
        print(f"   💡 This might be due to missing test data or configuration issues")
        return None


def main():
    """
    メイン関数: 4ステップでのRAGシステム構築・評価
    
    この関数は以下の4つのステップを順次実行します：
    1. 環境変数設定
    2. コーパス作成
    3. クエリエンジン検索
    4. 品質評価
    """
    print("🚀 Simple Hybrid RAG Example - 4 Clear Steps")
    print("=" * 60)
    print("This example demonstrates a complete RAG workflow in 4 simple steps:")
    print("1. Environment Variable Setup")
    print("2. Corpus Creation with CorpusManager")
    print("3. Query Engine Search & Answer Generation")
    print("4. Quality Evaluation with QualityLab")
    print()
    print("All components are automatically configured from environment variables!")
    
    try:
        # Step 1: 環境変数設定
        available_plugins = step1_setup_environment()
        
        # Step 2: コーパス作成
        corpus_manager = step2_create_corpus()
        
        # Step 3: クエリエンジン検索
        query_engine = step3_query_engine_search()
        
        # Step 4: 品質評価
        sample_queries = [
            "会社の主な事業内容は何ですか？",
            "AIソリューションの製品ラインナップを教えてください",
            "2023年度の売上高と営業利益はいくらですか？",
            "リモートワークの制度について教えてください",
            "情報セキュリティの取り組みはどのようなものがありますか？"
        ]
        evaluation_result = step4_quality_evaluation(query_engine, sample_queries)
        
        # 完了メッセージ
        print("\n" + "="*60)
        print("🎉 SUCCESS: RAG System Ready with Quality Evaluation!")
        print("="*60)
        print("Your RAG system is now fully configured, tested, and ready to use.")
        print()
        print("Key components initialized:")
        print(f"• CorpusManager: {len(corpus_manager.retrievers)} retrievers")
        print(f"• QueryEngine: {[type(r).__name__ for r in query_engine.retrievers]}")
        print(f"• Reranker: {type(query_engine.reranker).__name__ if query_engine.reranker else 'None'}")
        print(f"• Synthesizer: {type(query_engine.synthesizer).__name__ if query_engine.synthesizer else 'None'}")
        if evaluation_result:
            print(f"• QualityLab: Evaluation completed with score {evaluation_result.overall_score:.2f}")
        else:
            print(f"• QualityLab: Evaluation skipped due to configuration issues")
        print()
        print("💡 Next steps:")
        print("- Try your own queries with: query_engine.query('your question here')")
        print("- Run quality evaluations with: quality_lab.evaluate_rag_pipeline()")
        print("- Explore different environment variable configurations")
        print("- Add your own documents to the corpus")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Check the error messages above and ensure all dependencies are installed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)