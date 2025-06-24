#!/usr/bin/env python3
"""
Part 2: QueryEngine Tutorial Example
QueryEngineチュートリアル例

This example demonstrates comprehensive query processing using refinire-rag's QueryEngine
with retrieval, reranking, answer synthesis, and performance monitoring.

この例では、refinire-ragのQueryEngineを使用した包括的なクエリ処理を、
検索、再ランキング、回答合成、パフォーマンス監視とともに実演します。
"""

import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval import (
    SimpleRetriever, SimpleRetrieverConfig,
    SimpleReranker, SimpleRerankerConfig,
    SimpleReader, SimpleReaderConfig,
    HybridRetriever
)
from refinire_rag.keywordstore import TfidfKeywordStore
from refinire_rag.models.query import QueryResult


def setup_sample_corpus(temp_dir: Path) -> tuple:
    """
    Set up a sample corpus for QueryEngine demonstration
    QueryEngineデモ用のサンプルコーパスをセットアップ
    """
    
    print("📚 Setting up sample corpus for QueryEngine demo...")
    print("📚 QueryEngineデモ用サンプルコーパスをセットアップ中...")
    
    # Create sample documents / サンプル文書を作成
    documents_dir = temp_dir / "knowledge_base"
    documents_dir.mkdir(exist_ok=True)
    
    # AI Fundamentals / AI基礎
    (documents_dir / "ai_fundamentals.txt").write_text("""
人工知能（AI）とは、人間の知的活動をコンピュータで再現する技術です。
主な特徴：
- 学習能力（Learning）：データから知識を獲得
- 推論能力（Reasoning）：論理的思考と判断
- 認識能力（Perception）：視覚・聴覚情報の理解
- 創造能力（Creativity）：新しいアイデアの生成

AIの分類：
1. 弱いAI（Narrow AI）：特定タスクに特化
2. 強いAI（General AI）：人間レベルの汎用知能
3. 超AI（Super AI）：人間を超える知能

現在実用化されているのは主に弱いAIで、
音声認識、画像認識、自然言語処理などの分野で活用されています。
""", encoding='utf-8')
    
    # Machine Learning / 機械学習
    (documents_dir / "machine_learning.txt").write_text("""
機械学習（ML）は、明示的にプログラムすることなく
コンピュータがデータから学習し予測・分類を行う技術です。

主要なアプローチ：

1. 教師あり学習（Supervised Learning）
   - 分類（Classification）：カテゴリ予測
   - 回帰（Regression）：数値予測
   - 例：スパムメール検出、株価予測

2. 教師なし学習（Unsupervised Learning）
   - クラスタリング：データをグループ化
   - 次元削減：データの簡約化
   - 例：顧客セグメンテーション、異常検知

3. 強化学習（Reinforcement Learning）
   - エージェントが環境との相互作用で学習
   - 例：ゲームAI、ロボット制御

機械学習の応用分野は金融、医療、製造業、
マーケティングなど多岐にわたります。
""", encoding='utf-8')
    
    # Deep Learning / 深層学習
    (documents_dir / "deep_learning.txt").write_text("""
深層学習（Deep Learning）は、多層のニューラルネットワークを
使用する機械学習の手法です。

主要なアーキテクチャ：

1. フィードフォワードネットワーク（MLP）
   - 全結合層による基本的な構造
   - 用途：分類、回帰問題

2. 畳み込みニューラルネットワーク（CNN）
   - 画像処理に特化した構造
   - 用途：画像認識、物体検出

3. 再帰ニューラルネットワーク（RNN）
   - 時系列データ処理に適した構造
   - 改良版：LSTM、GRU
   - 用途：自然言語処理、音声認識

4. トランスフォーマー（Transformer）
   - 注意機構（Attention）を核とした構造
   - 用途：機械翻訳、言語モデル（GPT、BERT）

深層学習の発展により、従来困難とされていた
複雑なパターン認識が可能になりました。
""", encoding='utf-8')
    
    # Natural Language Processing / 自然言語処理
    (documents_dir / "nlp.txt").write_text("""
自然言語処理（NLP）は、人間の言語をコンピュータで
理解・生成・操作する技術分野です。

主要なタスク：

1. 基本処理
   - トークン化（Tokenization）：文章を単語に分割
   - 品詞タグ付け（POS Tagging）：単語の品詞を識別
   - 構文解析（Parsing）：文法構造の分析

2. 意味理解
   - 固有表現認識（NER）：人名、地名等の抽出
   - 感情分析（Sentiment Analysis）：文章の感情判定
   - 意図理解（Intent Recognition）：発話の意図推定

3. 生成タスク
   - 機械翻訳（Machine Translation）：言語間変換
   - 文書要約（Text Summarization）：要点抽出
   - 質問応答（Question Answering）：質問への回答生成

4. 応用システム
   - チャットボット：対話型システム
   - 検索エンジン：情報検索
   - 音声アシスタント：音声対話

近年の大規模言語モデル（LLM）の発展により、
NLPの性能は大幅に向上しています。
""", encoding='utf-8')
    
    # Computer Vision / コンピュータビジョン
    (documents_dir / "computer_vision.txt").write_text("""
コンピュータビジョン（CV）は、コンピュータが
視覚情報を理解・解釈する技術分野です。

主要な処理タスク：

1. 画像分類（Image Classification）
   - 画像全体のカテゴリを判定
   - 例：動物の種類識別、製品分類

2. 物体検出（Object Detection）
   - 画像内の物体位置と種類を特定
   - 例：自動運転での歩行者検出

3. セマンティックセグメンテーション
   - 画像の各ピクセルにラベル付与
   - 例：医療画像での病変部位特定

4. 顔認識（Face Recognition）
   - 個人の顔を識別・認証
   - 例：セキュリティシステム、写真管理

5. 動画解析（Video Analysis）
   - 時系列での物体追跡・行動認識
   - 例：監視システム、スポーツ解析

技術要素：
- 特徴抽出：エッジ、テクスチャ、形状の検出
- パターンマッチング：テンプレートとの照合
- 機械学習：深層学習による自動特徴学習

応用分野：自動運転、医療診断、製造業の品質管理、
エンターテインメントなど幅広く活用されています。
""", encoding='utf-8')
    
    # Initialize storage / ストレージ初期化
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    
    # Build corpus using semantic RAG / セマンティックRAGでコーパス構築
    corpus_manager = CorpusManager.create_semantic_rag(doc_store, vector_store)
    
    # Process all documents / 全文書を処理
    file_paths = [str(f) for f in documents_dir.glob("*.txt")]
    stats = corpus_manager.build_corpus(file_paths)
    
    print(f"✅ Corpus setup completed / コーパスセットアップ完了:")
    print(f"   Files processed / 処理ファイル数: {stats.total_files_processed}")
    print(f"   Documents created / 作成文書数: {stats.total_documents_created}")
    print(f"   Chunks created / 作成チャンク数: {stats.total_chunks_created}")
    print(f"   Processing time / 処理時間: {stats.total_processing_time:.3f}s")
    
    return doc_store, vector_store


def demonstrate_basic_queries(query_engine: QueryEngine):
    """
    Demonstrate basic query operations
    基本的なクエリ操作のデモンストレーション
    """
    
    print("\n" + "="*60)
    print("🔍 BASIC QUERY OPERATIONS DEMONSTRATION")
    print("🔍 基本クエリ操作のデモンストレーション")
    print("="*60)
    
    # Basic queries / 基本クエリ
    basic_queries = [
        "AIとは何ですか？",
        "機械学習の種類を教えて",
        "深層学習のアーキテクチャにはどんなものがありますか？",
        "自然言語処理の応用例は？",
        "コンピュータビジョンでできることは？"
    ]
    
    for i, query in enumerate(basic_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 50)
        
        try:
            # Execute query / クエリ実行
            start_time = time.time()
            result = query_engine.answer(query)
            end_time = time.time()
            
            # Display results / 結果表示
            print(f"🤖 Answer / 回答:")
            print(f"   {result.answer[:200]}...")
            print(f"")
            print(f"📊 Metrics / メトリクス:")
            print(f"   Processing time / 処理時間: {end_time - start_time:.3f}s")
            print(f"   Confidence / 信頼度: {result.confidence:.3f}")
            print(f"   Source count / ソース数: {len(result.sources)}")
            
            # Show source information / ソース情報表示
            if result.sources:
                print(f"   Top source / トップソース: {result.sources[0].metadata.get('source', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Query failed / クエリ失敗: {e}")


def demonstrate_advanced_configurations(doc_store, vector_store):
    """
    Demonstrate advanced QueryEngine configurations
    高度なQueryEngine設定のデモンストレーション
    """
    
    print("\n" + "="*60)
    print("⚙️  ADVANCED CONFIGURATIONS DEMONSTRATION")
    print("⚙️  高度な設定のデモンストレーション")
    print("="*60)
    
    # Configuration 1: Performance-optimized / 設定1: パフォーマンス最適化
    print("\n📌 Configuration 1: Performance-Optimized")
    print("📌 設定1: パフォーマンス最適化")
    print("-" * 40)
    
    fast_retriever = SimpleRetriever(
        vector_store, 
        config=SimpleRetrieverConfig(
            top_k=5,
            similarity_threshold=0.2,
            embedding_model="text-embedding-3-small"
        )
    )
    
    fast_reranker = SimpleReranker(
        config=SimpleRerankerConfig(
            top_k=3,
            boost_exact_matches=True
        )
    )
    
    fast_reader = SimpleReader(
        config=SimpleReaderConfig(
            llm_model="gpt-4o-mini",
            max_context_length=1000,
            temperature=0.1
        )
    )
    
    fast_engine = QueryEngine(
        document_store=doc_store,
        vector_store=vector_store,
        retriever=fast_retriever,
        reranker=fast_reranker,
        reader=fast_reader,
        config=QueryEngineConfig(
            enable_query_normalization=False,  # Disable for speed
            include_sources=True,
            max_response_time=5.0
        )
    )
    
    # Test performance configuration / パフォーマンス設定をテスト
    test_query = "機械学習とは何ですか？"
    start_time = time.time()
    result = fast_engine.answer(test_query)
    fast_time = time.time() - start_time
    
    print(f"✅ Fast Configuration Results / 高速設定結果:")
    print(f"   Query time / クエリ時間: {fast_time:.3f}s")
    print(f"   Answer length / 回答長: {len(result.answer)} chars")
    print(f"   Sources used / 使用ソース: {len(result.sources)}")
    
    # Configuration 2: Accuracy-optimized / 設定2: 精度最適化
    print("\n📌 Configuration 2: Accuracy-Optimized")
    print("📌 設定2: 精度最適化")
    print("-" * 40)
    
    accurate_retriever = SimpleRetriever(
        vector_store,
        config=SimpleRetrieverConfig(
            top_k=15,
            similarity_threshold=0.05,
            embedding_model="text-embedding-3-large"
        )
    )
    
    accurate_reranker = SimpleReranker(
        config=SimpleRerankerConfig(
            top_k=8,
            boost_exact_matches=True,
            length_penalty_factor=0.05
        )
    )
    
    accurate_reader = SimpleReader(
        config=SimpleReaderConfig(
            llm_model="gpt-4",
            max_context_length=2500,
            temperature=0.2,
            generation_instructions="""
            Provide detailed, accurate answers based on the context.
            Include specific examples and technical details where appropriate.
            Structure your response clearly with main points and supporting details.
            """
        )
    )
    
    accurate_engine = QueryEngine(
        document_store=doc_store,
        vector_store=vector_store,
        retriever=accurate_retriever,
        reranker=accurate_reranker,
        reader=accurate_reader,
        config=QueryEngineConfig(
            enable_query_normalization=True,
            include_sources=True,
            include_confidence=True,
            max_response_time=30.0
        )
    )
    
    # Test accuracy configuration / 精度設定をテスト
    start_time = time.time()
    result = accurate_engine.answer(test_query)
    accurate_time = time.time() - start_time
    
    print(f"✅ Accurate Configuration Results / 高精度設定結果:")
    print(f"   Query time / クエリ時間: {accurate_time:.3f}s")
    print(f"   Answer length / 回答長: {len(result.answer)} chars")
    print(f"   Sources used / 使用ソース: {len(result.sources)}")
    print(f"   Confidence / 信頼度: {result.confidence:.3f}")
    
    # Configuration 3: Hybrid retrieval / 設定3: ハイブリッド検索
    print("\n📌 Configuration 3: Hybrid Retrieval")
    print("📌 設定3: ハイブリッド検索")
    print("-" * 40)
    
    # Setup keyword store / キーワードストア設定
    keyword_store = TfidfKeywordStore()
    
    # Build keyword index from documents / 文書からキーワードインデックス構築
    # Note: In real implementation, this would be done during corpus building
    # 注意: 実際の実装では、これはコーパス構築時に行われます
    
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_store=keyword_store,
        vector_weight=0.7,
        keyword_weight=0.3
    )
    
    hybrid_engine = QueryEngine(
        document_store=doc_store,
        vector_store=vector_store,
        retriever=hybrid_retriever,
        reranker=fast_reranker,
        reader=fast_reader
    )
    
    # Test hybrid configuration / ハイブリッド設定をテスト
    start_time = time.time()
    result = hybrid_engine.answer("deep learning neural network")
    hybrid_time = time.time() - start_time
    
    print(f"✅ Hybrid Configuration Results / ハイブリッド設定結果:")
    print(f"   Query time / クエリ時間: {hybrid_time:.3f}s")
    print(f"   Vector + Keyword search / ベクトル + キーワード検索")
    print(f"   Sources found / 発見ソース: {len(result.sources)}")
    
    print(f"\n📊 Configuration Comparison / 設定比較:")
    print(f"   Fast config / 高速設定: {fast_time:.3f}s")
    print(f"   Accurate config / 高精度設定: {accurate_time:.3f}s")
    print(f"   Hybrid config / ハイブリッド設定: {hybrid_time:.3f}s")


def demonstrate_query_analysis(query_engine: QueryEngine):
    """
    Demonstrate detailed query result analysis
    詳細なクエリ結果分析のデモンストレーション
    """
    
    print("\n" + "="*60)
    print("🔬 QUERY RESULT ANALYSIS DEMONSTRATION")
    print("🔬 クエリ結果分析のデモンストレーション")
    print("="*60)
    
    analysis_queries = [
        "CNNとRNNの違いは何ですか？",
        "自然言語処理でできることを教えて",
        "AIの倫理的な課題について"  # This may have lower confidence
    ]
    
    for i, query in enumerate(analysis_queries, 1):
        print(f"\n📝 Analysis Query {i}: {query}")
        print("-" * 50)
        
        # Execute with detailed timing / 詳細なタイミングで実行
        start_time = time.time()
        result = query_engine.answer(query)
        total_time = time.time() - start_time
        
        # Detailed analysis / 詳細分析
        print(f"🔍 Detailed Analysis / 詳細分析:")
        print(f"   Total processing time / 総処理時間: {total_time:.3f}s")
        print(f"   Answer quality / 回答品質:")
        print(f"     - Length / 長さ: {len(result.answer)} characters")
        print(f"     - Word count / 単語数: {len(result.answer.split())}")
        print(f"     - Confidence / 信頼度: {result.confidence:.3f}")
        
        # Confidence interpretation / 信頼度解釈
        if result.confidence > 0.8:
            confidence_level = "High / 高"
        elif result.confidence > 0.5:
            confidence_level = "Medium / 中"
        else:
            confidence_level = "Low / 低"
        
        print(f"     - Confidence level / 信頼度レベル: {confidence_level}")
        
        # Source analysis / ソース分析
        print(f"   Source analysis / ソース分析:")
        print(f"     - Source count / ソース数: {len(result.sources)}")
        
        for j, source in enumerate(result.sources[:3]):
            relevance = source.metadata.get('relevance_score', 'N/A')
            source_title = source.metadata.get('title', f'Document {j+1}')
            print(f"     - Source {j+1}: {source_title}")
            print(f"       Relevance / 関連度: {relevance}")
            print(f"       Length / 長さ: {len(source.content)} chars")
        
        # Answer preview / 回答プレビュー
        print(f"   Answer preview / 回答プレビュー:")
        print(f"     {result.answer[:150]}...")
        
        # Quality indicators / 品質指標
        quality_indicators = []
        if result.sources:
            quality_indicators.append("✓ Has sources / ソースあり")
        if result.confidence > 0.7:
            quality_indicators.append("✓ High confidence / 高信頼度")
        if len(result.answer) > 50:
            quality_indicators.append("✓ Detailed answer / 詳細回答")
        
        print(f"   Quality indicators / 品質指標: {', '.join(quality_indicators)}")


def demonstrate_performance_monitoring(query_engine: QueryEngine):
    """
    Demonstrate performance monitoring across multiple queries
    複数クエリでのパフォーマンス監視のデモンストレーション
    """
    
    print("\n" + "="*60)
    print("📊 PERFORMANCE MONITORING DEMONSTRATION")
    print("📊 パフォーマンス監視のデモンストレーション")
    print("="*60)
    
    # Performance test queries / パフォーマンステストクエリ
    test_queries = [
        "人工知能とは何ですか？",
        "機械学習のアルゴリズムの種類は？",
        "深層学習の応用分野を教えて",
        "自然言語処理の技術要素は？",
        "コンピュータビジョンでできることは？",
        "ニューラルネットワークの仕組みは？",
        "強化学習の特徴は？",
        "AIの歴史について教えて"
    ]
    
    print(f"🚀 Running performance test with {len(test_queries)} queries...")
    print(f"🚀 {len(test_queries)}クエリでパフォーマンステスト実行中...")
    
    results = []
    total_start_time = time.time()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n⏱️  Query {i}/{len(test_queries)}: {query[:30]}...")
        
        # Execute with timing / タイミング測定で実行
        start_time = time.time()
        try:
            result = query_engine.answer(query)
            end_time = time.time()
            
            query_time = end_time - start_time
            
            # Collect metrics / メトリクス収集
            metrics = {
                'query': query,
                'time': query_time,
                'confidence': result.confidence,
                'source_count': len(result.sources),
                'answer_length': len(result.answer),
                'success': True
            }
            
            print(f"     ✅ Success: {query_time:.3f}s, confidence: {result.confidence:.3f}")
            
        except Exception as e:
            query_time = time.time() - start_time
            metrics = {
                'query': query,
                'time': query_time,
                'confidence': 0.0,
                'source_count': 0,
                'answer_length': 0,
                'success': False,
                'error': str(e)
            }
            
            print(f"     ❌ Failed: {e}")
        
        results.append(metrics)
    
    total_time = time.time() - total_start_time
    
    # Performance analysis / パフォーマンス分析
    print(f"\n📈 Performance Analysis / パフォーマンス分析:")
    print("="*50)
    
    successful_results = [r for r in results if r['success']]
    failed_count = len(results) - len(successful_results)
    
    if successful_results:
        avg_time = sum(r['time'] for r in successful_results) / len(successful_results)
        avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
        avg_sources = sum(r['source_count'] for r in successful_results) / len(successful_results)
        avg_answer_length = sum(r['answer_length'] for r in successful_results) / len(successful_results)
        
        min_time = min(r['time'] for r in successful_results)
        max_time = max(r['time'] for r in successful_results)
        
        print(f"📊 Overall Statistics / 全体統計:")
        print(f"   Total queries / 総クエリ数: {len(test_queries)}")
        print(f"   Successful / 成功: {len(successful_results)}")
        print(f"   Failed / 失敗: {failed_count}")
        print(f"   Success rate / 成功率: {len(successful_results)/len(test_queries)*100:.1f}%")
        print(f"   Total time / 総時間: {total_time:.3f}s")
        print(f"   Throughput / スループット: {len(test_queries)/total_time:.2f} queries/sec")
        
        print(f"\n⏱️  Timing Statistics / タイミング統計:")
        print(f"   Average response time / 平均応答時間: {avg_time:.3f}s")
        print(f"   Fastest query / 最速クエリ: {min_time:.3f}s")
        print(f"   Slowest query / 最遅クエリ: {max_time:.3f}s")
        
        print(f"\n🎯 Quality Statistics / 品質統計:")
        print(f"   Average confidence / 平均信頼度: {avg_confidence:.3f}")
        print(f"   Average sources per query / 平均ソース数: {avg_sources:.1f}")
        print(f"   Average answer length / 平均回答長: {avg_answer_length:.0f} characters")
        
        # Performance categories / パフォーマンスカテゴリ
        fast_queries = [r for r in successful_results if r['time'] < avg_time * 0.8]
        slow_queries = [r for r in successful_results if r['time'] > avg_time * 1.2]
        
        print(f"\n🚀 Performance Categories / パフォーマンスカテゴリ:")
        print(f"   Fast queries (< {avg_time * 0.8:.3f}s): {len(fast_queries)}")
        print(f"   Normal queries: {len(successful_results) - len(fast_queries) - len(slow_queries)}")
        print(f"   Slow queries (> {avg_time * 1.2:.3f}s): {len(slow_queries)}")
        
        if slow_queries:
            print(f"   Slowest query: {slow_queries[0]['query'][:50]}... ({max(r['time'] for r in slow_queries):.3f}s)")
    
    # Engine statistics / エンジン統計
    try:
        engine_stats = query_engine.get_engine_stats()
        print(f"\n🔧 Engine Statistics / エンジン統計:")
        print(f"   Queries processed: {engine_stats.get('queries_processed', 'N/A')}")
        print(f"   Cache hits: {engine_stats.get('cache_hits', 'N/A')}")
        print(f"   Average retrieval time: {engine_stats.get('average_retrieval_time', 'N/A')}")
        
    except Exception as e:
        print(f"   Engine statistics not available: {e}")


def demonstrate_error_handling(query_engine: QueryEngine):
    """
    Demonstrate error handling and recovery
    エラーハンドリングと復旧のデモンストレーション
    """
    
    print("\n" + "="*60)
    print("🛠️  ERROR HANDLING DEMONSTRATION")
    print("🛠️  エラーハンドリングのデモンストレーション")
    print("="*60)
    
    # Test cases with potential issues / 問題を起こす可能性のあるテストケース
    error_test_cases = [
        {
            "query": "",  # Empty query
            "description": "Empty query test / 空クエリテスト"
        },
        {
            "query": "x" * 1000,  # Very long query
            "description": "Extremely long query test / 極端に長いクエリテスト"
        },
        {
            "query": "What is quantum supremacy in blockchain AI?",  # Complex/nonsensical query
            "description": "Complex nonsensical query test / 複雑で意味不明なクエリテスト"
        },
        {
            "query": "Tell me about flying unicorns",  # Query with no relevant sources
            "description": "No relevant sources test / 関連ソースなしテスト"
        }
    ]
    
    for i, test_case in enumerate(error_test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['description']}")
        print(f"Query: '{test_case['query'][:50]}{'...' if len(test_case['query']) > 50 else ''}'")
        print("-" * 40)
        
        try:
            # Attempt query with timeout / タイムアウト付きでクエリ試行
            start_time = time.time()
            result = query_engine.answer(test_case['query'])
            end_time = time.time()
            
            # Analyze result / 結果分析
            print(f"✅ Query completed / クエリ完了:")
            print(f"   Time: {end_time - start_time:.3f}s")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Sources found: {len(result.sources)}")
            print(f"   Answer length: {len(result.answer)}")
            
            # Check for low-quality results / 低品質結果をチェック
            if result.confidence < 0.3:
                print(f"⚠️  Warning: Low confidence result / 警告: 低信頼度結果")
            
            if len(result.sources) == 0:
                print(f"⚠️  Warning: No sources found / 警告: ソースが見つかりません")
            
            if len(result.answer) < 20:
                print(f"⚠️  Warning: Very short answer / 警告: 非常に短い回答")
                
        except Exception as e:
            print(f"❌ Query failed / クエリ失敗: {e}")
            print(f"   Error type: {type(e).__name__}")
            
            # Provide recovery suggestions / 復旧提案を提供
            print(f"💡 Recovery suggestions / 復旧提案:")
            if "timeout" in str(e).lower():
                print(f"   - Try a simpler query / より簡単なクエリを試す")
                print(f"   - Increase timeout limit / タイムアウト制限を増加")
            elif "empty" in str(e).lower():
                print(f"   - Provide a non-empty query / 空でないクエリを提供")
            else:
                print(f"   - Check query format / クエリ形式を確認")
                print(f"   - Verify corpus content / コーパス内容を確認")


def main():
    """
    Main demonstration function
    メインデモンストレーション関数
    """
    
    print("🚀 Part 2: QueryEngine Tutorial")
    print("🚀 Part 2: QueryEngineチュートリアル")
    print("="*60)
    print("Comprehensive demonstration of query processing with refinire-rag")
    print("refinire-ragを使用したクエリ処理の包括的なデモンストレーション")
    print("")
    print("Features demonstrated / デモ機能:")
    print("✓ Basic query operations / 基本クエリ操作")
    print("✓ Advanced configurations / 高度な設定")
    print("✓ Query result analysis / クエリ結果分析")
    print("✓ Performance monitoring / パフォーマンス監視")
    print("✓ Error handling / エラーハンドリング")
    
    # Create temporary directory / 一時ディレクトリを作成
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Setup sample corpus / サンプルコーパスをセットアップ
        print(f"\n📁 Setup: Creating sample corpus in {temp_dir}")
        print(f"📁 セットアップ: {temp_dir} にサンプルコーパスを作成中")
        doc_store, vector_store = setup_sample_corpus(temp_dir)
        
        # Initialize basic QueryEngine / 基本QueryEngineを初期化
        print(f"\n🔍 Initializing basic QueryEngine...")
        print(f"🔍 基本QueryEngineを初期化中...")
        
        query_engine = QueryEngine(
            document_store=doc_store,
            vector_store=vector_store,
            retriever=SimpleRetriever(vector_store),
            reranker=SimpleReranker(),
            reader=SimpleReader(),
            config=QueryEngineConfig(
                enable_query_normalization=True,
                include_sources=True,
                include_confidence=True
            )
        )
        
        print(f"✅ QueryEngine initialized successfully")
        print(f"✅ QueryEngine初期化成功")
        
        # Demonstration sequence / デモシーケンス
        demonstrate_basic_queries(query_engine)
        demonstrate_advanced_configurations(doc_store, vector_store)
        demonstrate_query_analysis(query_engine)
        demonstrate_performance_monitoring(query_engine)
        demonstrate_error_handling(query_engine)
        
        print("\n" + "="*60)
        print("🎉 TUTORIAL COMPLETE / チュートリアル完了")
        print("="*60)
        print("✅ All QueryEngine demonstrations completed successfully!")
        print("✅ すべてのQueryEngineデモが正常に完了しました！")
        print("")
        print("📚 What you learned / 学習内容:")
        print("   • Basic query operations and result analysis")
        print("     基本クエリ操作と結果分析")
        print("   • Advanced component configurations")
        print("     高度なコンポーネント設定")
        print("   • Performance optimization techniques")
        print("     パフォーマンス最適化技術")
        print("   • Error handling and recovery strategies")
        print("     エラーハンドリングと復旧戦略")
        print("   • Query result quality assessment")
        print("     クエリ結果品質評価")
        print("")
        print(f"📁 Generated files available in: {temp_dir}")
        print(f"📁 生成ファイルの場所: {temp_dir}")
        print("")
        print("Next step / 次のステップ:")
        print("→ Part 3: Evaluation Tutorial (評価チュートリアル)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Tutorial failed / チュートリアル失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup (comment out to inspect generated files)
        # クリーンアップ（生成ファイルを確認する場合はコメントアウト）
        # shutil.rmtree(temp_dir)
        pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)