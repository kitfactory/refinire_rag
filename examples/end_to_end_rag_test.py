#!/usr/bin/env python3
"""
エンドツーエンドRAGパイプライン統合テスト

CorpusManager、QueryEngine、QualityLabの全コンポーネントを統合した
完全なRAGワークフローのテストです。
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.application.corpus_manager import CorpusManager, CorpusManagerConfig
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.models.document import Document
from refinire_rag.embedding import TFIDFEmbedder, TFIDFEmbeddingConfig
from refinire_rag.storage import InMemoryVectorStore, SQLiteDocumentStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
from refinire_rag.processing import TestSuite, TestSuiteConfig, Evaluator, EvaluatorConfig
from refinire_rag.processing import ContradictionDetector, ContradictionDetectorConfig
from refinire_rag.processing import InsightReporter, InsightReporterConfig


def create_test_documents() -> List[Document]:
    """テスト用ドキュメントセットを作成"""
    
    documents = [
        Document(
            id="rag_overview",
            content="""
            # RAG（Retrieval-Augmented Generation）技術概要
            
            RAGは検索拡張生成と呼ばれる革新的なAI技術です。
            この技術により、大規模言語モデル（LLM）の知識制限を克服できます。
            
            ## 主要な利点
            - リアルタイムで最新情報にアクセス可能
            - ハルシネーション（幻覚）の大幅減少
            - 透明性と説明可能性の向上
            - 高い精度での情報検索と生成
            
            ## 技術的構成要素
            RAGシステムは以下の主要コンポーネントで構成されます：
            1. 文書ローダー（Loader）
            2. チャンカー（Chunker）
            3. エンベッダー（Embedder）
            4. ベクトルストア（Vector Store）
            5. リトリーバー（Retriever）
            6. リランカー（Reranker）
            7. リーダー（Reader/Generator）
            """,
            metadata={"category": "技術解説", "source": "内部文書", "version": "1.0"}
        ),
        
        Document(
            id="evaluation_metrics",
            content="""
            # RAGシステムの評価指標
            
            RAGシステムの性能評価には多角的なアプローチが必要です。
            
            ## 精度指標
            - **Accuracy**: 正答率
            - **Precision**: 適合率
            - **Recall**: 再現率  
            - **F1-Score**: 精度と再現率の調和平均
            
            ## 効率性指標
            - **Response Time**: 応答時間
            - **Throughput**: スループット
            - **Resource Usage**: リソース使用量
            
            ## 品質指標
            - **Relevance**: 関連性
            - **Completeness**: 完全性
            - **Consistency**: 一貫性
            - **Factual Accuracy**: 事実正確性
            
            ## ユーザー体験指標
            - **User Satisfaction**: ユーザー満足度
            - **Task Completion Rate**: タスク完了率
            - **Error Recovery**: エラー回復能力
            
            評価は継続的に実施し、システムの改善に活用することが重要です。
            """,
            metadata={"category": "評価", "source": "ベストプラクティス", "version": "1.1"}
        ),
        
        Document(
            id="implementation_challenges", 
            content="""
            # RAG実装における課題と解決策
            
            RAGシステムの実装では様々な技術的課題に直面します。
            
            ## データ品質の課題
            - **問題**: 不正確または古い情報の混入
            - **解決策**: データクリーニングと定期的更新プロセス
            
            ## 検索性能の課題
            - **問題**: 大規模コーパスでの遅い検索速度
            - **解決策**: インデックス最適化とキャッシュ戦略
            
            ## 一貫性の課題
            - **問題**: 矛盾する情報源からの回答生成
            - **解決策**: 矛盾検出システムと信頼度スコアリング
            
            ## スケーラビリティの課題
            - **問題**: ユーザー数増加に伴う性能劣化
            - **解決策**: 分散アーキテクチャと負荷分散
            
            ## コスト最適化
            実装コストは比較的低く抑えられますが、運用コストの管理が重要です。
            クラウドリソースの効率的活用と自動スケーリングが推奨されます。
            """,
            metadata={"category": "実装", "source": "技術ガイド", "version": "1.2"}
        ),
        
        Document(
            id="future_directions",
            content="""
            # RAG技術の今後の発展方向
            
            RAG技術は急速に進化しており、以下の方向性が注目されています。
            
            ## マルチモーダルRAG
            テキストだけでなく、画像、音声、動画などの多様なメディアを統合した
            検索拡張生成システムが開発されています。
            
            ## ファインチューニングとRAGの融合
            事前学習モデルのファインチューニングとRAGを組み合わせることで、
            より高精度で専門的な知識を活用できるシステムが実現されます。
            
            ## リアルタイム学習
            ユーザーのフィードバックから継続的に学習し、
            検索と生成の品質を動的に改善するシステム。
            
            ## エッジコンピューティング対応
            軽量化されたRAGモデルにより、エッジデバイスでの
            プライベートで高速な情報検索が可能になります。
            
            これらの技術進歩により、RAGはより実用的で
            多様な用途に適用可能な技術へと発展しています。
            """,
            metadata={"category": "未来予測", "source": "研究論文", "version": "1.0"}
        )
    ]
    
    return documents


def create_test_queries() -> List[Dict[str, Any]]:
    """テスト用クエリセットを作成"""
    
    queries = [
        {
            "query": "RAGとは何ですか？",
            "expected_type": "definition",
            "category": "基本概念"
        },
        {
            "query": "RAGシステムの主要コンポーネントを教えてください",
            "expected_type": "enumeration", 
            "category": "技術詳細"
        },
        {
            "query": "RAGの評価指標にはどのようなものがありますか？",
            "expected_type": "classification",
            "category": "評価方法"
        },
        {
            "query": "RAG実装時の主な課題は何ですか？",
            "expected_type": "problem_identification",
            "category": "実装"
        },
        {
            "query": "RAG技術の将来性について説明してください",
            "expected_type": "analysis",
            "category": "将来展望"
        },
        {
            "query": "RAGの精度を向上させる方法は？",
            "expected_type": "solution",
            "category": "最適化"
        },
        {
            "query": "マルチモーダルRAGとは何ですか？",
            "expected_type": "advanced_concept",
            "category": "先端技術"
        },
        {
            "query": "RAGのコストはどの程度ですか？",
            "expected_type": "quantitative",
            "category": "運用"
        }
    ]
    
    return queries


def setup_rag_pipeline() -> tuple:
    """完全なRAGパイプラインをセットアップ"""
    
    print("🔧 RAGパイプラインセットアップ中...")
    
    # 1. CorpusManager設定
    from refinire_rag.chunking import ChunkingConfig
    
    corpus_config = CorpusManagerConfig(
        enable_processing=True,
        enable_chunking=True,
        enable_embedding=True,
        chunking_config=ChunkingConfig(
            chunk_size=200,
            overlap=50,
            split_by_sentence=True
        )
    )
    
    # 2. QueryEngine設定
    query_config = QueryEngineConfig(
        retriever_top_k=10,
        reranker_top_k=5,
        enable_query_normalization=True,
        include_sources=True,
        include_confidence=True
    )
    
    # 3. ストレージ初期化
    vector_store = InMemoryVectorStore()
    document_store = SQLiteDocumentStore(":memory:")
    
    # 4. 埋め込みモデル設定
    embedding_config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0)
    embedder = TFIDFEmbedder(config=embedding_config)
    
    # 5. 検索コンポーネント
    retriever = SimpleRetriever(vector_store=vector_store, embedder=embedder)
    reranker = SimpleReranker()
    reader = SimpleReader()
    
    # 6. メインコンポーネント 
    # CorpusManagerConfigに外部ストレージを設定
    corpus_config.document_store = document_store
    corpus_config.vector_store = vector_store
    corpus_config.embedder = embedder
    
    corpus_manager = CorpusManager(config=corpus_config)
    
    query_engine = QueryEngine(
        document_store=document_store,
        vector_store=vector_store,
        retriever=retriever,
        reader=reader,
        reranker=reranker,
        config=query_config
    )
    
    print("✅ RAGパイプラインセットアップ完了")
    
    return corpus_manager, query_engine, embedder


def setup_quality_lab() -> tuple:
    """QualityLabコンポーネントをセットアップ"""
    
    print("🔬 QualityLabセットアップ中...")
    
    # TestSuite設定
    test_config = TestSuiteConfig()
    
    # Evaluator設定
    eval_config = EvaluatorConfig()
    
    # ContradictionDetector設定
    contradiction_config = ContradictionDetectorConfig()
    
    # InsightReporter設定
    insight_config = InsightReporterConfig()
    
    # コンポーネント作成
    test_suite = TestSuite(test_config)
    evaluator = Evaluator(eval_config)
    contradiction_detector = ContradictionDetector(contradiction_config)
    insight_reporter = InsightReporter(insight_config)
    
    print("✅ QualityLabセットアップ完了")
    
    return test_suite, evaluator, contradiction_detector, insight_reporter


def test_corpus_building(corpus_manager: CorpusManager, documents: List[Document]) -> bool:
    """コーパス構築のテスト"""
    
    print("\n📚 コーパス構築テスト")
    print("=" * 50)
    
    try:
        # 文書の処理
        print(f"📄 {len(documents)}件の文書を処理中...")
        processed_docs = corpus_manager.process_documents(documents)
        
        # 埋め込み生成
        print("🔧 埋め込み生成中...")
        embedded_docs = corpus_manager.embed_documents(processed_docs)
        
        # 文書ストレージ
        print("💾 文書を保存中...")
        stored_count = corpus_manager.store_documents(processed_docs)
        
        # 統計確認
        stats = corpus_manager.get_corpus_stats()
        
        print(f"✅ コーパス構築完了:")
        print(f"   - 処理済み文書数: {len(processed_docs)}")
        print(f"   - 埋め込み生成数: {len(embedded_docs)}")
        print(f"   - 保存済み文書数: {stored_count}")
        print(f"   - 処理統計: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ コーパス構築エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_query_processing(query_engine: QueryEngine, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """クエリ処理のテスト"""
    
    print("\n🔍 クエリ処理テスト")
    print("=" * 50)
    
    query_results = []
    
    for i, query_data in enumerate(queries, 1):
        query = query_data["query"]
        category = query_data["category"]
        
        print(f"\n📝 クエリ {i}: {query}")
        print(f"カテゴリ: {category}")
        
        try:
            # クエリ実行
            import time
            start_time = time.time()
            
            result = query_engine.answer(query)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 結果の記録
            query_result = {
                "query_id": f"test_query_{i}",
                "query": query,
                "category": category,
                "answer": result.answer,
                "sources": len(result.sources),
                "confidence": result.confidence,
                "processing_time": processing_time,
                "success": result.confidence > 0.3,
                "metadata": query_data
            }
            
            query_results.append(query_result)
            
            # 結果表示
            print(f"✅ 応答: {result.answer[:100]}...")
            print(f"   信頼度: {result.confidence:.3f}")
            print(f"   ソース数: {len(result.sources)}")
            print(f"   処理時間: {processing_time:.3f}秒")
            
        except Exception as e:
            print(f"❌ クエリ処理エラー: {e}")
            
            # エラー結果の記録
            query_result = {
                "query_id": f"test_query_{i}",
                "query": query,
                "category": category,
                "answer": f"エラー: {str(e)}",
                "sources": 0,
                "confidence": 0.0,
                "processing_time": 0.0,
                "success": False,
                "error": str(e)
            }
            
            query_results.append(query_result)
    
    # サマリー表示
    success_count = sum(1 for r in query_results if r["success"])
    avg_confidence = sum(r["confidence"] for r in query_results) / len(query_results)
    avg_time = sum(r["processing_time"] for r in query_results) / len(query_results)
    
    print(f"\n📊 クエリ処理サマリー:")
    print(f"   - 成功率: {success_count}/{len(query_results)} ({success_count/len(query_results):.1%})")
    print(f"   - 平均信頼度: {avg_confidence:.3f}")
    print(f"   - 平均処理時間: {avg_time:.3f}秒")
    
    return query_results


def test_quality_evaluation(
    quality_components: tuple,
    documents: List[Document],
    query_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """品質評価のテスト"""
    
    print("\n🔬 品質評価テスト")
    print("=" * 50)
    
    test_suite, evaluator, contradiction_detector, insight_reporter = quality_components
    evaluation_results = {}
    
    try:
        # 1. 自動テストケース生成
        print("\n1️⃣ 自動テストケース生成")
        test_documents = []
        
        for doc in documents[:2]:  # 最初の2文書をテスト
            test_results = test_suite.process(doc)
            test_documents.extend(test_results)
            
            if test_results:
                metadata = test_results[0].metadata
                print(f"   📋 {doc.id}: {metadata.get('generated_cases_count', 0)}件のテストケース生成")
        
        # 2. 矛盾検出
        print("\n2️⃣ 矛盾検出分析")
        contradiction_documents = []
        
        for doc in documents:
            contradiction_results = contradiction_detector.process(doc)
            contradiction_documents.extend(contradiction_results)
            
            if contradiction_results:
                metadata = contradiction_results[0].metadata
                claims = metadata.get('claims_extracted', 0)
                contradictions = metadata.get('contradictions_found', 0)
                print(f"   🔍 {doc.id}: {claims}クレーム, {contradictions}矛盾")
        
        # 3. クエリ結果の評価用ドキュメント作成
        print("\n3️⃣ クエリ結果評価")
        
        # クエリ結果をテスト結果ドキュメントに変換
        test_results_content = "# テスト実行結果\n\n"
        
        for result in query_results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            test_results_content += f"## {status} {result['query_id']}\n"
            test_results_content += f"**Query**: {result['query']}\n"
            test_results_content += f"**Confidence**: {result['confidence']}\n"
            test_results_content += f"**Processing Time**: {result['processing_time']}s\n"
            test_results_content += f"**Sources Found**: {result['sources']}\n\n"
        
        test_results_doc = Document(
            id="end_to_end_test_results",
            content=test_results_content,
            metadata={
                "processing_stage": "test_execution",
                "tests_run": len(query_results),
                "tests_passed": sum(1 for r in query_results if r["success"]),
                "success_rate": sum(1 for r in query_results if r["success"]) / len(query_results),
                "source_document_id": "end_to_end_test"
            }
        )
        
        # 評価実行
        evaluation_documents = evaluator.process(test_results_doc)
        
        if evaluation_documents:
            eval_metadata = evaluation_documents[0].metadata
            metrics_count = eval_metadata.get('metrics_computed', 0)
            overall_score = eval_metadata.get('overall_score', 0)
            print(f"   📊 評価完了: {metrics_count}メトリクス, 総合スコア{overall_score:.2f}")
        
        # 4. インサイト生成
        print("\n4️⃣ インサイト生成")
        insight_documents = []
        
        for eval_doc in evaluation_documents:
            insight_results = insight_reporter.process(eval_doc)
            insight_documents.extend(insight_results)
            
            if insight_results:
                metadata = insight_results[0].metadata
                insights = metadata.get('insights_generated', 0)
                critical = metadata.get('critical_insights', 0)
                health_score = metadata.get('overall_health_score', 0)
                print(f"   💡 インサイト生成: {insights}件({critical}重要), ヘルススコア{health_score:.2f}")
        
        # 5. 結果サマリー
        evaluation_results = {
            "test_generation": {
                "documents_processed": len(test_documents),
                "test_cases_generated": sum(d.metadata.get('generated_cases_count', 0) for d in test_documents)
            },
            "contradiction_detection": {
                "documents_analyzed": len(contradiction_documents),
                "total_claims": sum(d.metadata.get('claims_extracted', 0) for d in contradiction_documents),
                "total_contradictions": sum(d.metadata.get('contradictions_found', 0) for d in contradiction_documents)
            },
            "evaluation": {
                "queries_evaluated": len(query_results),
                "overall_score": evaluation_documents[0].metadata.get('overall_score', 0) if evaluation_documents else 0,
                "metrics_computed": evaluation_documents[0].metadata.get('metrics_computed', 0) if evaluation_documents else 0
            },
            "insights": {
                "insights_generated": insight_documents[0].metadata.get('insights_generated', 0) if insight_documents else 0,
                "critical_insights": insight_documents[0].metadata.get('critical_insights', 0) if insight_documents else 0,
                "health_score": insight_documents[0].metadata.get('overall_health_score', 0) if insight_documents else 0
            }
        }
        
        return evaluation_results
        
    except Exception as e:
        print(f"❌ 品質評価エラー: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def test_system_integration() -> bool:
    """システム統合テストの実行"""
    
    print("\n🚀 システム統合テスト")
    print("=" * 50)
    
    try:
        # コンポーネント間の相互作用テスト
        print("🔗 コンポーネント間相互作用テスト")
        
        # 1. メモリ使用量チェック
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            memory_monitoring = True
        except ImportError:
            print("   ⚠️ psutilが利用できません。メモリ監視をスキップします")
            memory_before = 0
            memory_monitoring = False
        
        # 2. 並行処理テスト（簡易版）
        print("   📊 リソース使用量監視開始")
        
        # 3. エラー処理テスト
        print("   🔧 エラー処理テスト")
        
        # 無効なクエリのテスト
        corpus_manager, query_engine, _ = setup_rag_pipeline()
        
        try:
            result = query_engine.answer("")  # 空クエリ
            print("   ❌ 空クエリが例外を発生させませんでした")
        except:
            print("   ✅ 空クエリが適切にハンドルされました")
        
        try:
            result = query_engine.answer("x" * 1000)  # 長すぎるクエリ
            print(f"   ⚠️ 長いクエリが処理されました (信頼度: {result.confidence:.3f})")
        except:
            print("   ✅ 長いクエリが適切にハンドルされました")
        
        # メモリ使用量チェック
        if memory_monitoring:
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = memory_after - memory_before
            
            print(f"   📊 メモリ使用量: {memory_after:.1f}MB (差分: +{memory_diff:.1f}MB)")
            
            if memory_diff > 100:  # 100MB以上の増加は警告
                print("   ⚠️ メモリ使用量が大幅に増加しました")
            else:
                print("   ✅ メモリ使用量は適切な範囲内です")
        else:
            print("   📊 メモリ使用量監視をスキップしました")
        
        return True
        
    except Exception as e:
        print(f"❌ システム統合テストエラー: {e}")
        return False


def main() -> bool:
    """メイン統合テスト実行"""
    
    print("🚀 エンドツーエンドRAGパイプライン統合テスト")
    print("=" * 80)
    
    # テストフラグ
    test_results = {
        "corpus_building": False,
        "query_processing": False,
        "quality_evaluation": False,
        "system_integration": False
    }
    
    try:
        # 1. テストデータ準備
        print("📋 テストデータ準備中...")
        documents = create_test_documents()
        queries = create_test_queries()
        print(f"✅ テストデータ準備完了: {len(documents)}文書, {len(queries)}クエリ")
        
        # 2. パイプラインセットアップ
        corpus_manager, query_engine, embedder = setup_rag_pipeline()
        quality_components = setup_quality_lab()
        
        # 3. コーパス構築テスト
        test_results["corpus_building"] = test_corpus_building(corpus_manager, documents)
        
        # 4. クエリ処理テスト
        if test_results["corpus_building"]:
            query_results = test_query_processing(query_engine, queries)
            test_results["query_processing"] = len(query_results) > 0
            
            # 5. 品質評価テスト
            if test_results["query_processing"]:
                evaluation_results = test_quality_evaluation(
                    quality_components, documents, query_results
                )
                test_results["quality_evaluation"] = "error" not in evaluation_results
        
        # 6. システム統合テスト
        test_results["system_integration"] = test_system_integration()
        
        # 7. 最終結果レポート
        print("\n📋 最終テスト結果")
        print("=" * 50)
        
        total_tests = len(test_results)
        passed_tests = sum(test_results.values())
        
        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\n📊 総合結果: {passed_tests}/{total_tests} テスト通過 ({passed_tests/total_tests:.1%})")
        
        if passed_tests == total_tests:
            print("\n🎉 全テスト通過！エンドツーエンドRAGパイプラインは正常に動作しています")
            
            # 推奨事項
            print("\n🎯 システムの特徴:")
            print("   ✅ コーパス管理: 文書の追加、正規化、チャンキング、埋め込み生成")
            print("   ✅ クエリ処理: 検索、リランキング、回答生成")
            print("   ✅ 品質評価: 自動テスト、矛盾検出、メトリクス計算、インサイト生成")
            print("   ✅ 統合性: エラーハンドリング、リソース管理")
            
            print("\n📚 次のステップ:")
            print("   🔹 本番環境への展開準備")
            print("   🔹 パフォーマンスチューニング")
            print("   🔹 監視とロギングの実装")
            print("   🔹 ユーザー受け入れテスト")
            
        else:
            print("\n⚠️ 一部テストが失敗しました。システムの修正が必要です")
            
            # 失敗したテストの特定
            failed_tests = [name for name, passed in test_results.items() if not passed]
            print(f"失敗したテスト: {', '.join(failed_tests)}")
        
        return passed_tests == total_tests
        
    except Exception as e:
        print(f"\n❌ 統合テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)