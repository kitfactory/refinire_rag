#!/usr/bin/env python3
"""
QualityLab統合例

TestSuite、Evaluator、ContradictionDetector、InsightReporterを
統合したRAGシステム品質評価のサンプルです。
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.processing.test_suite import TestSuite, TestSuiteConfig
from refinire_rag.processing.evaluator import Evaluator, EvaluatorConfig
from refinire_rag.processing.contradiction_detector import ContradictionDetector, ContradictionDetectorConfig
from refinire_rag.processing.insight_reporter import InsightReporter, InsightReporterConfig
from refinire_rag.processing.document_pipeline import DocumentPipeline
from refinire_rag.models.document import Document


def create_quality_lab_pipeline():
    """QualityLabパイプラインを作成"""
    
    print("🔬 QualityLab統合パイプラインを作成中...")
    
    # 各コンポーネントの設定
    test_suite_config = TestSuiteConfig(
        auto_generate_cases=True,
        max_cases_per_document=5,
        include_negative_cases=True
    )
    
    evaluator_config = EvaluatorConfig(
        include_category_analysis=True,
        include_failure_analysis=True,
        accuracy_threshold=0.8,
        response_time_threshold=2.0,
        confidence_threshold=0.7
    )
    
    contradiction_config = ContradictionDetectorConfig(
        enable_claim_extraction=True,
        enable_nli_detection=True,
        contradiction_threshold=0.7,
        check_within_document=True,
        check_across_documents=True
    )
    
    insight_config = InsightReporterConfig(
        enable_trend_analysis=True,
        enable_comparative_analysis=True,
        enable_root_cause_analysis=True,
        include_executive_summary=True,
        include_action_items=True
    )
    
    # コンポーネント作成
    test_suite = TestSuite(test_suite_config)
    evaluator = Evaluator(evaluator_config)
    contradiction_detector = ContradictionDetector(contradiction_config)
    insight_reporter = InsightReporter(insight_config)
    
    print("✅ QualityLabコンポーネントを作成完了")
    
    return {
        "test_suite": test_suite,
        "evaluator": evaluator,
        "contradiction_detector": contradiction_detector,
        "insight_reporter": insight_reporter
    }


def create_sample_documents():
    """評価用サンプルドキュメントを作成"""
    
    documents = [
        Document(
            id="rag_analysis",
            content="""
            RAG（Retrieval-Augmented Generation）は革新的な技術です。
            この技術により、LLMの知識制限を克服できます。
            RAGシステムは高い精度を実現します。
            しかし、RAGシステムの精度は低いという報告もあります。
            実装コストは比較的低く抑えられます。
            評価指標としてBLEU、ROUGE、BERTScoreが使用されます。
            """,
            metadata={"title": "RAG分析", "category": "技術解説"}
        ),
        
        Document(
            id="evaluation_results",
            content="""
            # テスト結果
            
            ## ✅ PASS test_rag_basic
            **Query**: RAGとは何ですか？
            **Confidence**: 0.85
            **Processing Time**: 1.2s
            **Sources Found**: 2
            
            ## ❌ FAIL test_complex_query
            **Query**: 複雑な技術的問題について
            **Confidence**: 0.3
            **Processing Time**: 4.5s
            **Sources Found**: 0
            
            ## ✅ PASS test_simple_fact
            **Query**: 基本的な事実について
            **Confidence**: 0.9
            **Processing Time**: 0.8s
            **Sources Found**: 3
            """,
            metadata={
                "processing_stage": "test_execution",
                "tests_run": 3,
                "tests_passed": 2,
                "success_rate": 0.67,
                "source_document_id": "rag_analysis"
            }
        ),
        
        Document(
            id="system_metrics",
            content="""
            # システム評価レポート
            
            ## 主要メトリクス
            - **精度 (Accuracy)**: 0.75
            - **適合率 (Precision)**: 0.82
            - **再現率 (Recall)**: 0.68
            - **F1スコア**: 0.744
            - **平均信頼度**: 0.72
            - **平均応答時間**: 2.1秒
            - **ソース精度**: 0.8
            - **一貫性**: 0.65
            """,
            metadata={
                "processing_stage": "evaluation",
                "overall_score": 0.75,
                "source_document_id": "evaluation_results"
            }
        )
    ]
    
    return documents


def demo_quality_lab_workflow():
    """QualityLabワークフローのデモ"""
    
    print("\n🚀 QualityLabワークフローデモ")
    print("="*60)
    
    # パイプライン作成
    components = create_quality_lab_pipeline()
    documents = create_sample_documents()
    
    print(f"📄 {len(documents)}件のドキュメントを処理します")
    
    # 処理結果を保存
    all_results = []
    
    # 1. テストケース生成・実行
    print("\n1️⃣ TestSuite: テストケース生成・実行")
    test_suite = components["test_suite"]
    
    for doc in documents[:1]:  # 最初のドキュメントのみテスト生成
        results = test_suite.process(doc)
        all_results.extend(results)
        
        result_doc = results[0]
        print(f"   📋 {doc.id}: {result_doc.metadata['generated_cases_count']}件のテストケース生成")
    
    # 2. 矛盾検出
    print("\n2️⃣ ContradictionDetector: 矛盾検出")
    contradiction_detector = components["contradiction_detector"]
    
    for doc in documents:
        results = contradiction_detector.process(doc)
        all_results.extend(results)
        
        result_doc = results[0]
        claims_extracted = result_doc.metadata['claims_extracted']
        contradictions_found = result_doc.metadata['contradictions_found']
        print(f"   🔍 {doc.id}: {claims_extracted}クレーム, {contradictions_found}矛盾")
    
    # 3. 評価メトリクス計算
    print("\n3️⃣ Evaluator: メトリクス計算")
    evaluator = components["evaluator"]
    
    evaluation_docs = [doc for doc in documents if doc.metadata.get("processing_stage") in ["test_execution", "evaluation"]]
    
    for doc in evaluation_docs:
        results = evaluator.process(doc)
        all_results.extend(results)
        
        result_doc = results[0]
        metrics_computed = result_doc.metadata['metrics_computed']
        overall_score = result_doc.metadata['overall_score']
        print(f"   📊 {doc.id}: {metrics_computed}メトリクス, スコア{overall_score:.2f}")
    
    # 4. インサイト生成
    print("\n4️⃣ InsightReporter: インサイト生成")
    insight_reporter = components["insight_reporter"]
    
    insight_docs = [doc for doc in all_results if doc.metadata.get("processing_stage") == "evaluation"]
    
    for doc in insight_docs:
        results = insight_reporter.process(doc)
        all_results.extend(results)
        
        result_doc = results[0]
        insights_generated = result_doc.metadata['insights_generated']
        critical_insights = result_doc.metadata['critical_insights']
        health_score = result_doc.metadata['overall_health_score']
        print(f"   💡 {doc.id}: {insights_generated}インサイト({critical_insights}重要), ヘルス{health_score:.2f}")
    
    return all_results, components


def demo_integrated_pipeline():
    """統合パイプラインのデモ"""
    
    print("\n🔗 統合パイプラインデモ")
    print("="*60)
    
    # 全コンポーネントを含むパイプラインを作成
    components = create_quality_lab_pipeline()
    
    # パイプライン構築（矛盾検出→評価→インサイト生成）
    quality_pipeline = DocumentPipeline([
        components["contradiction_detector"],
        components["evaluator"], 
        components["insight_reporter"]
    ])
    
    # テストデータ
    test_document = Document(
        id="integrated_test",
        content="""
        # 統合テスト結果
        
        ## 主要メトリクス
        - **精度**: 0.65
        - **応答時間**: 3.2秒  
        - **信頼度**: 0.45
        - **F1スコア**: 0.612
        
        ## テスト結果サマリー
        - 総テスト数: 10
        - 成功数: 6
        - 失敗数: 4
        - 成功率: 60%
        """,
        metadata={
            "processing_stage": "evaluation",
            "overall_score": 0.65,
            "tests_run": 10,
            "tests_passed": 6,
            "success_rate": 0.6
        }
    )
    
    print("📋 統合パイプラインを実行中...")
    
    # パイプライン実行
    final_results = quality_pipeline.process(test_document)
    
    print(f"✅ パイプライン完了: {len(final_results)}件の結果生成")
    
    # 最終結果の表示
    for result in final_results:
        stage = result.metadata.get("processing_stage", "unknown")
        print(f"\n📄 結果ドキュメント: {result.id} ({stage})")
        
        if stage == "insight_reporting":
            # インサイトレポートの要約を表示
            insights_count = result.metadata.get("insights_generated", 0)
            critical_count = result.metadata.get("critical_insights", 0)
            health_score = result.metadata.get("overall_health_score", 0)
            
            print(f"   💡 インサイト: {insights_count}件 (緊急: {critical_count}件)")
            print(f"   🏥 ヘルススコア: {health_score:.2f}")
            
            # レポート内容の一部を表示
            content_lines = result.content.split('\n')[:15]
            print(f"   📝 レポート抜粋:")
            for line in content_lines:
                if line.strip():
                    print(f"      {line}")
            if len(result.content.split('\n')) > 15:
                print(f"      ... (続きあり)")
    
    return final_results


def demo_quality_lab_summary():
    """QualityLabサマリーのデモ"""
    
    print("\n📋 QualityLabサマリーデモ")
    print("="*60)
    
    # ワークフロー実行
    results, components = demo_quality_lab_workflow()
    
    # 各コンポーネントのサマリーを取得
    print("\n📊 コンポーネント別サマリー:")
    
    # TestSuiteサマリー
    test_summary = components["test_suite"].get_test_summary()
    print(f"\n🧪 TestSuite:")
    if "total_tests" in test_summary:
        print(f"   - 総テスト数: {test_summary['total_tests']}")
        print(f"   - 成功率: {test_summary['success_rate']:.1%}")
        print(f"   - 平均信頼度: {test_summary['average_confidence']:.3f}")
    else:
        print(f"   - {test_summary.get('message', 'データなし')}")
    
    # Evaluatorサマリー  
    eval_summary = components["evaluator"].get_summary_metrics()
    print(f"\n📈 Evaluator:")
    if eval_summary:
        print(f"   - 総合スコア: {eval_summary.get('overall_score', 0):.2f}")
        print(f"   - 精度: {eval_summary.get('accuracy', 0):.1%}")
        print(f"   - F1スコア: {eval_summary.get('f1_score', 0):.3f}")
    else:
        print("   - データなし")
    
    # ContradictionDetectorサマリー
    contradiction_summary = components["contradiction_detector"].get_contradiction_summary()
    print(f"\n🔍 ContradictionDetector:")
    print(f"   - 総矛盾数: {contradiction_summary['total_contradictions']}")
    print(f"   - 総クレーム数: {contradiction_summary['total_claims']}")
    print(f"   - 一貫性ステータス: {contradiction_summary['consistency_status']}")
    
    # InsightReporterサマリー
    insight_summary = components["insight_reporter"].get_insight_summary()
    print(f"\n💡 InsightReporter:")
    if "total_insights" in insight_summary:
        print(f"   - 総インサイト数: {insight_summary['total_insights']}")
        print(f"   - 平均信頼度: {insight_summary['average_confidence']:.3f}")
        print(f"   - 推奨事項数: {insight_summary['recommendations_count']}")
    else:
        print(f"   - {insight_summary.get('message', 'データなし')}")


def main():
    """メイン関数"""
    
    print("🚀 QualityLab統合デモ")
    print("RAGシステムの包括的な品質評価を実行します")
    
    try:
        # 1. 基本ワークフロー
        demo_quality_lab_workflow()
        
        # 2. 統合パイプライン
        demo_integrated_pipeline()
        
        # 3. サマリー表示
        demo_quality_lab_summary()
        
        print("\n🎉 QualityLab統合デモが完了しました！")
        print("\n📚 QualityLabの主な機能:")
        print("   ✅ TestSuite: 自動テストケース生成・実行")
        print("   ✅ Evaluator: 包括的メトリクス計算・分析")
        print("   ✅ ContradictionDetector: クレーム抽出・矛盾検出")
        print("   ✅ InsightReporter: 閾値ベースインサイト・推奨事項生成")
        print("   ✅ 統合パイプライン: 全コンポーネントの連携動作")
        
        print("\n🎯 QualityLabにより実現できること:")
        print("   📊 システム性能の定量的評価")
        print("   🔍 データ品質・一貫性の検証")
        print("   💡 改善領域の特定と優先順位付け")
        print("   📋 ステークホルダー向けレポート生成")
        print("   🚀 継続的品質改善の支援")
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)