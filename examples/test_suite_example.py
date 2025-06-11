#!/usr/bin/env python3
"""
TestSuite使用例

RAGシステムの評価を実行するサンプルです。
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.processing.test_suite import TestSuite, TestSuiteConfig, TestCase
from refinire_rag.models.document import Document


def demo_test_case_generation():
    """テストケース生成のデモ"""
    
    print("🧪 TestSuite - テストケース生成デモ")
    print("=" * 60)
    
    # TestSuite設定
    config = TestSuiteConfig(
        auto_generate_cases=True,
        max_cases_per_document=4,
        include_negative_cases=True
    )
    
    test_suite = TestSuite(config)
    
    # サンプルドキュメント
    documents = [
        Document(
            id="rag_doc",
            content="""
            RAG（Retrieval-Augmented Generation）は、検索拡張生成技術です。
            この技術は、大規模言語モデルと外部知識ベースを組み合わせることで、
            より正確で根拠のある回答を生成します。
            RAGの主な利点は、ハルシネーションの減少、知識の更新容易性、
            専門ドメインへの適応性です。
            評価方法としては、BLEU、ROUGE、BERTScoreなどが使用されます。
            """,
            metadata={"title": "RAG概要", "category": "技術解説"}
        ),
        
        Document(
            id="vector_doc",
            content="""
            ベクトル検索は、意味的類似性に基づく検索技術です。
            文書やクエリを高次元ベクトル空間に埋め込み、
            コサイン類似度などを使用して関連性を計算します。
            従来のキーワード検索では発見できない
            文脈的に関連する情報を見つけることができます。
            実装方法にはFaiss、Chroma、Weaviateなどがあります。
            """,
            metadata={"title": "ベクトル検索", "category": "技術解説"}
        )
    ]
    
    print("📝 ドキュメントからテストケースを生成中...")
    
    all_test_cases = []
    
    for doc in documents:
        result_docs = test_suite.process(doc)
        print(f"\n📄 Document: {doc.id}")
        print(f"   生成されたテストケース数: {result_docs[0].metadata['generated_cases_count']}")
        print(f"   カテゴリ: {result_docs[0].metadata['categories']}")
        
        # 実際に生成されたテストケースを表示
        print("\n生成されたテストケース:")
        for case in test_suite.test_cases[-result_docs[0].metadata['generated_cases_count']:]:
            print(f"   - ID: {case.id}")
            print(f"     Query: {case.query}")
            print(f"     Category: {case.category}")
            print(f"     Expected Sources: {case.expected_sources}")
            print()
    
    # テストケースをファイルに保存
    output_file = Path(__file__).parent / "generated_test_cases.json"
    test_suite.save_test_cases(str(output_file))
    print(f"💾 テストケースを保存: {output_file}")
    
    return test_suite


def demo_test_execution():
    """テスト実行のデモ"""
    
    print("\n🔬 TestSuite - テスト実行デモ")
    print("=" * 60)
    
    # 事前定義されたテストケースでTestSuiteを作成
    config = TestSuiteConfig(
        auto_generate_cases=False,
        evaluation_criteria={
            "answer_relevance": 0.4,
            "source_accuracy": 0.3,
            "response_time": 0.2,
            "confidence": 0.1
        }
    )
    
    test_suite = TestSuite(config)
    
    # 手動でテストケースを追加
    test_cases = [
        TestCase(
            id="test_rag_definition",
            query="RAGとは何ですか？",
            expected_sources=["rag_doc"],
            category="definition"
        ),
        TestCase(
            id="test_vector_search",
            query="ベクトル検索の仕組みを説明してください",
            expected_sources=["vector_doc"],
            category="how_to"
        ),
        TestCase(
            id="test_irrelevant",
            query="今日の天気はどうですか？",
            expected_sources=[],
            category="negative"
        )
    ]
    
    test_suite.test_cases = test_cases
    
    # 評価対象ドキュメント
    documents = [
        Document(
            id="rag_doc",
            content="RAGは検索拡張生成技術で、LLMと外部知識を組み合わせます。",
            metadata={"title": "RAG概要"}
        ),
        Document(
            id="vector_doc", 
            content="ベクトル検索は意味的類似性で文書を検索する技術です。",
            metadata={"title": "ベクトル検索"}
        )
    ]
    
    print("🔍 テストを実行中...")
    
    for doc in documents:
        result_docs = test_suite.process(doc)
        result_doc = result_docs[0]
        
        print(f"\n📊 Document: {doc.id} の評価結果")
        print(f"   実行テスト数: {result_doc.metadata['tests_run']}")
        print(f"   成功テスト数: {result_doc.metadata['tests_passed']}")
        print(f"   成功率: {result_doc.metadata['success_rate']:.1%}")
    
    # 全体的なサマリーを表示
    summary = test_suite.get_test_summary()
    print(f"\n📈 全体サマリー:")
    print(f"   総テスト数: {summary['total_tests']}")
    print(f"   成功数: {summary['passed_tests']}")
    print(f"   全体成功率: {summary['success_rate']:.1%}")
    print(f"   平均信頼度: {summary['average_confidence']:.3f}")
    print(f"   平均処理時間: {summary['average_processing_time']:.3f}秒")
    
    print("\n📊 カテゴリ別統計:")
    for category, stats in summary['category_stats'].items():
        success_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
        print(f"   {category}: {stats['passed']}/{stats['total']} ({success_rate:.1%})")
    
    # 結果をファイルに保存
    output_file = Path(__file__).parent / "test_results.json"
    test_suite.save_test_results(str(output_file))
    print(f"\n💾 テスト結果を保存: {output_file}")


def demo_comprehensive_evaluation():
    """包括的評価のデモ"""
    
    print("\n🏆 TestSuite - 包括的評価デモ")
    print("=" * 60)
    
    # 生成と実行の両方を行う設定
    config = TestSuiteConfig(
        auto_generate_cases=True,
        max_cases_per_document=2,
        include_negative_cases=True,
        evaluation_criteria={
            "answer_relevance": 0.5,
            "source_accuracy": 0.3,
            "response_time": 0.2
        }
    )
    
    test_suite = TestSuite(config)
    
    # 評価用ドキュメント
    evaluation_docs = [
        Document(
            id="comprehensive_doc",
            content="""
            情報検索システムの評価は多面的なアプローチが必要です。
            精度（Precision）と再現率（Recall）は基本的な指標です。
            F1スコアはこれらの調和平均として計算されます。
            ユーザビリティテストも重要な評価手法の一つです。
            RAGシステムでは、検索精度と生成品質の両方を評価する必要があります。
            """,
            metadata={"title": "評価手法", "category": "評価"}
        )
    ]
    
    print("📋 包括的評価を実行中...")
    
    for doc in evaluation_docs:
        # 1. テストケース生成
        print(f"\n1️⃣ テストケース生成: {doc.id}")
        generation_results = test_suite.process(doc)
        gen_result = generation_results[0]
        
        print(f"   生成されたケース数: {gen_result.metadata['generated_cases_count']}")
        
        # 2. 生成されたケースで評価実行
        print(f"\n2️⃣ 評価実行: {doc.id}")
        test_suite.config.auto_generate_cases = False  # 評価モードに切り替え
        evaluation_results = test_suite.process(doc)
        eval_result = evaluation_results[0]
        
        print(f"   実行されたテスト数: {eval_result.metadata['tests_run']}")
        print(f"   成功率: {eval_result.metadata['success_rate']:.1%}")
    
    # 最終サマリー
    final_summary = test_suite.get_test_summary()
    print(f"\n🎯 最終評価結果:")
    print(f"   総合成功率: {final_summary['success_rate']:.1%}")
    print(f"   システム信頼度: {final_summary['average_confidence']:.3f}")
    
    # 評価品質の判定
    if final_summary['success_rate'] >= 0.8:
        print("   ✅ 評価: 優秀なシステム")
    elif final_summary['success_rate'] >= 0.6:
        print("   🟡 評価: 良好なシステム")
    else:
        print("   🔴 評価: 改善が必要")


def main():
    """メイン関数"""
    
    print("🚀 TestSuite 実装例")
    print("RAGシステムの評価機能をテストします")
    
    try:
        # 1. テストケース生成デモ
        test_suite = demo_test_case_generation()
        
        # 2. テスト実行デモ
        demo_test_execution()
        
        # 3. 包括的評価デモ
        demo_comprehensive_evaluation()
        
        print("\n🎉 TestSuite デモが完了しました！")
        print("\n📚 TestSuiteの主な機能:")
        print("   ✅ ドキュメントからの自動テストケース生成")
        print("   ✅ 手動テストケースの実行")
        print("   ✅ ネガティブテストケースの生成")
        print("   ✅ カテゴリ別評価統計")
        print("   ✅ 結果の保存と読み込み")
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)