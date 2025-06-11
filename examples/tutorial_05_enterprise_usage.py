#!/usr/bin/env python3
"""
企業部門別RAGシステムの実装例

人事部と営業部が独立したRAGシステムを持ち、
データ分離とアクセス制御を実装するサンプルです。
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.use_cases.corpus_manager import CorpusManager, CorpusManagerConfig
from refinire_rag.use_cases.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.models.document import Document
from refinire_rag.embedding import TFIDFEmbedder, TFIDFEmbeddingConfig
from refinire_rag.storage import InMemoryVectorStore, SQLiteDocumentStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
from refinire_rag.processing import TestSuite, TestSuiteConfig, Evaluator, EvaluatorConfig
from refinire_rag.processing import ContradictionDetector, InsightReporter
from refinire_rag.chunking import ChunkingConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DepartmentRAGManager:
    """部門別RAG管理クラス"""
    
    def __init__(self, department: str, data_dir: str = "data"):
        self.department = department
        self.data_dir = Path(data_dir)
        self.department_dir = self.data_dir / department
        self.department_dir.mkdir(parents=True, exist_ok=True)
        
        # 部門別ストレージ
        self.document_store = SQLiteDocumentStore(
            str(self.department_dir / f"{department}_documents.db")
        )
        self.vector_store = InMemoryVectorStore()
        
        # 共有埋め込みモデル（効率性のため）
        self.embedder = TFIDFEmbedder(TFIDFEmbeddingConfig(min_df=1, max_df=1.0))
        
        # コーパスマネージャー設定
        self.corpus_config = CorpusManagerConfig(
            enable_processing=True,
            enable_chunking=True,
            enable_embedding=True,
            chunking_config=ChunkingConfig(
                chunk_size=300,
                overlap=50,
                split_by_sentence=True
            ),
            document_store=self.document_store,
            vector_store=self.vector_store,
            embedder=self.embedder
        )
        
        # コンポーネント初期化
        self.corpus_manager = CorpusManager(config=self.corpus_config)
        
        # クエリエンジン設定
        self.retriever = SimpleRetriever(
            vector_store=self.vector_store, 
            embedder=self.embedder
        )
        self.query_engine = QueryEngine(
            document_store=self.document_store,
            vector_store=self.vector_store,
            retriever=self.retriever,
            reader=SimpleReader(),
            reranker=SimpleReranker()
        )
        
        # 品質監視
        from refinire_rag.processing import ContradictionDetectorConfig, InsightReporterConfig
        
        self.quality_components = {
            'test_suite': TestSuite(TestSuiteConfig()),
            'evaluator': Evaluator(EvaluatorConfig()),
            'contradiction_detector': ContradictionDetector(ContradictionDetectorConfig()),
            'insight_reporter': InsightReporter(InsightReporterConfig())
        }
        
        logger.info(f"{department}部門のRAGシステムを初期化しました")
    
    def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """部門にドキュメントを追加"""
        
        logger.info(f"{self.department}部門に{len(documents)}件のドキュメントを追加中...")
        
        # コーパスマネージャーでドキュメント処理
        processed_docs = self.corpus_manager.process_documents(documents)
        embedded_docs = self.corpus_manager.embed_documents(processed_docs)
        stored_count = self.corpus_manager.store_documents(processed_docs)
        
        stats = self.corpus_manager.get_corpus_stats()
        
        result = {
            'department': self.department,
            'processed_documents': len(processed_docs),
            'embedded_documents': len(embedded_docs),
            'stored_documents': stored_count,
            'corpus_stats': stats
        }
        
        logger.info(f"{self.department}部門: {stored_count}件のドキュメントを処理・保存完了")
        return result
    
    def query(self, question: str, user_department: str) -> Dict[str, Any]:
        """部門別クエリ実行（アクセス制御付き）"""
        
        # アクセス制御チェック
        if user_department != self.department:
            logger.warning(f"アクセス拒否: {user_department}部門のユーザーが{self.department}部門データにアクセス試行")
            return {
                'department': self.department,
                'user_department': user_department,
                'question': question,
                'answer': "アクセス権限がありません。所属部門のデータのみアクセス可能です。",
                'confidence': 0.0,
                'sources': [],
                'access_denied': True
            }
        
        # クエリ実行
        start_time = datetime.now()
        result = self.query_engine.answer(question)
        end_time = datetime.now()
        
        # ログ記録
        self._log_query(user_department, question, result, end_time - start_time)
        
        return {
            'department': self.department,
            'user_department': user_department,
            'question': question,
            'answer': result.answer,
            'confidence': result.confidence,
            'sources': [s.content[:100] + "..." for s in result.sources],
            'processing_time': (end_time - start_time).total_seconds(),
            'access_denied': False
        }
    
    def _log_query(self, user_department: str, question: str, result, processing_time):
        """クエリログ記録"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'department': self.department,
            'user_department': user_department,
            'question': question,
            'confidence': result.confidence,
            'sources_count': len(result.sources),
            'processing_time': processing_time.total_seconds()
        }
        
        logger.info(f"クエリログ: {log_entry}")
    
    def evaluate_quality(self) -> Dict[str, Any]:
        """部門RAGシステムの品質評価"""
        
        logger.info(f"{self.department}部門の品質評価を実行中...")
        
        # サンプルドキュメントから評価
        sample_docs = self._get_sample_documents()
        evaluation_results = {}
        
        for doc in sample_docs:
            # テストケース生成
            test_results = self.quality_components['test_suite'].process(doc)
            
            # 矛盾検出
            contradiction_results = self.quality_components['contradiction_detector'].process(doc)
            
            # 評価メトリクス計算
            if test_results:
                eval_results = self.quality_components['evaluator'].process(test_results[0])
                
                # インサイト生成
                if eval_results:
                    insight_results = self.quality_components['insight_reporter'].process(eval_results[0])
        
        return {
            'department': self.department,
            'evaluation_completed': True,
            'quality_score': 0.85,  # サンプル値
            'recommendations': [
                "定期的なドキュメント更新を推奨",
                "クエリパフォーマンスは良好",
                "データ一貫性チェックを継続"
            ]
        }
    
    def _get_sample_documents(self) -> List[Document]:
        """評価用サンプルドキュメント取得"""
        # 実装では実際のドキュメントストアから取得
        return []


class EnterpriseRAGSystem:
    """企業全体のRAGシステム管理"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.departments: Dict[str, DepartmentRAGManager] = {}
        
        logger.info("企業RAGシステムを初期化しました")
    
    def add_department(self, department_name: str) -> DepartmentRAGManager:
        """新しい部門のRAGシステムを追加"""
        
        if department_name in self.departments:
            logger.warning(f"{department_name}部門は既に存在します")
            return self.departments[department_name]
        
        dept_rag = DepartmentRAGManager(department_name, str(self.data_dir))
        self.departments[department_name] = dept_rag
        
        logger.info(f"{department_name}部門のRAGシステムを追加しました")
        return dept_rag
    
    def get_department(self, department_name: str) -> Optional[DepartmentRAGManager]:
        """部門のRAGシステムを取得"""
        return self.departments.get(department_name)
    
    def query_department(self, department_name: str, question: str, user_department: str) -> Dict[str, Any]:
        """指定部門にクエリを実行"""
        
        dept_rag = self.get_department(department_name)
        if not dept_rag:
            return {
                'error': f"{department_name}部門のRAGシステムが見つかりません",
                'available_departments': list(self.departments.keys())
            }
        
        return dept_rag.query(question, user_department)
    
    def get_system_overview(self) -> Dict[str, Any]:
        """システム全体の概要を取得"""
        
        return {
            'total_departments': len(self.departments),
            'departments': list(self.departments.keys()),
            'data_directory': str(self.data_dir),
            'system_status': 'operational'
        }


def create_hr_documents() -> List[Document]:
    """人事部のサンプルドキュメントを作成"""
    
    return [
        Document(
            id="hr_vacation_policy",
            content="""
            # 有給休暇ポリシー
            
            ## 概要
            当社の有給休暇制度について説明します。
            
            ## 付与日数
            - 入社1年目: 10日間
            - 入社2年目: 11日間
            - 入社3年目以降: 20日間（最大）
            
            ## 取得手順
            1. 事前に上司に相談
            2. 人事システムで申請
            3. 承認後に取得可能
            
            ## 注意事項
            - 繁忙期の取得は制限される場合があります
            - 年度末に未消化分は失効します
            - 病気休暇は別途規定があります
            """,
            metadata={"department": "人事部", "category": "ポリシー", "version": "2024.1"}
        ),
        
        Document(
            id="hr_performance_review",
            content="""
            # 人事評価制度
            
            ## 評価サイクル
            年2回（4月、10月）に実施します。
            
            ## 評価項目
            1. **業績評価** (60%)
               - 目標達成度
               - 成果の質
               - 顧客満足度
            
            2. **行動評価** (40%)
               - チームワーク
               - リーダーシップ
               - 専門スキル向上
            
            ## 評価プロセス
            1. 自己評価の提出
            2. 上司との面談
            3. 多面評価の実施
            4. 最終評価の決定
            5. フィードバック面談
            
            ## 評価結果の活用
            - 昇進・昇格の判断材料
            - 賞与・昇給の決定
            - 研修計画の策定
            """,
            metadata={"department": "人事部", "category": "評価制度", "version": "2024.1"}
        ),
        
        Document(
            id="hr_training_program",
            content="""
            # 社員研修プログラム
            
            ## 新入社員研修
            - 期間: 入社後3ヶ月
            - 内容: 会社概要、ビジネスマナー、基本スキル
            - 担当: 人事部・各部門メンター
            
            ## 継続教育プログラム
            1. **技術研修**
               - 月1回の技術セミナー
               - 外部研修参加支援
               - 資格取得奨励金制度
            
            2. **マネジメント研修**
               - 管理職向け研修（年4回）
               - リーダーシップ開発プログラム
               - コーチング研修
            
            3. **語学研修**
               - 英語・中国語クラス
               - 語学試験費用補助
               - 留学制度
            
            ## 研修申請方法
            1. 年次研修計画の策定
            2. 上司の承認取得
            3. 人事部への申請
            4. 予算確保後実施
            """,
            metadata={"department": "人事部", "category": "研修制度", "version": "2024.1"}
        )
    ]


def create_sales_documents() -> List[Document]:
    """営業部のサンプルドキュメントを作成"""
    
    return [
        Document(
            id="sales_product_catalog",
            content="""
            # 製品カタログ 2024年版
            
            ## エンタープライズソリューション
            
            ### CloudSync Pro
            **価格**: 月額500,000円〜
            **主要機能**:
            - 大容量データ同期
            - 高度なセキュリティ
            - 24/7サポート
            - カスタマイズ可能
            
            **対象顧客**: 大企業（従業員1000名以上）
            
            ### BusinessHub Standard
            **価格**: 月額100,000円〜
            **主要機能**:
            - 基本的なワークフロー
            - 標準レポート機能
            - 営業時間サポート
            - テンプレート提供
            
            **対象顧客**: 中小企業（従業員50-1000名）
            
            ### StartupKit
            **価格**: 月額20,000円〜
            **主要機能**:
            - 基本機能のみ
            - セルフサービス
            - コミュニティサポート
            - 簡単セットアップ
            
            **対象顧客**: スタートアップ（従業員50名未満）
            """,
            metadata={"department": "営業部", "category": "製品情報", "version": "2024.1"}
        ),
        
        Document(
            id="sales_pricing_strategy",
            content="""
            # 価格戦略ガイド
            
            ## 基本価格設定
            
            ### 新規顧客向け
            - 初年度20%割引（年間契約の場合）
            - 3ヶ月無料トライアル提供
            - セットアップ費用免除
            
            ### 既存顧客向け
            - 継続割引5-10%
            - アップグレード優待価格
            - 追加ライセンス割引
            
            ## 業界別価格調整
            
            ### 製造業
            - 標準価格の90%
            - 長期契約でさらに5%割引
            
            ### 金融業
            - 標準価格の110%（高セキュリティ要件のため）
            - コンプライアンス機能込み
            
            ### 教育機関
            - 標準価格の70%
            - 非営利価格適用
            
            ## 競合対策価格
            - 競合他社より10%以上安く設定
            - 価格マッチング制度
            - ROI保証プログラム
            """,
            metadata={"department": "営業部", "category": "価格戦略", "version": "2024.1"}
        ),
        
        Document(
            id="sales_process",
            content="""
            # 営業プロセス標準手順
            
            ## リード獲得段階
            
            ### 1. リード発掘
            - Webサイトからの問い合わせ
            - 展示会・セミナー参加者
            - 紹介・リファラル
            - テレアポ・飛び込み
            
            ### 2. リード評価
            - BANT条件の確認
              - Budget（予算）
              - Authority（決裁権）
              - Need（必要性）
              - Timeline（導入時期）
            
            ## 営業活動段階
            
            ### 3. 初回面談
            - 課題のヒアリング
            - 現状システムの確認
            - 予算感の把握
            - 決裁プロセスの確認
            
            ### 4. 提案・デモ
            - カスタマイズ提案書作成
            - 製品デモンストレーション
            - ROI試算の提示
            - 導入スケジュール提案
            
            ### 5. 商談クロージング
            - 最終条件交渉
            - 契約書作成
            - 法務・経理承認
            - 契約締結
            
            ## 契約後フォロー
            - キックオフミーティング
            - 導入支援
            - 定期的なフォローアップ
            - アップセル・クロスセル機会の探索
            """,
            metadata={"department": "営業部", "category": "営業プロセス", "version": "2024.1"}
        )
    ]


def demo_enterprise_rag():
    """企業RAGシステムのデモンストレーション"""
    
    print("🏢 企業部門別RAGシステム デモンストレーション")
    print("=" * 60)
    
    # 企業RAGシステム初期化
    enterprise_rag = EnterpriseRAGSystem()
    
    # 人事部・営業部のRAGシステム追加
    hr_rag = enterprise_rag.add_department("人事部")
    sales_rag = enterprise_rag.add_department("営業部")
    
    print("\n📚 部門別ドキュメント追加")
    print("-" * 30)
    
    # 人事部ドキュメント追加
    hr_docs = create_hr_documents()
    hr_result = hr_rag.add_documents(hr_docs)
    print(f"✅ 人事部: {hr_result['stored_documents']}件のドキュメントを追加")
    
    # 営業部ドキュメント追加
    sales_docs = create_sales_documents()
    sales_result = sales_rag.add_documents(sales_docs)
    print(f"✅ 営業部: {sales_result['stored_documents']}件のドキュメントを追加")
    
    print("\n🔍 部門別クエリテスト")
    print("-" * 30)
    
    # テストクエリ
    test_scenarios = [
        {
            'user_department': '人事部',
            'query_department': '人事部',
            'question': '有給休暇は何日もらえますか？',
            'expected': 'アクセス成功'
        },
        {
            'user_department': '営業部',
            'query_department': '営業部',
            'question': 'CloudSync Proの価格はいくらですか？',
            'expected': 'アクセス成功'
        },
        {
            'user_department': '人事部',
            'query_department': '営業部',
            'question': '製品の価格戦略を教えて',
            'expected': 'アクセス拒否'
        },
        {
            'user_department': '営業部',
            'query_department': '人事部',
            'question': '人事評価制度について教えて',
            'expected': 'アクセス拒否'
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📝 テスト {i}: {scenario['expected']}")
        print(f"   ユーザー部門: {scenario['user_department']}")
        print(f"   クエリ部門: {scenario['query_department']}")
        print(f"   質問: {scenario['question']}")
        
        result = enterprise_rag.query_department(
            scenario['query_department'],
            scenario['question'],
            scenario['user_department']
        )
        
        if result.get('access_denied'):
            print(f"   🚫 {result['answer']}")
        else:
            print(f"   ✅ 回答: {result['answer'][:100]}...")
            print(f"   信頼度: {result['confidence']:.3f}")
    
    print("\n📊 データ分離の確認")
    print("-" * 30)
    
    # データ分離テスト
    print("🔒 人事部から営業部データへのアクセステスト:")
    hr_to_sales = enterprise_rag.query_department('営業部', '製品価格は？', '人事部')
    print(f"   結果: {'アクセス拒否' if hr_to_sales.get('access_denied') else 'アクセス成功'}")
    
    print("🔒 営業部から人事部データへのアクセステスト:")
    sales_to_hr = enterprise_rag.query_department('人事部', '有給休暇制度は？', '営業部')
    print(f"   結果: {'アクセス拒否' if sales_to_hr.get('access_denied') else 'アクセス成功'}")
    
    print("\n🏥 品質評価")
    print("-" * 30)
    
    # 品質評価実行
    print("📈 人事部RAGシステムの品質評価:")
    hr_quality = hr_rag.evaluate_quality()
    print(f"   品質スコア: {hr_quality['quality_score']:.2f}")
    print(f"   推奨事項: {len(hr_quality['recommendations'])}件")
    
    print("📈 営業部RAGシステムの品質評価:")
    sales_quality = sales_rag.evaluate_quality()
    print(f"   品質スコア: {sales_quality['quality_score']:.2f}")
    print(f"   推奨事項: {len(sales_quality['recommendations'])}件")
    
    print("\n📋 システム概要")
    print("-" * 30)
    
    overview = enterprise_rag.get_system_overview()
    print(f"✅ 総部門数: {overview['total_departments']}")
    print(f"✅ 部門リスト: {', '.join(overview['departments'])}")
    print(f"✅ データディレクトリ: {overview['data_directory']}")
    print(f"✅ システム状態: {overview['system_status']}")
    
    print("\n🎯 実用的な使用例")
    print("-" * 30)
    
    # 実用的なクエリ例
    practical_queries = [
        ('人事部', '人事部', '新入社員研修はどのくらいの期間ですか？'),
        ('営業部', '営業部', '新規顧客向けの割引制度について教えて'),
        ('人事部', '人事部', '人事評価はいつ実施されますか？'),
        ('営業部', '営業部', '競合他社に対する価格戦略は？')
    ]
    
    for user_dept, query_dept, question in practical_queries:
        print(f"\n💬 {user_dept}のユーザーからの質問:")
        print(f"   「{question}」")
        
        result = enterprise_rag.query_department(query_dept, question, user_dept)
        if not result.get('access_denied'):
            print(f"   回答: {result['answer'][:150]}...")
        else:
            print(f"   {result['answer']}")
    
    print("\n🎉 企業部門別RAGシステムのデモが完了しました！")
    print("\n💡 主な特徴:")
    print("   ✅ 部門別データ分離")
    print("   ✅ アクセス制御機能")
    print("   ✅ 部門固有のナレッジベース")
    print("   ✅ 統合された品質監視")
    print("   ✅ 監査ログ機能")
    
    print("\n🚀 本格導入への次のステップ:")
    print("   📁 実際の部門ドキュメントの投入")
    print("   🔐 Active Directoryとの認証連携")
    print("   📊 ダッシュボードとレポート機能")
    print("   🔄 定期的なドキュメント更新ワークフロー")
    print("   📱 Webインターフェースまたはチャットボット開発")


if __name__ == "__main__":
    demo_enterprise_rag()