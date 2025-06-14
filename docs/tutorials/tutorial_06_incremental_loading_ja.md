# チュートリアル 6: 増分文書ローディング

このチュートリアルでは、チュートリアル5の企業RAGシステムを基に、増分ローディング機能を使用して大規模な文書コレクションを効率的に管理する方法を学びます。

## 概要

増分ローディングにより以下が可能になります：
- 新規・更新された文書のみを処理
- 未変更ファイルをスキップして効率的な更新
- 大規模文書リポジトリの効率的な処理
- 文書系譜と履歴の維持

## 前提条件

このチュートリアルはチュートリアル5（企業RAGの利用）の内容を拡張するため、事前にチュートリアル5を完了してください。

## 実装

### ステップ 1: 増分ローディング対応企業RAG

```python
#!/usr/bin/env python3
"""
チュートリアル 6: 増分文書ローディング
企業環境でのRAGシステムに増分ローディング機能を追加
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# srcをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.loaders.incremental_loader import IncrementalLoader
from refinire_rag.application.corpus_manager import CorpusManager, CorpusManagerConfig
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.storage import SQLiteDocumentStore, InMemoryVectorStore
from refinire_rag.embedding import TFIDFEmbedder, TFIDFEmbeddingConfig
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
from refinire_rag.processing import Normalizer, NormalizerConfig, TokenBasedChunker, ChunkingConfig
from refinire_rag.models.document import Document


class IncrementalEnterpriseRAG:
    """
    増分ローディング対応の企業RAGシステム
    """
    
    def __init__(self, department: str, base_dir: Path):
        self.department = department
        self.base_dir = base_dir
        self.docs_dir = base_dir / f"{department}_docs"
        self.db_path = str(base_dir / f"{department}_rag.db")
        self.cache_path = str(base_dir / f".{department}_cache.json")
        
        # 部署ディレクトリを作成
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # コンポーネントを初期化
        self._setup_components()
        
        # 統計情報
        self.stats = {
            "total_documents": 0,
            "last_update": None,
            "incremental_runs": 0,
            "total_processing_time": 0.0
        }
    
    def _setup_components(self):
        """RAGコンポーネントを初期化"""
        
        # ストレージ
        self.document_store = SQLiteDocumentStore(self.db_path)
        self.vector_store = InMemoryVectorStore(similarity_metric="cosine")
        
        # 埋め込み
        embedder_config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0, max_features=5000)
        self.embedder = TFIDFEmbedder(config=embedder_config)
        
        # 処理コンポーネント
        normalizer_config = NormalizerConfig(
            dictionary_file_path=str(self.base_dir / f"{self.department}_dictionary.md"),
            whole_word_only=False
        )
        
        chunking_config = ChunkingConfig(
            chunk_size=300,
            overlap=30,
            split_by_sentence=True
        )
        
        # プロセッサ付きコーパスマネージャー
        corpus_config = CorpusManagerConfig(
            document_store=self.document_store,
            vector_store=self.vector_store,
            embedder=self.embedder,
            processors=[
                Normalizer(normalizer_config),
                TokenBasedChunker(chunking_config)
            ],
            enable_progress_reporting=True
        )
        
        self.corpus_manager = CorpusManager(corpus_config)
        
        # クエリエンジン
        retriever = SimpleRetriever(self.vector_store, self.embedder)
        reranker = SimpleReranker()
        reader = SimpleReader()
        
        query_config = QueryEngineConfig(
            retriever=retriever,
            reranker=reranker,
            reader=reader,
            normalizer=Normalizer(normalizer_config)
        )
        
        self.query_engine = QueryEngine(query_config)
        
        # 増分ローダー
        self.incremental_loader = IncrementalLoader(
            document_store=self.document_store,
            base_loader=self.corpus_manager._loader,
            cache_file=self.cache_path
        )
    
    def add_documents(self, documents: Dict[str, str], force_update: bool = False):
        """
        部署の文書コレクションに文書を追加または更新
        
        Args:
            documents: ファイル名 -> 内容の辞書
            force_update: ファイルが変更されていなくても強制更新
        """
        print(f"\\n📁 {self.department}部署に文書を追加中...")
        
        # 文書をファイルに書き込み
        for filename, content in documents.items():
            file_path = self.docs_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   作成: {filename}")
        
        # 増分処理
        start_time = time.time()
        
        force_reload = set()
        if force_update:
            force_reload = {str(self.docs_dir / filename) for filename in documents.keys()}
        
        results = self.incremental_loader.process_incremental(
            sources=[self.docs_dir],
            force_reload=force_reload
        )
        
        processing_time = time.time() - start_time
        
        # 新しい文書がある場合は埋め込み器をフィット
        all_docs = results['new'] + results['updated']
        if all_docs and not self.embedder.is_fitted():
            print(f"   {len(all_docs)}件の文書で埋め込み器をフィット中...")
            texts = [doc.content for doc in all_docs if doc.content.strip()]
            if texts:
                self.embedder.fit(texts)
        
        # コーパスマネージャーパイプラインで処理
        if all_docs:
            print(f"   {len(all_docs)}件の文書をRAGパイプラインで処理中...")
            corpus_results = self.corpus_manager.process_documents(all_docs)
            
            # 埋め込み生成と保存
            embedded_docs = self.corpus_manager.embed_documents(corpus_results)
            print(f"   {len(embedded_docs)}件の文書の埋め込みを生成")
        
        # 統計情報を更新
        self.stats["total_documents"] = self.document_store.get_stats().total_documents
        self.stats["last_update"] = datetime.now().isoformat()
        self.stats["incremental_runs"] += 1
        self.stats["total_processing_time"] += processing_time
        
        # 結果をレポート
        print(f"\\n📊 処理結果:")
        print(f"   新規文書: {len(results['new'])}件")
        print(f"   更新文書: {len(results['updated'])}件")
        print(f"   スキップ文書: {len(results['skipped'])}件")
        print(f"   処理時間: {processing_time:.2f}秒")
        
        return results
    
    def update_document(self, filename: str, new_content: str):
        """特定の文書を更新"""
        print(f"\\n📝 文書を更新: {filename}")
        
        file_path = self.docs_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        # 更新を処理
        results = self.add_documents({filename: new_content})
        return results
    
    def delete_document(self, filename: str):
        """文書を削除してクリーンアップ"""
        print(f"\\n🗑️ 文書を削除: {filename}")
        
        file_path = self.docs_dir / filename
        if file_path.exists():
            file_path.unlink()
            print(f"   ファイルを削除: {filename}")
            
            # ストアからクリーンアップ
            deleted_docs = self.incremental_loader.cleanup_deleted_files([self.docs_dir])
            print(f"   ストアから{len(deleted_docs)}件の文書をクリーンアップ")
            
            return deleted_docs
        else:
            print(f"   ファイルが見つかりません: {filename}")
            return []
    
    def query(self, question: str) -> Dict[str, Any]:
        """部署のRAGシステムにクエリ"""
        return self.query_engine.answer(question)
    
    def get_statistics(self) -> Dict[str, Any]:
        """包括的なシステム統計を取得"""
        cache_stats = self.incremental_loader.get_cache_stats()
        store_stats = self.document_store.get_stats()
        
        return {
            **self.stats,
            "cache_statistics": cache_stats,
            "document_store": {
                "total_documents": store_stats.total_documents,
                "db_path": self.db_path
            },
            "vector_store": {
                "total_vectors": len(self.vector_store.vectors),
                "dimension": getattr(self.vector_store, 'dimension', '不明')
            }
        }


def demo_incremental_enterprise_rag():
    """企業RAGでの増分ローディングをデモンストレーション"""
    
    print("🏢 チュートリアル 6: 増分文書ローディング")
    print("=" * 60)
    
    # デモ環境をセットアップ
    demo_dir = Path("enterprise_incremental_demo")
    demo_dir.mkdir(exist_ok=True)
    
    try:
        # 人事部RAGシステムを作成
        hr_rag = IncrementalEnterpriseRAG("hr", demo_dir)
        
        print("\\n🎯 ステップ 1: 初期文書ローディング")
        print("-" * 40)
        
        # 初期文書
        initial_docs = {
            "employee_handbook.md": '''
# 従業員ハンドブック v1.0

## 勤務時間
- 平日: 9:00-18:00
- 昼休み: 12:00-13:00

## 休暇制度
- 年次有給休暇: 20日
- 夏季休暇: 3日
- 年末年始: 5日

## 福利厚生
- 健康保険
- 厚生年金
- 雇用保険
            ''',
            
            "remote_work_policy.md": '''
# リモートワーク規定

## 対象者
正社員および契約社員

## 申請手続き
1. 上司に事前申請
2. 人事部承認
3. 実施報告

## 勤務環境
- 安定したインターネット接続
- 適切な作業スペース
- セキュリティ対策
            '''
        }
        
        # 初期文書を追加
        results1 = hr_rag.add_documents(initial_docs)
        
        print("\\n🔍 ステップ 2: 初期クエリテスト")
        print("-" * 40)
        
        result = hr_rag.query("リモートワークの申請手続きは？")
        print(f"質問: リモートワークの申請手続きは？")
        print(f"回答: {result['answer']}")
        
        print("\\n⏱️ ステップ 3: 増分更新（変更なし）")
        print("-" * 40)
        
        # 変更なしで再実行 - すべてスキップされるはず
        results2 = hr_rag.add_documents({})
        
        print("\\n📝 ステップ 4: 既存文書の更新")
        print("-" * 40)
        
        # 従業員ハンドブックを更新
        updated_handbook = '''
# 従業員ハンドブック v2.0

## 勤務時間
- 平日: 9:00-17:30 (変更)
- 昼休み: 12:00-13:00

## 休暇制度
- 年次有給休暇: 25日 (改善)
- 夏季休暇: 5日 (改善)
- 年末年始: 5日
- リフレッシュ休暇: 3日 (新規)

## 福利厚生
- 健康保険
- 厚生年金
- 雇用保険
- 住宅手当 (新規)

最終更新: 2024年6月
        '''
        
        results3 = hr_rag.update_document("employee_handbook.md", updated_handbook)
        
        print("\\n📄 ステップ 5: 新規文書の追加")
        print("-" * 40)
        
        new_docs = {
            "performance_review.md": '''
# 人事評価制度

## 評価期間
- 上半期: 4月-9月
- 下半期: 10月-3月

## 評価項目
1. 業務成果
2. プロセス
3. 行動特性

## 評価プロセス
1. 自己評価
2. 上司評価
3. フィードバック面談
4. 評価確定

## 昇進・昇格
評価結果に基づき年2回検討
            '''
        }
        
        results4 = hr_rag.add_documents(new_docs)
        
        print("\\n🔍 ステップ 6: 更新された知識のテスト")
        print("-" * 40)
        
        # 更新された情報をテスト
        questions = [
            "勤務時間は何時までですか？",
            "年次有給休暇は何日もらえますか？",
            "人事評価はいつ行われますか？"
        ]
        
        for question in questions:
            result = hr_rag.query(question)
            print(f"\\n質問: {question}")
            print(f"回答: {result['answer']}")
        
        print("\\n🗑️ ステップ 7: 文書の削除")
        print("-" * 40)
        
        # 文書を削除
        deleted = hr_rag.delete_document("remote_work_policy.md")
        
        print("\\n📊 ステップ 8: 最終統計")
        print("-" * 40)
        
        stats = hr_rag.get_statistics()
        print(f"📈 システム統計:")
        print(f"   総文書数: {stats['total_documents']}")
        print(f"   増分実行回数: {stats['incremental_runs']}")
        print(f"   総処理時間: {stats['total_processing_time']:.2f}秒")
        print(f"   最終更新: {stats['last_update']}")
        print(f"   キャッシュファイル数: {stats['cache_statistics']['total_files']}")
        print(f"   ベクトルストアサイズ: {stats['vector_store']['total_vectors']}")
        
        print("\\n🎯 増分ローディングの利点")
        print("-" * 40)
        print("✅ 効率的な更新 - 変更されたファイルのみを処理")
        print("✅ 高速再処理 - 未変更文書をスキップ")
        print("✅ 大規模対応 - 数千の文書を処理可能")
        print("✅ 自動クリーンアップ - 削除された文書を除去")
        print("✅ 変更検出 - 複数の検証方法")
        print("✅ 企業対応 - 本番環境デプロイメント適用可能")
        
        print("\\n💡 本番環境での使用ヒント")
        print("-" * 40)
        print("1. スケジュールで増分更新を実行（夜間/時間毎）")
        print("2. 最適化のためキャッシュ統計を監視")
        print("3. 体系的更新でforce_reloadを使用")
        print("4. リアルタイム更新でファイル監視を実装")
        print("5. 災害復旧用キャッシュファイルをバックアップ")
        
    except Exception as e:
        print(f"❌ デモでエラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # クリーンアップオプション
        cleanup = input("\\nデモファイルを削除しますか？ (y/N): ").lower().strip()
        if cleanup == 'y':
            import shutil
            if demo_dir.exists():
                shutil.rmtree(demo_dir)
            print("✅ デモファイルをクリーンアップしました")
        else:
            print(f"📁 デモファイルを保持: {demo_dir}")
    
    print("\\n🎉 チュートリアル 6 完了！")
    print("\\n次回: チュートリアル 7では高度なクエリ最適化とキャッシュについて説明します")


if __name__ == "__main__":
    demo_incremental_enterprise_rag()