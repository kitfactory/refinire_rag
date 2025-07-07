# refinire-rag コンセプト文書

## 概要
refinire-ragは、Refinireライブラリのサブパッケージとして提供されるRAG（Retrieval-Augmented Generation）機能を実装するPythonライブラリです。モジュラーアーキテクチャを採用し、ユースケースをRefinire Stepサブクラスとして実装し、単一責務のバックエンドモジュールを提供します。

## アーキテクチャ

### ユースケースクラス（Refinire Steps）
- **CorpusManager**: 文書の読み込み、正規化、チャンク分割、埋め込み生成、保存
- **QueryEngine**: 文書検索、再ランキング、回答生成
- **QualityLab**: 評価データ作成、自動RAG評価、矛盾検出、レポート生成

### バックエンドモジュール（すべてDocumentProcessorを実装）
- **Loader**: 外部ファイル → Document変換
- **DictionaryMaker**: LLMベースのドメイン固有用語抽出と累積MDディクショナリ
- **Normalizer**: MDディクショナリベースの表現バリエーション正規化
- **GraphBuilder**: LLMベースの関係抽出と累積MDナレッジグラフ
- **Chunker**: トークンベースのチャンク分割
- **VectorStoreProcessor**: チャンク → ベクター生成と保存（Embedderを統合）
- **Retriever**: 文書検索
- **Reranker**: 候補再ランキング
- **Reader**: LLMベースの回答生成
- **TestSuite**: 評価実行器
- **Evaluator**: メトリクス集約
- **ContradictionDetector**: 主張抽出 + NLI検出
- **InsightReporter**: 閾値ベースの解釈とレポート

## 現在の実装状況

### ✅ 完了機能
- 環境変数ベースの設定システム
- 組み込みおよび外部プラグインサポートを備えたプラグインレジストリ
- ベクトルとキーワード検索によるハイブリッド検索
- 複数の再ランカー実装（ヒューリスティック、RRF、LLM）
- SQLite文書ストレージ
- OpenAI埋め込み統合
- 3ステップ教育的例（hybrid_rag_example.py）

### 🔧 現在のアーキテクチャ改善
- 全コンポーネント across でのプラグインパターン実装
- 設定されたストアからの自動リトリーバー作成
- 欠損コンポーネントに対する優雅な代替メカニズム

## 将来の拡張

### 🚀 優先度の高い拡張

#### 1. RefinireAgent Context統合
**ステータス**: コンセプトが特定済み、未実装
**説明**: QueryEngineの検索結果をRefinireAgentのContextProviderとして適切に提供する機能

**現在の課題**:
- SimpleAnswerSynthesizerは検索結果を単純なテキストとして結合している
- QueryEngineの検索結果（List[SearchResult]）がRefinireAgentのContextとして適切に渡されていない
- 現在のLLMは「提供されたコンテキストで回答が見つかりません」と回答することが多い

**提案する解決策**:
```python
# 将来の実装例
from refinire import Context, RefinireAgent

class RefinireContextProvider:
    def create_context_from_search_results(self, search_results: List[SearchResult]) -> Context:
        # SearchResultsをRefinire ContextオブジェクトとしてラップDictionary API
        # メタデータ、スコア、文書IDなどの情報を保持
        pass

class EnhancedAnswerSynthesizer(AnswerSynthesizer):
    def synthesize(self, query: str, contexts: List[SearchResult]) -> str:
        # RefinireAgentにContextProviderとして検索結果を渡す
        context = self.context_provider.create_context_from_search_results(contexts)
        agent = RefinireAgent(context_provider=context)
        return agent.generate_answer(query)
```

**利点**:
- より正確で文脈を理解した回答生成
- RefinireAgentの高度な推論能力の活用
- メタデータとスコア情報を保持した文脈提供
- Refinireエコシステムとの完全な統合

**実装タスク**:
1. Refinire ContextとRefinireAgentのAPI調査
2. SearchResult → Contextの変換機能実装
3. EnhancedAnswerSynthesizerの作成
4. hybrid_rag_example.pyでのテスト
5. 既存のSimpleAnswerSynthesizerとの互換性維持

---

#### 2. 高度なクエリ処理
- コーパス辞書を使用したクエリ正規化
- 多段階クエリ分解
- ドメイン知識グラフを使用したクエリ拡張

#### 3. 強化された検索手法
- 密なパッセージ検索統合
- クロスエンコーダー再ランキング
- 時間的・空間的検索機能

#### 4. プロダクション機能
- 埋め込みと検索結果のキャッシュ層
- バッチ処理機能
- APIエンドポイント統合
- 監視とログ機能の強化

#### 5. 品質評価
- 自動化された評価パイプライン
- 検索手法のA/Bテストフレームワーク
- バイアス検出と軽減ツール

## 技術的決定

### 設定管理
- デフォルト値を持つ環境変数ベースの設定
- コンポーネントの発見とインスタンス化のプラグインパターン
- キーワード引数 → 環境変数 → デフォルト値の階層

### 統合戦略
- ユースケース調整のためのRefinire Stepサブクラス
- 処理パイプライン統一のためのDocumentProcessorインターフェース
- 拡張可能なコンポーネントエコシステムのためのプラグインレジストリ

### テスト戦略
- 全コンポーネントの包括的単体テスト
- エンドツーエンドワークフローの統合テスト
- 例駆動のドキュメント化とテスト

## 依存関係

### コア依存関係
- Python 3.10+
- LLM統合とエージェントフレームワークのためのRefinireライブラリ
- 埋め込みと言語モデルのためのOpenAI API

### オプションのプラグイン依存関係
- refinire-rag-chroma: Chromaベクトルストア統合
- refinire-rag-bm25s-j: BM25sキーワード検索統合

### 開発依存関係
- テストフレームワークのためのpytest
- カバレッジレポートのためのpytest-cov
- パッケージ管理のためのuv