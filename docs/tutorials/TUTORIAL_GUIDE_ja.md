# refinire-rag チュートリアルガイド

## はじめに

このチュートリアルガイドでは、refinire-ragライブラリを使用してRAG（Retrieval-Augmented Generation）システムを構築する方法を段階的に学習できます。初心者から上級者まで、あなたのレベルに応じたラーニングパスを提供します。

## 学習前の準備

### 1. 環境セットアップ
```bash
# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# refinire-ragのインストール
pip install refinire-rag

# 必要な環境変数の設定
export OPENAI_API_KEY="your-api-key-here"
export REFINIRE_RAG_LLM_MODEL="gpt-4o-mini"
```

### 2. 前提知識
- Python基礎知識
- RAG（検索拡張生成）の基本概念
- LLM（大規模言語モデル）の基本的な理解

## 学習ロードマップ

### 🎯 レベル1: 基礎理解

**目標**: RAGシステムの基本概念を理解し、簡単なRAGシステムを構築する

#### 1.1 RAGの基本概念
- **学習内容**: RAGとは何か、なぜ必要か
- **資料**: [tutorial_01_basic_rag.md](docs/tutorials/tutorial_01_basic_rag.md)
- **実習**: 基本的なRAGシステムの動作確認

#### 1.2 refinire-ragの全体像
- **学習内容**: ライブラリの構成と主要コンポーネント
- **資料**: [tutorial_overview.md](docs/tutorials/tutorial_overview.md)
- **実習**: サンプルコードの実行

#### 1.3 クイックスタート
- **学習内容**: 最短でRAGシステムを動作させる
- **資料**: [quickstart_guide.py](examples/quickstart_guide.py)
- **実習**: 10分でRAGシステムを構築

**✅ レベル1完了チェック**
- [ ] RAGの基本概念を説明できる
- [ ] refinire-ragの主要コンポーネントを理解している
- [ ] 基本的なRAGシステムを動作させることができる

### 🚀 レベル2: コア機能習得

**目標**: refinire-ragの3つの主要コンポーネントを使いこなす

#### 2.1 Part 1: コーパス作成・管理
- **学習内容**: CorpusManagerを使った文書の読み込み、処理、インデックス化
- **資料**: [tutorial_part1_corpus_creation.md](docs/tutorials/tutorial_part1_corpus_creation.md)
- **実習**: [tutorial_part1_corpus_creation_example.py](examples/tutorial_part1_corpus_creation_example.py)

**重要ポイント**:
- 文書の読み込みと前処理
- チャンキング戦略の選択
- ベクトル化とインデックス構築
- 増分読み込み（Incremental Loading）

#### 2.2 Part 2: クエリエンジン
- **学習内容**: QueryEngineを使った検索、リランキング、回答生成
- **資料**: [tutorial_part2_query_engine.md](docs/tutorials/tutorial_part2_query_engine.md)
- **実習**: [tutorial_part2_query_engine_example.py](examples/tutorial_part2_query_engine_example.py)

**重要ポイント**:
- 検索戦略の設定
- リランキング手法の選択
- 回答生成の最適化
- パフォーマンスチューニング

#### 2.3 Part 3: 品質評価
- **学習内容**: QualityLabを使ったRAGシステムの評価と改善
- **資料**: [tutorial_part3_evaluation.md](docs/tutorials/tutorial_part3_evaluation.md)
- **実習**: [tutorial_part3_evaluation_example.py](examples/tutorial_part3_evaluation_example.py)

**重要ポイント**:
- 評価データの作成
- 多面的な品質評価
- 矛盾検出
- 改善提案の解釈

**✅ レベル2完了チェック**
- [ ] 独自のデータでコーパスを作成できる
- [ ] 効果的な検索・回答生成を設定できる
- [ ] RAGシステムの品質を定量的に評価できる

### 🎯 レベル3: 統合・実用化

**目標**: 3つのコンポーネントを統合し、実用的なRAGシステムを構築する

#### 3.1 エンドツーエンド統合
- **学習内容**: Complete RAG Tutorialによる統合システム構築
- **資料**: [complete_rag_tutorial.py](examples/complete_rag_tutorial.py)
- **実習**: 完全なRAGワークフローの実装

#### 3.2 コーパス管理の実践
- **学習内容**: 実際のデータでのコーパス管理
- **資料**: [tutorial_02_corpus_management.md](docs/tutorials/tutorial_02_corpus_management.md)
- **実習**: [corpus_manager_demo.py](examples/corpus_manager_demo.py)

#### 3.3 パフォーマンス最適化
- **学習内容**: システムパフォーマンスの監視と最適化
- **実習**: パフォーマンステストの実行

**✅ レベル3完了チェック**
- [ ] エンドツーエンドのRAGシステムを構築できる
- [ ] 実際のデータでパフォーマンスを最適化できる
- [ ] システムの品質を継続的に監視できる

### 🏆 レベル4: 高度な活用

**目標**: 高度な機能を活用し、企業レベルのRAGシステムを構築する

#### 4.1 正規化・知識グラフ活用
- **学習内容**: テキスト正規化と知識グラフを活用した高度なRAG
- **資料**: [tutorial_04_normalization.md](docs/tutorials/tutorial_04_normalization.md)
- **実習**: 正規化機能の実装

#### 4.2 企業レベルの活用
- **学習内容**: 企業環境での実用的なRAGシステム構築
- **資料**: [tutorial_05_enterprise_usage.md](docs/tutorials/tutorial_05_enterprise_usage.md)
- **実習**: [tutorial_05_enterprise_usage.py](examples/tutorial_05_enterprise_usage.py)

#### 4.3 増分データ処理
- **学習内容**: 大規模データの効率的な処理
- **資料**: [tutorial_06_incremental_loading.md](docs/tutorials/tutorial_06_incremental_loading.md)
- **実習**: [incremental_loading_demo.py](examples/incremental_loading_demo.py)

#### 4.4 高度な評価手法
- **学習内容**: 詳細な評価指標と改善手法
- **資料**: [tutorial_07_rag_evaluation.md](docs/tutorials/tutorial_07_rag_evaluation.md)
- **実習**: [tutorial_07_evaluation_example.py](examples/tutorial_07_evaluation_example.py)

**✅ レベル4完了チェック**
- [ ] 高度な正規化機能を活用できる
- [ ] 企業レベルのRAGシステムを設計できる
- [ ] 大規模データの効率的な処理を実装できる
- [ ] 高度な評価手法を適用できる

## 学習支援リソース

### 📚 ドキュメント
- **API リファレンス**: [docs/api/](docs/api/)
- **アーキテクチャガイド**: [docs/design/architecture.md](docs/design/architecture.md)
- **開発ガイド**: [docs/development/](docs/development/)

### 💻 実践例
- **サンプルコード**: [examples/](examples/)
- **テストデータ**: [examples/data/](examples/data/)
- **設定例**: [docs/development/processor_config_example.md](docs/development/processor_config_example.md)

### 🔧 開発支援
- **プラグイン開発**: [docs/development/plugin_development.md](docs/development/plugin_development.md)
- **カスタマイズガイド**: [docs/development/answer_synthesizer_customization.md](docs/development/answer_synthesizer_customization.md)

## よくある質問（FAQ）

### Q1: どのレベルから始めればよいですか？
**A**: RAGの経験がない場合は**レベル1**から、機械学習の経験がある場合は**レベル2**から始めることをお勧めします。

### Q2: 学習にはどのくらいの時間がかかりますか？
**A**: 学習時間は個人の経験や目標によって異なります。速度よりも理解と実践に焦点を当て、自分のペースでレベルを進めてください。

### Q3: 自分のデータで試したい場合はどうすればよいですか？
**A**: **レベル2**完了後、Part 1のチュートリアルを参考に独自データでコーパスを作成してください。

### Q4: 企業での活用を考えています。どこから始めればよいですか？
**A**: **レベル3**完了後、[tutorial_05_enterprise_usage.md](docs/tutorials/tutorial_05_enterprise_usage.md)を参考に企業向けの実装を検討してください。

### Q5: エラーが発生した場合はどうすればよいですか？
**A**: 
1. 環境設定を確認（API キー、依存関係）
2. サンプルコードと比較
3. デバッグ出力を有効化
4. [トラブルシューティング](docs/development/troubleshooting.md)を参照

## 次のステップ

### 🎯 レベル1を完了したら
- [tutorial_part1_corpus_creation.md](docs/tutorials/tutorial_part1_corpus_creation.md)に進む
- 実際のデータでコーパス作成を試す

### 🚀 レベル2を完了したら
- [complete_rag_tutorial.py](examples/complete_rag_tutorial.py)を実行
- 統合システムの動作を確認

### 🏆 レベル3を完了したら
- 高度な機能の学習に進む
- 企業レベルでの活用を検討

### 🌟 レベル4を完了したら
- 独自のプラグイン開発に挑戦
- コミュニティへの貢献を検討

## プラグインシステム活用ガイド

### 🔌 プラグインの基本概念

refinire-ragは環境変数ベースのプラグインアーキテクチャを採用しており、様々なコンポーネントを独立したパッケージとして利用できます。

#### 利用可能なプラグインタイプ
- **検索・取得系**: VectorStore（Chroma等）、KeywordSearch（BM25s等）、Retriever
- **文書処理系**: Loader、Splitter、Filter、Metadata
- **評価・品質管理系**: Evaluator、ContradictionDetector、TestSuite
- **ストレージ系**: DocumentStore、EvaluationStore

### 🎯 プラグインの切り替え方法

#### 1. 環境変数による基本的な切り替え

```bash
# 開発環境：軽量プラグイン
export REFINIRE_RAG_RETRIEVERS="inmemory_vector"

# 本番環境：高性能プラグイン
export REFINIRE_RAG_RETRIEVERS="chroma,bm25s"
export REFINIRE_RAG_CHROMA_HOST="localhost"
export REFINIRE_RAG_CHROMA_PORT="8000"
export REFINIRE_RAG_BM25S_INDEX_PATH="./bm25s_index"
```

#### 2. 利用可能なプラグインの確認

```python
from refinire_rag.registry import PluginRegistry

# 利用可能なプラグインタイプを確認
available_plugins = PluginRegistry.get_all_plugins_info()

print("=== 利用可能なプラグイン ===")
for group, plugins in available_plugins.items():
    print(f"\n{group.upper()}:")
    for name, info in plugins.items():
        print(f"  - {name}: {info['description']}")
```

#### 3. 動的なプラグイン作成

```python
# 特定のプラグインを動的に作成
vector_store = PluginRegistry.create_plugin('vector_stores', 'chroma', 
                                           host='localhost', port=8000)

# 複数のプラグインを環境変数から作成
from refinire_rag.factories import PluginFactory
retrievers = PluginFactory.create_retrievers_from_env()
```

### 🛠 プラグイン開発の基本

#### 1. プロジェクト構造
```
my-refinire-rag-plugin/
├── src/
│   └── my_refinire_plugin/
│       ├── __init__.py
│       ├── vector_store.py      # プラグイン実装
│       ├── config.py           # 設定クラス
│       └── env_template.py     # 環境変数テンプレート
├── tests/
├── pyproject.toml             # エントリポイント設定
└── README.md
```

#### 2. 必須実装パターン

```python
# 統一設定パターンの例
class CustomVectorStore(VectorStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import os
        # 設定優先順位: kwargs > 環境変数 > デフォルト値
        self.host = kwargs.get('host', 
                              os.getenv('REFINIRE_RAG_CUSTOM_HOST', 'localhost'))
        self.port = int(kwargs.get('port', 
                                  os.getenv('REFINIRE_RAG_CUSTOM_PORT', '8080')))
    
    def get_config(self) -> Dict[str, Any]:
        """現在の設定を辞書として返却（必須）"""
        return {
            'host': self.host,
            'port': self.port,
            'plugin_type': self.__class__.__name__
        }
```

#### 3. エントリポイント設定

```toml
# pyproject.toml
[project.entry-points."refinire_rag.vector_stores"]
custom = "my_refinire_plugin:CustomVectorStore"

[project.entry-points."refinire_rag.oneenv_templates"]
custom = "my_refinire_plugin.env_template:custom_env_template"
```

### 📚 プラグイン関連のリソース

#### 開発ガイド
- **詳細な開発ガイド**: [docs/development/plugin_development_guide.md](docs/development/plugin_development_guide.md)
- **設計戦略**: [docs/design/plugin_strategy.md](docs/design/plugin_strategy.md)
- **統一設定パターン**: [docs/development/plugin_development.md](docs/development/plugin_development.md)

#### 実装例
- **プラグイン設定例**: [docs/development/plugin_setup_example.py](docs/development/plugin_setup_example.py)
- **統合テスト**: [tests/integration/plugins/](tests/integration/plugins/)

#### 環境変数管理
- **oneenv統合**: プラグイン固有の環境変数テンプレートを自動生成
- **設定優先順位**: kwargs > 環境変数 > デフォルト値

### 🔧 高度なプラグイン活用

#### 1. 複数プラグインの組み合わせ

```python
# 複数のVectorStoreとKeywordSearchを組み合わせ
import os
os.environ["REFINIRE_RAG_RETRIEVERS"] = "chroma,bm25s"

# CorpusManagerが自動的に複数のretrieverを統合管理
corpus_manager = CorpusManager.from_env()
print(f"統合された retrievers: {len(corpus_manager.retrievers)}")
```

#### 2. 環境別プラグイン切り替え

```python
import os

# 環境による動的切り替え
if os.getenv("ENVIRONMENT") == "development":
    os.environ["REFINIRE_RAG_RETRIEVERS"] = "inmemory_vector"
elif os.getenv("ENVIRONMENT") == "production": 
    os.environ["REFINIRE_RAG_RETRIEVERS"] = "chroma,bm25s"
elif os.getenv("ENVIRONMENT") == "testing":
    os.environ["REFINIRE_RAG_RETRIEVERS"] = "inmemory_vector"
```

#### 3. カスタムプラグインの配布

```bash
# プラグインパッケージのビルドと配布
cd my-refinire-rag-plugin
python -m build
twine upload dist/*

# インストールして自動的に利用可能に
pip install my-refinire-rag-plugin
```

### ⚡ プラグイン開発のベストプラクティス

#### 必須チェックリスト
- [ ] 統一設定パターン（`**kwargs`コンストラクタ）の実装
- [ ] `get_config()`メソッドの実装
- [ ] 環境変数命名規則の遵守（`REFINIRE_RAG_{PLUGIN}_{SETTING}`）
- [ ] エラー処理の適切な実装
- [ ] 適切なエントリポイントグループの選択
- [ ] oneenvテンプレートの提供

#### 開発原則
1. **単一責任**: 一つのプラグインは一つの明確な責務を持つ
2. **環境変数サポート**: 全設定で環境変数をサポート
3. **遅延初期化**: 重い処理は実際の使用時まで遅延
4. **エラー耐性**: 接続失敗等のエラーに適切に対応
5. **テスト容易性**: 設定を注入してテストしやすい設計

### 🚀 プラグイン学習の進め方

#### 初心者向け
1. 既存プラグインの利用方法を習得
2. 環境変数による設定変更を実践
3. 簡単なカスタムプラグインを作成

#### 中級者向け
1. 複数プラグインの組み合わせ活用
2. 統一設定パターンの理解と実装
3. oneenv統合による環境変数管理

#### 上級者向け
1. 複雑なプラグインアーキテクチャの設計
2. プラグイン間の依存関係管理
3. プラグインエコシステムへの貢献

プラグインシステムにより、refinire-ragは高度な拡張性と柔軟性を提供します。あなたのニーズに合わせてプラグインを選択・開発し、より強力なRAGシステムを構築してください。

## 学習サポート

### 📖 学習進捗管理
各レベルのチェックリストを使用して、学習の進捗を管理してください。

### 💡 実践的な学習のコツ
1. **小さく始める**: 最初は簡単な例から始めて、徐々に複雑にする
2. **実際のデータで試す**: 理論だけでなく、実際のデータで動作確認する
3. **評価を重視する**: 定期的にシステムの品質を評価する
4. **コミュニティを活用する**: 疑問点は積極的に質問する

---

**準備はできましたか？**

refinire-ragの学習を始めましょう。あなたのレベルに応じたチュートリアルからスタートして、段階的にRAGシステムの構築スキルを身に付けてください。

**Happy Learning! 🚀**