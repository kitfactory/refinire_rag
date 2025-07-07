# refinire-rag チュートリアル

## 概要

refinire-ragの学習と実装のための包括的なチュートリアルコレクションへようこそ。これらのチュートリアルは、RAGシステムの構築、評価、本番展開に必要なすべての知識を提供します。

## 学習パス

### 🚀 クイックスタート
RAGシステムをすぐに開始したい場合：
- [クイックスタートガイド](../why_refinire_rag_ja.md) - refinire-ragを選ぶ理由
- [包括的チュートリアルガイド](TUTORIAL_GUIDE_ja.md) - 学習ロードマップ

### 📚 レベル別学習

#### Level 1: 基礎 (Foundation)
**前提知識**: Python基礎、基本的なNLP概念  
**学習時間**: 2-3時間  
**目標**: RAGの基礎概念を理解し、シンプルなシステムを構築

- [基本RAGチュートリアル](tutorial_01_basic_rag_ja.md) - RAGの基礎
- [コーパス作成](tutorial_part1_corpus_creation.md) - 文書コーパスの構築
- [QueryEngine基礎](tutorial_part2_query_engine.md) - クエリエンジンの実装

#### Level 2: 実践応用 (Application)
**前提知識**: Level 1完了、RAG基礎概念の理解  
**学習時間**: 4-6時間  
**目標**: 本格的なRAGシステムを構築し、カスタマイズする

- [コーパス管理](tutorial_02_corpus_management.md) - 高度な文書管理
- [クエリエンジン](tutorial_03_query_engine.md) - 高度な検索と回答生成
- [正規化処理](tutorial_04_normalization.md) - 文書正規化とクエリ最適化

#### Level 3: 高度な実装 (Advanced Implementation)
**前提知識**: Level 2完了、本番環境の理解  
**学習時間**: 6-8時間  
**目標**: エンタープライズ対応システムと高度な評価を実装

- [企業利用](tutorial_05_enterprise_usage_ja.md) - エンタープライズ環境での実装
- [インクリメンタル読み込み](tutorial_06_incremental_loading_ja.md) - 大規模データの効率的処理
- [RAG評価](tutorial_part3_evaluation.md) - 包括的な評価システム

#### Level 4: 専門家レベル (Expert)
**前提知識**: Level 3完了、高度なMLOps知識  
**学習時間**: 8-12時間  
**目標**: カスタムコンポーネント開発とプラグインシステム活用

- [評価メトリクス詳細](evaluation_metrics_guide_ja.md) - 評価指標の深い理解
- プラグイン開発 (planning) - カスタムコンポーネント作成
- MLOps統合 (planning) - 本番運用の最適化

## 主要コンポーネント別ガイド

### 📖 コーパス管理
- **文書読み込み**: 様々な形式の文書をシステムに取り込む
- **前処理**: 文書のクリーニングと正規化
- **チャンク分割**: 適切なサイズでの文書分割
- **埋め込み生成**: ベクトル化による意味表現

**関連チュートリアル**: 
- [コーパス作成](tutorial_part1_corpus_creation.md)
- [コーパス管理](tutorial_02_corpus_management.md)

### 🔍 検索・クエリエンジン
- **検索戦略**: ベクトル検索、キーワード検索、ハイブリッド検索
- **再ランキング**: 検索結果の精度向上
- **回答生成**: LLMによる自然な回答作成

**関連チュートリアル**:
- [QueryEngine基礎](tutorial_part2_query_engine.md)
- [クエリエンジン](tutorial_03_query_engine.md)

### 📊 評価・品質管理
- **自動評価**: QAペア生成と自動評価
- **メトリクス**: BLEU、ROUGE、RAG特化指標
- **品質監視**: 継続的な品質モニタリング

**関連チュートリアル**:
- [RAG評価](tutorial_part3_evaluation.md)
- [評価メトリクス詳細](evaluation_metrics_guide_ja.md)

## 使用例・シナリオ

### 🏢 ビジネス用途
- **社内文書検索**: 企業の知識ベース構築
- **カスタマーサポート**: FAQとサポート自動化
- **研究支援**: 学術論文・技術文書の検索

### 🛠️ 技術用途
- **API ドキュメント検索**: 開発者向け情報検索
- **コード理解**: ソースコード解析と説明
- **技術仕様書管理**: 仕様書の構造化と検索

### 🎓 教育用途
- **教材検索**: 学習リソースの効率的発見
- **質問応答システム**: 学習者の疑問解決
- **知識評価**: 理解度の測定と評価

## 前提知識とセットアップ

### 必要な前提知識
- **Python**: 基本的なプログラミングスキル
- **NLP基礎**: 自然言語処理の基本概念
- **機械学習**: 埋め込み、ベクトルの理解
- **API使用**: RESTful APIの基本

### 環境設定
```bash
# Python 3.10+ が必要
python --version

# 仮想環境の作成と有効化
uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# refinire-ragのインストール
uv pip install -e .

# 環境変数の設定
export OPENAI_API_KEY="your-api-key"
export REFINIRE_RAG_LLM_MODEL="gpt-4o-mini"
```

### 推奨学習順序

1. **概念理解** (30分)
   - [Why refinire-rag?](../why_refinire_rag_ja.md)
   - [コンセプト文書](../concept_ja.md)

2. **基礎実装** (2-3時間)
   - [基本RAG](tutorial_01_basic_rag_ja.md)
   - [コーパス作成](tutorial_part1_corpus_creation.md)

3. **実践応用** (4-6時間)
   - [QueryEngine](tutorial_part2_query_engine.md)
   - [評価システム](tutorial_part3_evaluation.md)

4. **高度な機能** (6-8時間)
   - [企業利用](tutorial_05_enterprise_usage_ja.md)
   - [インクリメンタル処理](tutorial_06_incremental_loading_ja.md)

## よくある質問

### Q: どのチュートリアルから始めるべきですか？
A: RAGが初めての場合は[基本RAGチュートリアル](tutorial_01_basic_rag_ja.md)から、経験者は[包括的ガイド](TUTORIAL_GUIDE_ja.md)で適切なレベルを選択してください。

### Q: 企業環境での実装について知りたいです
A: [企業利用チュートリアル](tutorial_05_enterprise_usage_ja.md)で、セキュリティ、スケーラビリティ、運用面について詳しく説明しています。

### Q: 評価方法がよくわかりません
A: [評価メトリクス詳細ガイド](evaluation_metrics_guide_ja.md)で、各指標の意味と使い分けを具体的に説明しています。

### Q: カスタマイズはどこまで可能ですか？
A: refinire-ragは高度にモジュラー設計されており、プラグインシステムによる拡張が可能です。詳細は各チュートリアルのカスタマイズセクションを参照してください。

## サポートとコミュニティ

- **Issues**: [GitHub Issues](https://github.com/kitfactory/refinire_rag/issues)
- **API リファレンス**: [API ドキュメント](../api/)
- **例とサンプル**: [examples/](../../examples/)

## 貢献

チュートリアルの改善、新しい例の追加、翻訳の向上にご協力いただける場合は、プルリクエストをお送りください。

---

**注記**: 各チュートリアルは独立して学習可能ですが、体系的な学習のため推奨順序に従うことをお勧めします。