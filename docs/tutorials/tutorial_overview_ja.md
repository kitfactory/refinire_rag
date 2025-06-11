# refinire-rag チュートリアル

このチュートリアルでは、refinire-ragライブラリを使用したRAG（Retrieval-Augmented Generation）システムの構築方法を段階的に学習します。

## チュートリアル構成

### 1. 基本編
- [チュートリアル1: 基本的なRAGパイプライン](tutorial_01_basic_rag_ja.md)
- [チュートリアル2: コーパス管理とドキュメント処理](tutorial_02_corpus_management_ja.md)
- [チュートリアル3: クエリエンジンと回答生成](tutorial_03_query_engine_ja.md)

### 2. 応用編
- [チュートリアル4: 高度な正規化とクエリ処理](tutorial_04_normalization_ja.md)
- [チュートリアル5: エンタープライズ利用 - 部門別RAG](tutorial_05_enterprise_usage_ja.md)
- [チュートリアル6: 増分ドキュメントローディング](tutorial_06_incremental_loading_ja.md)
- [チュートリアル7: RAGシステムの評価](tutorial_07_rag_evaluation_ja.md)

### 3. 本番環境編（計画中）
- チュートリアル8: 本番環境デプロイメントとモニタリング
- チュートリアル9: パフォーマンス最適化とスケーリング

## 前提知識

このチュートリアルを始める前に、以下の基本知識があることを推奨します：

- Python プログラミングの基礎
- 機械学習・自然言語処理の基本概念
- RAG（Retrieval-Augmented Generation）の概要理解

## システム要件

- Python 3.10以上
- メモリ: 4GB以上推奨
- ディスク容量: 1GB以上の空き容量

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/your-org/refinire-rag.git
cd refinire-rag

# 仮想環境の作成
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係のインストール
pip install -e .
```

## 学習の進め方

1. **基本編から順番に学習**: 各チュートリアルは前のチュートリアルの知識を前提としています
2. **実際にコードを実行**: 提供されるサンプルコードを実際に動かしてみてください
3. **カスタマイズに挑戦**: 基本を理解したら、自分のデータやユースケースに合わせてカスタマイズしてみてください

## サポート

チュートリアルで問題が発生した場合：

1. [README.md](../README.md)の設定を確認
2. [トラブルシューティングガイド](troubleshooting.md)を参照
3. [Issues](https://github.com/your-org/refinire-rag/issues)で質問を投稿

それでは、[チュートリアル1: 基本的なRAGパイプライン](tutorial_01_basic_rag_ja.md)から始めましょう！