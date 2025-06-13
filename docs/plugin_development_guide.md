# プラグイン開発ガイド

## 概要

refinire-ragでは、統一されたプラグインアーキテクチャを提供しており、開発者は一貫したルールに従ってカスタム実装を作成できます。

## プラグインの種類と使用ユースケース

refinire-ragは以下のプラグインタイプをサポートしています：

| プラグインタイプ | サブパッケージ | 実装インターフェース | 使用ユースケース | 目的 |
|---|---|---|---|---|
| DocumentProcessor | refinire.rag.document_processor | DocumentProcessor | CorpusManager | 文書の前処理・変換 |
| VectorStore | refinire.rag.vectorstore | VectorStore | CorpusManager, QueryEngine | ベクトルベースの文書検索 |
| KeywordStore | refinire.rag.keywordstore | KeywordStore | CorpusManager, QueryEngine | キーワードベースの文書検索 |
| Loader | refinire.rag.loader | Loader | CorpusManager | ファイル読み込み |
| Embedder | refinire.rag.embedder | Embedder | CorpusManager | テキストのベクトル化 |
| Reranker | refinire.rag.reranker | Reranker | QueryEngine | 検索結果の再ランキング |
| Synthesizer | refinire.rag.synthesizer | Synthesizer | QueryEngine | LLMを使用した回答生成 |
| Evaluator | refinire.rag.evaluator | Evaluator | QualityLab | 評価メトリクスの計算 |
| ContradictionDetector | refinire.rag.contradiction | ContradictionDetector | QualityLab | 文書間の矛盾検出 |

## DocumentProcessorについて

DocumentProcessorは、CorpusManagerで使用される文書処理の統一インターフェースです。文書の前処理、変換、正規化などの処理を実装する際に使用します。

詳細な実装方法については、[アーキテクチャ設計書](architecture.md)の「DocumentProcessor統一アーキテクチャ」セクションを参照してください。

## 実装インターフェースの詳細

各プラグインタイプには、以下の必須メソッドを実装する必要があります：

### VectorStore
```python
def retrieve(self, query_vector: np.ndarray, k: int = 10) -> List[SearchResult]:
    """クエリベクトルに基づいて文書を検索"""
    pass

def store_vectors(self, vectors: List[Vector]) -> None:
    """ベクトルを保存"""
    pass
```

### KeywordStore
```python
def search(self, query: str, k: int = 10) -> List[SearchResult]:
    """キーワード検索を実行"""
    pass

def index_documents(self, documents: List[Document]) -> None:
    """文書をインデックス化"""
    pass
```

### Loader
```python
def load_single(self, path: Path) -> Document:
    """単一ファイルを読み込み"""
    pass

def load_batch(self, paths: List[Path]) -> List[Document]:
    """複数ファイルを読み込み"""
    pass
```

### Embedder
```python
def embed(self, text: str) -> np.ndarray:
    """テキストをベクトル化"""
    pass

def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
    """複数テキストをベクトル化"""
    pass
```

### Reranker
```python
def rerank(self, documents: List[Document], query: str) -> List[Document]:
    """検索結果を再ランキング"""
    pass
```

### Synthesizer
```python
def synthesize(self, query: str, context: List[Document]) -> str:
    """コンテキストから回答を生成"""
    pass
```

### DocumentProcessor
```python
def process(self, document: Document) -> List[Document]:
    """文書を処理"""
    pass

def get_config_class(self) -> Type[DocumentProcessorConfig]:
    """設定クラスを取得"""
    pass
```

### Evaluator
```python
def evaluate(self, results: List[Result]) -> Metrics:
    """評価メトリクスを計算"""
    pass
```

### ContradictionDetector
```python
def detect(self, documents: List[Document]) -> List[Conflict]:
    """文書間の矛盾を検出"""
    pass
```

## パッケージ構造

プラグイン開発者は以下のパッケージ構造を推奨します：

```
my-refinire-plugin/
├── pyproject.toml
├── README.md
├── src/
│   └── my_refinire_plugin/
│       ├── __init__.py
│       ├── vectorstore.py
│       └── keywordstore.py
└── tests/
    ├── __init__.py
    ├── test_vectorstore.py
    └── test_keywordstore.py
```

## pyproject.tomlの設定

プラグインの`pyproject.toml`には以下の設定が必要です：

```toml
[project]
name = "my-refinire-plugin"
version = "0.1.0"
description = "My refinire-rag plugin"
requires-python = ">=3.10"
dependencies = [
    "refinire-rag>=0.1.0",
    "numpy>=1.24.0",
]

[project.entry-points."refinire.rag"]
vectorstore = "my_refinire_plugin.vectorstore:MyVectorStore"
keywordstore = "my_refinire_plugin.keywordstore:MyKeywordStore"
```

## 実装例

### VectorStoreプラグインの実装例

```python
from typing import List
import numpy as np
from refinire.rag.vectorstore import VectorStore
from refinire.rag.models import SearchResult, Vector

class MyVectorStore(VectorStore):
    """カスタムベクトルストアの実装例"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vectors = {}  # ベクトルを保存する辞書
    
    def retrieve(self, query_vector: np.ndarray, k: int = 10) -> List[SearchResult]:
        # クエリベクトルに基づいて検索を実行
        results = []
        for doc_id, vector in self.vectors.items():
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            results.append(SearchResult(
                document_id=doc_id,
                score=float(similarity)
            ))
        
        # スコアでソートして上位k件を返す
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    def store_vectors(self, vectors: List[Vector]) -> None:
        for vector in vectors:
            self.vectors[vector.document_id] = vector.embedding
```

## Refinireとの統合

プラグインはRefinireのStepクラスと統合可能です：

```python
from refinire import Step
from refinire.rag.models import Document

class MyRAGStep(Step):
    """カスタムRAGステップの実装例"""
    
    def __init__(self, vector_store: VectorStore, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vector_store = vector_store
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data["query"]
        # 検索と回答生成のロジックを実装
        return {"result": "回答"}
```

## テスト

プラグインには適切なテストを実装することを推奨します：

```python
import pytest
from refinire.rag.models import Document, Vector
from my_refinire_plugin.vectorstore import MyVectorStore

def test_vector_store_retrieval():
    store = MyVectorStore()
    # テストデータの準備
    test_vector = np.array([1.0, 0.0, 0.0])
    store.store_vectors([
        Vector(document_id="doc1", embedding=np.array([1.0, 0.0, 0.0])),
        Vector(document_id="doc2", embedding=np.array([0.0, 1.0, 0.0]))
    ])
    
    # 検索のテスト
    results = store.retrieve(test_vector, k=1)
    assert len(results) == 1
    assert results[0].document_id == "doc1"
    assert results[0].score > 0.9
```

## ベストプラクティス

1. **エラーハンドリング**
   - 適切な例外を定義し、エラーを適切に処理
   - ユーザーフレンドリーなエラーメッセージを提供

2. **メタデータの設定**
   - プラグインのバージョン情報を提供
   - 必要な依存関係を明示
   - 設定オプションを文書化

3. **パフォーマンス最適化**
   - バッチ処理をサポート
   - 非同期処理を検討
   - キャッシュを適切に使用

4. **テストカバレッジ**
   - ユニットテストを実装
   - エッジケースをカバー
   - パフォーマンステストを含める