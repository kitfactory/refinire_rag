# 統一検索API仕様書 / Unified Retrieval API Specification

## 概要 / Overview

refinire-ragの新しい統一検索アーキテクチャは、ベクトル検索、キーワード検索、ハイブリッド検索を統一されたインターフェースで提供します。

The new unified retrieval architecture in refinire-rag provides vector search, keyword search, and hybrid search through a unified interface.

## 📋 主要クラス一覧 / Main Classes

| クラス | 役割 | 継承 | 主要機能 |
|--------|------|------|----------|
| `Retriever` | 検索機能 | `QueryComponent` | 検索機能の基底クラス |
| `Indexer` | インデックス機能 | - | 文書管理機能の基底クラス |
| `VectorStore` | ベクトル検索ストア | `Retriever`, `Indexer` | ベクトル検索+保存 |
| `KeywordStore` | キーワード検索ストア | `Retriever`, `Indexer` | キーワード検索+保存 |
| `HybridRetriever` | ハイブリッド検索 | `Retriever` | 複数検索手法の統合 |

## 🔍 基底インターフェース / Base Interface

### Retrieverクラス

```python
from refinire_rag.retrieval import Retriever, SearchResult
from typing import List, Optional, Dict, Any

class Retriever(ABC):
    def retrieve(self, 
                 query: str, 
                 limit: Optional[int] = None,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        文書を検索します
        
        Args:
            query: 検索クエリテキスト
            limit: 最大結果数（Noneの場合はconfig.top_kを使用）
            metadata_filter: メタデータフィルタ
                例: {"department": "AI", "year": 2024}
        
        Returns:
            List[SearchResult]: スコア付き検索結果（関連度順）
        """
```

### Indexerクラス

```python
from refinire_rag.retrieval import Indexer

class Indexer:
    """Base class for document indexing capabilities
    
    Provides document indexing and management functionality that can be
    used by Retriever implementations to create searchable stores.
    
    検索可能なストアを作成するためにRetriever実装で使用できる
    文書インデックスと管理機能を提供します。
    """
```

## 🔧 実装クラス / Implementation Classes

### 1. VectorStore - ベクトル検索ストア

```python
from refinire_rag.retrieval import VectorStore, VectorStoreConfig
from refinire_rag.storage import InMemoryVectorStore
from refinire_rag.embedding import OpenAIEmbedder

# 設定
config = VectorStoreConfig(
    top_k=10,
    similarity_threshold=0.7,
    embedding_model="text-embedding-3-small"
)

# 初期化
backend_store = InMemoryVectorStore()
embedder = OpenAIEmbedder()
vector_store = VectorStore(backend_store, embedder, config)

# 文書のインデックス
from refinire_rag.models import Document

documents = [
    Document(id="doc1", content="機械学習について", metadata={"category": "AI"}),
    Document(id="doc2", content="自然言語処理の技術", metadata={"category": "NLP"})
]

vector_store.index_documents(documents)

# 検索
results = vector_store.retrieve(
    query="AI技術について",
    limit=5,
    metadata_filter={"category": "AI"}
)

for result in results:
    print(f"文書ID: {result.document_id}")
    print(f"スコア: {result.score}")
    print(f"内容: {result.document.content}")
```

### 2. KeywordStore - キーワード検索ストア

```python
from refinire_rag.retrieval import TFIDFKeywordStore, KeywordStoreConfig

# 設定
config = KeywordStoreConfig(
    top_k=10,
    algorithm="tfidf",
    similarity_threshold=0.1
)

# 初期化
keyword_store = TFIDFKeywordStore(config)

# 文書のインデックス
keyword_store.index_documents(documents)

# 検索
results = keyword_store.retrieve(
    query="機械学習 技術",
    limit=5,
    metadata_filter={"category": "AI"}
)
```

### 3. HybridRetriever - ハイブリッド検索

```python
from refinire_rag.retrieval import HybridRetriever, HybridRetrieverConfig

# 複数の検索器を準備
retrievers = [vector_store, keyword_store]

# ハイブリッド検索の設定
config = HybridRetrieverConfig(
    top_k=10,
    fusion_method="rrf",  # "rrf", "weighted", "max"
    retriever_weights=[0.7, 0.3],  # ベクトル検索70%, キーワード検索30%
    rrf_k=60
)

# 初期化
hybrid_retriever = HybridRetriever(retrievers, config)

# 検索（複数手法を自動統合）
results = hybrid_retriever.retrieve(
    query="AI技術の応用",
    limit=10,
    metadata_filter={"year": 2024}
)
```

## 📊 検索結果形式 / Search Result Format

### SearchResultクラス

```python
@dataclass
class SearchResult:
    document_id: str          # 文書ID
    document: Document        # 文書オブジェクト
    score: float             # 関連度スコア（0.0-1.0）
    metadata: Dict[str, Any] # 検索メタデータ

# 使用例
for result in results:
    print(f"文書: {result.document_id}")
    print(f"スコア: {result.score:.3f}")
    print(f"検索手法: {result.metadata['retrieval_method']}")
    print(f"内容: {result.document.content[:100]}...")
```

## 🔍 メタデータフィルタリング / Metadata Filtering

### 基本フィルタ

```python
# 完全一致
metadata_filter = {"department": "AI"}

# 複数条件（AND）
metadata_filter = {
    "department": "AI",
    "year": 2024,
    "status": "active"
}

# OR条件（リスト指定）
metadata_filter = {
    "department": ["AI", "ML", "NLP"]
}

# 範囲指定
metadata_filter = {
    "year": {"$gte": 2020, "$lte": 2024},
    "score": {"$gte": 0.8}
}

# 除外条件
metadata_filter = {
    "status": {"$ne": "archived"}
}
```

### 使用例

```python
# 2023年以降のAI部門の文書を検索
results = retriever.retrieve(
    query="機械学習モデル",
    metadata_filter={
        "department": "AI",
        "year": {"$gte": 2023},
        "status": "published"
    }
)
```

## ⚙️ 設定オプション / Configuration Options

### VectorStoreConfig

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `top_k` | int | 10 | 最大検索結果数 |
| `similarity_threshold` | float | 0.0 | 類似度閾値 |
| `embedding_model` | str | "text-embedding-3-small" | 埋め込みモデル |
| `batch_size` | int | 100 | バッチ処理サイズ |

### KeywordStoreConfig

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `top_k` | int | 10 | 最大検索結果数 |
| `algorithm` | str | "bm25" | 検索アルゴリズム |
| `index_path` | str | None | インデックス保存パス |

### HybridRetrieverConfig

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `fusion_method` | str | "rrf" | 統合手法（rrf/weighted/max） |
| `retriever_weights` | List[float] | None | 検索器の重み |
| `rrf_k` | int | 60 | RRFパラメータ |

## 🔄 統合手法 / Fusion Methods

### 1. Reciprocal Rank Fusion (RRF)

```python
config = HybridRetrieverConfig(fusion_method="rrf", rrf_k=60)
```

**特徴**: ランキングベースの統合。スコアの違いに影響されにくい。

### 2. Weighted Fusion

```python
config = HybridRetrieverConfig(
    fusion_method="weighted",
    retriever_weights=[0.7, 0.3]  # 重み比率
)
```

**特徴**: スコアの重み付き平均。検索器の重要度を調整可能。

### 3. Max Score Fusion

```python
config = HybridRetrieverConfig(fusion_method="max")
```

**特徴**: 各文書の最高スコアを採用。シンプルで高速。

## 💡 使用パターン / Usage Patterns

### パターン1: シンプルなベクトル検索

```python
from refinire_rag.retrieval import VectorStore
from refinire_rag.storage import InMemoryVectorStore
from refinire_rag.embedding import TFIDFEmbedder

# 軽量セットアップ
vector_store = VectorStore(
    backend_store=InMemoryVectorStore(),
    embedder=TFIDFEmbedder()
)

# 文書インデックス + 検索
vector_store.index_documents(documents)
results = vector_store.retrieve("検索クエリ")
```

### パターン2: プロダクション環境

```python
# ChromaDBプラグイン使用
from refinire_rag_chroma import ChromaVectorStore
from refinire_rag.embedding import OpenAIEmbedder
from refinire_rag.retrieval import VectorStore

# 本格的セットアップ
vector_store = VectorStore(
    backend_store=ChromaVectorStore("production_collection"),
    embedder=OpenAIEmbedder(api_key="your-key"),
    config=VectorStoreConfig(
        top_k=20,
        similarity_threshold=0.75,
        embedding_model="text-embedding-3-large"
    )
)
```

### パターン3: 部署別検索システム

```python
# 部署ごとに検索器を分離
departments = ["AI", "Sales", "HR"]
department_stores = {}

for dept in departments:
    store = VectorStore(
        backend_store=ChromaVectorStore(f"dept_{dept.lower()}"),
        embedder=OpenAIEmbedder()
    )
    department_stores[dept] = store

# 部署指定検索
def search_by_department(query: str, department: str):
    if department in department_stores:
        return department_stores[department].retrieve(query)
    else:
        # 全部署横断検索
        hybrid = HybridRetriever(list(department_stores.values()))
        return hybrid.retrieve(query)
```

## 📈 パフォーマンス最適化 / Performance Optimization

### バッチ処理

```python
# 大量文書の効率的インデックス
large_documents = [...]  # 10,000件の文書

# バッチサイズを調整
config = VectorStoreConfig(batch_size=500)
vector_store = VectorStore(backend_store, embedder, config)

# バッチで処理（メモリ効率が良い）
vector_store.index_document_batch(large_documents, batch_size=500)
```

### 並列検索

```python
# 複数検索の並列実行
import asyncio

async def parallel_search(queries: List[str]):
    tasks = []
    for query in queries:
        task = asyncio.create_task(
            asyncio.to_thread(retriever.retrieve, query)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

### メモリ管理

```python
# 定期的なインデックスクリア
if vector_store.get_document_count() > 100000:
    # 古い文書を削除
    old_documents = get_old_documents()
    for doc_id in old_documents:
        vector_store.remove_document(doc_id)
```

## 🚨 エラーハンドリング / Error Handling

```python
from refinire_rag.exceptions import RefinireRAGError, EmbeddingError

try:
    results = vector_store.retrieve("クエリ")
except EmbeddingError as e:
    print(f"埋め込み生成エラー: {e}")
except RefinireRAGError as e:
    print(f"RAGシステムエラー: {e}")
except Exception as e:
    print(f"予期しないエラー: {e}")

# 統計情報でエラー監視
stats = vector_store.get_processing_stats()
if stats["errors_encountered"] > 0:
    print(f"エラー発生数: {stats['errors_encountered']}")
```

## 📊 統計・監視 / Statistics and Monitoring

```python
# 詳細統計情報
stats = retriever.get_processing_stats()

print(f"検索実行回数: {stats['queries_processed']}")
print(f"平均処理時間: {stats['processing_time'] / max(stats['queries_processed'], 1):.3f}秒")
print(f"エラー率: {stats['errors_encountered'] / max(stats['queries_processed'], 1) * 100:.1f}%")

# ハイブリッド検索の詳細
if isinstance(retriever, HybridRetriever):
    print(f"使用検索器: {stats['retriever_types']}")
    print(f"統合手法: {stats['fusion_method']}")
```

## 🔄 移行ガイド / Migration Guide

### 既存コードからの移行

#### Before (既存のSimpleRetriever)
```python
from refinire_rag.retrieval import SimpleRetriever

retriever = SimpleRetriever(vector_store, embedder)
results = retriever.retrieve("query")
```

#### After (新しいVectorStore)
```python
from refinire_rag.retrieval import VectorStore

vector_store = VectorStore(backend_store, embedder)
results = vector_store.retrieve("query", metadata_filter={"dept": "AI"})
```

### プラグイン対応

#### ChromaDBプラグイン使用
```python
# 新しいVectorStoreでChromaDBを使用
from refinire_rag_chroma import ChromaVectorStore
from refinire_rag.retrieval import VectorStore

# ChromaDBバックエンドを使用
chroma_backend = ChromaVectorStore("my_collection")
vector_store = VectorStore(chroma_backend, embedder)
```

---

## 📞 サポート / Support

- **ドキュメント**: [完全ガイド](../tutorials/tutorial_overview.md)
- **サンプルコード**: [examples/](../../examples/)
- **Issue報告**: [GitHub Issues](https://github.com/kitfactory/refinire-rag/issues)

統一検索APIにより、refinire-ragはより柔軟で強力な検索機能を提供します。