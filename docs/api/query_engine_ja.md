# QueryEngine - クエリ処理エンジン

QueryEngineは、コーパス状態に基づく自動正規化と柔軟なコンポーネント構成により、クエリから回答までの完全なワークフローを統合管理します。

## 概要

QueryEngineは、以下のワークフローでインテリジェントなクエリ処理を提供します：

1. **クエリ正規化** - コーパス状態に基づく自動正規化
2. **文書検索** - 複数検索器対応の柔軟な構成
3. **結果再ランク** - 関連性最適化のための任意の再ランク機能
4. **回答生成** - コンテキストを考慮した回答合成

```python
from refinire_rag.application import QueryEngine, QueryEngineConfig
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, AnswerSynthesizer

# コンポーネントの作成
retriever = SimpleRetriever(vector_store, embedder)
reranker = SimpleReranker()
synthesizer = AnswerSynthesizer()

# QueryEngineの作成
query_engine = QueryEngine(
    corpus_name="knowledge_base",
    retrievers=retriever,  # 単一検索器またはリスト
    synthesizer=synthesizer,
    reranker=reranker,  # 任意
    config=QueryEngineConfig()
)
```

## パブリックAPIメソッド

### __init__

コンポーネントと設定でQueryEngineを初期化します。

```python
QueryEngine(
    corpus_name: str,
    retrievers: Union[Retriever, List[Retriever]],
    synthesizer: AnswerSynthesizer,
    reranker: Optional[Reranker] = None,
    config: Optional[QueryEngineConfig] = None
)
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|----|---------|-------------|
| `corpus_name` | `str` | 必須 | このクエリエンジンのコーパス名 |
| `retrievers` | `Union[Retriever, List[Retriever]]` | 必須 | 単一検索器または検索器のリスト |
| `synthesizer` | `AnswerSynthesizer` | 必須 | 回答生成用のコンポーネント |
| `reranker` | `Optional[Reranker]` | `None` | 結果再ランク用の任意のコンポーネント |
| `config` | `Optional[QueryEngineConfig]` | `None` | エンジンの設定 |

### set_normalizer

クエリ処理用の正規化器を設定します。

```python
set_normalizer(normalizer: Optional[Normalizer]) -> None
```

### query

ユーザークエリに対して回答を生成します。

```python
query(query: str, context: Optional[Dict[str, Any]] = None) -> QueryResult
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|----|---------|-------------|
| `query` | `str` | 必須 | ユーザークエリ文字列 |
| `context` | `Optional[Dict[str, Any]]` | `None` | 任意のコンテキストパラメータ (top_k, filters, etc.) |

### add_retriever

エンジンに新しい検索器を追加します。

```python
add_retriever(retriever: Retriever) -> None
```

### remove_retriever

インデックスによって検索器を削除します。

```python
remove_retriever(index: int) -> bool
```

### get_engine_stats

包括的なエンジン統計を取得します。

```python
get_engine_stats() -> Dict[str, Any]
```

### clear_cache

キャッシュされたデータをクリアします。

```python
clear_cache() -> None
```

## QueryEngineConfig

QueryEngineの動作を制御する設定クラスです。

```python
@dataclass
class QueryEngineConfig:
    # クエリ処理設定
    enable_query_normalization: bool = True
    
    # コンポーネント設定
    retriever_top_k: int = 10                    # 検索器あたりの結果数
    total_top_k: int = 20                        # 統合後の総結果数
    reranker_top_k: int = 5                      # 再ランク後の最終結果数
    synthesizer_max_context: int = 2000          # 回答生成用の最大コンテキスト
    
    # パフォーマンス設定
    enable_caching: bool = True
    cache_ttl: int = 3600                        # キャッシュTTL（秒）
    
    # 出力設定
    include_sources: bool = True
    include_confidence: bool = True
    include_processing_metadata: bool = True
    
    # 複数検索器設定
    deduplicate_results: bool = True             # 重複文書の削除
    combine_scores: str = "max"                  # スコア統合方法
```

## 使用例

### 基本的なクエリ処理

```python
from refinire_rag.application import QueryEngine
from refinire_rag.retrieval import SimpleRetriever, AnswerSynthesizer

# コンポーネントの設定
retriever = SimpleRetriever(vector_store, embedder)
synthesizer = AnswerSynthesizer()

# QueryEngineの作成
query_engine = QueryEngine(
    corpus_name="knowledge_base",
    retrievers=retriever,
    synthesizer=synthesizer
)

# クエリの処理
result = query_engine.query("RAGはどのように動作しますか？")
print(f"回答: {result.answer}")
print(f"ソース数: {len(result.sources)}")
```

### 複数検索器の設定

```python
from refinire_rag.retrieval import SimpleRetriever, HybridRetriever

# 異なる検索戦略のための複数検索器
vector_retriever = SimpleRetriever(vector_store, embedder)
hybrid_retriever = HybridRetriever(vector_store, keyword_store)

query_engine = QueryEngine(
    corpus_name="knowledge_base",
    retrievers=[vector_retriever, hybrid_retriever],  # 検索器のリスト
    synthesizer=synthesizer,
    config=QueryEngineConfig(
        total_top_k=30,  # 複数検索器からより多くの結果を取得
        reranker_top_k=5
    )
)
```

### 正規化機能付きの高度な設定

```python
from refinire_rag.processing import Normalizer

# 正規化機能付きの設定
config = QueryEngineConfig(
    enable_query_normalization=True,
    retriever_top_k=15,
    total_top_k=25,
    reranker_top_k=8,
    include_processing_metadata=True,
    deduplicate_results=True,
    combine_scores="average"
)

query_engine = QueryEngine(
    corpus_name="knowledge_base",
    retrievers=[vector_retriever, hybrid_retriever],
    synthesizer=synthesizer,
    reranker=reranker,
    config=config
)

# クエリ処理用の正規化器を設定
normalizer = Normalizer(dictionary_path="./knowledge_base_dictionary.md")
query_engine.set_normalizer(normalizer)
```

### コンテキストパラメータ付きクエリ

```python
# カスタムコンテキスト付きクエリ
result = query_engine.query(
    query="機械学習はどのように動作しますか？",
    context={
        "retriever_top_k": 20,  # デフォルトを上書き
        "rerank_top_k": 10,     # デフォルトを上書き
        "filters": {"document_type": "technical"}
    }
)

print(f"回答: {result.answer}")
print(f"信頼度: {result.confidence}")
print(f"処理時間: {result.processing_time}")
```

## QueryResultオブジェクト

`query`メソッドは以下の構造を持つ`QueryResult`オブジェクトを返します：

```python
@dataclass
class QueryResult:
    answer: str                          # 生成された回答
    confidence: float                    # 信頼度スコア
    sources: List[SearchResult]          # ソース文書
    processing_time: float               # 総処理時間
    query_metadata: Dict[str, Any]       # 処理メタデータ
    
    # 設定に基づく任意フィールド
    normalized_query: Optional[str]      # 正規化されたクエリ（有効化時）
    retrieval_results: Optional[List]    # 生の検索結果
    reranking_results: Optional[List]    # 再ランク結果
```

## エンジン統計

```python
# 包括的なエンジン統計の取得
stats = query_engine.get_engine_stats()

print(f"処理されたクエリ総数: {stats['total_queries']}")
print(f"平均処理時間: {stats['avg_processing_time']:.2f}秒")
print(f"キャッシュヒット率: {stats['cache_hit_rate']:.1%}")
print(f"検索器数: {stats['num_retrievers']}")
```

## ベストプラクティス

1. **複数検索器戦略**: 包括的なカバレッジのために複数検索器を使用
2. **正規化**: 一貫性向上のためにクエリ正規化を有効化
3. **キャッシュ**: 性能向上のためにキャッシュを有効に保つ
4. **コンテキストパラメータ**: 動的なクエリ調整にコンテキストパラメータを使用
5. **統計監視**: 性能最適化のためにエンジン統計を監視

## 完全な例

```python
from refinire_rag.application import QueryEngine, QueryEngineConfig
from refinire_rag.retrieval import SimpleRetriever, HybridRetriever, AnswerSynthesizer, SimpleReranker
from refinire_rag.processing import Normalizer

def create_query_engine():
    # コンポーネントの初期化
    vector_retriever = SimpleRetriever(vector_store, embedder)
    hybrid_retriever = HybridRetriever(vector_store, keyword_store)
    reranker = SimpleReranker()
    synthesizer = AnswerSynthesizer()
    
    # エンジンの設定
    config = QueryEngineConfig(
        enable_query_normalization=True,
        retriever_top_k=15,
        total_top_k=25,
        reranker_top_k=8,
        include_sources=True,
        include_confidence=True,
        deduplicate_results=True
    )
    
    # QueryEngineの作成
    query_engine = QueryEngine(
        corpus_name="knowledge_base",
        retrievers=[vector_retriever, hybrid_retriever],
        synthesizer=synthesizer,
        reranker=reranker,
        config=config
    )
    
    # 正規化の設定
    normalizer = Normalizer(dictionary_path="./knowledge_base_dictionary.md")
    query_engine.set_normalizer(normalizer)
    
    return query_engine

# 使用
query_engine = create_query_engine()

# クエリの処理
result = query_engine.query("RAGの利点は何ですか？")
print(f"回答: {result.answer}")
print(f"信頼度: {result.confidence:.2f}")
print(f"使用されたソース数: {len(result.sources)}")

# 統計の取得
stats = query_engine.get_engine_stats()
print(f"総クエリ数: {stats['total_queries']}")
```