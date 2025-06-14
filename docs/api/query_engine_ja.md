# QueryEngine - クエリ処理エンジン

クエリ処理と回答生成を管理するアプリケーションクラスです。

## 概要

QueryEngineは、ユーザーのクエリを受け取り、以下の処理を統合的に管理します：

1. **クエリ正規化** - 表現揺らぎの統一
2. **文書検索** - 関連文書の検索
3. **再ランキング** - 検索結果の再評価
4. **回答生成** - LLMによる回答生成

```python
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader

# コンポーネントの作成
retriever = SimpleRetriever(vector_store, embedder)
reranker = SimpleReranker()
reader = SimpleReader()

# QueryEngineの作成
query_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=retriever,
    reader=reader,
    reranker=reranker,
    config=QueryEngineConfig()
)
```

## QueryEngineConfig

QueryEngineの動作を制御する設定クラスです。

```python
class QueryEngineConfig(BaseModel):
    enable_query_normalization: bool = True      # クエリ正規化を有効化
    auto_detect_corpus_state: bool = True       # コーパス状態を自動検出
    retriever_top_k: int = 10                   # 検索結果の上位K件
    reranker_top_k: int = 5                     # 再ランキング後の上位K件
    include_sources: bool = True                # ソース情報を含める
    include_confidence: bool = True             # 信頼度を含める
    include_processing_metadata: bool = False   # 処理メタデータを含める
    query_timeout: float = 30.0                 # クエリタイムアウト（秒）
```

### 設定例

```python
# 基本設定
config = QueryEngineConfig()

# カスタム設定
config = QueryEngineConfig(
    enable_query_normalization=True,
    retriever_top_k=20,
    reranker_top_k=3,
    include_processing_metadata=True
)

# QueryEngineに適用
query_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=retriever,
    reader=reader,
    reranker=reranker,
    config=config
)
```

## 主要メソッド

### answer

クエリに対する回答を生成します。

```python
def answer(self, query: str) -> QueryResult:
    """
    クエリに対する回答を生成
    
    Args:
        query: ユーザーのクエリ
        
    Returns:
        QueryResult: 回答結果
    """
    
# 使用例
result = query_engine.answer("RAGとは何ですか？")
print(f"回答: {result.answer}")
print(f"信頼度: {result.confidence}")
print(f"ソース数: {len(result.sources)}")
```

### search

文書検索のみを実行します（回答生成なし）。

```python
def search(self, query: str) -> List[SearchResult]:
    """
    文書検索を実行
    
    Args:
        query: 検索クエリ
        
    Returns:
        List[SearchResult]: 検索結果リスト
    """
    
# 使用例
results = query_engine.search("ベクトル検索")
for result in results:
    print(f"文書ID: {result.document_id}, スコア: {result.score}")
```

### get_engine_stats

エンジンの統計情報を取得します。

```python
stats = query_engine.get_engine_stats()
print(f"処理クエリ数: {stats['queries_processed']}")
print(f"平均応答時間: {stats['average_response_time']}")
print(f"正規化率: {stats['normalization_rate']}")
```

## クエリ正規化

コーパスに正規化が適用されている場合、クエリも自動的に正規化されます。

```python
# 正規化辞書の設定
normalizer_config = NormalizerConfig(
    dictionary_file_path="dictionary.md",
    normalize_variations=True,
    whole_word_only=False
)
query_engine.normalizer = Normalizer(normalizer_config)

# クエリ処理
result = query_engine.answer("検索強化生成について教えて")
# → 内部で "検索拡張生成について教えて" に正規化される

print(f"元のクエリ: {result.query}")
print(f"正規化後: {result.normalized_query}")
```

## QueryResult

クエリ処理の結果を表すモデルです。

```python
class QueryResult:
    query: str                    # 元のクエリ
    answer: str                   # 生成された回答
    sources: List[SearchResult]   # 参照したソース
    confidence: float             # 回答の信頼度 (0.0-1.0)
    metadata: Dict[str, Any]      # メタデータ
    processing_time: float        # 処理時間（秒）
    normalized_query: str         # 正規化後のクエリ
```

### 結果の活用例

```python
result = query_engine.answer("RAGの利点は？")

# 基本情報
print(f"質問: {result.query}")
print(f"回答: {result.answer}")
print(f"信頼度: {result.confidence:.2%}")

# ソース情報
print("\n参照ソース:")
for i, source in enumerate(result.sources[:3], 1):
    print(f"{i}. {source.metadata.get('title', 'Unknown')}")
    print(f"   スコア: {source.score:.3f}")
    print(f"   内容: {source.content[:100]}...")

# メタデータ
if result.metadata.get('query_normalized'):
    print(f"\nクエリ正規化: {result.normalized_query}")
print(f"処理時間: {result.processing_time:.3f}秒")
```

## コンポーネントのカスタマイズ

### カスタムRetriever

```python
from refinire_rag.retrieval.base import Retriever

class CustomRetriever(Retriever):
    def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
        # カスタム検索ロジック
        pass

# 使用
custom_retriever = CustomRetriever()
query_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=custom_retriever,
    reader=reader
)
```

### カスタムReader

```python
from refinire_rag.retrieval.base import Reader

class CustomReader(Reader):
    def generate_answer(
        self, 
        query: str, 
        contexts: List[str]
    ) -> str:
        # カスタム回答生成ロジック
        pass
```

## エラーハンドリング

```python
try:
    result = query_engine.answer(query)
except TimeoutError:
    print("クエリ処理がタイムアウトしました")
except RetrievalError as e:
    print(f"検索エラー: {e}")
except GenerationError as e:
    print(f"回答生成エラー: {e}")
```

## パフォーマンス最適化

### キャッシング

```python
# キャッシュ付きQueryEngine
query_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=retriever,
    reader=reader,
    cache_enabled=True,
    cache_ttl=3600  # 1時間
)
```

### バッチ処理

```python
# 複数クエリのバッチ処理
queries = ["質問1", "質問2", "質問3"]
results = query_engine.batch_answer(queries)
```

## ベストプラクティス

1. **適切なtop_k設定**: 検索精度とパフォーマンスのバランス
2. **正規化の活用**: 日本語クエリでは特に効果的
3. **タイムアウト設定**: 長時間処理を防ぐ
4. **エラーハンドリング**: ユーザー体験の向上

```python
# 推奨設定例
config = QueryEngineConfig(
    enable_query_normalization=True,
    retriever_top_k=15,    # 十分な候補を取得
    reranker_top_k=3,      # 最終的に3件に絞る
    include_sources=True,   # デバッグと信頼性のため
    query_timeout=10.0      # 10秒でタイムアウト
)
```