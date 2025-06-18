# QueryEngine インターフェース

QueryEngineは、RAGシステムのクエリ処理とアンサー生成を統合する中心的なコンポーネントです。

## コーパス名の重要性

QueryEngineには必須パラメータとして`corpus_name`を指定します。コーパス名は以下の用途で使用されます：

- **正規化処理**: コーパス固有の辞書やルールの適用
- **ログ管理**: どのコーパスに対するクエリかの識別  
- **統計管理**: コーパス別の性能指標の追跡
- **マルチコーパス環境**: 複数コーパスを扱う際の識別子
- **デバッグ**: エラー発生時のコーパス特定

## 主要クラス

### QueryEngine

```python
class QueryEngine:
    """Query processing and answer generation engine
    
    完全なクエリからアンサーまでのワークフローを調整：
    1. クエリ正規化（コーパスが正規化されている場合）
    2. ベクトル類似度を使用した文書検索
    3. 関連性最適化のための結果再ランキング
    4. コンテキスト付きアンサー生成
    
    エンジンは自動的にコーパス処理状態に適応し、
    コーパス構築時に使用されたのと同じ正規化を適用します。
    """
```

### 初期化

```python
def __init__(self, 
             corpus_name: str,
             retrievers: Union[Retriever, List[Retriever]],
             synthesizer: AnswerSynthesizer,
             reranker: Optional[Reranker] = None,
             config: Optional[QueryEngineConfig] = None):
    """QueryEngineを初期化
    
    Args:
        corpus_name: このクエリエンジンのコーパス名
                    正規化、ログ、統計で使用される識別子
        retrievers: 文書検索のためのRetrieverコンポーネント（単一またはリスト）
                   単一のRetrieverまたはRetrieverのリスト
        synthesizer: アンサー生成のためのAnswerSynthesizerコンポーネント
                    回答生成のためのAnswerSynthesizerコンポーネント
        reranker: 結果再ランキングのためのオプションのRerankerコンポーネント
                 結果再ランキングのためのオプションのRerankerコンポーネント
        config: エンジンの設定
               エンジンの設定
    """
```

## 主要メソッド

### 1. query() - メインインターフェース

```python
def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryResult:
    """クエリに対するアンサーを生成
    
    Args:
        query (str): ユーザークエリ
        context (Dict[str, Any], optional): オプションのコンテキストパラメータ
            - top_k (int): 取得する結果の最大数
            - rerank_top_k (int): 再ランキングする結果数
            - metadata_filter (Dict): メタデータフィルタ
            - filters (Dict): 検索制約
            
    Returns:
        QueryResult: アンサーとメタデータを含む結果
            - query (str): 元のクエリ
            - normalized_query (str): 正規化されたクエリ（該当する場合）
            - answer (str): 生成されたアンサー
            - sources (List[SearchResult]): ソース文書
            - confidence (float): 信頼度スコア
            - metadata (Dict): 処理メタデータ
    """
```

### 2. get_engine_stats() - 統計情報

```python
def get_engine_stats(self) -> Dict[str, Any]:
    """包括的なエンジン統計を取得
    
    Returns:
        Dict[str, Any]: 統計情報
            - queries_processed (int): 処理されたクエリ数
            - total_processing_time (float): 総処理時間
            - queries_normalized (int): 正規化されたクエリ数
            - average_retrieval_count (float): 平均検索結果数
            - average_response_time (float): 平均応答時間
            - retriever_stats (Dict): Retrieverの統計
            - synthesizer_stats (Dict): AnswerSynthesizerの統計
            - reranker_stats (Dict): Rerankerの統計（使用時）
            - normalizer_stats (Dict): Normalizerの統計（使用時）
            - corpus_state (Dict): コーパス状態情報
            - config (Dict): 設定情報
    """
```

### 3. Retriever管理メソッド

```python
def add_retriever(self, retriever: Retriever):
    """新しいRetrieverをエンジンに追加
    
    Args:
        retriever: 追加するRetriever
    """

def remove_retriever(self, index: int) -> bool:
    """インデックスでRetrieverを削除
    
    Args:
        index: 削除するRetrieverのインデックス
        
    Returns:
        bool: 削除成功時True、無効なインデックス時False
    """

def set_normalizer(self, normalizer: Optional[Normalizer]):
    """クエリ処理用のNormalizerを設定
    
    Args:
        normalizer: クエリ正規化のためのNormalizerインスタンス
    """
```

### 4. 管理メソッド

```python
def clear_cache(self):
    """キャッシュされたデータをクリア"""
```

## 設定クラス

### QueryEngineConfig

```python
@dataclass
class QueryEngineConfig:
    """QueryEngineの設定"""
    
    # クエリ処理設定
    enable_query_normalization: bool = True      # クエリ正規化を有効化
    auto_detect_corpus_state: bool = True        # コーパス状態の自動検出
    
    # コンポーネント設定
    retriever_top_k: int = 10                    # 各Retrieverの上位K件
    total_top_k: int = 20                        # 複数Retriever結合後の総上位K件
    reranker_top_k: int = 5                      # Rerankerの上位K件
    synthesizer_max_context: int = 2000          # Synthesizerの最大コンテキスト
    
    # マルチRetriever設定
    deduplicate_results: bool = True             # 重複文書の除去
    combine_scores: str = "max"                  # スコア結合方法: "max", "average", "sum"
    
    # パフォーマンス設定
    enable_caching: bool = True                  # キャッシングを有効化
    cache_ttl: int = 3600                        # キャッシュ生存時間（秒）
    
    # 出力設定
    include_sources: bool = True                 # ソース情報を含める
    include_confidence: bool = True              # 信頼度を含める
    include_processing_metadata: bool = True     # 処理メタデータを含める
```

## 結果クラス

### QueryResult

```python
@dataclass
class QueryResult:
    """最終的なクエリ結果とアンサー"""
    query: str                                   # 元のクエリ
    normalized_query: Optional[str] = None       # 正規化されたクエリ
    answer: str = ""                            # 生成されたアンサー
    sources: List[SearchResult] = None          # ソース文書
    confidence: float = 0.0                     # 信頼度スコア
    metadata: Dict[str, Any] = None             # メタデータ
```

### SearchResult

```python
@dataclass
class SearchResult:
    """検索結果"""
    document_id: str                            # 文書ID
    document: Document                          # 文書オブジェクト
    score: float                               # 関連度スコア
    metadata: Dict[str, Any] = None            # メタデータ
```

## 使用例

### 基本的な使用（単一Retriever）

```python
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval.simple_reranker import SimpleReranker
from refinire_rag.retrieval.simple_reader import SimpleAnswerSynthesizer

# コンポーネントの設定
vector_store = InMemoryVectorStore()  # VectorStoreはRetrieverインターフェースも実装
reranker = SimpleReranker()
synthesizer = SimpleAnswerSynthesizer()

# QueryEngineの作成（単一Retriever）
config = QueryEngineConfig(
    retriever_top_k=10,
    reranker_top_k=5,
    include_sources=True
)

query_engine = QueryEngine(
    corpus_name="my_ai_corpus",  # コーパス名を指定
    retrievers=vector_store,     # 単一のRetriever
    synthesizer=synthesizer,
    reranker=reranker,
    config=config
)

# クエリの実行
result = query_engine.query("機械学習とは何ですか？")

print(f"質問: {result.query}")
print(f"回答: {result.answer}")
print(f"信頼度: {result.confidence:.2f}")
print(f"ソース数: {len(result.sources)}")
print(f"処理時間: {result.metadata['processing_time']:.3f}秒")
```

### 複数Retrieverの使用

```python
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval.hybrid_retriever import HybridRetriever
from refinire_rag.keywordstore.tfidf_keyword_store import TFIDFKeywordStore

# 複数のRetrieverを作成
vector_retriever = InMemoryVectorStore()
keyword_retriever = TFIDFKeywordStore()
hybrid_retriever = HybridRetriever()

# マルチRetriever設定
config = QueryEngineConfig(
    retriever_top_k=10,        # 各Retrieverから10件ずつ
    total_top_k=15,            # 結合後は15件に絞る
    reranker_top_k=5,          # 最終的に5件
    deduplicate_results=True,  # 重複除去を有効化
    combine_scores="max"       # 重複時は最高スコアを採用
)

query_engine = QueryEngine(
    corpus_name="hybrid_ai_corpus",  # コーパス名を指定
    retrievers=[vector_retriever, keyword_retriever, hybrid_retriever],  # 複数のRetriever
    synthesizer=synthesizer,
    reranker=reranker,
    config=config
)

result = query_engine.query("機械学習とは何ですか？")
```

### 動的Retriever管理

```python
# 初期は単一のRetrieverで開始
query_engine = QueryEngine(
    corpus_name="dynamic_corpus",
    retrievers=vector_retriever,
    synthesizer=synthesizer
)

print(f"初期Retriever数: {len(query_engine.retrievers)}")  # 1

# 実行時にRetrieverを追加
query_engine.add_retriever(keyword_retriever)
query_engine.add_retriever(hybrid_retriever)

print(f"追加後Retriever数: {len(query_engine.retrievers)}")  # 3

# Retrieverを削除（インデックスで指定）
success = query_engine.remove_retriever(1)  # インデックス1のRetrieverを削除
print(f"削除成功: {success}")  # True
print(f"削除後Retriever数: {len(query_engine.retrievers)}")  # 2

# クエリ実行は現在のRetriever構成で動作
result = query_engine.query("クエリ実行")
```

### 高度な使用（コンテキストパラメータ付き）

```python
# より多くの結果を取得し、特定のメタデータでフィルタリング
context = {
    "top_k": 20,                    # より多くの候補を検索
    "rerank_top_k": 8,              # より多くの結果を再ランキング
    "metadata_filter": {            # メタデータフィルタ
        "category": "AI",
        "year": {"$gte": 2020}
    }
}

result = query_engine.query(
    "最新のディープラーニング手法について教えてください",
    context=context
)

# 詳細なソース情報の表示
for i, source in enumerate(result.sources):
    print(f"ソース {i+1}:")
    print(f"  文書ID: {source.document_id}")
    print(f"  スコア: {source.score:.3f}")
    print(f"  メタデータ: {source.metadata}")
    print(f"  内容（抜粋）: {source.document.content[:100]}...")
    print()
```

### 統計情報の取得

```python
# エンジンの統計情報を取得
stats = query_engine.get_engine_stats()

print("=== QueryEngine Statistics ===")
print(f"コーパス名: {stats['corpus_name']}")
print(f"処理済みクエリ数: {stats['queries_processed']}")
print(f"総処理時間: {stats['total_processing_time']:.2f}秒")
print(f"平均応答時間: {stats['average_response_time']:.3f}秒")
print(f"正規化済みクエリ数: {stats['queries_normalized']}")
print(f"平均検索結果数: {stats['average_retrieval_count']:.1f}")

print("\n=== Component Statistics ===")
print(f"Retriever: {stats['retriever_stats']}")
print(f"Reranker: {stats['reranker_stats']}")
print(f"Synthesizer: {stats['synthesizer_stats']}")

print(f"\n=== Corpus State ===")
print(f"正規化あり: {stats['corpus_state'].get('has_normalization', False)}")
print(f"文書あり: {stats['corpus_state'].get('has_documents', False)}")
```

### エラーハンドリング

```python
try:
    result = query_engine.query("複雑なクエリ")
    
    # エラーの確認
    if "error" in result.metadata:
        print(f"エラーが発生しました: {result.metadata['error']}")
        print(f"エラー時の回答: {result.answer}")
    else:
        print(f"正常な回答: {result.answer}")
        
except Exception as e:
    print(f"例外が発生しました: {e}")
```

## パフォーマンスとモニタリング

### 処理時間の最適化

1. **retriever_top_k**: 初期検索結果数（多すぎると遅い）
2. **reranker_top_k**: 再ランキング結果数（quality vs speed）
3. **synthesizer_max_context**: コンテキスト長（長すぎると遅い）

### 品質の最適化

1. **enable_query_normalization**: クエリ正規化で検索精度向上
2. **include_sources**: ソース情報で回答の信頼性向上
3. **reranker使用**: 関連性の高い結果の優先順位付け

### モニタリング指標

- **average_response_time**: 応答速度の監視
- **confidence**: 回答品質の指標
- **source_count**: 検索効果の測定
- **queries_normalized**: 正規化効果の確認

QueryEngineは柔軟で拡張可能な設計により、様々なRAGアプリケーションのニーズに対応できます。