# models - データモデル定義

refinire-ragで使用される基本的なデータモデルの定義です。

## Document

ドキュメントを表す基本モデルです。

```python
from refinire_rag.models.document import Document

class Document(BaseModel):
    """ドキュメントモデル"""
    
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
```

### 属性

- `id` (str): ドキュメントの一意識別子
- `content` (str): ドキュメントの本文
- `metadata` (Dict[str, Any]): メタデータ（タイトル、著者、カテゴリなど）
- `created_at` (Optional[datetime]): 作成日時
- `updated_at` (Optional[datetime]): 更新日時

### 使用例

```python
# ドキュメントの作成
doc = Document(
    id="doc1",
    content="RAGは検索拡張生成技術です。",
    metadata={
        "title": "RAG概要",
        "category": "技術",
        "processing_stage": "original"
    }
)

# メタデータの更新
doc.metadata["tags"] = ["AI", "検索", "生成"]
```

## Chunk

チャンク（文書の断片）を表すモデルです。

```python
from refinire_rag.models.chunk import Chunk

class Chunk(BaseModel):
    """チャンクモデル"""
    
    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    start_char: Optional[int] = None
    end_char: Optional[int] = None
```

### 属性

- `id` (str): チャンクの一意識別子
- `document_id` (str): 元ドキュメントのID
- `content` (str): チャンクの本文
- `chunk_index` (int): ドキュメント内でのチャンク順序
- `metadata` (Dict[str, Any]): メタデータ
- `start_char` (Optional[int]): 元ドキュメントでの開始文字位置
- `end_char` (Optional[int]): 元ドキュメントでの終了文字位置

### 使用例

```python
# チャンクの作成
chunk = Chunk(
    id="chunk1",
    document_id="doc1",
    content="RAGは検索拡張生成技術です。",
    chunk_index=0,
    start_char=0,
    end_char=20,
    metadata={"overlap": True}
)
```

## EmbeddingResult

埋め込みベクトルの結果を表すモデルです。

```python
from refinire_rag.models.embedding import EmbeddingResult

class EmbeddingResult(BaseModel):
    """埋め込み結果モデル"""
    
    text: str
    vector: np.ndarray
    model_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### 属性

- `text` (str): 埋め込み対象のテキスト
- `vector` (np.ndarray): 埋め込みベクトル
- `model_name` (str): 使用したモデル名
- `metadata` (Dict[str, Any]): 追加メタデータ

### 使用例

```python
# 埋め込み結果の作成
result = EmbeddingResult(
    text="RAGは検索拡張生成技術です。",
    vector=np.array([0.1, 0.2, 0.3, ...]),
    model_name="tfidf",
    metadata={"dimension": 768}
)
```

## QueryResult

クエリ処理の結果を表すモデルです。

```python
from refinire_rag.models.query import QueryResult

class QueryResult(BaseModel):
    """クエリ結果モデル"""
    
    query: str
    answer: str
    sources: List[SearchResult] = Field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: Optional[float] = None
    normalized_query: Optional[str] = None
```

### 属性

- `query` (str): 元のクエリ
- `answer` (str): 生成された回答
- `sources` (List[SearchResult]): 参照した情報源
- `confidence` (float): 回答の信頼度（0.0-1.0）
- `metadata` (Dict[str, Any]): 処理メタデータ
- `processing_time` (Optional[float]): 処理時間（秒）
- `normalized_query` (Optional[str]): 正規化後のクエリ

### 使用例

```python
# クエリ結果の作成
result = QueryResult(
    query="RAGとは何ですか？",
    answer="RAGは検索拡張生成技術で、LLMと外部知識を組み合わせます。",
    sources=[search_result1, search_result2],
    confidence=0.85,
    processing_time=0.234,
    normalized_query="検索拡張生成とは何ですか？"
)
```

## SearchResult

検索結果を表すモデルです。

```python
from refinire_rag.models.search import SearchResult

class SearchResult(BaseModel):
    """検索結果モデル"""
    
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_id: Optional[str] = None
```

### 属性

- `document_id` (str): ドキュメントID
- `content` (str): 検索結果の内容
- `score` (float): 関連性スコア
- `metadata` (Dict[str, Any]): メタデータ
- `chunk_id` (Optional[str]): チャンクID（チャンク検索の場合）

## ProcessingStats

処理統計を表すモデルです。

```python
from refinire_rag.models.stats import ProcessingStats

class ProcessingStats(BaseModel):
    """処理統計モデル"""
    
    total_documents_created: int = 0
    total_chunks_created: int = 0
    total_processing_time: float = 0.0
    documents_by_stage: Dict[str, int] = Field(default_factory=dict)
    errors_encountered: int = 0
    pipeline_stages_executed: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### 属性

- `total_documents_created` (int): 作成されたドキュメント総数
- `total_chunks_created` (int): 作成されたチャンク総数
- `total_processing_time` (float): 総処理時間（秒）
- `documents_by_stage` (Dict[str, int]): ステージ別ドキュメント数
- `errors_encountered` (int): 発生したエラー数
- `pipeline_stages_executed` (int): 実行されたパイプラインステージ数
- `metadata` (Dict[str, Any]): 追加統計情報