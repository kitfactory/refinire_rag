# processing - ドキュメント処理パイプライン

ドキュメント処理のためのパイプライン構成要素です。

## DocumentProcessor

全ての処理コンポーネントの基底インターフェースです。

```python
from refinire_rag.processing.base import DocumentProcessor

class DocumentProcessor(ABC):
    """ドキュメント処理の基底インターフェース"""
    
    @abstractmethod
    def process(self, document: Document) -> List[Document]:
        """ドキュメントを処理する"""
        pass
```

### 実装クラス

- `Normalizer` - 正規化処理
- `Chunker` - チャンキング処理
- `DictionaryMaker` - 辞書作成
- `GraphBuilder` - グラフ構築
- `VectorStoreProcessor` - ベクトル化と保存

## Normalizer

辞書ベースの表現揺らぎ正規化を行います。

```python
from refinire_rag.processing.normalizer import Normalizer, NormalizerConfig

# 設定
config = NormalizerConfig(
    dictionary_file_path="dictionary.md",
    normalize_variations=True,
    expand_abbreviations=True,
    whole_word_only=False,  # 日本語対応
    case_sensitive=False
)

# 正規化器の作成
normalizer = Normalizer(config)

# 使用例
doc = Document(id="doc1", content="検索強化生成について")
normalized_docs = normalizer.process(doc)
# → "検索拡張生成について"
```

### NormalizerConfig

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| dictionary_file_path | str | 必須 | 辞書ファイルパス |
| normalize_variations | bool | True | 表現揺らぎを正規化 |
| expand_abbreviations | bool | True | 略語を展開 |
| whole_word_only | bool | False | 単語境界でのみマッチ |
| case_sensitive | bool | False | 大文字小文字を区別 |

## Chunker

ドキュメントを検索可能な単位に分割します。

```python
from refinire_rag.processing.chunker import Chunker, ChunkingConfig

# 設定
config = ChunkingConfig(
    chunk_size=500,
    overlap=50,
    split_by_sentence=True,
    min_chunk_size=100
)

# チャンカーの作成
chunker = Chunker(config)

# 使用例
doc = Document(id="doc1", content="長い文書...")
chunks = chunker.process(doc)
# → [Chunk1, Chunk2, ...]
```

### ChunkingConfig

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| chunk_size | int | 500 | チャンクサイズ（トークン数） |
| overlap | int | 50 | チャンク間のオーバーラップ |
| split_by_sentence | bool | True | 文境界で分割 |
| min_chunk_size | int | 100 | 最小チャンクサイズ |

## DictionaryMaker

ドキュメントから用語辞書を作成・更新します。

```python
from refinire_rag.processing.dictionary_maker import DictionaryMaker, DictionaryMakerConfig

# 設定
config = DictionaryMakerConfig(
    dictionary_file_path="dictionary.md",
    focus_on_technical_terms=True,
    extract_abbreviations=True,
    min_term_frequency=2
)

# 辞書作成器の作成
dict_maker = DictionaryMaker(config)

# 使用例
doc = Document(id="doc1", content="RAG（Retrieval-Augmented Generation）は...")
dict_maker.process(doc)
# → 辞書ファイルが更新される
```

### DictionaryMakerConfig

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| dictionary_file_path | str | 必須 | 辞書ファイルパス |
| focus_on_technical_terms | bool | True | 技術用語に焦点 |
| extract_abbreviations | bool | True | 略語を抽出 |
| min_term_frequency | int | 1 | 最小出現頻度 |

## DocumentPipeline

複数のProcessorを連結してパイプラインを構築します。

```python
from refinire_rag.processing.document_pipeline import DocumentPipeline

# パイプラインの構築
pipeline = DocumentPipeline([
    DictionaryMaker(dict_config),
    Normalizer(norm_config),
    Chunker(chunk_config)
])

# 使用例
doc = Document(id="doc1", content="検索強化生成について...")
results = pipeline.process(doc)
# → 辞書作成 → 正規化 → チャンキング
```

### メソッド

- `process(document: Document) -> List[Document]` - ドキュメントを処理
- `add_processor(processor: DocumentProcessor)` - プロセッサを追加
- `remove_processor(index: int)` - プロセッサを削除

## VectorStoreProcessor

ドキュメントをベクトル化して保存します。

```python
from refinire_rag.processing.vector_store_processor import VectorStoreProcessor

# 設定
config = VectorStoreProcessorConfig(
    embedder_type="tfidf",
    embedder_config={"min_df": 1, "max_df": 1.0},
    batch_size=100
)

# プロセッサの作成
processor = VectorStoreProcessor(vector_store, config)

# 使用例
doc = Document(id="doc1", content="RAGは検索拡張生成技術です")
processor.process(doc)
# → ベクトル化してストアに保存
```

## GraphBuilder

ドキュメントから知識グラフを構築します。

```python
from refinire_rag.processing.graph_builder import GraphBuilder, GraphBuilderConfig

# 設定
config = GraphBuilderConfig(
    graph_file_path="knowledge_graph.md",
    extract_entities=True,
    extract_relations=True,
    min_confidence=0.7
)

# グラフビルダーの作成
graph_builder = GraphBuilder(config)

# 使用例
doc = Document(id="doc1", content="RAGはLLMと外部知識を組み合わせます")
graph_builder.process(doc)
# → グラフファイルが更新される
```

## 使用パターン

### 1. Simple RAGパイプライン

```python
pipeline = DocumentPipeline([
    Chunker(chunk_config),
    VectorStoreProcessor(vector_store, vector_config)
])
```

### 2. Semantic RAGパイプライン

```python
pipeline = DocumentPipeline([
    DictionaryMaker(dict_config),
    Normalizer(norm_config),
    Chunker(chunk_config),
    VectorStoreProcessor(vector_store, vector_config)
])
```

### 3. Knowledge RAGパイプライン

```python
pipeline = DocumentPipeline([
    DictionaryMaker(dict_config),
    GraphBuilder(graph_config),
    Normalizer(norm_config),
    Chunker(chunk_config),
    VectorStoreProcessor(vector_store, vector_config)
])
```