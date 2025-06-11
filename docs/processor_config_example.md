# DocumentProcessor Configuration Pattern

このドキュメントでは、改善された`DocumentProcessor`の設定パターンについて説明します。

## 設計原則

各`DocumentProcessor`実装は：
1. 自身の設定クラスを定義する
2. `get_config_class()`メソッドで設定クラスを返す
3. 設定クラスは`DocumentProcessorConfig`を継承する

## 実装例

### 1. カスタム設定クラスの定義

```python
from dataclasses import dataclass
from refinire_rag.processing import DocumentProcessorConfig

@dataclass
class MyProcessorConfig(DocumentProcessorConfig):
    """カスタムプロセッサーの設定"""
    
    # 基底クラスから継承される共通フィールド：
    # - preserve_metadata: bool = True
    # - add_processing_info: bool = True
    # - fail_on_error: bool = False
    
    # このプロセッサー固有のフィールド
    my_parameter: str = "default_value"
    threshold: float = 0.8
    enable_feature: bool = True
    max_items: int = 100
```

### 2. プロセッサーの実装

```python
from typing import Type, List, Optional
from refinire_rag import Document, DocumentProcessor

class MyCustomProcessor(DocumentProcessor):
    """カスタム文書プロセッサー"""
    
    @classmethod
    def get_config_class(cls) -> Type[MyProcessorConfig]:
        """このプロセッサーの設定クラスを返す"""
        return MyProcessorConfig
    
    def process(self, document: Document, config: Optional[MyProcessorConfig] = None) -> List[Document]:
        """文書を処理"""
        # 提供された設定またはインスタンス設定を使用
        proc_config = config or self.config
        
        # 設定に基づいて処理を実行
        if proc_config.enable_feature:
            # 機能が有効な場合の処理
            pass
        
        # 処理結果の文書を作成
        result_doc = Document(
            id=f"{document.id}_processed",
            content=document.content,
            metadata={
                **document.metadata,
                "threshold_used": proc_config.threshold,
                "processing_stage": "my_custom_stage"
            }
        )
        
        return [result_doc]
```

### 3. 使用例

```python
# デフォルト設定で使用
processor = MyCustomProcessor()

# カスタム設定で使用
custom_config = MyProcessorConfig(
    threshold=0.95,
    enable_feature=False,
    max_items=50
)
processor = MyCustomProcessor(custom_config)

# 処理時に設定を上書き
doc = Document(id="test", content="...", metadata={...})
temp_config = MyProcessorConfig(threshold=0.7)
results = processor.process(doc, temp_config)
```

## 既存の実装例

### ChunkingConfig (Chunker)

```python
@dataclass
class ChunkingConfig(DocumentProcessorConfig):
    """文書チャンキングの設定"""
    chunk_size: int = 512
    overlap: int = 50
    split_by_sentence: bool = True
    preserve_formatting: bool = False
    min_chunk_size: int = 10
    max_chunk_size: int = 1024
```

### 将来の実装例

#### NormalizationConfig

```python
@dataclass
class NormalizationConfig(DocumentProcessorConfig):
    """テキスト正規化の設定"""
    use_dictionary: bool = True
    expand_abbreviations: bool = True
    add_semantic_tags: bool = True
    preserve_formatting: bool = False
    language: str = "ja"
```

#### DictionaryConfig

```python
@dataclass
class DictionaryConfig(DocumentProcessorConfig):
    """辞書生成の設定"""
    min_term_frequency: int = 2
    extract_abbreviations: bool = True
    extract_synonyms: bool = True
    language: str = "ja"
```

#### GraphConfig

```python
@dataclass
class GraphConfig(DocumentProcessorConfig):
    """グラフ構築の設定"""
    extract_entities: bool = True
    extract_relationships: bool = True
    min_entity_confidence: float = 0.7
    language: str = "ja"
    graph_format: str = "networkx"
```

## 利点

1. **型安全性**: 各プロセッサーが自身の設定型を定義
2. **自己文書化**: 設定クラスがプロセッサーの要求を明確に示す
3. **検証**: `validate_config()`により誤った設定型の使用を防ぐ
4. **拡張性**: 新しいプロセッサーは独自の設定を簡単に定義可能
5. **発見可能性**: `get_config_class()`により動的に設定要件を確認可能

## パイプラインでの使用

```python
pipeline = DocumentPipeline([
    TextNormalizer(NormalizationConfig(lowercase=True)),
    DictionaryMaker(DictionaryConfig(min_term_frequency=3)),
    Chunker(ChunkingConfig(chunk_size=256, overlap=32))
])

# 各プロセッサーは独自の設定型を持ちながら協調動作
results = pipeline.process_document(document)
```

この設計により、各`DocumentProcessor`が必要とする設定を明確に定義し、型安全で拡張可能なシステムを実現します。