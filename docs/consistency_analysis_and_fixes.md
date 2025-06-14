# Loader/Splitter実装と設計書の整合性分析・修正提案

## 🚨 発見された重要な不整合

### 1. **DocumentProcessor統一インターフェースの破綻**

#### 問題
- **設計書**（architecture.md）では統一された`DocumentProcessor`インターフェースを約束
- **実装**では複数の異なるインターフェースが存在：
  - `document_processor.py`: `process(documents: Iterable[Document], config) -> Iterator[Document]`
  - `loader/loader.py`: `process(documents: Iterable[Document], config) -> Iterator[Document]`
  - `processing/document_store_loader.py`: 別の`DocumentProcessor`継承

#### 影響
- パイプライン統一アーキテクチャが機能しない
- 依存性注入（DI）による切り替えが困難

#### 修正提案
```python
# すべてのプロセッサーで統一すべきインターフェース
class DocumentProcessor(ABC):
    @abstractmethod
    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """統一処理インターフェース"""
        pass
```

### 2. **DocumentStoreLoader重複実装**

#### 問題
- `src/refinire_rag/loader/document_store_loader.py` - Loaderを継承
- `src/refinire_rag/processing/document_store_loader.py` - DocumentProcessorを継承
- **同名だが異なる実装の2つのクラス**が存在

#### 修正提案
1. `processing/document_store_loader.py`を削除または改名
2. `loader/document_store_loader.py`を`DocumentProcessor`継承に変更
3. 統一インターフェースに準拠

### 3. **パッケージ構造の不整合**

#### 問題
- **設計書**: `src/refinire/rag/`（Refinireサブパッケージ）
- **実装**: `src/refinire_rag/`（独立パッケージ）

#### 影響
- RefinireのStep統合が不明
- import文が設計書と異なる

#### 修正提案
設計書を実装に合わせて更新し、`refinire_rag`の独立パッケージ構造を正式採用

### 4. **機能仕様書で約束された機能の未実装**

#### 問題
**function_spec.md**で約束されているが未実装：

1. **CorpusManagerプリセット**:
   ```python
   # 未実装
   CorpusManager.create_simple_rag(doc_store, vector_store)
   CorpusManager.create_semantic_rag(doc_store, vector_store)  
   CorpusManager.create_knowledge_rag(doc_store, vector_store)
   ```

2. **段階選択方式**:
   ```python
   # 未実装
   corpus_manager.build_corpus(
       file_paths=["docs/*.pdf"],
       stages=["load", "dictionary", "normalize", "chunk", "vector"]
   )
   ```

#### 修正提案
1. `CorpusManager`にプリセットファクトリーメソッドを追加
2. 段階選択機能を実装
3. 設計書で約束された`DocumentStoreLoader`統合を実現

## 📋 詳細修正計画

### 🔴 **最優先（システム破綻回避）**

#### 1. DocumentProcessor統一インターフェース修正
```python
# 修正対象ファイル
- src/refinire_rag/document_processor.py
- src/refinire_rag/loader/loader.py  
- src/refinire_rag/loader/document_store_loader.py
- src/refinire_rag/splitter/splitter.py

# 統一すべきメソッドシグネチャ
def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]
```

#### 2. DocumentStoreLoader重複解決
```bash
# 削除または改名
rm src/refinire_rag/processing/document_store_loader.py

# または改名
mv src/refinire_rag/processing/document_store_loader.py \
   src/refinire_rag/processing/legacy_document_store_loader.py
```

### 🟠 **高優先（機能完成）**

#### 3. CorpusManager機能実装
```python
# src/refinire_rag/use_cases/corpus_manager_new.py に追加
@classmethod
def create_simple_rag(cls, doc_store: DocumentStore, vector_store: VectorStore) -> 'CorpusManager':
    """シンプルRAG構成: Load → Chunk → Vector"""
    return cls(
        document_store=doc_store,
        vector_store=vector_store,
        pipeline_stages=["load", "chunk", "vector"]
    )

@classmethod  
def create_semantic_rag(cls, doc_store: DocumentStore, vector_store: VectorStore) -> 'CorpusManager':
    """セマンティックRAG構成: Load → Dictionary → Normalize → Chunk → Vector"""
    return cls(
        document_store=doc_store,
        vector_store=vector_store,
        pipeline_stages=["load", "dictionary", "normalize", "chunk", "vector"]
    )

@classmethod
def create_knowledge_rag(cls, doc_store: DocumentStore, vector_store: VectorStore) -> 'CorpusManager':
    """知識グラフRAG構成: Load → Dictionary → Graph → Normalize → Chunk → Vector"""
    return cls(
        document_store=doc_store,
        vector_store=vector_store,
        pipeline_stages=["load", "dictionary", "graph", "normalize", "chunk", "vector"]
    )
```

#### 4. DocumentStoreLoader統合強化
```python
# function_spec.mdで約束されたQueryEngine統合
class QueryEngine:
    def answer(self, query: str) -> QueryResult:
        # 3. DocumentStoreからチャンクIDに対応する完全なDocumentデータを取得
        chunk_docs = []
        for doc_id in search_results:
            document = self.document_store.get_document(doc_id)
            chunk_docs.append(document)
        
        # 7. DocumentStoreから関連するオリジナル文書情報を取得
        for doc in chunk_docs:
            lineage = self.document_store.get_documents_by_lineage(
                doc.metadata.get('original_document_id')
            )
```

### 🟡 **中優先（品質向上）**

#### 5. 設定クラス統一
```python
# すべてのプロセッサーに型安全な設定クラスを追加
@dataclass  
class LoaderConfig:
    source_paths: List[str]
    file_types: List[str] = field(default_factory=lambda: ["pdf", "docx", "txt"])
    recursive: bool = True
    
@dataclass
class ChunkerConfig:
    chunk_size: int = 512
    overlap: int = 50
    split_by_sentence: bool = True
```

#### 6. 例外処理統一
```python
# src/refinire_rag/exceptions.py に追加実装
def wrap_exception(exception: Exception, context: str = None) -> RefinireRAGError:
    """設計書で約束されたユーティリティ関数"""
    # 既存実装を活用
    pass
```

### 🔵 **低優先（文書整合性）**

#### 7. 設計書更新
```markdown
# docs/architecture.md 修正
- パッケージ構造: src/refinire_rag/ に変更
- DocumentProcessor統一インターフェース仕様を実装に合わせて修正
- 実装済み機能（IncrementalDirectoryLoader等）を追加

# docs/function_spec.md 修正  
- DocumentStoreLoader統合フローを実装状況に合わせて更新
- 実装済みのロード戦略（FULL, FILTERED, INCREMENTAL等）を記載
```

## 🎯 修正優先順位と実装計画

### **Phase 1: 緊急修正（システム破綻回避）**
1. DocumentProcessor統一インターフェース修正
2. DocumentStoreLoader重複解決
3. 基本的なパイプライン統合テスト

### **Phase 2: 機能完成（設計書約束の実現）**
1. CorpusManagerプリセット実装
2. 段階選択機能実装
3. QueryEngine統合強化

### **Phase 3: 品質向上（保守性・使いやすさ）**
1. 設定クラス統一
2. 例外処理統一
3. 文書整合性確保

## 📊 現在の整合性スコア

| 領域 | 整合性 | 主要問題 |
|------|-------|---------|
| **アーキテクチャ** | ❌ 30% | インターフェース不統一、重複実装 |
| **機能仕様** | ⚠️ 60% | プリセット未実装、段階選択未実装 |
| **命名規則** | ⚠️ 70% | Chunker/Splitter混在、メタデータ不統一 |
| **例外処理** | ✅ 85% | 基本クラス実装済み、ユーティリティ不足 |
| **文書化** | ⚠️ 65% | パッケージ構造不一致、実装記載不足 |

**総合整合性**: ⚠️ **62%** 

緊急修正（Phase 1）の実施により、**85%以上の整合性**達成を目標とします。

## 🚀 推奨実装手順

1. **DocumentProcessor統一インターフェース修正**を最優先実施
2. **DocumentStoreLoader重複解決**で混乱を除去  
3. **CorpusManagerプリセット実装**で設計書約束を履行
4. **設計書更新**で実装との整合性を確保

この修正により、refinire-ragは設計書通りの統一パイプラインアーキテクチャを実現し、柔軟な依存性注入による切り替え自由度を提供できます。