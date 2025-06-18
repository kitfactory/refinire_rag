# Plugin Development Guide

## 概要

refinire-ragは環境変数ベースのプラグインアーキテクチャを採用しており、Retriever、VectorStore、DocumentStoreなどのコンポーネントを独立したパッケージとして開発・配布できます。現在はChroma・BM25sプラグインが利用可能で、将来的には様々なコンポーネントのプラグイン化が予定されています。本ガイドでは、プラグインの開発方法、oneenvとの統合、entry pointsを使った自動発見機能について説明します。

## プラグインアーキテクチャの設計思想

### 統合レジストリシステム

refinire-ragは内蔵コンポーネントと外部プラグインを統一的に管理するレジストリシステムを採用しています：

- **内蔵コンポーネント**：refinire-ragに標準で組み込まれているコンポーネント（エントリポイント不要）
- **外部プラグイン**：独立したパッケージとして配布されるプラグイン（エントリポイント使用）
- **統一アクセス**：どちらも同じAPI（`PluginRegistry`）で利用可能

### 環境変数ベース設定

内蔵・外部問わず、同じ環境変数命名規則に従います：

```bash
# メインの設定（内蔵＋外部プラグイン混在可能）
REFINIRE_RAG_RETRIEVERS=inmemory_vector,chroma,bm25s  # 内蔵,外部,外部
REFINIRE_RAG_DOCUMENT_STORES=sqlite                   # 内蔵

# プラグイン固有の設定（外部プラグインのみ）
REFINIRE_RAG_CHROMA_HOST=localhost
REFINIRE_RAG_CHROMA_PORT=8000
REFINIRE_RAG_BM25S_INDEX_PATH=./bm25s_index
```

### 内蔵コンポーネントの自動登録

システム起動時に内蔵コンポーネントが自動的にレジストリに登録されます：

```python
# 利用可能なコンポーネント確認（内蔵＋外部）
from refinire_rag.registry import PluginRegistry

vector_stores = PluginRegistry.list_available_plugins('vector_stores')
# 結果例: ['inmemory_vector', 'pickle_vector', 'chroma']  # 内蔵,内蔵,外部

# 内蔵かどうかの判定
is_builtin = PluginRegistry.is_builtin('vector_stores', 'inmemory_vector')  # True
```

### oneenvとの統合

各プラグインは独自のoneenvテンプレートを提供し、環境変数の管理を統一的に行います。

### Entry Pointsによる外部プラグイン発見

外部プラグインは`pyproject.toml`のentry pointsを通じて自動的に発見され、内蔵コンポーネントと統合されます。

## プラグインの種類

refinire-ragでは以下のコンポーネントがプラグイン化可能です：

### 検索・取得系

| プラグインタイプ | Entry Points グループ | 責務 | 実装例 |
|---|---|---|---|
| **Retriever** | `refinire_rag.retrievers` | 文書検索とランキング | SimpleRetriever, HybridRetriever |
| **VectorStore** | `refinire_rag.vector_stores` | ベクトル保存と検索 | Chroma※, InMemoryVectorStore |
| **KeywordSearch** | `refinire_rag.keyword_stores` | キーワードベース検索 | BM25s※, TFIDFKeywordStore |
| **Reranker** | `refinire_rag.rerankers` | 検索結果の再ランキング | SimpleReranker（開発予定） |
| **AnswerSynthesizer** | `refinire_rag.synthesizers` | 回答生成・統合 | AnswerSynthesizer（開発予定） |

### 文書処理系

| プラグインタイプ | Entry Points グループ | 責務 | 実装例 |
|---|---|---|---|
| **DocumentProcessor** | `refinire_rag.processors` | 文書処理パイプライン | Normalizer, Chunker, DictionaryMaker |
| **Loader** | `refinire_rag.loaders` | 文書ロード・変換 | TextLoader, CSVLoader, HTMLLoader |
| **Splitter** | `refinire_rag.splitters` | 文書分割戦略 | RecursiveCharacterSplitter, CodeSplitter |
| **Filter** | `refinire_rag.filters` | ファイルフィルタリング | ExtensionFilter, SizeFilter, DateFilter |
| **Metadata** | `refinire_rag.metadata` | メタデータ付与 | FileInfoMetadata, PathMapMetadata |

### 埋め込み・ベクトル化系

| プラグインタイプ | Entry Points グループ | 責務 | 実装例 |
|---|---|---|---|
| **Embedder** | `refinire_rag.embedders` | テキストのベクトル化 | OpenAIEmbedder, TFIDFEmbedder |
| **VectorIndexer** | `refinire_rag.indexers` | ベクトルインデックス作成 | （開発予定） |

### 評価・品質管理系

| プラグインタイプ | Entry Points グループ | 責務 | 実装例 |
|---|---|---|---|
| **Evaluator** | `refinire_rag.evaluators` | 評価指標計算 | BLEUEvaluator, ROUGEEvaluator, LLMJudgeEvaluator |
| **ContradictionDetector** | `refinire_rag.contradiction_detectors` | 矛盾検出 | ContradictionDetector |
| **TestSuite** | `refinire_rag.test_suites` | テスト実行 | TestSuite |
| **InsightReporter** | `refinire_rag.reporters` | レポート生成 | InsightReporter |

### ストレージ系

| プラグインタイプ | Entry Points グループ | 責務 | 実装例 |
|---|---|---|---|
| **DocumentStore** | `refinire_rag.document_stores` | 文書メタデータ保存 | SQLiteStore |
| **EvaluationStore** | `refinire_rag.evaluation_stores` | 評価結果保存 | SQLiteEvaluationStore |
| **CorpusStore** | `refinire_rag.corpus_stores` | コーパス管理 | SQLiteCorpusStore |

### ナレッジ処理系

| プラグインタイプ | Entry Points グループ | 責務 | 実装例 |
|---|---|---|---|
| **DictionaryMaker** | `refinire_rag.dictionary_makers` | 専門用語辞書作成 | DictionaryMaker |
| **GraphBuilder** | `refinire_rag.graph_builders` | 知識グラフ構築 | GraphBuilder |
| **Normalizer** | `refinire_rag.normalizers` | テキスト正規化 | Normalizer |

**注：** ※マークがついているものは外部プラグインとして開発予定・可能なコンポーネントです。

## プラグイン開発手順

### 1. プロジェクト構造

```
refinire-rag-chroma/
├── src/
│   └── refinire_rag_chroma/
│       ├── __init__.py
│       ├── retriever.py          # メインのRetrieverクラス
│       ├── vector_store.py       # ChromaVectorStoreクラス
│       ├── config.py             # 設定クラス
│       └── env_template.py       # oneenvテンプレート
├── tests/
├── pyproject.toml               # entry points設定
├── README.md
└── .env.template                # 環境変数テンプレート
```

### 2. pyproject.toml設定

#### Retriever/VectorStoreプラグインの例（Chroma）

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "refinire-rag-chroma"
version = "0.1.0"
description = "Chroma integration plugin for refinire-rag"
dependencies = [
    "refinire-rag",
    "chromadb>=0.4.0",
    "oneenv>=0.3.0"
]

[project.entry-points."refinire_rag.retrievers"]
chroma = "refinire_rag_chroma:ChromaRetriever"

[project.entry-points."refinire_rag.vector_stores"]
chroma = "refinire_rag_chroma:ChromaVectorStore"

[project.entry-points."refinire_rag.oneenv_templates"]
chroma = "refinire_rag_chroma.env_template:chroma_env_template"
```

#### 評価プラグインの例（BERT Score）

```toml
[project]
name = "refinire-rag-bertscore"
version = "0.1.0"
description = "BERTScore evaluation plugin for refinire-rag"
dependencies = [
    "refinire-rag",
    "bert-score>=0.3.12",
    "oneenv>=0.3.0"
]

[project.entry-points."refinire_rag.evaluators"]
bertscore = "refinire_rag_bertscore:BERTScoreEvaluator"

[project.entry-points."refinire_rag.oneenv_templates"]
bertscore = "refinire_rag_bertscore.env_template:bertscore_env_template"
```

#### Rerankerプラグインの例（Sentence-BERT）

```toml
[project]
name = "refinire-rag-sbert-reranker"
version = "0.1.0"
description = "Sentence-BERT reranking plugin for refinire-rag"
dependencies = [
    "refinire-rag",
    "sentence-transformers>=2.2.0",
    "oneenv>=0.3.0"
]

[project.entry-points."refinire_rag.rerankers"]
sbert = "refinire_rag_sbert_reranker:SentenceBERTReranker"

[project.entry-points."refinire_rag.oneenv_templates"]
sbert_reranker = "refinire_rag_sbert_reranker.env_template:sbert_reranker_env_template"
```

### 3. oneenvテンプレート実装

```python
# src/refinire_rag_chroma/env_template.py
from oneenv import EnvTemplate, EnvVarConfig
from oneenv.templates import template_enhanced

@template_enhanced
def chroma_env_template() -> EnvTemplate:
    """Chroma plugin environment variables template
    
    Chromaプラグイン用環境変数テンプレート
    """
    return EnvTemplate(
        name="refinire-rag-chroma",
        description="Environment variables for Chroma integration",
        variables=[
            # Critical variables
            EnvVarConfig(
                name="REFINIRE_RAG_CHROMA_HOST",
                description="Chroma server host",
                default="localhost",
                importance="Critical",
                group="Connection"
            ),
            
            # Important variables
            EnvVarConfig(
                name="REFINIRE_RAG_CHROMA_PORT",
                description="Chroma server port",
                default="8000",
                importance="Important",
                group="Connection"
            ),
            EnvVarConfig(
                name="REFINIRE_RAG_CHROMA_COLLECTION",
                description="Default collection name",
                default="refinire_rag",
                importance="Important",
                group="Storage"
            ),
            EnvVarConfig(
                name="REFINIRE_RAG_CHROMA_DISTANCE_METRIC",
                description="Distance metric for similarity search",
                default="cosine",
                importance="Important",
                group="Search"
            ),
            
            # Optional variables
            EnvVarConfig(
                name="REFINIRE_RAG_CHROMA_BATCH_SIZE",
                description="Batch size for bulk operations",
                default="100",
                importance="Optional",
                group="Performance"
            )
        ],
        groups={
            "Connection": "Chroma server connection settings",
            "Storage": "Data storage configuration",
            "Search": "Search and retrieval settings",
            "Performance": "Performance optimization settings"
        }
    )
```

### 4. 設定クラス実装

```python
# src/refinire_rag_chroma/config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ChromaConfig:
    """Configuration class for Chroma plugin
    
    Chromaプラグインの設定クラス
    """
    
    @property
    def host(self) -> str:
        return os.getenv("REFINIRE_RAG_CHROMA_HOST", "localhost")
    
    @property
    def port(self) -> int:
        return int(os.getenv("REFINIRE_RAG_CHROMA_PORT", "8000"))
    
    @property
    def collection_name(self) -> str:
        return os.getenv("REFINIRE_RAG_CHROMA_COLLECTION", "refinire_rag")
    
    @property
    def distance_metric(self) -> str:
        return os.getenv("REFINIRE_RAG_CHROMA_DISTANCE_METRIC", "cosine")
    
    @property
    def batch_size(self) -> int:
        return int(os.getenv("REFINIRE_RAG_CHROMA_BATCH_SIZE", "100"))
    
    @property
    def connection_url(self) -> str:
        return f"http://{self.host}:{self.port}"
```

### 5. Retrieverクラス実装

```python
# src/refinire_rag_chroma/retriever.py
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from refinire_rag.retrieval.base import Retriever
from refinire_rag.models.document import Document
from refinire_rag.models.search_result import SearchResult
from .config import ChromaConfig

logger = logging.getLogger(__name__)

class ChromaRetriever(Retriever):
    """Chroma-based document retriever
    
    Chromaベースの文書検索器
    """
    
    def __init__(self, config: Optional[ChromaConfig] = None):
        """Initialize ChromaRetriever
        
        ChromaRetrieverを初期化
        
        Args:
            config: Optional configuration object. If None, uses environment variables
                   オプションの設定オブジェクト。Noneの場合は環境変数を使用
        """
        self.config = config or ChromaConfig()
        self._client = None
        self._collection = None
        
        logger.info(f"Initialized ChromaRetriever with host: {self.config.host}:{self.config.port}")
    
    @property
    def client(self):
        """Lazy initialization of Chroma client
        
        Chromaクライアントの遅延初期化
        """
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=self.config.host,
                port=self.config.port,
                settings=Settings()
            )
        return self._client
    
    @property
    def collection(self):
        """Get or create collection
        
        コレクションの取得または作成
        """
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(
                    name=self.config.collection_name
                )
            except Exception:
                # Collection doesn't exist, create it
                self._collection = self.client.create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": self.config.distance_metric}
                )
                logger.info(f"Created new collection: {self.config.collection_name}")
        return self._collection
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for documents using Chroma
        
        Chromaを使用して文書を検索
        
        Args:
            query: Search query string
                  検索クエリ文字列
            top_k: Number of results to return
                  返却する結果数
            filters: Optional metadata filters
                    オプションのメタデータフィルター
        
        Returns:
            List of search results
            検索結果のリスト
        """
        try:
            # Prepare where clause for filtering
            where_clause = None
            if filters:
                where_clause = self._convert_filters_to_where_clause(filters)
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause
            )
            
            # Convert to SearchResult objects
            search_results = []
            for i in range(len(results['ids'][0])):
                search_result = SearchResult(
                    document_id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                    metadata=results['metadatas'][0][i] or {}
                )
                search_results.append(search_result)
            
            logger.info(f"Retrieved {len(search_results)} documents for query: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching in Chroma: {e}")
            return []
    
    def _convert_filters_to_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert refinire-rag filters to Chroma where clause
        
        refinire-ragフィルターをChroma where句に変換
        """
        # Simple implementation - can be extended for complex filtering
        return filters
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to Chroma collection
        
        Chromaコレクションに文書を追加
        """
        try:
            # Prepare data for batch insertion
            ids = [doc.id for doc in documents]
            documents_text = [doc.content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Add to collection in batches
            batch_size = self.config.batch_size
            for i in range(0, len(documents), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_docs = documents_text[i:i + batch_size]
                batch_meta = metadatas[i:i + batch_size]
                
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta
                )
            
            logger.info(f"Added {len(documents)} documents to Chroma collection")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to Chroma: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all documents from collection
        
        コレクションからすべての文書を削除
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.config.collection_name)
            self._collection = None  # Reset collection reference
            
            logger.info(f"Cleared Chroma collection: {self.config.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing Chroma collection: {e}")
            return False
```

### 6. パッケージ初期化

```python
# src/refinire_rag_chroma/__init__.py
"""
Chroma integration plugin for refinire-rag

This plugin provides Chroma database integration for vector storage and retrieval.

Environment Variables:
- REFINIRE_RAG_CHROMA_HOST: Chroma server host (default: localhost)
- REFINIRE_RAG_CHROMA_PORT: Chroma server port (default: 8000)
- REFINIRE_RAG_CHROMA_COLLECTION: Collection name (default: refinire_rag)
- REFINIRE_RAG_CHROMA_DISTANCE_METRIC: Distance metric (default: cosine)
- REFINIRE_RAG_CHROMA_BATCH_SIZE: Batch size for operations (default: 100)
"""

from .retriever import ChromaRetriever
from .vector_store import ChromaVectorStore
from .config import ChromaConfig

__all__ = ['ChromaRetriever', 'ChromaVectorStore', 'ChromaConfig']
__version__ = '0.1.0'
```

## 評価プラグインの実装例

### BERTScoreEvaluatorプラグイン

```python
# src/refinire_rag_bertscore/evaluator.py
import logging
from typing import List, Dict, Any, Optional
from bert_score import score

from refinire_rag.evaluation.base_evaluator import ReferenceBasedEvaluator
from refinire_rag.models.evaluation_result import EvaluationResult
from .config import BERTScoreConfig

logger = logging.getLogger(__name__)

class BERTScoreEvaluator(ReferenceBasedEvaluator):
    """BERTScore-based evaluation for semantic similarity
    
    BERTScoreによるセマンティック類似度評価
    """
    
    def __init__(self, config: Optional[BERTScoreConfig] = None):
        """Initialize BERTScoreEvaluator
        
        BERTScoreEvaluatorを初期化
        """
        self.config = config or BERTScoreConfig()
        logger.info(f"Initialized BERTScoreEvaluator with model: {self.config.model_name}")
    
    def evaluate(self, 
                 predictions: List[str], 
                 references: List[str],
                 **kwargs) -> EvaluationResult:
        """Evaluate using BERTScore
        
        BERTScoreを使用して評価
        """
        try:
            # Calculate BERTScore
            P, R, F1 = score(
                predictions, 
                references,
                model_type=self.config.model_name,
                lang=self.config.language,
                verbose=self.config.verbose,
                device=self.config.device
            )
            
            # Calculate metrics
            precision = P.mean().item()
            recall = R.mean().item()
            f1_score = F1.mean().item()
            
            # Individual scores
            individual_scores = [
                {
                    "precision": p.item(),
                    "recall": r.item(), 
                    "f1": f.item()
                }
                for p, r, f in zip(P, R, F1)
            ]
            
            return EvaluationResult(
                metric_name="bertscore",
                overall_score=f1_score,
                component_scores={
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score
                },
                individual_scores=individual_scores,
                metadata={
                    "model_name": self.config.model_name,
                    "language": self.config.language,
                    "num_predictions": len(predictions)
                }
            )
            
        except Exception as e:
            logger.error(f"BERTScore evaluation failed: {e}")
            return EvaluationResult(
                metric_name="bertscore",
                overall_score=0.0,
                error=str(e)
            )

# src/refinire_rag_bertscore/config.py
import os
from dataclasses import dataclass

@dataclass
class BERTScoreConfig:
    """Configuration for BERTScore evaluator"""
    
    @property
    def model_name(self) -> str:
        return os.getenv("REFINIRE_RAG_BERTSCORE_MODEL", "microsoft/deberta-xlarge-mnli")
    
    @property
    def language(self) -> str:
        return os.getenv("REFINIRE_RAG_BERTSCORE_LANGUAGE", "en")
    
    @property
    def device(self) -> str:
        return os.getenv("REFINIRE_RAG_BERTSCORE_DEVICE", "cuda")
    
    @property
    def verbose(self) -> bool:
        return os.getenv("REFINIRE_RAG_BERTSCORE_VERBOSE", "false").lower() == "true"
```

### Rerankerプラグインの実装例

```python
# src/refinire_rag_sbert_reranker/reranker.py
import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

from refinire_rag.retrieval.base import Reranker
from refinire_rag.models.search_result import SearchResult
from .config import SentenceBERTRerankerConfig

logger = logging.getLogger(__name__)

class SentenceBERTReranker(Reranker):
    """Sentence-BERT based reranking for search results
    
    Sentence-BERTベースの検索結果リランキング
    """
    
    def __init__(self, config: Optional[SentenceBERTRerankerConfig] = None):
        """Initialize SentenceBERTReranker
        
        SentenceBERTRerankerを初期化
        """
        self.config = config or SentenceBERTRerankerConfig()
        self._model = None
        logger.info(f"Initialized SentenceBERTReranker with model: {self.config.model_name}")
    
    @property
    def model(self):
        """Lazy loading of the model"""
        if self._model is None:
            if self.config.use_cross_encoder:
                self._model = CrossEncoder(
                    self.config.model_name,
                    device=self.config.device
                )
            else:
                self._model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device
                )
        return self._model
    
    def rerank(self, 
               query: str, 
               results: List[SearchResult],
               top_k: Optional[int] = None) -> List[SearchResult]:
        """Rerank search results using Sentence-BERT
        
        Sentence-BERTを使用して検索結果をリランキング
        """
        if not results:
            return results
        
        try:
            if self.config.use_cross_encoder:
                # Cross-encoder for direct relevance scoring
                pairs = [(query, result.content) for result in results]
                scores = self.model.predict(pairs)
                
                # Update scores and sort
                for result, score in zip(results, scores):
                    result.score = float(score)
                    result.metadata["rerank_score"] = float(score)
                    result.metadata["reranker"] = "sbert_cross_encoder"
            
            else:
                # Bi-encoder for similarity scoring
                query_embedding = self.model.encode([query])
                doc_embeddings = self.model.encode([result.content for result in results])
                
                # Calculate cosine similarity
                similarities = torch.cosine_similarity(
                    torch.tensor(query_embedding),
                    torch.tensor(doc_embeddings)
                )
                
                # Update scores
                for result, similarity in zip(results, similarities):
                    result.score = float(similarity)
                    result.metadata["rerank_score"] = float(similarity)
                    result.metadata["reranker"] = "sbert_bi_encoder"
            
            # Sort by new scores
            reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
            
            # Apply top_k limit if specified
            if top_k is not None:
                reranked_results = reranked_results[:top_k]
            
            logger.info(f"Reranked {len(results)} results, returning top {len(reranked_results)}")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results  # Return original results on error

# src/refinire_rag_sbert_reranker/config.py
import os
from dataclasses import dataclass

@dataclass 
class SentenceBERTRerankerConfig:
    """Configuration for Sentence-BERT reranker"""
    
    @property
    def model_name(self) -> str:
        return os.getenv("REFINIRE_RAG_SBERT_MODEL", "ms-marco-MiniLM-L-6-v2")
    
    @property
    def use_cross_encoder(self) -> bool:
        return os.getenv("REFINIRE_RAG_SBERT_USE_CROSS_ENCODER", "true").lower() == "true"
    
    @property
    def device(self) -> str:
        return os.getenv("REFINIRE_RAG_SBERT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def batch_size(self) -> int:
        return int(os.getenv("REFINIRE_RAG_SBERT_BATCH_SIZE", "32"))
```

## プラグイン発見とロードシステム

### 統合レジストリシステム

```python
# refinire-rag core: src/refinire_rag/registry/plugin_registry.py
import importlib.metadata
import logging
from typing import Dict, Type, List, Optional, Any

logger = logging.getLogger(__name__)

class PluginRegistry:
    """Universal registry for all plugin types
    
    すべてのプラグインタイプ用の統合レジストリ
    """
    
    _registries: Dict[str, Dict[str, Type]] = {}
    _discovered_groups: set = set()
    
    # Plugin group definitions
    PLUGIN_GROUPS = {
        'retrievers': 'refinire_rag.retrievers',
        'vector_stores': 'refinire_rag.vector_stores', 
        'keyword_stores': 'refinire_rag.keyword_stores',
        'rerankers': 'refinire_rag.rerankers',
        'synthesizers': 'refinire_rag.synthesizers',
        'evaluators': 'refinire_rag.evaluators',
        'embedders': 'refinire_rag.embedders',
        'loaders': 'refinire_rag.loaders',
        'processors': 'refinire_rag.processors',
        'splitters': 'refinire_rag.splitters',
        'filters': 'refinire_rag.filters',
        'metadata': 'refinire_rag.metadata',
        'document_stores': 'refinire_rag.document_stores',
        'evaluation_stores': 'refinire_rag.evaluation_stores',
        'contradiction_detectors': 'refinire_rag.contradiction_detectors',
        'test_suites': 'refinire_rag.test_suites',
        'reporters': 'refinire_rag.reporters',
        'oneenv_templates': 'refinire_rag.oneenv_templates'
    }
    
    @classmethod
    def discover_plugins(cls, group_name: str = None) -> None:
        """Discover plugins from entry points
        
        entry pointsからプラグインを発見
        """
        groups_to_discover = [group_name] if group_name else cls.PLUGIN_GROUPS.keys()
        
        for group in groups_to_discover:
            if group in cls._discovered_groups:
                continue
                
            entry_point_group = cls.PLUGIN_GROUPS.get(group)
            if not entry_point_group:
                continue
                
            cls._registries[group] = {}
            
            try:
                for entry_point in importlib.metadata.entry_points(group=entry_point_group):
                    try:
                        plugin_class = entry_point.load()
                        cls._registries[group][entry_point.name] = plugin_class
                        logger.info(f"Discovered {group} plugin: {entry_point.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load {group} plugin {entry_point.name}: {e}")
                
                cls._discovered_groups.add(group)
                logger.info(f"Discovered {len(cls._registries[group])} {group} plugins")
                
            except Exception as e:
                logger.error(f"Error discovering {group} plugins: {e}")
    
    @classmethod
    def get_plugin_class(cls, group: str, name: str) -> Optional[Type]:
        """Get plugin class by group and name
        
        グループと名前でプラグインクラスを取得
        """
        cls.discover_plugins(group)
        return cls._registries.get(group, {}).get(name)
    
    @classmethod
    def list_available_plugins(cls, group: str) -> List[str]:
        """List all available plugin names for a group
        
        グループの利用可能なプラグイン名のリストを取得
        """
        cls.discover_plugins(group)
        return list(cls._registries.get(group, {}).keys())
    
    @classmethod
    def create_plugin(cls, group: str, name: str, **kwargs) -> Any:
        """Create plugin instance by group and name
        
        グループと名前でプラグインインスタンスを作成
        """
        plugin_class = cls.get_plugin_class(group, name)
        if plugin_class is None:
            available = cls.list_available_plugins(group)
            raise ValueError(f"Unknown {group} plugin: {name}. Available: {available}")
        
        return plugin_class(**kwargs)
    
    @classmethod
    def get_all_plugins_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive information about all available plugins
        
        すべての利用可能なプラグインの包括的な情報を取得
        """
        # Discover all plugin groups
        for group in cls.PLUGIN_GROUPS.keys():
            cls.discover_plugins(group)
        
        info = {}
        for group, plugins in cls._registries.items():
            info[group] = {}
            for name, plugin_class in plugins.items():
                info[group][name] = {
                    "class": plugin_class.__name__,
                    "module": plugin_class.__module__,
                    "description": getattr(plugin_class, "__doc__", "No description"),
                    "version": getattr(plugin_class, "__version__", "Unknown")
                }
        
        return info


### 環境変数ベースファクトリー（統合版）

```python
# src/refinire_rag/factories/plugin_factory.py
import os
import logging
from typing import List, Dict, Any, Optional

from ..registry.plugin_registry import PluginRegistry

logger = logging.getLogger(__name__)

class PluginFactory:
    """Universal factory for creating plugins from environment variables
    
    環境変数からプラグインを作成する統合ファクトリー
    """
    
    @staticmethod
    def create_plugins_from_env(group: str, env_var: str) -> List[Any]:
        """Create plugins based on environment variable
        
        環境変数に基づいてプラグインを作成
        
        Args:
            group: Plugin group (e.g., 'retrievers', 'evaluators')
            env_var: Environment variable name (e.g., 'REFINIRE_RAG_RETRIEVERS')
        """
        plugins_config = os.getenv(env_var, "").strip()
        if not plugins_config:
            logger.warning(f"{env_var} not set, no {group} will be created")
            return []
        
        plugin_names = [name.strip() for name in plugins_config.split(",")]
        plugins = []
        
        for name in plugin_names:
            if not name:
                continue
            
            try:
                plugin = PluginRegistry.create_plugin(group, name)
                plugins.append(plugin)
                logger.info(f"Created {group} plugin: {name}")
                
            except Exception as e:
                logger.error(f"Failed to create {group} plugin '{name}': {e}")
        
        return plugins
    
    @staticmethod
    def create_retrievers_from_env() -> List[Any]:
        """Create retrievers from REFINIRE_RAG_RETRIEVERS"""
        return PluginFactory.create_plugins_from_env('retrievers', 'REFINIRE_RAG_RETRIEVERS')
    
    @staticmethod
    def create_evaluators_from_env() -> List[Any]:
        """Create evaluators from REFINIRE_RAG_EVALUATORS"""
        return PluginFactory.create_plugins_from_env('evaluators', 'REFINIRE_RAG_EVALUATORS')
    
    @staticmethod
    def create_rerankers_from_env() -> List[Any]:
        """Create rerankers from REFINIRE_RAG_RERANKERS"""
        rerankers = PluginFactory.create_plugins_from_env('rerankers', 'REFINIRE_RAG_RERANKERS')
        return rerankers[0] if rerankers else None  # Usually single reranker
    
    @staticmethod
    def get_available_plugins(group: str) -> Dict[str, Any]:
        """Get information about available plugins for a group"""
        return PluginRegistry.get_all_plugins_info().get(group, {})

class RetrieverRegistry:
    """Registry for automatically discovering and loading retriever plugins
    
    Retrieverプラグインの自動発見とロード用レジストリ
    """
    
    _registry: Dict[str, Type] = {}
    _discovered = False
    
    @classmethod
    def discover_retrievers(cls) -> None:
        """Discover retrievers from entry points
        
        entry pointsからretrieverを発見
        """
        if cls._discovered:
            return
        
        try:
            # Discover retrievers from entry points
            for entry_point in importlib.metadata.entry_points(group='refinire_rag.retrievers'):
                try:
                    retriever_class = entry_point.load()
                    cls._registry[entry_point.name] = retriever_class
                    logger.info(f"Discovered retriever plugin: {entry_point.name}")
                except Exception as e:
                    logger.warning(f"Failed to load retriever plugin {entry_point.name}: {e}")
            
            cls._discovered = True
            logger.info(f"Discovered {len(cls._registry)} retriever plugins")
            
        except Exception as e:
            logger.error(f"Error discovering retriever plugins: {e}")
    
    @classmethod
    def get_retriever_class(cls, name: str) -> Optional[Type]:
        """Get retriever class by name
        
        名前でretrieverクラスを取得
        """
        cls.discover_retrievers()
        return cls._registry.get(name)
    
    @classmethod
    def list_available_retrievers(cls) -> List[str]:
        """List all available retriever names
        
        利用可能なretriever名のリストを取得
        """
        cls.discover_retrievers()
        return list(cls._registry.keys())
    
    @classmethod
    def create_retriever(cls, name: str, **kwargs):
        """Create retriever instance by name
        
        名前でretrieverインスタンスを作成
        """
        retriever_class = cls.get_retriever_class(name)
        if retriever_class is None:
            raise ValueError(f"Unknown retriever: {name}. Available: {cls.list_available_retrievers()}")
        
        return retriever_class(**kwargs)
```

### 環境変数ベースファクトリー

```python
# src/refinire_rag/factories/retriever_factory.py
import os
import logging
from typing import List, Optional, Dict, Any

from ..registry.retriever_registry import RetrieverRegistry

logger = logging.getLogger(__name__)

class RetrieverFactory:
    """Factory for creating retrievers from environment variables
    
    環境変数からretrieverを作成するファクトリー
    """
    
    @staticmethod
    def create_retrievers_from_env() -> List:
        """Create retrievers based on REFINIRE_RAG_RETRIEVERS environment variable
        
        REFINIRE_RAG_RETRIEVERS環境変数に基づいてretrieverを作成
        
        Returns:
            List of configured retriever instances
            設定されたretrieverインスタンスのリスト
        """
        retrievers_config = os.getenv("REFINIRE_RAG_RETRIEVERS", "").strip()
        if not retrievers_config:
            logger.warning("REFINIRE_RAG_RETRIEVERS not set, no retrievers will be created")
            return []
        
        retriever_names = [name.strip() for name in retrievers_config.split(",")]
        retrievers = []
        
        # Ensure plugins are discovered
        RetrieverRegistry.discover_retrievers()
        
        for name in retriever_names:
            if not name:
                continue
            
            try:
                # Create retriever instance
                # Each plugin handles its own configuration from environment variables
                retriever = RetrieverRegistry.create_retriever(name)
                retrievers.append(retriever)
                logger.info(f"Created retriever: {name}")
                
            except Exception as e:
                logger.error(f"Failed to create retriever '{name}': {e}")
        
        return retrievers
    
    @staticmethod
    def get_available_retrievers() -> Dict[str, Any]:
        """Get information about available retrievers
        
        利用可能なretrieverの情報を取得
        """
        RetrieverRegistry.discover_retrievers()
        available = {}
        
        for name in RetrieverRegistry.list_available_retrievers():
            try:
                retriever_class = RetrieverRegistry.get_retriever_class(name)
                available[name] = {
                    "class": retriever_class.__name__,
                    "module": retriever_class.__module__,
                    "description": getattr(retriever_class, "__doc__", "No description")
                }
            except Exception as e:
                available[name] = {"error": str(e)}
        
        return available
```

## oneenvテンプレート統合

### テンプレート発見システム

```python
# src/refinire_rag/oneenv_integration.py
import importlib.metadata
import logging
from typing import List, Dict, Any
from oneenv import EnvTemplate

logger = logging.getLogger(__name__)

class OneenvTemplateRegistry:
    """Registry for oneenv templates from plugins
    
    プラグインからのoneenvテンプレート用レジストリ
    """
    
    @staticmethod
    def discover_plugin_templates() -> List[EnvTemplate]:
        """Discover and load oneenv templates from all plugins
        
        すべてのプラグインからoneenvテンプレートを発見・ロード
        """
        templates = []
        
        try:
            # Discover templates from entry points
            for entry_point in importlib.metadata.entry_points(group='refinire_rag.oneenv_templates'):
                try:
                    template_func = entry_point.load()
                    template = template_func()
                    templates.append(template)
                    logger.info(f"Loaded oneenv template: {entry_point.name}")
                except Exception as e:
                    logger.warning(f"Failed to load oneenv template {entry_point.name}: {e}")
        
        except Exception as e:
            logger.error(f"Error discovering oneenv templates: {e}")
        
        return templates
    
    @staticmethod
    def generate_combined_template() -> EnvTemplate:
        """Generate combined template from core and all plugins
        
        コアと全プラグインから統合テンプレートを生成
        """
        from .env_template import refinire_rag_env_template
        
        # Start with core template
        core_template = refinire_rag_env_template()
        combined_variables = list(core_template.variables)
        combined_groups = dict(core_template.groups) if core_template.groups else {}
        
        # Add plugin templates
        plugin_templates = OneenvTemplateRegistry.discover_plugin_templates()
        for template in plugin_templates:
            combined_variables.extend(template.variables)
            if template.groups:
                combined_groups.update(template.groups)
        
        return EnvTemplate(
            name="refinire-rag-complete",
            description="Complete environment variables for refinire-rag with all plugins",
            variables=combined_variables,
            groups=combined_groups
        )
```

## 統合使用例

### 1. 複数プラグインを使用した高度なRAGシステム

```python
# advanced_rag_system.py
import os
from refinire_rag.application import CorpusManager, QueryEngine, QualityLab
from refinire_rag.factories import PluginFactory

# 環境変数設定（現在利用可能なプラグイン）
os.environ["REFINIRE_RAG_RETRIEVERS"] = "chroma,bm25s"  # 現在サポート：Chroma, BM25s
os.environ["REFINIRE_RAG_DOCUMENT_STORES"] = "sqlite"  # DocumentStoreプラグイン
# 将来的に開発予定：Faiss, Elasticsearch等

# プラグイン固有設定
os.environ["REFINIRE_RAG_CHROMA_HOST"] = "localhost"
os.environ["REFINIRE_RAG_CHROMA_PORT"] = "8000" 
os.environ["REFINIRE_RAG_BM25S_INDEX_PATH"] = "./bm25s_index"

# CorpusManager を環境変数から自動作成
corpus_manager = CorpusManager.from_env()

# CorpusManager の情報確認
corpus_info = corpus_manager.get_corpus_info()
print("=== Corpus Manager Configuration ===")
print(f"Document Store: {corpus_info['document_store']['type']}")
print("Retrievers:")
for retriever_info in corpus_info['retrievers']:
    print(f"  - {retriever_info['type']}: {retriever_info['capabilities']}")

# 文書のインポート（複数retrieversに自動的に保存）
stats = corpus_manager.import_original_documents(
    corpus_name="knowledge_base",
    directory="./documents",
    glob="**/*.{md,txt,pdf}",
    create_dictionary=True,
    create_knowledge_graph=True
)

print(f"Imported {stats.total_documents_created} documents")
print(f"Created {stats.total_chunks_created} chunks")
print(f"Processed by {len(corpus_manager.retrievers)} retrievers")

# QueryEngine の作成（同じretrieversを使用）
query_engine = QueryEngine(
    corpus_name="knowledge_base",
    retrievers=corpus_manager.retrievers,  # [ChromaRetriever, BM25sRetriever]
    # reranker=reranker,        # 将来実装予定
    synthesizer=synthesizer
)

# QualityLab での評価（内蔵evaluator使用）
quality_lab = QualityLab(
    corpus_manager=corpus_manager
    # evaluators=evaluators     # 将来プラグイン化予定
)

# クエリ実行と評価
result = query_engine.query("How does RAG work?")
print(f"Answer: {result.answer}")
print(f"Sources from {len(result.sources)} retrievers")

# 動的なretriever管理
print("\n=== Dynamic Retriever Management ===")

# 特定タイプのretrieversを取得
vector_retrievers = corpus_manager.get_retrievers_by_type("vector")
keyword_retrievers = corpus_manager.get_retrievers_by_type("keyword")

print(f"Vector retrievers: {[type(r).__name__ for r in vector_retrievers]}")
print(f"Keyword retrievers: {[type(r).__name__ for r in keyword_retrievers]}")

# 新しいretrieverを動的に追加（例）
# new_retriever = PluginRegistry.create_plugin('retrievers', 'faiss')  # 将来実装予定
# corpus_manager.add_retriever(new_retriever)

# コーパス統計の確認
print(f"\nCorpus Statistics:")
print(f"  Total documents: {corpus_info['stats']['total_documents_created']}")
print(f"  Total chunks: {corpus_info['stats']['total_chunks_created']}")
print(f"  Processing time: {corpus_info['stats']['total_processing_time']:.2f}s")

# 包括的評価実行
evaluation_results = quality_lab.run_full_evaluation(
    qa_set_name="comprehensive_test",
    corpus_name="knowledge_base",
    query_engine=query_engine,
    num_qa_pairs=100
)

print(f"Overall score: {evaluation_results['overall_score']:.2f}")
```

### 2. 環境変数による動的プラグイン切り替え

```python
# dynamic_plugin_switching.py
from refinire_rag.registry import PluginRegistry

# 利用可能なプラグインの確認
available_plugins = PluginRegistry.get_all_plugins_info()

print("=== Available Plugins ===")
for group, plugins in available_plugins.items():
    print(f"\n{group.upper()}:")
    for name, info in plugins.items():
        print(f"  - {name}: {info['description']}")

# 環境変数による動的切り替え例
import os

# 開発環境：軽量プラグイン
if os.getenv("ENVIRONMENT") == "development":
    os.environ["REFINIRE_RAG_RETRIEVERS"] = "inmemory_vector"  # 内蔵の軽量retriever
    
# 本番環境：現在利用可能なプラグイン  
elif os.getenv("ENVIRONMENT") == "production":
    os.environ["REFINIRE_RAG_RETRIEVERS"] = "chroma,bm25s"  # 現在サポート済み
    # 将来追加予定：faiss, elasticsearch等

# テスト環境：テスト用設定
elif os.getenv("ENVIRONMENT") == "testing":
    os.environ["REFINIRE_RAG_RETRIEVERS"] = "inmemory_vector"  # テスト用軽量設定
```

### 3. カスタム評価パイプライン

```python
# custom_evaluation_pipeline.py
from refinire_rag.factories import PluginFactory

# 注意：以下は将来の評価プラグイン実装例です
# 現在は内蔵evaluatorを使用してください

# 将来実装予定の評価プラグイン設定例
# os.environ["REFINIRE_RAG_EVALUATORS"] = "bertscore,rouge"  # 開発予定
# os.environ["REFINIRE_RAG_BERTSCORE_MODEL"] = "microsoft/deberta-xlarge-mnli"

# 現在は QualityLab の内蔵評価機能を使用
from refinire_rag.application import QualityLab

quality_lab = QualityLab(corpus_manager=corpus_manager)

# 内蔵評価実行
evaluation_results = quality_lab.evaluate_query_engine(
    query_engine=query_engine,
    qa_pairs=qa_pairs
)
print(f"Overall Score: {evaluation_results['overall_score']:.3f}")
```

### 4. プラグイン情報の動的取得

```python
# plugin_introspection.py
from refinire_rag.registry import PluginRegistry
from refinire_rag.factories import PluginFactory

# 現在利用可能なプラグイン一覧の確認（内蔵 + 外部）
retrievers = PluginRegistry.list_available_plugins('retrievers')
vector_stores = PluginRegistry.list_available_plugins('vector_stores')
document_stores = PluginRegistry.list_available_plugins('document_stores')

print(f"Available retrievers: {retrievers}")
print(f"Available vector stores: {vector_stores}")
print(f"Available document stores: {document_stores}")

# 内蔵コンポーネントと外部プラグインの区別
all_info = PluginRegistry.get_all_plugins_info()

print("\n=== Detailed Plugin Information ===")
for group in ['retrievers', 'vector_stores', 'document_stores']:
    print(f"\n{group.upper()}:")
    if group in all_info:
        for name, info in all_info[group].items():
            plugin_type = "Built-in" if info['builtin'] else "External"
            print(f"  - {name}: {info['class']} ({plugin_type})")

# プラグインの動的作成とテスト
print("\n=== Testing Plugin Creation ===")
for name in vector_stores:
    try:
        store = PluginRegistry.create_plugin('vector_stores', name)
        is_builtin = PluginRegistry.is_builtin('vector_stores', name)
        source = "built-in" if is_builtin else "external plugin"
        print(f"Successfully created {source}: {name} -> {type(store).__name__}")
    except Exception as e:
        print(f"Failed to create {name}: {e}")

# 内蔵コンポーネント一覧の取得
print("\n=== Built-in Components by Group ===")
builtin_components = PluginFactory.list_builtin_components()
for group, components in builtin_components.items():
    if components:
        print(f"{group}: {components}")
```

### oneenv統合例

```bash
# .env ファイル生成
python -c "
from refinire_rag.oneenv_integration import OneenvTemplateRegistry
template = OneenvTemplateRegistry.generate_combined_template()
template.save_env_file('.env.template')
"

# 環境変数設定支援
python -c "
from refinire_rag.oneenv_integration import OneenvTemplateRegistry
template = OneenvTemplateRegistry.generate_combined_template()
template.interactive_setup()
"
```

## プラグイン配布とインストール

### PyPIでの配布

```bash
# プラグインパッケージのビルドと配布
cd refinire-rag-chroma
python -m build
twine upload dist/*
```

### インストールと使用

```bash
# プラグインインストール
pip install refinire-rag-chroma

# 自動的にentry pointsが登録され、利用可能になる
python -c "
from refinire_rag.registry import RetrieverRegistry
print(RetrieverRegistry.list_available_retrievers())
"
# 出力: ['chroma', 'bm25s', ...]
```

## 最適化とベストプラクティス

### 1. 環境変数命名規則

- プレフィックス: `REFINIRE_RAG_[PLUGIN_NAME]_`
- 大文字・アンダースコア区切り
- 意味のある名前付け

### 2. oneenvテンプレート設計

- 重要度の適切な設定（Critical/Important/Optional）
- グループ化による論理的整理
- デフォルト値の提供

### 3. エラーハンドリング

- 設定不備時の適切なフォールバック
- 詳細なログ出力
- ユーザーフレンドリーなエラーメッセージ

### 4. パフォーマンス考慮

- 遅延初期化（lazy loading）
- 接続プールの適切な管理
- バッチ処理の最適化

このプラグインアーキテクチャにより、refinire-ragエコシステムは柔軟に拡張可能で、oneenvによる統一的な環境変数管理と組み合わせることで、使いやすく保守性の高いシステムを実現できます。