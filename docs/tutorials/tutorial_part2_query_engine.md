# Part 2: QueryEngine検索チュートリアル

## Overview / 概要

This tutorial demonstrates how to use refinire-rag's QueryEngine for intelligent document search and answer generation. The QueryEngine integrates retrieval, reranking, and answer synthesis to provide accurate, contextual responses from your corpus.

このチュートリアルでは、refinire-ragのQueryEngineを使用したインテリジェントな文書検索と回答生成の方法を説明します。QueryEngineは、検索、再ランキング、回答合成を統合して、コーパスから正確で文脈に沿った回答を提供します。

## Learning Objectives / 学習目標

- Configure and initialize QueryEngine components / QueryEngineコンポーネントの設定と初期化
- Understand retrieval and reranking mechanisms / 検索と再ランキングメカニズムの理解
- Master query normalization and processing / クエリ正規化と処理のマスター
- Implement custom answer synthesis / カスタム回答合成の実装
- Monitor and optimize query performance / クエリパフォーマンスの監視と最適化

## Prerequisites / 前提条件

```bash
# Complete Part 1: Corpus Creation first
# Part 1: コーパス作成を先に完了

# Set environment variables for LLM integration
export OPENAI_API_KEY="your-api-key"
export REFINIRE_RAG_LLM_MODEL="gpt-4o-mini"
export REFINIRE_RAG_EMBEDDING_MODEL="text-embedding-3-small"
```

## Quick Start Example / クイックスタート例

```python
from refinire_rag.application.query_engine import QueryEngine
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader

# Initialize QueryEngine with existing corpus
query_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=SimpleRetriever(vector_store),
    reranker=SimpleReranker(),
    reader=SimpleReader()
)

# Simple query
result = query_engine.answer("What is artificial intelligence?")
print(f"Answer: {result.answer}")
print(f"Sources: {len(result.sources)} documents")
print(f"Confidence: {result.confidence:.2f}")
```

## 1. QueryEngine Architecture / QueryEngine アーキテクチャ

### 1.1 Core Components / コアコンポーネント

```python
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.retrieval import (
    SimpleRetriever, SimpleRetrieverConfig,
    SimpleReranker, SimpleRerankerConfig, 
    SimpleReader, SimpleReaderConfig
)

# Component configuration / コンポーネント設定
retriever_config = SimpleRetrieverConfig(
    top_k=10,                        # Number of initial candidates
    similarity_threshold=0.1,        # Minimum similarity score
    embedding_model="text-embedding-3-small"
)

reranker_config = SimpleRerankerConfig(
    top_k=5,                         # Final number of results
    boost_exact_matches=True,        # Boost exact keyword matches
    length_penalty_factor=0.1        # Penalty for very long/short docs
)

reader_config = SimpleReaderConfig(
    llm_model="gpt-4o-mini",         # Language model for synthesis
    max_context_length=2000,         # Maximum context tokens
    temperature=0.2,                 # Generation temperature
    include_sources=True             # Include source citations
)

# Create components / コンポーネント作成
retriever = SimpleRetriever(vector_store, config=retriever_config)
reranker = SimpleReranker(config=reranker_config)
reader = SimpleReader(config=reader_config)
```

### 1.2 QueryEngine Configuration / QueryEngine設定

```python
# Engine-level configuration / エンジンレベル設定
engine_config = QueryEngineConfig(
    enable_query_normalization=True,    # Normalize queries using dictionary
    auto_detect_corpus_state=True,      # Auto-detect corpus capabilities
    retriever_top_k=10,                 # Override retriever top_k
    reranker_top_k=5,                   # Override reranker top_k
    include_sources=True,                # Include source documents
    include_confidence=True,             # Calculate confidence scores
    max_response_time=30.0,             # Maximum response time (seconds)
    enable_caching=True                  # Enable response caching
)

# Initialize QueryEngine / QueryEngine初期化
query_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=retriever,
    reranker=reranker,
    reader=reader,
    config=engine_config
)
```

## 2. Basic Query Operations / 基本クエリ操作

### 2.1 Simple Queries / シンプルクエリ

```python
# Direct question answering / 直接的な質問応答
result = query_engine.answer("What is machine learning?")

print(f"Question / 質問: What is machine learning?")
print(f"Answer / 回答: {result.answer}")
print(f"Confidence / 信頼度: {result.confidence:.3f}")
print(f"Processing time / 処理時間: {result.metadata['processing_time']:.3f}s")

# Access source documents / ソース文書へのアクセス
for i, source in enumerate(result.sources):
    print(f"Source {i+1}: {source.metadata.get('title', 'Untitled')}")
    print(f"  Content preview: {source.content[:100]}...")
    print(f"  Relevance score: {source.metadata.get('relevance_score', 'N/A')}")
```

### 2.2 Queries with Context / 文脈付きクエリ

```python
# Multi-turn conversation / 複数ターン会話
context = {
    "previous_questions": [
        "What is artificial intelligence?",
        "What are the main types of AI?"
    ],
    "user_preferences": {
        "detail_level": "intermediate",
        "include_examples": True
    }
}

result = query_engine.answer(
    query="How does deep learning differ from traditional ML?",
    context=context
)

print(f"Contextual answer: {result.answer}")
```

### 2.3 Filtered Queries / フィルタ付きクエリ

```python
# Search with metadata filters / メタデータフィルタ付き検索
filters = {
    "category": "AI",
    "difficulty": ["beginner", "intermediate"],
    "source": "Technical Documentation"
}

result = query_engine.answer(
    query="Explain neural networks",
    filters=filters
)

print(f"Filtered result from {len(result.sources)} matching documents")
```

## 3. Advanced Query Features / 高度なクエリ機能

### 3.1 Query Normalization / クエリ正規化

```python
# Enable automatic query normalization / 自動クエリ正規化を有効化
# Requires dictionary generated in Part 1
# Part 1で生成された辞書が必要

# Query with technical terms / 技術用語を含むクエリ
original_query = "How does ML work?"
result = query_engine.answer(original_query)

# Check if query was normalized / クエリが正規化されたかチェック
if result.metadata.get('query_normalized'):
    normalized_query = result.metadata.get('normalized_query')
    print(f"Original: {original_query}")
    print(f"Normalized: {normalized_query}")
    print(f"Normalization improved search accuracy")
```

### 3.2 Multi-Retriever Integration / マルチ検索統合

```python
from refinire_rag.retrieval import HybridRetriever
from refinire_rag.keywordstore import TfidfKeywordStore

# Setup hybrid retrieval (vector + keyword)
# ハイブリッド検索の設定（ベクトル + キーワード）
keyword_store = TfidfKeywordStore()
hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    keyword_store=keyword_store,
    vector_weight=0.7,     # 70% weight for vector search
    keyword_weight=0.3     # 30% weight for keyword search
)

# QueryEngine with hybrid retrieval
hybrid_query_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=hybrid_retriever,
    reranker=reranker,
    reader=reader
)

result = hybrid_query_engine.answer("machine learning algorithms")
print(f"Hybrid search found {len(result.sources)} relevant documents")
```

### 3.3 Custom Answer Instructions / カスタム回答指示

```python
# Custom instructions for domain-specific responses
# ドメイン特化回答のためのカスタム指示
custom_reader_config = SimpleReaderConfig(
    llm_model="gpt-4o-mini",
    generation_instructions="""
    You are an AI technology expert. When answering questions:
    1. Provide clear, technical explanations
    2. Include practical examples and use cases
    3. Explain complex concepts in simple terms
    4. Always cite specific sources from the provided context
    5. If uncertain, clearly state limitations
    
    Format your response as:
    - Direct answer (2-3 sentences)
    - Technical details (if applicable)
    - Practical examples
    - Related concepts
    """,
    temperature=0.1,
    max_tokens=500
)

custom_reader = SimpleReader(config=custom_reader_config)
custom_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=retriever,
    reranker=reranker,
    reader=custom_reader
)

result = custom_engine.answer("What are the applications of computer vision?")
```

## 4. Query Result Analysis / クエリ結果分析

### 4.1 Detailed Result Inspection / 詳細結果検査

```python
def analyze_query_result(result):
    """
    Analyze and display detailed query results
    詳細なクエリ結果を分析・表示
    """
    
    print("="*60)
    print("QUERY RESULT ANALYSIS / クエリ結果分析")
    print("="*60)
    
    # Basic metrics / 基本メトリクス
    print(f"\n📊 Basic Metrics / 基本メトリクス:")
    print(f"   Answer length: {len(result.answer)} characters")
    print(f"   Confidence score: {result.confidence:.3f}")
    print(f"   Source count: {len(result.sources)}")
    print(f"   Processing time: {result.metadata.get('processing_time', 0):.3f}s")
    
    # Query processing details / クエリ処理詳細
    print(f"\n🔍 Query Processing / クエリ処理:")
    print(f"   Original query: {result.metadata.get('original_query', 'N/A')}")
    print(f"   Normalized: {result.metadata.get('query_normalized', False)}")
    if result.metadata.get('normalized_query'):
        print(f"   Normalized query: {result.metadata['normalized_query']}")
    
    # Retrieval analysis / 検索分析
    print(f"\n📚 Retrieval Analysis / 検索分析:")
    print(f"   Retrieved documents: {result.metadata.get('retrieved_count', 0)}")
    print(f"   Reranked documents: {result.metadata.get('reranked_count', 0)}")
    print(f"   Final sources used: {len(result.sources)}")
    
    # Source quality / ソース品質
    print(f"\n🎯 Source Quality / ソース品質:")
    for i, source in enumerate(result.sources[:3]):  # Top 3 sources
        relevance = source.metadata.get('relevance_score', 0)
        title = source.metadata.get('title', f'Document {i+1}')
        print(f"   {i+1}. {title}: relevance={relevance:.3f}")
    
    # Answer quality indicators / 回答品質指標
    print(f"\n✅ Quality Indicators / 品質指標:")
    answer_length = len(result.answer.split())
    print(f"   Answer word count: {answer_length}")
    print(f"   Has sources: {'Yes' if result.sources else 'No'}")
    print(f"   Confidence level: {'High' if result.confidence > 0.8 else 'Medium' if result.confidence > 0.5 else 'Low'}")

# Usage example / 使用例
result = query_engine.answer("Explain the difference between AI and ML")
analyze_query_result(result)
```

### 4.2 Performance Monitoring / パフォーマンス監視

```python
def monitor_query_performance(query_engine, test_queries):
    """
    Monitor QueryEngine performance across multiple queries
    複数クエリでQueryEngineのパフォーマンスを監視
    """
    
    print("🚀 PERFORMANCE MONITORING / パフォーマンス監視")
    print("="*60)
    
    results = []
    total_time = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}/{len(test_queries)}: {query}")
        
        import time
        start_time = time.time()
        result = query_engine.answer(query)
        end_time = time.time()
        
        query_time = end_time - start_time
        total_time += query_time
        
        results.append({
            'query': query,
            'time': query_time,
            'confidence': result.confidence,
            'source_count': len(result.sources),
            'answer_length': len(result.answer)
        })
        
        print(f"   ⏱️  Time: {query_time:.3f}s")
        print(f"   🎯 Confidence: {result.confidence:.3f}")
        print(f"   📚 Sources: {len(result.sources)}")
    
    # Summary statistics / 要約統計
    print(f"\n📊 Summary Statistics / 要約統計:")
    avg_time = total_time / len(test_queries)
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    avg_sources = sum(r['source_count'] for r in results) / len(results)
    
    print(f"   Average response time: {avg_time:.3f}s")
    print(f"   Average confidence: {avg_confidence:.3f}")
    print(f"   Average sources per query: {avg_sources:.1f}")
    print(f"   Total queries processed: {len(test_queries)}")
    print(f"   Queries per second: {len(test_queries)/total_time:.2f}")

# Test queries for monitoring / 監視用テストクエリ
test_queries = [
    "What is artificial intelligence?",
    "How does machine learning work?", 
    "What are the applications of deep learning?",
    "Explain neural networks",
    "What is the difference between supervised and unsupervised learning?"
]

monitor_query_performance(query_engine, test_queries)
```

## 5. QueryEngine Optimization / QueryEngine最適化

### 5.1 Component Configuration Tuning / コンポーネント設定調整

```python
# Performance-optimized configuration / パフォーマンス最適化設定
fast_config = {
    "retriever": SimpleRetrieverConfig(
        top_k=5,                    # Fewer candidates for speed
        similarity_threshold=0.2,   # Higher threshold
        embedding_model="text-embedding-3-small"
    ),
    "reranker": SimpleRerankerConfig(
        top_k=3,                    # Fewer final results
        boost_exact_matches=True
    ),
    "reader": SimpleReaderConfig(
        llm_model="gpt-4o-mini",    # Faster model
        max_context_length=1000,    # Shorter context
        temperature=0.1             # Lower temperature for consistency
    )
}

# Accuracy-optimized configuration / 精度最適化設定
accurate_config = {
    "retriever": SimpleRetrieverConfig(
        top_k=20,                   # More candidates for accuracy
        similarity_threshold=0.05,  # Lower threshold for recall
        embedding_model="text-embedding-3-large"
    ),
    "reranker": SimpleRerankerConfig(
        top_k=8,                    # More final results
        boost_exact_matches=True,
        length_penalty_factor=0.05  # Less aggressive length penalty
    ),
    "reader": SimpleReaderConfig(
        llm_model="gpt-4",          # More capable model
        max_context_length=3000,    # Longer context
        temperature=0.2             # Balanced temperature
    )
}
```

### 5.2 Caching and Optimization / キャッシュと最適化

```python
from refinire_rag.application.query_engine import QueryEngineConfig

# Enable advanced caching / 高度なキャッシュを有効化
optimized_config = QueryEngineConfig(
    enable_caching=True,
    cache_ttl=3600,                # Cache for 1 hour
    cache_max_size=1000,           # Max cache entries
    enable_query_normalization=True,
    similarity_cache_threshold=0.95, # Cache similar queries
    max_response_time=10.0         # Response timeout
)

optimized_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=retriever,
    reranker=reranker,
    reader=reader,
    config=optimized_config
)

# Warm up cache with common queries / よくあるクエリでキャッシュをウォームアップ
common_queries = [
    "What is AI?",
    "Machine learning basics",
    "Deep learning overview"
]

for query in common_queries:
    optimized_engine.answer(query)
    
print("Cache warmed up with common queries")
```

## 6. Error Handling and Debugging / エラーハンドリングとデバッグ

### 6.1 Robust Query Processing / 堅牢なクエリ処理

```python
def robust_query(query_engine, query, max_retries=3):
    """
    Robust query processing with error handling and retries
    エラーハンドリングとリトライを含む堅牢なクエリ処理
    """
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Attempt {attempt + 1}: {query}")
            
            result = query_engine.answer(query)
            
            # Validate result quality / 結果品質を検証
            if len(result.answer) < 10:
                raise ValueError("Answer too short")
            
            if result.confidence < 0.1:
                raise ValueError("Confidence too low")
            
            if not result.sources:
                raise ValueError("No sources found")
            
            print(f"✅ Success on attempt {attempt + 1}")
            return result
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            
            if attempt == max_retries - 1:
                print(f"🚫 All {max_retries} attempts failed")
                
                # Return fallback response / フォールバック応答を返す
                from refinire_rag.models.query import QueryResult
                fallback_result = QueryResult(
                    query=query,
                    answer="I apologize, but I couldn't process your query at this time. Please try rephrasing your question or contact support.",
                    confidence=0.0,
                    sources=[],
                    metadata={'error': str(e), 'fallback': True}
                )
                return fallback_result
    
    return None

# Usage / 使用方法
result = robust_query(query_engine, "What is quantum computing?")
```

### 6.2 Debugging Tools / デバッグツール

```python
def debug_query_pipeline(query_engine, query):
    """
    Debug QueryEngine pipeline step by step
    QueryEngineパイプラインをステップバイステップでデバッグ
    """
    
    print("🔧 QUERY PIPELINE DEBUG / クエリパイプラインデバッグ")
    print("="*60)
    
    # Step 1: Query preprocessing / ステップ1: クエリ前処理
    print(f"\n1️⃣ Original Query / 元クエリ: {query}")
    
    # Check query normalization if enabled
    # 有効な場合、クエリ正規化をチェック
    if hasattr(query_engine, '_normalize_query'):
        normalized = query_engine._normalize_query(query)
        print(f"   Normalized / 正規化済み: {normalized}")
    
    # Step 2: Retrieval / ステップ2: 検索
    print(f"\n2️⃣ Retrieval Phase / 検索フェーズ:")
    try:
        retrieved_docs = query_engine.retriever.retrieve(query)
        print(f"   Retrieved documents / 検索文書数: {len(retrieved_docs)}")
        
        for i, doc in enumerate(retrieved_docs[:3]):
            score = doc.metadata.get('similarity_score', 'N/A')
            print(f"   {i+1}. Score: {score}, Length: {len(doc.content)}")
            
    except Exception as e:
        print(f"   ❌ Retrieval failed: {e}")
        return
    
    # Step 3: Reranking / ステップ3: 再ランキング
    print(f"\n3️⃣ Reranking Phase / 再ランキングフェーズ:")
    try:
        reranked_docs = query_engine.reranker.rerank(query, retrieved_docs)
        print(f"   Reranked documents / 再ランキング文書数: {len(reranked_docs)}")
        
        for i, doc in enumerate(reranked_docs[:3]):
            score = doc.metadata.get('rerank_score', 'N/A')
            print(f"   {i+1}. Rerank Score: {score}")
            
    except Exception as e:
        print(f"   ❌ Reranking failed: {e}")
        reranked_docs = retrieved_docs
    
    # Step 4: Answer synthesis / ステップ4: 回答合成
    print(f"\n4️⃣ Answer Synthesis / 回答合成:")
    try:
        answer = query_engine.reader.synthesize(query, reranked_docs)
        print(f"   Generated answer / 生成回答: {len(answer)} characters")
        print(f"   Preview / プレビュー: {answer[:100]}...")
        
    except Exception as e:
        print(f"   ❌ Answer synthesis failed: {e}")
        return
    
    print(f"\n✅ Pipeline completed successfully / パイプライン正常完了")

# Debug example / デバッグ例
debug_query_pipeline(query_engine, "How do neural networks learn?")
```

## 7. Complete Example / 完全な例

```python
#!/usr/bin/env python3
"""
Complete QueryEngine tutorial example
完全なQueryEngineチュートリアル例
"""

from pathlib import Path
from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader

def main():
    # Setup corpus (assuming Part 1 completed)
    # コーパスセットアップ（Part 1完了前提）
    print("🚀 QueryEngine Tutorial / QueryEngineチュートリアル")
    print("="*60)
    
    # Initialize storage / ストレージ初期化
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    
    # Create sample corpus / サンプルコーパス作成
    print("📚 Setting up sample corpus...")
    corpus_manager = CorpusManager.create_semantic_rag(doc_store, vector_store)
    
    # Create sample documents / サンプル文書作成
    sample_docs = [
        "AI is artificial intelligence technology.",
        "Machine learning is a subset of AI that learns from data.",
        "Deep learning uses neural networks with multiple layers."
    ]
    
    # Build corpus / コーパス構築
    stats = corpus_manager.build_corpus(sample_docs)
    print(f"✅ Corpus built: {stats.total_chunks_created} chunks")
    
    # Initialize QueryEngine / QueryEngine初期化
    print("\n🔍 Initializing QueryEngine...")
    
    query_engine = QueryEngine(
        document_store=doc_store,
        vector_store=vector_store,
        retriever=SimpleRetriever(vector_store),
        reranker=SimpleReranker(),
        reader=SimpleReader(),
        config=QueryEngineConfig(
            enable_query_normalization=True,
            include_sources=True,
            include_confidence=True
        )
    )
    
    # Demo queries / デモクエリ
    queries = [
        "What is AI?",
        "How does machine learning work?",
        "Explain deep learning"
    ]
    
    print("\n💬 Running demo queries...")
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        result = query_engine.answer(query)
        
        print(f"🤖 Answer: {result.answer}")
        print(f"🎯 Confidence: {result.confidence:.3f}")
        print(f"📚 Sources: {len(result.sources)}")
    
    # Engine statistics / エンジン統計
    print("\n📊 Engine Statistics:")
    stats = query_engine.get_engine_stats()
    print(f"   Queries processed: {stats.get('queries_processed', 0)}")
    print(f"   Average response time: {stats.get('average_response_time', 0):.3f}s")
    
    print("\n🎉 Tutorial completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    print("✅ All done!" if success else "❌ Failed")
```

## Next Steps / 次のステップ

After mastering QueryEngine, proceed to:
QueryEngineをマスターした後、次に進む：

1. **Part 3: Evaluation** - Learn to evaluate your RAG system performance
   **Part 3: Evaluation** - RAGシステムのパフォーマンス評価を学習
2. **Advanced Topics** - Custom retrievers, specialized readers, production deployment
   **高度なトピック** - カスタム検索、特殊リーダー、本番デプロイ

## Resources / リソース

- [QueryEngine API Documentation](../api/query_engine.md)
- [Retrieval Components Reference](../api/retrieval.md)
- [Performance Tuning Guide](../development/performance_optimization.md)
- [Example Scripts](../../examples/)