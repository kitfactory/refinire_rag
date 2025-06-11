# チュートリアル3: クエリエンジンと回答生成

このチュートリアルでは、QueryEngineを使用した高度なクエリ処理と回答生成を学習します。

## 学習目標

- QueryEngineの基本構造を理解する
- クエリ正規化の効果を体験する
- Retriever、Reranker、Readerの役割を学ぶ
- 回答品質を向上させる設定をマスターする

## QueryEngineのアーキテクチャ

QueryEngineは以下のコンポーネントで構成されます：

```
クエリ → [正規化] → [検索] → [再ランキング] → [回答生成] → 回答
          ↓        ↓         ↓           ↓
       Normalizer → Retriever → Reranker → Reader
```

### コンポーネントの役割

1. **Normalizer**: クエリの表現揺らぎを統一
2. **Retriever**: ベクトル類似度による文書検索
3. **Reranker**: 検索結果の関連性による再順序付け
4. **Reader**: LLMによる回答生成

## ステップ1: 正規化コーパスの準備

まず、正規化処理を含むコーパスを構築します：

```python
import tempfile
from pathlib import Path

def setup_normalized_corpus():
    """正規化されたコーパスを構築"""
    
    from refinire_rag.use_cases.corpus_manager_new import CorpusManager
    from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
    from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
    from refinire_rag.models.document import Document
    from refinire_rag.embedding import TFIDFEmbedder, TFIDFEmbeddingConfig
    from refinire_rag.storage.vector_store import VectorEntry
    
    # 一時ディレクトリ作成
    temp_dir = Path(tempfile.mkdtemp())
    
    # 辞書ファイル作成
    dict_file = temp_dir / "query_dictionary.md"
    dict_file.write_text("""# クエリ正規化辞書

## AI技術用語

- **RAG** (Retrieval-Augmented Generation): 検索拡張生成
  - 表現揺らぎ: 検索拡張生成, 検索強化生成, RAGシステム, 検索拡張技術

- **ベクトル検索** (Vector Search): ベクトル検索
  - 表現揺らぎ: ベクトル検索, 意味検索, セマンティック検索, 意味的検索

- **LLM** (Large Language Model): 大規模言語モデル
  - 表現揺らぎ: 大規模言語モデル, 言語モデル, LLMモデル, 大規模LM

- **チャンキング** (Chunking): チャンキング
  - 表現揺らぎ: チャンキング, 文書分割, テキスト分割, チャンク化
""", encoding='utf-8')
    
    # サンプルドキュメント
    documents = [
        Document(
            id="doc1",
            content="""
            検索拡張生成（RAG）は、情報検索と言語生成を組み合わせた
            革新的なAI技術です。従来のLLMの知識制限を克服し、
            外部データベースから関連情報を検索して、
            より正確で最新の回答を生成できます。
            RAGシステムは企業の質疑応答システムや
            知識管理プラットフォームで広く活用されています。
            """,
            metadata={"title": "RAG技術概要", "category": "技術解説"}
        ),
        
        Document(
            id="doc2", 
            content="""
            ベクトル検索は、文書とクエリを高次元ベクトル空間で
            表現し、数学的類似度で関連性を計算する検索手法です。
            従来のキーワードマッチングと異なり、
            文脈や意味を理解した検索が可能になります。
            コサイン類似度、ユークリッド距離、内積などの
            計算方法が使用され、近似最近傍探索（ANN）により
            高速な検索を実現します。
            """,
            metadata={"title": "ベクトル検索技術", "category": "技術解説"}
        ),
        
        Document(
            id="doc3",
            content="""
            チャンキングは、長い文書を検索・処理しやすい
            小さな単位に分割する重要な前処理技術です。
            適切なチャンクサイズとオーバーラップの設定により、
            文脈を保持しながら効率的な検索を実現します。
            文の境界、段落、意味的なまとまりを考慮した
            分割手法が存在し、RAGシステムの性能に
            大きな影響を与えます。
            """,
            metadata={"title": "チャンキング技術", "category": "技術解説"}
        ),
        
        Document(
            id="doc4",
            content="""
            大規模言語モデル（LLM）の評価には、
            多面的なアプローチが必要です。
            BLEU、ROUGE、BERTScoreなどの自動評価指標に加え、
            人手評価による流暢さ、正確性、有用性の評価が重要です。
            RAGシステムでは、検索精度と生成品質の
            両方を考慮した総合的な評価が求められます。
            """,
            metadata={"title": "LLM評価手法", "category": "評価"}
        )
    ]
    
    # ストレージ初期化
    document_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    
    # 手動でコーパス構築（正規化を模擬）
    print("📚 正規化コーパスを構築中...")
    
    # ドキュメントストアに保存
    for doc in documents:
        # 元文書として保存
        doc.metadata["processing_stage"] = "original"
        document_store.store_document(doc)
        
        # 正規化版も作成（実際の正規化処理は省略）
        normalized_doc = Document(
            id=doc.id,
            content=doc.content,
            metadata={
                **doc.metadata,
                "processing_stage": "normalized",
                "normalization_stats": {
                    "dictionary_file_used": str(dict_file),
                    "total_replacements": 2,
                    "variations_normalized": 1
                }
            }
        )
        document_store.store_document(normalized_doc)
    
    # ベクトル化
    embedder_config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0)
    embedder = TFIDFEmbedder(config=embedder_config)
    
    corpus_texts = [doc.content for doc in documents]
    embedder.fit(corpus_texts)
    
    for doc in documents:
        embedding_result = embedder.embed_text(doc.content)
        vector_entry = VectorEntry(
            document_id=doc.id,
            content=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
            embedding=embedding_result.vector.tolist(),
            metadata=doc.metadata
        )
        vector_store.add_vector(vector_entry)
    
    print(f"✅ {len(documents)}件の文書でコーパスを構築")
    return document_store, vector_store, embedder, str(dict_file), temp_dir
```

## ステップ2: 基本的なQueryEngine作成

基本設定でQueryEngineを作成し、動作を確認：

```python
def create_basic_query_engine(document_store, vector_store, embedder):
    """基本的なQueryEngineを作成"""
    
    from refinire_rag.use_cases.query_engine import QueryEngine, QueryEngineConfig
    from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
    
    print("🤖 基本QueryEngineを作成中...")
    
    # コンポーネント作成
    retriever = SimpleRetriever(vector_store, embedder=embedder)
    reranker = SimpleReranker()
    reader = SimpleReader()
    
    # 基本設定
    config = QueryEngineConfig(
        enable_query_normalization=True,
        auto_detect_corpus_state=True,
        retriever_top_k=10,
        reranker_top_k=3,
        include_sources=True,
        include_confidence=True
    )
    
    # QueryEngine作成
    query_engine = QueryEngine(
        document_store=document_store,
        vector_store=vector_store,
        retriever=retriever,
        reader=reader,
        reranker=reranker,
        config=config
    )
    
    print("✅ 基本QueryEngineを作成")
    return query_engine
```

## ステップ3: クエリ正規化のテスト

表現揺らぎを含むクエリで正規化効果を確認：

```python
def test_query_normalization(query_engine, dict_path):
    """クエリ正規化効果をテスト"""
    
    from refinire_rag.processing.normalizer import Normalizer, NormalizerConfig
    
    print("\\n" + "="*60)
    print("🔄 クエリ正規化テスト")
    print("="*60)
    
    # 手動で正規化機能を設定（自動検出が動作しない場合の対応）
    normalizer_config = NormalizerConfig(
        dictionary_file_path=dict_path,
        normalize_variations=True,
        expand_abbreviations=True,
        whole_word_only=False  # 日本語対応
    )
    query_engine.normalizer = Normalizer(normalizer_config)
    query_engine.corpus_state = {
        "has_normalization": True,
        "dictionary_path": dict_path
    }
    
    # 表現揺らぎを含むテストクエリ
    test_queries = [
        {
            "query": "検索強化生成について教えて",
            "expected": "検索拡張生成について教えて",
            "description": "検索強化生成 → 検索拡張生成"
        },
        {
            "query": "意味検索の仕組みは？",
            "expected": "ベクトル検索の仕組みは？",
            "description": "意味検索 → ベクトル検索"
        },
        {
            "query": "文書分割の方法を説明して",
            "expected": "チャンキングの方法を説明して",
            "description": "文書分割 → チャンキング"
        },
        {
            "query": "LLMモデルの評価指標は？",
            "expected": "大規模言語モデルの評価指標は？",
            "description": "LLMモデル → 大規模言語モデル"
        }
    ]
    
    print("📝 正規化前後の比較:")
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        description = test_case["description"]
        
        print(f"\\n📌 テスト {i}: {description}")
        print(f"   元クエリ: 「{query}」")
        
        try:
            result = query_engine.answer(query)
            
            normalized = result.metadata.get("query_normalized", False)
            normalized_query = result.normalized_query
            
            if normalized and normalized_query:
                print(f"   ✅ 正規化後: 「{normalized_query}」")
                print(f"   🔄 適用: 成功")
            else:
                print(f"   ❌ 正規化: 未適用")
            
            print(f"   📊 検索結果: {result.metadata.get('source_count', 0)}件")
            print(f"   ⏱️ 処理時間: {result.metadata.get('processing_time', 0):.3f}秒")
            
        except Exception as e:
            print(f"   ❌ エラー: {e}")
```

## ステップ4: コンポーネント別設定の最適化

各コンポーネントの設定を調整して性能を向上：

```python
def demo_component_optimization(document_store, vector_store, embedder):
    """コンポーネント最適化のデモ"""
    
    from refinire_rag.retrieval import (
        SimpleRetrieverConfig, SimpleRerankerConfig, SimpleReaderConfig
    )
    
    print("\\n" + "="*60)
    print("⚙️ コンポーネント最適化デモ")
    print("="*60)
    
    # 最適化された設定
    print("🔧 最適化設定を適用中...")
    
    # Retriever設定
    retriever_config = SimpleRetrieverConfig(
        top_k=15,  # より多くの候補を検索
        similarity_threshold=0.1,  # 低い閾値で幅広く検索
        embedding_model="tfidf-optimized"
    )
    
    # Reranker設定
    reranker_config = SimpleRerankerConfig(
        top_k=5,  # 上位5件に絞り込み
        boost_exact_matches=True,  # 完全一致にボーナス
        length_penalty_factor=0.1  # 適切な長さを優遇
    )
    
    # Reader設定
    reader_config = SimpleReaderConfig(
        max_context_length=2500,  # より多くの文脈を使用
        llm_model="gpt-4o-mini",
        temperature=0.1,  # 一貫性を重視
        max_tokens=600,  # より詳細な回答
        include_sources=True
    )
    
    # 最適化されたコンポーネント作成
    optimized_retriever = SimpleRetriever(vector_store, embedder=embedder, config=retriever_config)
    optimized_reranker = SimpleReranker(config=reranker_config)
    optimized_reader = SimpleReader(config=reader_config)
    
    # QueryEngine設定
    engine_config = QueryEngineConfig(
        enable_query_normalization=True,
        auto_detect_corpus_state=True,
        retriever_top_k=15,
        reranker_top_k=5,
        include_sources=True,
        include_confidence=True,
        include_processing_metadata=True
    )
    
    # 最適化QueryEngine作成
    optimized_engine = QueryEngine(
        document_store=document_store,
        vector_store=vector_store,
        retriever=optimized_retriever,
        reader=optimized_reader,
        reranker=optimized_reranker,
        config=engine_config
    )
    
    print("✅ 最適化QueryEngineを作成")
    
    # 性能比較テスト
    test_queries = [
        "RAGシステムの技術的な仕組みを詳しく説明してください",
        "ベクトル検索とキーワード検索の違いは何ですか？",
        "効果的なチャンキング戦略について教えて",
        "LLMの評価方法とその課題は？"
    ]
    
    print("\\n📊 最適化効果の比較:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\n📌 クエリ {i}: {query}")
        print("-" * 50)
        
        try:
            result = optimized_engine.answer(query)
            
            print(f"🤖 回答:")
            # 回答を見やすく表示
            answer_lines = result.answer.split('\\n')
            for line in answer_lines:
                if line.strip():
                    print(f"   {line.strip()}")
            
            print(f"\\n📊 詳細統計:")
            print(f"   - 検索文書数: {result.metadata.get('source_count', 0)}")
            print(f"   - 信頼度: {result.confidence:.3f}")
            print(f"   - 処理時間: {result.metadata.get('processing_time', 0):.3f}秒")
            
            # ソース情報
            if result.sources:
                print(f"   - 主要ソース:")
                for j, source in enumerate(result.sources[:3], 1):
                    title = source.metadata.get('title', f'Document {source.document_id}')
                    print(f"     {j}. {title} (スコア: {source.score:.3f})")
            
            # コンポーネント統計
            metadata = result.metadata
            if metadata.get('include_processing_metadata'):
                print(f"   - リランカー使用: {'Yes' if metadata.get('reranker_used') else 'No'}")
                if 'retrieval_stats' in metadata:
                    ret_stats = metadata['retrieval_stats']
                    print(f"   - 検索統計: {ret_stats.get('queries_processed', 0)}クエリ処理済み")
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            import traceback
            traceback.print_exc()
    
    return optimized_engine
```

## ステップ5: 回答品質の評価

生成された回答の品質を評価：

```python
def evaluate_answer_quality(query_engine):
    """回答品質の評価"""
    
    print("\\n" + "="*60)
    print("📈 回答品質評価")
    print("="*60)
    
    # 評価用クエリセット
    evaluation_queries = [
        {
            "query": "RAGの主な利点は何ですか？",
            "category": "基本知識",
            "expected_keywords": ["ハルシネーション", "知識更新", "外部データ"]
        },
        {
            "query": "ベクトル検索の計算方法を教えて",
            "category": "技術詳細", 
            "expected_keywords": ["コサイン類似度", "高次元", "ベクトル空間"]
        },
        {
            "query": "効果的なチャンキング戦略とは？",
            "category": "実践応用",
            "expected_keywords": ["チャンクサイズ", "オーバーラップ", "文脈"]
        }
    ]
    
    print("📝 評価結果:")
    
    total_score = 0
    for i, eval_case in enumerate(evaluation_queries, 1):
        query = eval_case["query"]
        category = eval_case["category"]
        expected_keywords = eval_case["expected_keywords"]
        
        print(f"\\n📌 評価 {i}: {category}")
        print(f"   クエリ: {query}")
        
        try:
            result = query_engine.answer(query)
            answer = result.answer
            
            # キーワード含有率チェック
            keyword_score = 0
            found_keywords = []
            
            for keyword in expected_keywords:
                if keyword in answer:
                    keyword_score += 1
                    found_keywords.append(keyword)
            
            keyword_ratio = keyword_score / len(expected_keywords)
            
            # 回答長チェック
            answer_length = len(answer.strip())
            length_score = 1.0 if 50 <= answer_length <= 500 else 0.5
            
            # ソース活用度チェック
            source_score = min(result.metadata.get('source_count', 0) / 3.0, 1.0)
            
            # 総合スコア計算
            overall_score = (keyword_ratio * 0.4 + length_score * 0.3 + source_score * 0.3)
            total_score += overall_score
            
            print(f"   🤖 回答: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"   📊 評価:")
            print(f"     - キーワード含有: {keyword_score}/{len(expected_keywords)} ({keyword_ratio:.1%})")
            print(f"     - 回答長: {answer_length}文字 (適切: {'Yes' if length_score == 1.0 else 'No'})")
            print(f"     - ソース活用: {result.metadata.get('source_count', 0)}件")
            print(f"     - 総合スコア: {overall_score:.2f}/1.00")
            
            if found_keywords:
                print(f"     - 発見キーワード: {', '.join(found_keywords)}")
        
        except Exception as e:
            print(f"   ❌ 評価エラー: {e}")
    
    # 全体評価
    average_score = total_score / len(evaluation_queries)
    print(f"\\n🏆 全体評価結果:")
    print(f"   - 平均スコア: {average_score:.2f}/1.00")
    
    if average_score >= 0.8:
        print(f"   - 評価: 優秀 🌟")
    elif average_score >= 0.6:
        print(f"   - 評価: 良好 👍")
    elif average_score >= 0.4:
        print(f"   - 評価: 改善必要 📈")
    else:
        print(f"   - 評価: 要大幅改善 🔧")
```

## ステップ6: 統計情報の分析

QueryEngineの詳細統計を分析：

```python
def analyze_engine_statistics(query_engine):
    """エンジン統計の分析"""
    
    print("\\n" + "="*60)
    print("📊 エンジン統計分析")
    print("="*60)
    
    # 統計情報取得
    stats = query_engine.get_engine_stats()
    
    # 基本統計
    print("📈 基本統計:")
    print(f"   - 処理クエリ数: {stats.get('queries_processed', 0)}")
    print(f"   - 正規化クエリ数: {stats.get('queries_normalized', 0)}")
    print(f"   - 平均応答時間: {stats.get('average_response_time', 0):.3f}秒")
    print(f"   - 平均検索件数: {stats.get('average_retrieval_count', 0):.1f}")
    
    # 正規化率
    total_queries = stats.get('queries_processed', 1)
    normalized_queries = stats.get('queries_normalized', 0)
    normalization_rate = normalized_queries / total_queries * 100
    print(f"   - クエリ正規化率: {normalization_rate:.1f}%")
    
    # コンポーネント統計
    print("\\n🔧 コンポーネント統計:")
    
    components = ['retriever_stats', 'reranker_stats', 'reader_stats']
    for component in components:
        if component in stats:
            comp_stats = stats[component]
            comp_name = component.replace('_stats', '').title()
            
            print(f"   {comp_name}:")
            print(f"     - 処理回数: {comp_stats.get('queries_processed', 0)}")
            print(f"     - 処理時間: {comp_stats.get('processing_time', 0):.3f}秒")
            print(f"     - エラー数: {comp_stats.get('errors_encountered', 0)}")
    
    # 設定情報
    print("\\n⚙️ 設定情報:")
    config_info = stats.get('config', {})
    for key, value in config_info.items():
        print(f"   - {key}: {value}")
    
    # コーパス状態
    print("\\n📚 コーパス状態:")
    corpus_state = stats.get('corpus_state', {})
    for key, value in corpus_state.items():
        print(f"   - {key}: {value}")
```

## 完全なサンプルプログラム

```python
#!/usr/bin/env python3
"""
チュートリアル3: クエリエンジンと回答生成
"""

def main():
    """メイン関数"""
    
    print("🚀 クエリエンジンと回答生成 チュートリアル")
    print("="*60)
    print("QueryEngineを使用した高度なクエリ処理と回答生成を学習します")
    
    try:
        # ステップ1: 正規化コーパス準備
        print("\\n📚 ステップ1: 正規化コーパス準備")
        document_store, vector_store, embedder, dict_path, temp_dir = setup_normalized_corpus()
        
        # ステップ2: 基本QueryEngine作成
        print("\\n🤖 ステップ2: 基本QueryEngine作成")
        basic_engine = create_basic_query_engine(document_store, vector_store, embedder)
        
        # ステップ3: クエリ正規化テスト
        test_query_normalization(basic_engine, dict_path)
        
        # ステップ4: コンポーネント最適化
        optimized_engine = demo_component_optimization(document_store, vector_store, embedder)
        
        # ステップ5: 回答品質評価
        evaluate_answer_quality(optimized_engine)
        
        # ステップ6: 統計分析
        analyze_engine_statistics(optimized_engine)
        
        print("\\n🎉 チュートリアル3が完了しました！")
        print("\\n📚 学習内容:")
        print("   ✅ QueryEngineの基本アーキテクチャ")
        print("   ✅ クエリ正規化による検索精度向上")
        print("   ✅ Retriever/Reranker/Readerの最適化")
        print("   ✅ 回答品質の定量的評価")
        print("   ✅ 統計情報による性能分析")
        
    except Exception as e:
        print(f"\\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
```

このチュートリアルにより、QueryEngineの高度な機能を活用した効果的なRAGシステムの構築方法を学習できます。次は[チュートリアル4: 高度な正規化とクエリ処理](tutorial_04_normalization.md)で、さらに詳細な正規化技術を学習しましょう。