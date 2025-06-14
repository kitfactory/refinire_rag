#!/usr/bin/env python3
"""
チュートリアル4: 高度な正規化とクエリ処理

このサンプルでは、refinire-ragの高度な正規化機能と、
クエリ処理の最適化技術を実装します。
"""

import sys
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.processing.normalizer import Normalizer, NormalizerConfig
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
from refinire_rag.embedding import TFIDFEmbedder, TFIDFEmbeddingConfig
from refinire_rag.models.document import Document
from refinire_rag.storage.vector_store import VectorEntry


def create_comprehensive_dictionary(temp_dir: Path):
    """包括的な正規化辞書を作成"""
    
    dict_file = temp_dir / "comprehensive_dictionary.md"
    dict_file.write_text("""# 包括的な正規化辞書

## AI・機械学習用語

### 核心技術
- **RAG** (Retrieval-Augmented Generation): 検索拡張生成
  - 表現揺らぎ: 検索拡張生成, 検索強化生成, 検索増強生成, RAGシステム, RAG技術, 検索拡張技術
  - 略語: RAG, R.A.G.

- **LLM** (Large Language Model): 大規模言語モデル
  - 表現揺らぎ: 大規模言語モデル, 言語モデル, LLMモデル, 大規模LM, 大型言語モデル
  - 略語: LLM, L.L.M.

- **NLP** (Natural Language Processing): 自然言語処理
  - 表現揺らぎ: 自然言語処理, 言語処理, NLP処理, 自然言語解析
  - 略語: NLP, N.L.P.

### 検索技術
- **ベクトル検索** (Vector Search): ベクトル検索
  - 表現揺らぎ: ベクトル検索, 意味検索, セマンティック検索, 意味的検索, ベクトル類似度検索
  - 関連技術: 埋め込み検索, エンベディング検索

- **類似度計算** (Similarity Calculation): 類似度計算
  - 表現揺らぎ: 類似度計算, 類似性計算, 類似度測定, 類似性評価
  - 具体的手法: コサイン類似度, ユークリッド距離, 内積計算

### 文書処理
- **チャンキング** (Chunking): チャンキング
  - 表現揺らぎ: チャンキング, 文書分割, テキスト分割, チャンク化, 文書セグメンテーション
  - 関連技術: 文書分割, セグメント分割

- **埋め込み** (Embedding): 埋め込み
  - 表現揺らぎ: 埋め込み, エンベディング, ベクトル表現, ベクトル化, 特徴量抽出

## 評価・品質
- **ハルシネーション** (Hallucination): ハルシネーション
  - 表現揺らぎ: ハルシネーション, 幻覚, 虚偽生成, 誤情報生成, 捏造

- **BLEU** (BLEU Score): BLEU
  - 表現揺らぎ: BLEU, BLEUスコア, BLEU評価, ブルースコア

## 技術設定パターン

### 日本語固有パターン
- **文字種変換**: ひらがな↔カタカナ
  - AI → エーアイ, AI → エイアイ
  - ML → エムエル, ML → マシンラーニング

### 英数字表記揺らぎ
- **英字大小文字**: OpenAI ↔ openai ↔ OPENAI
- **数字表記**: 1つ ↔ 一つ ↔ ひとつ
- **単位表記**: 10MB ↔ 10メガバイト ↔ 10メガB

### 送り仮名・語尾変化
- **動詞活用**: 行う ↔ 行なう, 表す ↔ 表わす
- **形容詞語尾**: 新しい ↔ 新らしい, 正しい ↔ 正かしい
""", encoding='utf-8')
    
    return str(dict_file)


def create_domain_specific_dictionaries(temp_dir: Path):
    """ドメイン特化辞書を作成"""
    
    # 医療ドメイン辞書
    medical_dict = temp_dir / "medical_dictionary.md"
    medical_dict.write_text("""# 医療ドメイン特化辞書

## 医療AI用語
- **診断支援AI** (Diagnostic AI): 診断支援AI
  - 表現揺らぎ: 診断支援AI, AI診断, 診断AI, 医療診断AI, コンピュータ診断

- **画像診断** (Medical Imaging): 画像診断
  - 表現揺らぎ: 画像診断, 医用画像, 放射線診断, イメージング診断

## 疾患名
- **COVID-19** (COVID-19): COVID-19
  - 表現揺らぎ: COVID-19, Covid-19, コロナ, 新型コロナ, SARS-CoV-2
""", encoding='utf-8')
    
    # 金融ドメイン辞書
    finance_dict = temp_dir / "finance_dictionary.md"
    finance_dict.write_text("""# 金融ドメイン特化辞書

## 金融AI用語
- **ロボアドバイザー** (Robo-advisor): ロボアドバイザー
  - 表現揺らぎ: ロボアドバイザー, ロボ・アドバイザー, AI投資顧問, 自動投資

- **リスク管理** (Risk Management): リスク管理
  - 表現揺らぎ: リスク管理, リスクマネジメント, 危険管理, リスク制御
""", encoding='utf-8')
    
    return str(medical_dict), str(finance_dict)


def demo_advanced_normalization_configs(temp_dir: Path):
    """高度な正規化設定のデモ"""
    
    print("\n" + "="*60)
    print("🔧 高度な正規化設定デモ")
    print("="*60)
    
    # 包括辞書作成
    dict_path = create_comprehensive_dictionary(temp_dir)
    
    # 異なる設定での正規化比較
    configs = [
        {
            "name": "基本設定",
            "config": NormalizerConfig(
                dictionary_file_path=dict_path,
                normalize_variations=True,
                expand_abbreviations=True,
                whole_word_only=False
            )
        },
        {
            "name": "厳密マッチ設定",
            "config": NormalizerConfig(
                dictionary_file_path=dict_path,
                normalize_variations=True,
                expand_abbreviations=True,
                whole_word_only=True
            )
        },
        {
            "name": "部分マッチ設定",
            "config": NormalizerConfig(
                dictionary_file_path=dict_path,
                normalize_variations=True,
                expand_abbreviations=True,
                whole_word_only=False
            )
        }
    ]
    
    # テスト文書
    test_texts = [
        "検索強化生成とRAGシステムについて説明します",
        "LLMモデルの性能をBLEUスコアで評価",
        "セマンティック検索とベクトル類似度計算",
        "AI診断支援システムの開発",
        "コサイン類似度を使った意味検索の実装"
    ]
    
    print("📝 設定別正規化結果比較:")
    print("-" * 70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📌 テスト文 {i}: 「{text}」")
        print("-" * 50)
        
        for config_info in configs:
            config_name = config_info["name"]
            normalizer = Normalizer(config_info["config"])
            
            doc = Document(id=f"test_{i}", content=text, metadata={})
            normalized_docs = normalizer.process(doc)
            
            if normalized_docs:
                normalized_text = normalized_docs[0].content
                changes = "✅変更あり" if text != normalized_text else "🔄変更なし"
                
                print(f"  {config_name:12}: 「{normalized_text}」 {changes}")
            else:
                print(f"  {config_name:12}: ❌ 処理失敗")


def demo_query_normalization_optimization(temp_dir: Path):
    """クエリ正規化最適化のデモ"""
    
    print("\n" + "="*60)
    print("🎯 クエリ正規化最適化デモ")
    print("="*60)
    
    # 辞書とコーパス準備
    dict_path = create_comprehensive_dictionary(temp_dir)
    
    # 正規化されたコーパスを作成
    documents = [
        Document(
            id="doc1",
            content="""
            検索拡張生成（RAG）は、大規模言語モデルの能力を向上させる
            革新的な技術です。外部知識ベースからの情報検索と
            言語生成を組み合わせることで、より正確で
            根拠のある回答を生成できます。
            """,
            metadata={"title": "RAG技術解説", "domain": "AI"}
        ),
        Document(
            id="doc2", 
            content="""
            ベクトル検索は、文書の意味的類似性を
            コサイン類似度や内積計算で評価する技術です。
            従来のキーワード検索では発見できない
            関連情報を見つけることができます。
            """,
            metadata={"title": "ベクトル検索技術", "domain": "Search"}
        ),
        Document(
            id="doc3",
            content="""
            チャンキングは、長い文書を検索しやすい
            小さな単位に分割する前処理技術です。
            適切なチャンクサイズとオーバーラップにより、
            文脈を保持しつつ効率的な検索を実現します。
            """,
            metadata={"title": "文書処理技術", "domain": "NLP"}
        )
    ]
    
    # ストレージ初期化
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    
    # 正規化器設定
    normalizer_config = NormalizerConfig(
        dictionary_file_path=dict_path,
        normalize_variations=True,
        expand_abbreviations=True,
        whole_word_only=False
    )
    normalizer = Normalizer(normalizer_config)
    
    # コーパス構築（正規化適用）
    normalized_docs = []
    for doc in documents:
        # 原文保存
        doc_store.store_document(doc)
        
        # 正規化版作成
        norm_results = normalizer.process(doc)
        if norm_results:
            norm_doc = norm_results[0]
            norm_doc.metadata["processing_stage"] = "normalized"
            doc_store.store_document(norm_doc)
            normalized_docs.append(norm_doc)
    
    # ベクトル化
    embedder_config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0)
    embedder = TFIDFEmbedder(config=embedder_config)
    
    corpus_texts = [doc.content for doc in normalized_docs]
    embedder.fit(corpus_texts)
    
    for doc in normalized_docs:
        embedding_result = embedder.embed_text(doc.content)
        vector_entry = VectorEntry(
            document_id=doc.id,
            content=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
            embedding=embedding_result.vector.tolist(),
            metadata=doc.metadata
        )
        vector_store.add_vector(vector_entry)
    
    # QueryEngine設定（正規化あり/なし比較）
    retriever = SimpleRetriever(vector_store, embedder=embedder)
    reranker = SimpleReranker()
    reader = SimpleReader()
    
    # 正規化なしエンジン
    engine_no_norm = QueryEngine(
        document_store=doc_store,
        vector_store=vector_store,
        retriever=retriever,
        reader=reader,
        reranker=reranker,
        config=QueryEngineConfig(
            enable_query_normalization=False,
            retriever_top_k=5,
            include_sources=True
        )
    )
    
    # 正規化ありエンジン
    engine_with_norm = QueryEngine(
        document_store=doc_store,
        vector_store=vector_store,
        retriever=retriever,
        reader=reader,
        reranker=reranker,
        config=QueryEngineConfig(
            enable_query_normalization=True,
            retriever_top_k=5,
            include_sources=True
        )
    )
    
    # 手動で正規化器を設定
    engine_with_norm.normalizer = normalizer
    engine_with_norm.corpus_state = {
        "has_normalization": True,
        "dictionary_path": dict_path
    }
    
    # 表現揺らぎを含むテストクエリ
    test_queries = [
        {
            "query": "検索強化生成の仕組みを教えて",
            "expected_improvement": "検索強化生成 → 検索拡張生成"
        },
        {
            "query": "意味検索とはどのような技術？",
            "expected_improvement": "意味検索 → ベクトル検索"
        },
        {
            "query": "文書分割の最適化方法は？",
            "expected_improvement": "文書分割 → チャンキング"
        },
        {
            "query": "LLMモデルとRAGシステムの関係",
            "expected_improvement": "LLMモデル → 大規模言語モデル"
        }
    ]
    
    print("📊 正規化効果比較:")
    print("-" * 70)
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected = test_case["expected_improvement"]
        
        print(f"\n📌 クエリ {i}: 「{query}」")
        print(f"   期待する正規化: {expected}")
        print("-" * 50)
        
        try:
            # 正規化なし結果
            result_no_norm = engine_no_norm.answer(query)
            
            # 正規化あり結果
            result_with_norm = engine_with_norm.answer(query)
            
            print(f"🔍 正規化なし:")
            print(f"   - 検索結果数: {result_no_norm.metadata.get('source_count', 0)}")
            print(f"   - 信頼度: {result_no_norm.confidence:.3f}")
            if result_no_norm.sources:
                print(f"   - 主要ソース: {result_no_norm.sources[0].metadata.get('title', 'Unknown')}")
            
            print(f"✨ 正規化あり:")
            normalized_query = result_with_norm.metadata.get('normalized_query', query)
            print(f"   - 正規化後クエリ: 「{normalized_query}」")
            print(f"   - 検索結果数: {result_with_norm.metadata.get('source_count', 0)}")
            print(f"   - 信頼度: {result_with_norm.confidence:.3f}")
            if result_with_norm.sources:
                print(f"   - 主要ソース: {result_with_norm.sources[0].metadata.get('title', 'Unknown')}")
            
            # 改善効果評価
            improvement = result_with_norm.confidence - result_no_norm.confidence
            if improvement > 0.05:
                print(f"   📈 信頼度改善: +{improvement:.3f} (大幅改善)")
            elif improvement > 0.01:
                print(f"   📈 信頼度改善: +{improvement:.3f} (改善)")
            elif improvement > -0.01:
                print(f"   ➡️ 信頼度変化: {improvement:+.3f} (変化なし)")
            else:
                print(f"   📉 信頼度低下: {improvement:.3f} (要調整)")
                
        except Exception as e:
            print(f"   ❌ エラー: {e}")
    
    return engine_with_norm


def demo_normalization_performance_optimization(temp_dir: Path):
    """正規化パフォーマンス最適化のデモ"""
    
    print("\n" + "="*60)
    print("⚡ 正規化パフォーマンス最適化")
    print("="*60)
    
    # 大規模辞書作成
    dict_path = create_comprehensive_dictionary(temp_dir)
    
    # 大量のテストドキュメント作成
    test_documents = []
    for i in range(50):  # 100から50に削減してテスト時間を短縮
        content = f"""
        テスト文書 {i}: 検索強化生成とLLMモデルについて。
        意味検索とベクトル類似度計算を使用。
        文書分割とチャンキングの最適化。
        AI診断支援システムの開発案件。
        """
        test_documents.append(Document(
            id=f"perf_test_{i}",
            content=content,
            metadata={"test_id": i}
        ))
    
    # 異なる最適化設定をテスト
    optimization_configs = [
        {
            "name": "基本設定",
            "config": NormalizerConfig(
                dictionary_file_path=dict_path,
                normalize_variations=True,
                expand_abbreviations=True,
                whole_word_only=False
            ),
            "parallel": False
        },
        {
            "name": "高速設定",
            "config": NormalizerConfig(
                dictionary_file_path=dict_path,
                normalize_variations=True,
                expand_abbreviations=True,
                whole_word_only=False
            ),
            "parallel": False
        },
        {
            "name": "並列処理",
            "config": NormalizerConfig(
                dictionary_file_path=dict_path,
                normalize_variations=True,
                expand_abbreviations=True,
                whole_word_only=False
            ),
            "parallel": True
        }
    ]
    
    print("📊 パフォーマンス比較結果:")
    print("-" * 60)
    
    results = {}
    
    for config_info in optimization_configs:
        config_name = config_info["name"]
        normalizer_config = config_info["config"]
        use_parallel = config_info["parallel"]
        
        print(f"\n🔧 {config_name}:")
        
        normalizer = Normalizer(normalizer_config)
        
        start_time = time.time()
        
        if use_parallel:
            # 並列処理で正規化
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(normalizer.process, doc) for doc in test_documents]
                normalized_results = [future.result() for future in futures]
        else:
            # 逐次処理で正規化
            normalized_results = [normalizer.process(doc) for doc in test_documents]
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 結果統計
        successful_docs = sum(1 for result in normalized_results if result)
        total_changes = 0
        
        for result in normalized_results:
            if result and len(result) > 0:
                original_doc = test_documents[normalized_results.index(result)]
                if result[0].content != original_doc.content:
                    total_changes += 1
        
        results[config_name] = {
            "time": processing_time,
            "successful": successful_docs,
            "changes": total_changes,
            "throughput": len(test_documents) / processing_time
        }
        
        print(f"   ⏱️ 処理時間: {processing_time:.3f}秒")
        print(f"   📊 成功率: {successful_docs}/{len(test_documents)} ({successful_docs/len(test_documents):.1%})")
        print(f"   🔄 変更文書数: {total_changes}")
        print(f"   🚀 スループット: {len(test_documents)/processing_time:.1f} docs/sec")
    
    # 最適化効果サマリー
    print(f"\n📈 最適化効果サマリー:")
    baseline_time = results["基本設定"]["time"]
    
    for config_name, result in results.items():
        if config_name != "基本設定":
            speedup = baseline_time / result["time"]
            print(f"   {config_name}: {speedup:.2f}x高速化")
    
    return results


def demo_domain_specific_normalization(temp_dir: Path):
    """ドメイン特化正規化のデモ"""
    
    print("\n" + "="*60)
    print("🏥 ドメイン特化正規化デモ")
    print("="*60)
    
    # ドメイン特化辞書作成
    medical_dict, finance_dict = create_domain_specific_dictionaries(temp_dir)
    general_dict = create_comprehensive_dictionary(temp_dir)
    
    # 各ドメインのテストケース
    domain_tests = [
        {
            "domain": "医療",
            "dict_path": medical_dict,
            "texts": [
                "COVID-19の診断支援AIシステム",
                "コロナウイルスの画像診断技術",
                "新型コロナの医用画像解析"
            ]
        },
        {
            "domain": "金融", 
            "dict_path": finance_dict,
            "texts": [
                "ロボ・アドバイザーによる投資支援",
                "AI投資顧問のリスクマネジメント",
                "自動投資システムの危険管理"
            ]
        },
        {
            "domain": "一般AI",
            "dict_path": general_dict,
            "texts": [
                "検索強化生成の実装方法",
                "LLMモデルと意味検索の組み合わせ",
                "文書分割の最適化技術"
            ]
        }
    ]
    
    print("📝 ドメイン別正規化結果:")
    print("-" * 50)
    
    for domain_test in domain_tests:
        domain = domain_test["domain"]
        dict_path = domain_test["dict_path"]
        texts = domain_test["texts"]
        
        print(f"\n🏷️ {domain}ドメイン:")
        print("-" * 30)
        
        # ドメイン特化正規化器
        normalizer_config = NormalizerConfig(
            dictionary_file_path=dict_path,
            normalize_variations=True,
            expand_abbreviations=True,
            whole_word_only=False
        )
        normalizer = Normalizer(normalizer_config)
        
        for i, text in enumerate(texts, 1):
            print(f"\n📌 テスト {i}:")
            print(f"   元文: 「{text}」")
            
            try:
                doc = Document(id=f"{domain}_{i}", content=text, metadata={})
                normalized_docs = normalizer.process(doc)
                
                if normalized_docs:
                    normalized_text = normalized_docs[0].content
                    if text != normalized_text:
                        print(f"   正規化後: 「{normalized_text}」")
                        print(f"   🔄 変更: あり")
                    else:
                        print(f"   🔄 変更: なし")
                else:
                    print(f"   ❌ 正規化失敗")
            except Exception as e:
                print(f"   ❌ エラー: {e}")
    
    # 複数辞書の統合テスト
    print(f"\n🔗 複数辞書統合テスト:")
    print("-" * 30)
    
    # 統合辞書作成
    combined_dict = temp_dir / "combined_dictionary.md"
    
    # 各辞書の内容を読み込んで統合
    combined_content = "# 統合辞書\n\n"
    
    for dict_path in [general_dict, medical_dict, finance_dict]:
        with open(dict_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # ヘッダーを調整
            content = content.replace("# ", "## ")
            combined_content += content + "\n\n"
    
    combined_dict.write_text(combined_content, encoding='utf-8')
    
    # 統合辞書での正規化テスト
    combined_normalizer = Normalizer(NormalizerConfig(
        dictionary_file_path=str(combined_dict),
        normalize_variations=True,
        expand_abbreviations=True,
        whole_word_only=False
    ))
    
    cross_domain_texts = [
        "COVID-19診断のRAGシステム開発",
        "金融AIとLLMモデルの統合",
        "医療画像のセマンティック検索"
    ]
    
    for i, text in enumerate(cross_domain_texts, 1):
        print(f"\n📌 クロスドメインテスト {i}:")
        print(f"   元文: 「{text}」")
        
        try:
            doc = Document(id=f"cross_{i}", content=text, metadata={})
            normalized_docs = combined_normalizer.process(doc)
            
            if normalized_docs:
                normalized_text = normalized_docs[0].content
                print(f"   統合正規化: 「{normalized_text}」")
                
                changes = []
                if "COVID-19" in text and "COVID-19" in normalized_text:
                    changes.append("医療用語正規化")
                if "RAG" in text:
                    changes.append("AI用語正規化")
                
                if changes:
                    print(f"   🎯 適用領域: {', '.join(changes)}")
                else:
                    print(f"   🔄 変更なし")
            else:
                print(f"   ❌ 正規化失敗")
        except Exception as e:
            print(f"   ❌ エラー: {e}")


def main():
    """メイン関数"""
    
    print("🚀 高度な正規化とクエリ処理 チュートリアル")
    print("="*60)
    print("辞書ベース正規化の詳細設定とクエリ処理最適化を学習します")
    
    # 一時ディレクトリ作成
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # ステップ1: 高度な辞書設計
        print("\n📚 ステップ1: 高度な辞書設計")
        create_comprehensive_dictionary(temp_dir)
        create_domain_specific_dictionaries(temp_dir)
        print("✅ 包括的辞書とドメイン特化辞書を作成")
        
        # ステップ2: 正規化設定の詳細カスタマイズ
        demo_advanced_normalization_configs(temp_dir)
        
        # ステップ3: クエリ時正規化の最適化
        optimized_engine = demo_query_normalization_optimization(temp_dir)
        
        # ステップ4: パフォーマンス最適化
        perf_results = demo_normalization_performance_optimization(temp_dir)
        
        # ステップ5: ドメイン特化正規化
        demo_domain_specific_normalization(temp_dir)
        
        print("\n🎉 チュートリアル4が完了しました！")
        print("\n📚 学習内容:")
        print("   ✅ 包括的な正規化辞書の設計")
        print("   ✅ 正規化設定の詳細カスタマイズ")
        print("   ✅ クエリ時正規化による検索精度向上")
        print("   ✅ パフォーマンス最適化技術")
        print("   ✅ ドメイン特化正規化の実装")
        
        print(f"\n📁 生成ファイル: {temp_dir}")
        print("\n🚀 次のステップ:")
        print("   • チュートリアル5: カスタムパイプラインの構築")
        print("   • より柔軟なパイプライン設計")
        print("   • カスタムコンポーネントの開発")
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)