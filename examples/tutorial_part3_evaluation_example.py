#!/usr/bin/env python3
"""
Part 3: RAG Evaluation Tutorial Example
RAG評価チュートリアル例

This example demonstrates comprehensive RAG system evaluation using refinire-rag's QualityLab
with automated QA generation, performance evaluation, contradiction detection, and reporting.

この例では、refinire-ragのQualityLabを使用した包括的なRAGシステム評価を、
自動QA生成、パフォーマンス評価、矛盾検出、レポート作成とともに実演します。
"""

import sys
import tempfile
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
from refinire_rag.models.document import Document
from refinire_rag.models.qa_pair import QAPair
from refinire_rag.models.evaluation_result import EvaluationResult


def setup_evaluation_corpus(temp_dir: Path) -> tuple:
    """
    Set up a comprehensive corpus for evaluation demonstration
    評価デモ用の包括的コーパスをセットアップ
    """
    
    print("📚 Setting up evaluation corpus...")
    print("📚 評価用コーパスをセットアップ中...")
    
    # Create knowledge base directory / 知識ベースディレクトリを作成
    kb_dir = temp_dir / "evaluation_kb"
    kb_dir.mkdir(exist_ok=True)
    
    # AI Overview / AI概要
    (kb_dir / "ai_overview.txt").write_text("""
人工知能（AI）概要

人工知能（Artificial Intelligence, AI）は、人間の知的活動をコンピュータで模倣・実現する技術分野です。

## 主要分野
1. 機械学習（Machine Learning, ML）
   - データから自動的に学習し、予測や判断を行う技術
   - 代表的アルゴリズム：線形回帰、決定木、ニューラルネットワーク

2. 深層学習（Deep Learning, DL）
   - 多層ニューラルネットワークを用いた機械学習手法
   - CNN（畳み込みニューラルネットワーク）、RNN（再帰ニューラルネットワーク）

3. 自然言語処理（Natural Language Processing, NLP）
   - 人間の言語をコンピュータで理解・生成する技術
   - 機械翻訳、感情分析、質問応答システム

4. コンピュータビジョン（Computer Vision, CV）
   - 画像や動画から情報を抽出・理解する技術
   - 物体検出、顔認識、医療画像解析

## AIの歴史
- 1950年代：アラン・チューリングによるチューリングテストの提案
- 1960年代：エキスパートシステムの開発
- 1980年代：ニューラルネットワークの復活
- 2010年代：深層学習の台頭、大規模言語モデルの登場

## 現在の応用分野
- 自動運転車
- 医療診断支援
- 金融取引の自動化
- 音声アシスタント（Siri, Alexa等）
- 推薦システム（Netflix, Amazon等）

AIは急速に発展し続け、今後も人間社会に大きな影響を与えると予想されています。
""", encoding='utf-8')
    
    # Machine Learning Details / 機械学習詳細
    (kb_dir / "machine_learning.txt").write_text("""
機械学習詳細解説

機械学習は、明示的にプログラムされることなく、データから学習する能力をコンピュータに与える技術です。

## 学習方式の分類

### 1. 教師あり学習（Supervised Learning）
正解データ（ラベル）を用いて学習する方式。

#### 分類（Classification）
- 目的：データを予め定義されたカテゴリに分類
- 例：スパムメール検出、画像認識、医療診断
- アルゴリズム：
  * ロジスティック回帰
  * サポートベクターマシン（SVM）
  * ランダムフォレスト
  * 勾配ブースティング

#### 回帰（Regression）
- 目的：連続値の予測
- 例：株価予測、気温予測、売上予測
- アルゴリズム：
  * 線形回帰
  * 多項式回帰
  * リッジ回帰、ラッソ回帰

### 2. 教師なし学習（Unsupervised Learning）
正解データなしでデータの構造やパターンを発見。

#### クラスタリング
- 目的：類似したデータをグループ化
- 例：顧客セグメンテーション、遺伝子分析
- アルゴリズム：
  * K-means
  * 階層クラスタリング
  * DBSCAN

#### 次元削減
- 目的：データの次元を減らしつつ重要な情報を保持
- 例：データ可視化、特徴選択
- アルゴリズム：
  * 主成分分析（PCA）
  * t-SNE
  * UMAP

### 3. 強化学習（Reinforcement Learning）
環境との相互作用を通じて最適な行動を学習。

- 構成要素：エージェント、環境、行動、報酬
- 例：ゲームAI、ロボット制御、自動取引
- アルゴリズム：
  * Q学習
  * Deep Q Network (DQN)
  * Policy Gradient

## 機械学習のプロセス

1. データ収集・前処理
2. 特徴量エンジニアリング
3. モデル選択
4. 学習・訓練
5. 評価・検証
6. パラメータ調整
7. 本番運用

## 評価指標

### 分類問題
- 精度（Accuracy）
- 適合率（Precision）
- 再現率（Recall）
- F1スコア
- AUC-ROC

### 回帰問題
- 平均絶対誤差（MAE）
- 平均二乗誤差（MSE）
- 決定係数（R²）

機械学習は現代のAIの基盤技術として、様々な分野で活用されています。
""", encoding='utf-8')
    
    # Deep Learning / 深層学習
    (kb_dir / "deep_learning.txt").write_text("""
深層学習技術解説

深層学習（Deep Learning）は、人間の脳の神経回路を模倣した多層ニューラルネットワークを用いる機械学習手法です。

## ニューラルネットワークの基礎

### 基本構造
- ニューロン（ノード）：情報処理の基本単位
- 重み（Weight）：ニューロン間の接続強度
- バイアス（Bias）：活性化の閾値調整
- 活性化関数：非線形変換を提供

### 学習メカニズム
1. 順伝播（Forward Propagation）：入力から出力への情報伝達
2. 損失計算：予測値と実際値の差を計算
3. 逆伝播（Backpropagation）：誤差を逆向きに伝播
4. 勾配降下法：重みとバイアスを更新

## 主要なアーキテクチャ

### 1. 畳み込みニューラルネットワーク（CNN）
画像処理に特化したアーキテクチャ。

#### 構成要素
- 畳み込み層（Convolution Layer）：特徴抽出
- プーリング層（Pooling Layer）：ダウンサンプリング
- 全結合層（Fully Connected Layer）：分類

#### 応用
- 画像分類：ImageNet、CIFAR-10
- 物体検出：YOLO、R-CNN
- 医療画像：X線、MRI解析
- 自動運転：道路標識認識

### 2. 再帰ニューラルネットワーク（RNN）
時系列データ処理に適したアーキテクチャ。

#### 改良版
- LSTM（Long Short-Term Memory）：長期依存関係の学習
- GRU（Gated Recurrent Unit）：LSTMの簡素化版

#### 応用
- 自然言語処理：機械翻訳、文章生成
- 音声認識：音声からテキストへの変換
- 時系列予測：株価、気象予測

### 3. トランスフォーマー（Transformer）
注意機構（Attention Mechanism）を核とした革新的アーキテクチャ。

#### 特徴
- Self-Attention：文脈内の関係性を捉える
- 並列処理：RNNより高速な学習
- 長距離依存関係：より長い文脈を理解

#### 代表モデル
- BERT：双方向エンコーダー
- GPT：生成型デコーダー
- T5：Text-to-Text Transfer Transformer

## 深層学習の発展

### 生成モデル
- GAN（Generative Adversarial Networks）：画像生成
- VAE（Variational Autoencoder）：データ生成
- Diffusion Models：高品質画像生成

### 大規模言語モデル
- GPT-3/4：1750億〜1兆パラメータ
- PaLM：5400億パラメータ
- ChatGPT：対話型AI

## 深層学習の課題

### 技術的課題
- 大量のデータと計算資源が必要
- ブラックボックス問題：解釈可能性の欠如
- 過学習：汎化性能の低下
- 敵対的攻撃：微小な変更による誤分類

### 社会的課題
- バイアス：学習データの偏りによる差別
- プライバシー：個人情報の保護
- 雇用への影響：自動化による職業の変化
- エネルギー消費：大規模モデルの環境負荷

深層学習は現在のAIブームを牽引する中核技術として、急速な発展を続けています。
""", encoding='utf-8')
    
    # NLP Applications / NLP応用
    (kb_dir / "nlp_applications.txt").write_text("""
自然言語処理の応用

自然言語処理（NLP）は、人間の言語をコンピュータで理解・生成・操作する技術分野です。

## 基本的なNLPタスク

### 1. 前処理（Preprocessing）
- トークン化（Tokenization）：文章を単語や文に分割
- 正規化（Normalization）：表記の統一
- ストップワード除去：意味の薄い語の削除
- ステミング・レンマ化：語形の正規化

### 2. 形態素解析
- 品詞タグ付け（POS Tagging）：単語の品詞を識別
- 固有表現認識（NER）：人名、地名、組織名等の抽出
- 構文解析（Parsing）：文の文法構造を分析

## 主要なNLPアプリケーション

### 1. 機械翻訳（Machine Translation）
#### 発展の歴史
- 規則ベース翻訳（1950年代〜）
- 統計的機械翻訳（1990年代〜）
- ニューラル機械翻訳（2010年代〜）

#### 現代の手法
- Transformer Based：Google Translate, DeepL
- 多言語モデル：mBERT, XLM-R
- ゼロショット翻訳：直接ペアなしで翻訳

#### 評価指標
- BLEU Score：n-gramベースの類似度
- ROUGE Score：要約品質評価
- 人手評価：流暢さ、正確性

### 2. 感情分析（Sentiment Analysis）
#### 分析レベル
- 文書レベル：文書全体の感情
- 文レベル：各文の感情
- アスペクトレベル：特定観点の感情

#### 応用分野
- ソーシャルメディア分析
- 製品レビュー分析
- 顧客満足度調査
- 株式市場予測

### 3. 質問応答システム（Question Answering）
#### システム分類
- 抽出型QA：文書から該当箇所を抽出
- 生成型QA：回答を生成
- 知識ベースQA：構造化知識を活用

#### 代表的システム
- SQuAD：読解ベンチマーク
- Natural Questions：実世界の質問
- MS MARCO：大規模QAデータセット

### 4. 文書要約（Text Summarization）
#### 要約手法
- 抽出型：重要文を選択
- 生成型：新しい文を生成
- ハイブリッド：両者の組み合わせ

#### 要約の種類
- 単一文書要約
- 複数文書要約
- 更新要約：新情報の追加

### 5. 対話システム（Dialogue Systems）
#### システム分類
- タスク指向：特定業務の遂行
- 雑談型：自然な会話を目指す
- 質問応答型：情報提供に特化

#### 技術要素
- 自然言語理解（NLU）
- 対話管理（DM）
- 自然言語生成（NLG）

## 最新の大規模言語モデル

### GPTシリーズ
- GPT-1（2018）：1.17億パラメータ
- GPT-2（2019）：15億パラメータ
- GPT-3（2020）：1750億パラメータ
- GPT-4（2023）：推定1兆パラメータ

### BERTとその発展
- BERT（2018）：双方向エンコーダー
- RoBERTa：BERTの改良版
- DeBERTa：Disentangled Attention

### 日本語特化モデル
- 京都大学BERT
- 東北大学BERT
- rinna GPT
- Japanese T5

## NLPの評価と課題

### 評価指標
- 自動評価：BLEU, ROUGE, METEOR
- 人手評価：流暢さ、正確性、適切性
- タスク特化指標：F1, Exact Match

### 現在の課題
- バイアスと公平性
- 解釈可能性
- 多言語対応
- 計算コストの削減
- データ効率の向上

NLPは現在、大規模言語モデルの登場により革命的な進歩を遂げており、
今後も人間とコンピュータの自然な対話を実現する重要技術として発展し続けます。
""", encoding='utf-8')
    
    # Setup storage and build corpus / ストレージをセットアップしてコーパスを構築
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    
    # Build corpus with semantic RAG / セマンティックRAGでコーパス構築
    corpus_manager = CorpusManager.create_semantic_rag(doc_store, vector_store)
    file_paths = [str(f) for f in kb_dir.glob("*.txt")]
    stats = corpus_manager.build_corpus(file_paths)
    
    print(f"✅ Evaluation corpus setup completed:")
    print(f"   Files processed: {stats.total_files_processed}")
    print(f"   Documents created: {stats.total_documents_created}")
    print(f"   Chunks created: {stats.total_chunks_created}")
    print(f"   Processing time: {stats.total_processing_time:.3f}s")
    
    return doc_store, vector_store, file_paths


def setup_query_engine(doc_store, vector_store) -> QueryEngine:
    """
    Set up QueryEngine for evaluation
    評価用QueryEngineをセットアップ
    """
    
    print("\n🔍 Setting up QueryEngine for evaluation...")
    print("🔍 評価用QueryEngineをセットアップ中...")
    
    # Initialize QueryEngine with optimal settings / 最適設定でQueryEngineを初期化
    query_engine = QueryEngine(
        document_store=doc_store,
        vector_store=vector_store,
        retriever=SimpleRetriever(vector_store),
        reranker=SimpleReranker(),
        reader=SimpleReader(),
        config=QueryEngineConfig(
            enable_query_normalization=True,
            include_sources=True,
            include_confidence=True,
            max_response_time=30.0
        )
    )
    
    print("✅ QueryEngine setup completed")
    return query_engine


def demonstrate_qa_generation(quality_lab: QualityLab, documents: List[Document]):
    """
    Demonstrate automated QA pair generation
    自動QAペア生成のデモンストレーション
    """
    
    print("\n" + "="*60)
    print("📝 QA PAIR GENERATION DEMONSTRATION")
    print("📝 QAペア生成のデモンストレーション")
    print("="*60)
    
    # Generate QA pairs with different question types / 異なる質問タイプでQAペアを生成
    print("\n📌 Generating QA pairs with multiple question types...")
    print("📌 複数の質問タイプでQAペアを生成中...")
    
    qa_pairs = quality_lab.generate_qa_pairs(
        documents=documents,
        num_pairs=15,
        question_types=["factual", "conceptual", "analytical", "comparative"]
    )
    
    print(f"✅ Generated {len(qa_pairs)} QA pairs")
    
    # Analyze QA pair distribution / QAペア分布を分析
    type_distribution = {}
    difficulty_distribution = {}
    
    for qa_pair in qa_pairs:
        q_type = qa_pair.metadata.get('question_type', 'unknown')
        difficulty = qa_pair.metadata.get('difficulty', 'unknown')
        
        type_distribution[q_type] = type_distribution.get(q_type, 0) + 1
        difficulty_distribution[difficulty] = difficulty_distribution.get(difficulty, 0) + 1
    
    print(f"\n📊 QA Pair Analysis / QAペア分析:")
    print(f"   Question type distribution / 質問タイプ分布:")
    for q_type, count in type_distribution.items():
        print(f"     {q_type}: {count}")
    
    print(f"   Difficulty distribution / 難易度分布:")
    for difficulty, count in difficulty_distribution.items():
        print(f"     {difficulty}: {count}")
    
    # Show sample QA pairs / サンプルQAペアを表示
    print(f"\n📖 Sample QA Pairs / サンプルQAペア:")
    for i, qa_pair in enumerate(qa_pairs[:3]):
        print(f"\n{i+1}. Document: {qa_pair.document_id}")
        print(f"   Type: {qa_pair.metadata.get('question_type', 'N/A')}")
        print(f"   Difficulty: {qa_pair.metadata.get('difficulty', 'N/A')}")
        print(f"   Question: {qa_pair.question}")
        print(f"   Expected Answer: {qa_pair.answer[:150]}...")
    
    return qa_pairs


def demonstrate_performance_evaluation(quality_lab: QualityLab, 
                                     query_engine: QueryEngine, 
                                     qa_pairs: List[QAPair]):
    """
    Demonstrate comprehensive performance evaluation
    包括的パフォーマンス評価のデモンストレーション
    """
    
    print("\n" + "="*60)
    print("🔬 PERFORMANCE EVALUATION DEMONSTRATION")
    print("🔬 パフォーマンス評価のデモンストレーション")
    print("="*60)
    
    print(f"\n📊 Evaluating QueryEngine with {len(qa_pairs)} test cases...")
    print(f"📊 {len(qa_pairs)}テストケースでQueryEngineを評価中...")
    
    # Run comprehensive evaluation / 包括的評価を実行
    start_time = time.time()
    evaluation_results = quality_lab.evaluate_query_engine(
        query_engine=query_engine,
        qa_pairs=qa_pairs,
        evaluation_metrics=["bleu", "rouge", "llm_judge"],
        include_contradiction_detection=True,
        detailed_analysis=True
    )
    
    evaluation_time = time.time() - start_time
    
    print(f"✅ Evaluation completed in {evaluation_time:.2f}s")
    
    # Analyze overall results / 全体結果を分析
    test_results = evaluation_results['test_results']
    passed_tests = sum(1 for result in test_results if result['passed'])
    pass_rate = (passed_tests / len(test_results)) * 100
    
    print(f"\n📈 Overall Results / 全体結果:")
    print(f"   Total tests: {len(test_results)}")
    print(f"   Passed tests: {passed_tests}")
    print(f"   Pass rate: {pass_rate:.1f}%")
    print(f"   Average processing time: {evaluation_time/len(test_results):.3f}s per test")
    
    # Performance by question type / 質問タイプ別パフォーマンス
    performance_by_type = {}
    for result in test_results:
        q_type = result.get('question_type', 'unknown')
        if q_type not in performance_by_type:
            performance_by_type[q_type] = {'total': 0, 'passed': 0, 'scores': []}
        
        performance_by_type[q_type]['total'] += 1
        if result['passed']:
            performance_by_type[q_type]['passed'] += 1
        performance_by_type[q_type]['scores'].append(result.get('score', 0))
    
    print(f"\n📊 Performance by Question Type / 質問タイプ別パフォーマンス:")
    for q_type, stats in performance_by_type.items():
        type_pass_rate = (stats['passed'] / stats['total']) * 100
        avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
        print(f"   {q_type.capitalize()}:")
        print(f"     Pass rate: {type_pass_rate:.1f}% ({stats['passed']}/{stats['total']})")
        print(f"     Average score: {avg_score:.3f}")
    
    # Metric analysis / メトリクス分析
    if 'metric_summaries' in evaluation_results:
        print(f"\n🎯 Metric Analysis / メトリクス分析:")
        for metric, summary in evaluation_results['metric_summaries'].items():
            print(f"   {metric.upper()}:")
            print(f"     Average: {summary.get('average', 0):.3f}")
            print(f"     Standard deviation: {summary.get('std_dev', 0):.3f}")
            print(f"     Range: {summary.get('min', 0):.3f} - {summary.get('max', 0):.3f}")
    
    # Show sample results / サンプル結果を表示
    print(f"\n📝 Sample Test Results / サンプルテスト結果:")
    for i, result in enumerate(test_results[:3]):
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"\n{i+1}. {status}")
        print(f"   Question: {result['query'][:80]}...")
        print(f"   Generated Answer: {result['generated_answer'][:100]}...")
        print(f"   Score: {result.get('score', 0):.3f}")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
        print(f"   Processing time: {result.get('processing_time', 0):.3f}s")
        
        if not result['passed'] and 'failure_reason' in result:
            print(f"   Failure reason: {result['failure_reason']}")
    
    return evaluation_results


def demonstrate_contradiction_detection(quality_lab: QualityLab, 
                                      documents: List[Document],
                                      query_engine: QueryEngine):
    """
    Demonstrate contradiction detection capabilities
    矛盾検出機能のデモンストレーション
    """
    
    print("\n" + "="*60)
    print("🕵️ CONTRADICTION DETECTION DEMONSTRATION")
    print("🕵️ 矛盾検出のデモンストレーション")
    print("="*60)
    
    # Test queries for contradiction detection / 矛盾検出用テストクエリ
    test_queries = [
        "機械学習とは何ですか？",
        "深層学習の主要なアーキテクチャは？",
        "自然言語処理の応用分野は？",
        "AIの歴史について教えて",
        "ニューラルネットワークの学習方法は？"
    ]
    
    print(f"\n🔍 Running contradiction detection with {len(test_queries)} queries...")
    print(f"🔍 {len(test_queries)}クエリで矛盾検出を実行中...")
    
    # Detect contradictions / 矛盾を検出
    contradiction_results = quality_lab.detect_contradictions(
        corpus_documents=documents,
        query_engine=query_engine,
        test_queries=test_queries
    )
    
    print(f"✅ Contradiction analysis completed")
    
    # Analyze contradiction results / 矛盾結果を分析
    contradictions = contradiction_results.get('contradictions', [])
    print(f"\n📊 Contradiction Analysis / 矛盾分析:")
    print(f"   Documents analyzed: {len(documents)}")
    print(f"   Queries tested: {len(test_queries)}")
    print(f"   Contradictions found: {len(contradictions)}")
    
    if contradictions:
        print(f"\n⚠️  Detected Contradictions / 検出された矛盾:")
        for i, contradiction in enumerate(contradictions[:3]):
            print(f"\n{i+1}. Contradiction Type: {contradiction.get('type', 'Unknown')}")
            print(f"   Confidence: {contradiction.get('confidence', 0):.3f}")
            print(f"   Statement 1: {contradiction.get('statement_1', '')[:100]}...")
            print(f"   Statement 2: {contradiction.get('statement_2', '')[:100]}...")
            print(f"   Source documents: {contradiction.get('source_documents', [])}")
    else:
        print(f"\n✅ No contradictions detected in the corpus")
    
    # Consistency check / 一貫性チェック
    print(f"\n🔄 Running consistency check...")
    print(f"🔄 一貫性チェック実行中...")
    
    similar_query_groups = [
        ["機械学習とは何ですか？", "MLの定義を教えて", "機械学習について説明して"],
        ["深層学習とは何ですか？", "ディープラーニングの説明", "DLの概要を教えて"]
    ]
    
    consistency_results = quality_lab.check_answer_consistency(
        query_engine=query_engine,
        similar_queries=similar_query_groups
    )
    
    print(f"✅ Consistency analysis completed")
    print(f"   Query groups tested: {len(similar_query_groups)}")
    print(f"   Average consistency score: {consistency_results.get('average_consistency', 0):.3f}")
    
    # Show consistency details / 一貫性詳細を表示
    if 'group_results' in consistency_results:
        print(f"\n📊 Consistency by Query Group / クエリグループ別一貫性:")
        for i, group_result in enumerate(consistency_results['group_results']):
            print(f"   Group {i+1}: {group_result.get('consistency_score', 0):.3f}")
            print(f"     Queries: {group_result.get('query_count', 0)}")
            print(f"     Similarity: {group_result.get('average_similarity', 0):.3f}")
    
    return contradiction_results, consistency_results


def demonstrate_advanced_evaluation(quality_lab: QualityLab,
                                   query_engine: QueryEngine,
                                   qa_pairs: List[QAPair]):
    """
    Demonstrate advanced evaluation techniques
    高度な評価技術のデモンストレーション
    """
    
    print("\n" + "="*60)
    print("🚀 ADVANCED EVALUATION DEMONSTRATION")
    print("🚀 高度な評価のデモンストレーション")
    print("="*60)
    
    # 1. LLM-based evaluation / LLMベース評価
    print(f"\n📌 1. LLM-based Quality Assessment")
    print(f"📌 1. LLMベース品質評価")
    print("-" * 40)
    
    llm_evaluation = quality_lab.evaluate_with_llm_judge(
        query_engine=query_engine,
        qa_pairs=qa_pairs[:5],  # Test with subset for demo
        criteria=[
            "accuracy",      # 正確性
            "completeness",  # 完全性
            "relevance",     # 関連性
            "clarity"        # 明確性
        ]
    )
    
    print(f"✅ LLM evaluation completed for {len(llm_evaluation['evaluations'])} answers")
    
    # Aggregate LLM scores / LLMスコアを集計
    criteria_scores = {}
    for evaluation in llm_evaluation['evaluations']:
        for criterion, score_data in evaluation['scores'].items():
            if criterion not in criteria_scores:
                criteria_scores[criterion] = []
            criteria_scores[criterion].append(score_data['score'])
    
    print(f"\n📊 LLM Judge Scores / LLM判定スコア:")
    for criterion, scores in criteria_scores.items():
        avg_score = sum(scores) / len(scores)
        print(f"   {criterion.capitalize()}: {avg_score:.2f}/5.0")
    
    # 2. Robustness testing / 堅牢性テスト
    print(f"\n📌 2. Robustness Testing")
    print(f"📌 2. 堅牢性テスト")
    print("-" * 40)
    
    # Generate adversarial cases / 敵対的ケースを生成
    adversarial_queries = [
        "",  # Empty query
        "これは意味不明な質問です quantum blockchain AI unicorn",  # Nonsensical
        "機械学習" * 50,  # Very long repetitive query
        "深層学習は機械学習と同じですか？違いますか？同じですか？",  # Contradictory
    ]
    
    robustness_results = []
    for query in adversarial_queries:
        try:
            start_time = time.time()
            result = query_engine.answer(query)
            end_time = time.time()
            
            robustness_results.append({
                'query': query[:50] + "..." if len(query) > 50 else query,
                'success': True,
                'time': end_time - start_time,
                'answer_length': len(result.answer),
                'confidence': result.confidence
            })
        except Exception as e:
            robustness_results.append({
                'query': query[:50] + "..." if len(query) > 50 else query,
                'success': False,
                'error': str(e)
            })
    
    print(f"✅ Robustness testing completed for {len(adversarial_queries)} adversarial cases")
    
    successful_cases = [r for r in robustness_results if r['success']]
    print(f"   Success rate: {len(successful_cases)}/{len(adversarial_queries)} ({len(successful_cases)/len(adversarial_queries)*100:.1f}%)")
    
    print(f"\n📊 Robustness Results / 堅牢性結果:")
    for i, result in enumerate(robustness_results):
        status = "✅" if result['success'] else "❌"
        print(f"   {i+1}. {status} Query: {result['query']}")
        if result['success']:
            print(f"      Time: {result['time']:.3f}s, Confidence: {result['confidence']:.3f}")
        else:
            print(f"      Error: {result['error']}")
    
    # 3. Performance benchmarking / パフォーマンスベンチマーク
    print(f"\n📌 3. Performance Benchmarking")
    print(f"📌 3. パフォーマンスベンチマーク")
    print("-" * 40)
    
    benchmark_queries = [
        "AIとは何ですか？",
        "機械学習の種類を教えて",
        "深層学習の応用分野は？",
        "自然言語処理でできることは？"
    ]
    
    benchmark_results = []
    total_start_time = time.time()
    
    for query in benchmark_queries:
        query_start = time.time()
        result = query_engine.answer(query)
        query_end = time.time()
        
        benchmark_results.append({
            'query': query,
            'time': query_end - query_start,
            'confidence': result.confidence,
            'source_count': len(result.sources),
            'answer_length': len(result.answer)
        })
    
    total_time = time.time() - total_start_time
    
    # Calculate benchmark metrics / ベンチマークメトリクスを計算
    avg_time = sum(r['time'] for r in benchmark_results) / len(benchmark_results)
    avg_confidence = sum(r['confidence'] for r in benchmark_results) / len(benchmark_results)
    avg_sources = sum(r['source_count'] for r in benchmark_results) / len(benchmark_results)
    
    print(f"✅ Benchmark completed for {len(benchmark_queries)} queries")
    print(f"\n📊 Benchmark Results / ベンチマーク結果:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Average time per query: {avg_time:.3f}s")
    print(f"   Throughput: {len(benchmark_queries)/total_time:.2f} queries/sec")
    print(f"   Average confidence: {avg_confidence:.3f}")
    print(f"   Average sources per answer: {avg_sources:.1f}")
    
    return llm_evaluation, robustness_results, benchmark_results


def demonstrate_report_generation(quality_lab: QualityLab, 
                                evaluation_results: Dict[str, Any],
                                temp_dir: Path):
    """
    Demonstrate comprehensive report generation
    包括的レポート生成のデモンストレーション
    """
    
    print("\n" + "="*60)
    print("📊 EVALUATION REPORT GENERATION")
    print("📊 評価レポート生成")
    print("="*60)
    
    # Generate comprehensive report / 包括的レポートを生成
    print(f"\n📝 Generating comprehensive evaluation report...")
    print(f"📝 包括的評価レポートを生成中...")
    
    report_path = temp_dir / "comprehensive_evaluation_report.md"
    
    try:
        report = quality_lab.generate_evaluation_report(
            evaluation_results=evaluation_results,
            output_file=str(report_path),
            include_detailed_analysis=True,
            include_recommendations=True
        )
        
        print(f"✅ Report generated successfully")
        print(f"   Report file: {report_path}")
        print(f"   Report length: {len(report)} characters")
        
        # Show report preview / レポートプレビューを表示
        print(f"\n📖 Report Preview / レポートプレビュー:")
        print("-" * 50)
        preview_lines = report.split('\n')[:15]
        for line in preview_lines:
            print(line)
        print("...")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return None
    
    # Generate executive summary / 要約レポートを生成
    print(f"\n📋 Generating executive summary...")
    print(f"📋 要約レポートを生成中...")
    
    try:
        executive_summary = quality_lab.generate_executive_summary(
            evaluation_results=evaluation_results,
            target_audience="technical_management"
        )
        
        summary_path = temp_dir / "executive_summary.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(executive_summary)
        
        print(f"✅ Executive summary generated")
        print(f"   Summary file: {summary_path}")
        print(f"   Summary length: {len(executive_summary)} characters")
        
        # Show summary preview / 要約プレビューを表示
        print(f"\n📄 Executive Summary Preview / 要約プレビュー:")
        print("-" * 50)
        summary_lines = executive_summary.split('\n')[:10]
        for line in summary_lines:
            if line.strip():
                print(line)
        print("...")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Executive summary generation failed: {e}")
    
    # Generate JSON report for programmatic access / プログラム用JSONレポートを生成
    json_path = temp_dir / "evaluation_results.json"
    try:
        # Prepare JSON-serializable data / JSON化可能データを準備
        json_data = {
            'evaluation_summary': {
                'total_tests': len(evaluation_results.get('test_results', [])),
                'passed_tests': sum(1 for r in evaluation_results.get('test_results', []) if r.get('passed', False)),
                'pass_rate': sum(1 for r in evaluation_results.get('test_results', []) if r.get('passed', False)) / len(evaluation_results.get('test_results', [])) * 100 if evaluation_results.get('test_results') else 0,
                'processing_time': evaluation_results.get('processing_time', 0)
            },
            'metric_summaries': evaluation_results.get('metric_summaries', {}),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 JSON report generated: {json_path}")
        
    except Exception as e:
        print(f"❌ JSON report generation failed: {e}")
    
    return str(report_path)


def main():
    """
    Main demonstration function
    メインデモンストレーション関数
    """
    
    print("🚀 Part 3: RAG Evaluation Tutorial")
    print("🚀 Part 3: RAG評価チュートリアル")
    print("="*60)
    print("Comprehensive demonstration of RAG system evaluation with refinire-rag")
    print("refinire-ragを使用したRAGシステム評価の包括的なデモンストレーション")
    print("")
    print("Features demonstrated / デモ機能:")
    print("✓ Automated QA pair generation / 自動QAペア生成")
    print("✓ Comprehensive performance evaluation / 包括的パフォーマンス評価")
    print("✓ Contradiction detection / 矛盾検出")
    print("✓ Advanced evaluation techniques / 高度な評価技術")
    print("✓ Report generation / レポート生成")
    
    # Create temporary directory / 一時ディレクトリを作成
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Setup evaluation corpus / 評価用コーパスをセットアップ
        print(f"\n📁 Setup: Creating evaluation corpus in {temp_dir}")
        print(f"📁 セットアップ: {temp_dir} に評価用コーパスを作成中")
        doc_store, vector_store, file_paths = setup_evaluation_corpus(temp_dir)
        
        # Setup QueryEngine / QueryEngineをセットアップ
        query_engine = setup_query_engine(doc_store, vector_store)
        
        # Load documents for evaluation / 評価用文書をロード
        documents = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc_id = Path(file_path).stem
                documents.append(Document(
                    id=doc_id,
                    content=content,
                    metadata={'source': file_path, 'topic': doc_id}
                ))
        
        # Initialize QualityLab / QualityLabを初期化
        print(f"\n🔬 Initializing QualityLab...")
        print(f"🔬 QualityLabを初期化中...")
        
        quality_lab_config = QualityLabConfig(
            qa_pairs_per_document=3,
            similarity_threshold=0.8,
            question_types=["factual", "conceptual", "analytical", "comparative"],
            evaluation_metrics=["bleu", "rouge", "llm_judge"],
            include_detailed_analysis=True,
            include_contradiction_detection=True,
            output_format="markdown"
        )
        
        quality_lab = QualityLab(
            corpus_name="evaluation_tutorial",
            config=quality_lab_config
        )
        
        print(f"✅ QualityLab initialized successfully")
        
        # Demonstration sequence / デモシーケンス
        qa_pairs = demonstrate_qa_generation(quality_lab, documents)
        evaluation_results = demonstrate_performance_evaluation(quality_lab, query_engine, qa_pairs)
        contradiction_results, consistency_results = demonstrate_contradiction_detection(quality_lab, documents, query_engine)
        llm_eval, robustness, benchmark = demonstrate_advanced_evaluation(quality_lab, query_engine, qa_pairs)
        report_path = demonstrate_report_generation(quality_lab, evaluation_results, temp_dir)
        
        print("\n" + "="*60)
        print("🎉 TUTORIAL COMPLETE / チュートリアル完了")
        print("="*60)
        print("✅ All RAG evaluation demonstrations completed successfully!")
        print("✅ すべてのRAG評価デモが正常に完了しました！")
        print("")
        print("📚 What you learned / 学習内容:")
        print("   • Automated QA pair generation with multiple question types")
        print("     複数質問タイプでの自動QAペア生成")
        print("   • Comprehensive performance evaluation with multiple metrics")
        print("     複数メトリクスでの包括的パフォーマンス評価")
        print("   • Contradiction detection and consistency analysis")
        print("     矛盾検出と一貫性分析")
        print("   • Advanced evaluation techniques (LLM judge, robustness testing)")
        print("     高度な評価技術（LLM判定、堅牢性テスト）")
        print("   • Comprehensive report generation")
        print("     包括的レポート生成")
        print("")
        print(f"📁 Generated files available in: {temp_dir}")
        print(f"📁 生成ファイルの場所: {temp_dir}")
        print("")
        print("Generated reports / 生成されたレポート:")
        if report_path:
            print(f"   • Comprehensive evaluation report: {report_path}")
        print(f"   • Executive summary: {temp_dir}/executive_summary.md")
        print(f"   • JSON data: {temp_dir}/evaluation_results.json")
        print("")
        print("Next step / 次のステップ:")
        print("→ End-to-End Integration Tutorial (エンドツーエンド統合チュートリアル)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Tutorial failed / チュートリアル失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup (comment out to inspect generated files)
        # クリーンアップ（生成ファイルを確認する場合はコメントアウト）
        # shutil.rmtree(temp_dir)
        pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)