#!/usr/bin/env python3
"""
Complete RAG Tutorial: End-to-End Integration
完全RAGチュートリアル：エンドツーエンド統合

This comprehensive example demonstrates the complete RAG workflow using refinire-rag:
Part 1: Corpus Creation (Document loading, processing, indexing)
Part 2: Query Engine (Search, retrieval, answer generation)  
Part 3: Evaluation (QA generation, performance assessment, reporting)

この包括的な例では、refinire-ragを使用した完全なRAGワークフローを実演します：
Part 1: コーパス作成（文書ロード、処理、インデックス）
Part 2: Query Engine（検索、取得、回答生成）
Part 3: 評価（QA生成、パフォーマンス評価、レポート）
"""

import sys
import tempfile
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Core imports
from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
from refinire_rag.models.document import Document


class CompleteRAGTutorial:
    """
    Complete RAG tutorial demonstration class
    完全RAGチュートリアルデモクラス
    """
    
    def __init__(self, work_dir: Path):
        """
        Initialize tutorial with working directory
        作業ディレクトリでチュートリアルを初期化
        """
        self.work_dir = work_dir
        self.knowledge_base_dir = work_dir / "knowledge_base"
        self.reports_dir = work_dir / "reports"
        self.data_dir = work_dir / "data"
        
        # Create directories / ディレクトリを作成
        for dir_path in [self.knowledge_base_dir, self.reports_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components / コンポーネントを初期化
        self.doc_store: Optional[SQLiteDocumentStore] = None
        self.vector_store: Optional[InMemoryVectorStore] = None
        self.corpus_manager: Optional[CorpusManager] = None
        self.query_engine: Optional[QueryEngine] = None
        self.quality_lab: Optional[QualityLab] = None
        
        # Store results / 結果を保存
        self.corpus_stats = None
        self.evaluation_results = None
        self.performance_metrics = {}
    
    def create_knowledge_base(self) -> List[str]:
        """
        Create comprehensive knowledge base for demonstration
        デモ用の包括的知識ベースを作成
        """
        
        print("📚 Creating comprehensive knowledge base...")
        print("📚 包括的知識ベースを作成中...")
        
        # AI Fundamentals / AI基礎
        ai_fundamentals = self.knowledge_base_dir / "01_ai_fundamentals.txt"
        ai_fundamentals.write_text("""
# 人工知能（AI）基礎

## 概要
人工知能（Artificial Intelligence, AI）は、人間の知的能力をコンピュータで再現する技術の総称です。
1950年代のアラン・チューリングによるチューリングテストの提案以来、長い歴史を持つ研究分野です。

## AIの分類

### 1. 弱いAI（Narrow AI）
- 特定の領域やタスクに特化したAI
- 現在実用化されているAIの大部分
- 例：音声認識、画像認識、翻訳システム

### 2. 強いAI（General AI）
- 人間と同等の汎用的な知能を持つAI
- まだ実現されていない理論的概念
- AGI（Artificial General Intelligence）とも呼ばれる

### 3. 超AI（Super AI）
- 人間の知能を超越したAI
- 理論的・仮想的な概念

## 主要技術領域

### 機械学習（Machine Learning）
データから自動的に学習し、予測や判断を行う技術

### 深層学習（Deep Learning）
多層ニューラルネットワークを用いた機械学習手法

### 自然言語処理（NLP）
人間の言語をコンピュータで理解・生成する技術

### コンピュータビジョン（Computer Vision）
画像や動画から情報を抽出・理解する技術

### ロボティクス（Robotics）
物理的な環境で自律的に動作するシステム

## 応用分野
- 医療：診断支援、創薬、手術支援
- 金融：不正検知、リスク評価、アルゴリズム取引
- 交通：自動運転、交通最適化
- 製造：品質管理、予知保全、工程最適化
- エンターテインメント：推薦システム、ゲームAI

## 社会的影響
- 雇用への影響：自動化による職業の変化
- プライバシー：個人データの保護
- 公平性：アルゴリズムバイアスの問題
- 安全性：AI安全性の確保

AIは今後も急速に発展し、人間社会に大きな変革をもたらすと予想されています。
""", encoding='utf-8')
        
        # Machine Learning Details / 機械学習詳細
        ml_details = self.knowledge_base_dir / "02_machine_learning.txt"
        ml_details.write_text("""
# 機械学習（Machine Learning）詳細

## 定義
機械学習は、コンピュータがデータから自動的に学習し、
明示的にプログラムされることなく予測や判断を行う技術です。

## 学習パラダイム

### 1. 教師あり学習（Supervised Learning）
正解データ（ラベル）を用いて学習する手法

#### 分類（Classification）
- 目的：データを事前定義されたクラスに分類
- 例：メール分類、画像認識、診断システム
- 評価指標：精度、適合率、再現率、F1スコア
- アルゴリズム：
  * ロジスティック回帰
  *決定木（Decision Tree）
  * ランダムフォレスト（Random Forest）
  * サポートベクターマシン（SVM）
  * ナイーブベイズ（Naive Bayes）

#### 回帰（Regression）
- 目的：連続値の予測
- 例：株価予測、売上予測、気温予測
- 評価指標：MAE、MSE、RMSE、R²
- アルゴリズム：
  * 線形回帰（Linear Regression）
  * 多項式回帰（Polynomial Regression）
  * リッジ回帰（Ridge Regression）
  * ラッソ回帰（Lasso Regression）

### 2. 教師なし学習（Unsupervised Learning）
正解データなしでデータの構造やパターンを発見

#### クラスタリング
- 目的：類似データのグループ化
- 例：顧客セグメンテーション、遺伝子分析
- アルゴリズム：
  * K-means
  * 階層クラスタリング
  * DBSCAN
  * ガウス混合モデル（GMM）

#### 次元削減
- 目的：高次元データの可視化・圧縮
- アルゴリズム：
  * 主成分分析（PCA）
  * 独立成分分析（ICA）
  * t-SNE
  * UMAP

#### 異常検知
- 目的：正常パターンから外れるデータの発見
- 例：不正検知、故障診断
- アルゴリズム：
  * Isolation Forest
  * One-Class SVM
  * Local Outlier Factor (LOF)

### 3. 強化学習（Reinforcement Learning）
環境との相互作用を通じて最適な行動戦略を学習

#### 基本概念
- エージェント：学習・行動主体
- 環境：エージェントが行動する場
- 状態：環境の現在の状況
- 行動：エージェントが選択できる動作
- 報酬：行動に対するフィードバック
- 方策：状態から行動への写像

#### 主要アルゴリズム
- Q学習（Q-Learning）
- SARSA
- Deep Q Network (DQN)
- Policy Gradient
- Actor-Critic
- Proximal Policy Optimization (PPO)

#### 応用例
- ゲームAI：囲碁、チェス、ビデオゲーム
- ロボット制御：歩行、操作
- 自動運転：経路計画、運転戦略
- 金融：アルゴリズム取引

## 機械学習パイプライン

### 1. データ収集・理解
- データソースの特定
- データ品質の評価
- 探索的データ分析（EDA）

### 2. データ前処理
- 欠損値処理
- 外れ値処理
- データ変換・正規化
- 特徴量エンジニアリング

### 3. モデル選択・構築
- アルゴリズムの選択
- ハイパーパラメータ調整
- 交差検証

### 4. モデル評価
- 性能指標の計算
- 過学習・未学習の検証
- モデル解釈

### 5. デプロイ・運用
- モデルの本番環境への展開
- 継続的監視・更新
- A/Bテスト

## 課題と限界
- データ品質への依存
- 解釈可能性の不足
- バイアスと公平性の問題
- 計算コストとリソース要件
- 汎化性能の限界

機械学習は現代AIの中核技術として、様々な分野で革新的な解決策を提供しています。
""", encoding='utf-8')
        
        # Deep Learning / 深層学習
        deep_learning = self.knowledge_base_dir / "03_deep_learning.txt"
        deep_learning.write_text("""
# 深層学習（Deep Learning）

## 概要
深層学習は、人間の脳の神経回路を模倣した多層ニューラルネットワークを用いる
機械学習の一分野です。2010年代以降のAIブームを牽引する中核技術です。

## ニューラルネットワークの基礎

### 基本構造
- ニューロン（ノード）：情報処理の基本単位
- 重み（Weight）：ニューロン間の接続強度
- バイアス（Bias）：活性化の閾値調整
- 活性化関数：非線形変換（ReLU、Sigmoid、Tanh等）

### 学習プロセス
1. 順伝播（Forward Propagation）：入力から出力への計算
2. 損失計算：予測値と正解値の誤差測定
3. 逆伝播（Backpropagation）：誤差の逆向き伝播
4. 勾配降下法：重みとバイアスの更新

## 主要アーキテクチャ

### 1. 畳み込みニューラルネットワーク（CNN）
画像処理に特化したアーキテクチャ

#### 構成要素
- 畳み込み層：局所的特徴の抽出
- プーリング層：ダウンサンプリング
- 全結合層：分類・回帰

#### 代表的モデル
- LeNet（1998）：手書き数字認識
- AlexNet（2012）：ImageNet革命
- VGG（2014）：深い畳み込み層
- ResNet（2015）：残差接続
- EfficientNet（2019）：効率的設計

#### 応用分野
- 画像分類：ImageNet、CIFAR-10
- 物体検出：YOLO、R-CNN系列
- セマンティックセグメンテーション
- 医療画像解析：X線、CT、MRI

### 2. 再帰ニューラルネットワーク（RNN）
時系列データ処理に適したアーキテクチャ

#### 基本構造
- 隠れ状態：前の時刻の情報を保持
- 入力ゲート、忘却ゲート、出力ゲート

#### 改良版
- LSTM（Long Short-Term Memory）
  * 長期依存関係の学習
  * 勾配消失問題の解決
- GRU（Gated Recurrent Unit）
  * LSTMの簡素化版
  * 計算効率の向上

#### 応用分野
- 自然言語処理：翻訳、要約、感情分析
- 音声認識・合成
- 時系列予測：株価、天気予報
- 音楽生成

### 3. トランスフォーマー（Transformer）
注意機構（Attention）を核とした革新的アーキテクチャ

#### 特徴
- Self-Attention：系列内の関係性を捉える
- Multi-Head Attention：複数の注意の組み合わせ
- 位置エンコーディング：順序情報の付与
- 並列処理：RNNより高速

#### 代表的モデル
- BERT（2018）：双方向エンコーダー
- GPT系列（2018-2023）：生成型デコーダー
- T5（2019）：Text-to-Text変換
- Vision Transformer（2020）：画像へのTransformer適用

#### 応用分野
- 機械翻訳：Google Translate
- 言語モデル：ChatGPT、Claude
- 検索エンジン：BERT搭載Google検索
- コード生成：GitHub Copilot

## 生成モデル

### 1. 生成的敵対ネットワーク（GAN）
- Generator：偽データ生成
- Discriminator：真偽判定
- 敵対的学習：互いの性能向上

#### 応用
- 画像生成：StyleGAN、BigGAN
- 超解像：SRGAN、ESRGAN
- データ拡張：少数データの増強

### 2. 変分オートエンコーダー（VAE）
- エンコーダー：データの潜在表現学習
- デコーダー：潜在表現からデータ復元
- 確率的生成

### 3. 拡散モデル（Diffusion Models）
- ノイズ除去プロセスの学習
- 高品質画像生成
- 例：DALL-E 2、Stable Diffusion、Midjourney

## 大規模言語モデル（LLM）

### 発展の歴史
- GPT-1（2018）：1.17億パラメータ
- BERT（2018）：3.4億パラメータ
- GPT-2（2019）：15億パラメータ
- GPT-3（2020）：1750億パラメータ
- PaLM（2022）：5400億パラメータ
- GPT-4（2023）：推定1兆パラメータ

### 能力
- 文章生成・要約
- 質問応答
- 翻訳
- コード生成
- 推論・計算

## 深層学習の課題

### 技術的課題
- 大量データ・計算資源の必要性
- ブラックボックス問題
- 過学習・汎化性能
- 敵対的攻撃への脆弱性
- カタストロフィック忘却

### 社会的課題
- バイアスと差別
- プライバシー侵害
- 雇用への影響
- エネルギー消費
- 情報の信頼性

## 最新動向
- マルチモーダルAI：テキスト・画像・音声の統合
- 少数ショット学習：少ないデータでの学習
- 連合学習：プライバシー保護学習
- ニューラルアーキテクチャ探索（NAS）
- 量子機械学習

深層学習は現在も急速に発展を続け、
AGI（汎用人工知能）の実現に向けた重要な技術として注目されています。
""", encoding='utf-8')
        
        # NLP Applications / NLP応用
        nlp_applications = self.knowledge_base_dir / "04_nlp_applications.txt"
        nlp_applications.write_text("""
# 自然言語処理（NLP）応用

## 概要
自然言語処理（Natural Language Processing, NLP）は、
人間の言語をコンピュータで理解・生成・操作する技術分野です。

## 基本的なNLPタスク

### 1. 前処理（Preprocessing）
#### テキスト正規化
- 大文字・小文字統一
- 数字・記号の処理
- 文字エンコーディング統一

#### トークン化（Tokenization）
- 文分割：文章を文に分割
- 単語分割：文を単語に分割
- サブワード分割：BPE、SentencePiece

#### 形態素解析
- 品詞タグ付け（POS Tagging）
- 語幹抽出（Stemming）
- 語形正規化（Lemmatization）

### 2. 言語理解タスク

#### 固有表現認識（NER）
- 人名、地名、組織名の抽出
- 日付、時間、金額の識別
- 応用：情報抽出、質問応答

#### 構文解析（Parsing）
- 依存構造解析：語の依存関係
- 句構造解析：文法構造の階層化
- 応用：機械翻訳、情報抽出

#### 意味解析
- 語義曖昧性解消（WSD）
- 意味役割ラベリング（SRL）
- 含意関係認識（RTE）

## 主要なNLPアプリケーション

### 1. 機械翻訳（Machine Translation）

#### 発展の歴史
- 規則ベース翻訳（1950年代-1980年代）
  * 言語学的規則の手動作成
  * 限定的な精度と範囲
  
- 統計的機械翻訳（1990年代-2010年代）
  * 大規模並列コーパスの活用
  * フレーズベース、階層フレーズベース
  
- ニューラル機械翻訳（2010年代-現在）
  * Encoder-Decoder モデル
  * Attention機構の導入
  * Transformer による革新

#### 現代の手法
- Google Translate：多言語ニューラル翻訳
- DeepL：高品質な翻訳サービス
- mBERT、XLM-R：多言語事前学習モデル
- ゼロショット翻訳：直接翻訳ペアなし

#### 評価指標
- BLEU Score：n-gramベース自動評価
- METEOR：語幹や同義語を考慮
- chrF：文字レベル評価
- 人手評価：流暢さ、正確性、適切性

### 2. 感情分析（Sentiment Analysis）

#### 分析レベル
- 文書レベル：文書全体の感情極性
- 文レベル：各文の感情
- アスペクトレベル：特定観点の感情
- 感情の強度：ポジティブ・ネガティブの程度

#### 手法
- 辞書ベース：感情辞書の活用
- 機械学習：特徴量エンジニアリング
- 深層学習：LSTM、BERT等

#### 応用分野
- ソーシャルメディア分析
- 製品レビュー分析
- 顧客満足度調査
- ブランド監視
- 株式市場予測

### 3. 質問応答システム（QA）

#### システム分類
- 抽出型QA：文書から該当箇所抽出
- 生成型QA：回答文の生成
- 知識ベースQA：構造化知識活用
- 会話型QA：対話的質問応答

#### 代表的データセット
- SQuAD：読解理解ベンチマーク
- Natural Questions：実世界質問
- MS MARCO：大規模検索QA
- JAQKET：日本語QAデータセット

#### 技術要素
- 文書検索（Retrieval）
- 読解理解（Reading Comprehension）
- 回答生成（Answer Generation）
- 複数ホップ推論

### 4. 文書要約（Text Summarization）

#### 要約手法
- 抽出型要約：重要文の選択・結合
- 生成型要約：新しい文の生成
- ハイブリッド：両手法の組み合わせ

#### 要約の種類
- 単一文書要約：1つの文書から要約
- 複数文書要約：複数文書の統合要約
- 更新要約：新情報の追加要約
- クエリ指向要約：特定観点の要約

#### 評価指標
- ROUGE：要約品質の自動評価
- Pyramid：重要度重み付き評価
- 人手評価：情報性、読みやすさ

### 5. 対話システム（Dialogue Systems）

#### システム分類
- タスク指向：特定業務の遂行
  * レストラン予約、航空券予約
  * カスタマーサポート
  
- 雑談型：自然な会話
  * 娯楽・コミュニケーション
  * 感情的サポート
  
- 質問応答型：情報提供
  * 知識検索・提供
  * 教育支援

#### 技術コンポーネント
- 自然言語理解（NLU）
  * 意図理解（Intent Recognition）
  * エンティティ抽出（Entity Extraction）
  
- 対話管理（Dialogue Management）
  * 対話状態追跡
  * 次行動決定
  
- 自然言語生成（NLG）
  * 応答文生成
  * 自然性の確保

### 6. 情報抽出（Information Extraction）

#### 抽出対象
- 固有表現：人名、地名、組織名
- 関係抽出：エンティティ間の関係
- イベント抽出：出来事の抽出
- 知識グラフ構築

#### 応用
- ニュース分析
- 科学文献からの知識抽出
- 企業情報の整理
- 法的文書の解析

## 最新の大規模言語モデル

### GPTシリーズの発展
- GPT-1（2018）：生成型事前学習
- GPT-2（2019）：スケールアップ
- GPT-3（2020）：Few-shot学習
- InstructGPT：人間フィードバック学習
- ChatGPT（2022）：対話特化
- GPT-4（2023）：マルチモーダル対応

### 日本語特化モデル
- 京都大学BERT
- 東北大学BERT
- rinna GPT
- Japanese T5
- Stockmark GPT
- CyberAgent LLM

### 多言語モデル
- mBERT：104言語対応
- XLM-R：100言語対応
- mT5：101言語対応
- BLOOM：176言語対応

## NLP評価の課題

### 自動評価の限界
- 語彙的類似性への偏重
- 意味的一致の不完全な捕捉
- 創造性・多様性の評価困難

### 人手評価の課題
- 主観性とアノテータ間一致
- コストと時間の制約
- スケーラビリティの問題

## 現在の課題と今後の展望

### 技術的課題
- 言語の曖昧性・多義性
- 文脈理解の限界
- 常識推論の不足
- 多言語・方言への対応

### 社会的課題
- バイアスと公平性
- プライバシー保護
- 偽情報対策
- 著作権・知的財産権

### 今後の展望
- マルチモーダルNLP：テキスト+画像・音声
- 効率的モデル：軽量化・高速化
- ドメイン適応：専門分野特化
- 説明可能AI：意思決定の透明性
- 人間とAIの協働：Human-in-the-loop

NLPは現在、大規模言語モデルの登場により革命的な進歩を遂げており、
人間とコンピュータの自然な対話を実現する重要技術として発展し続けています。
""", encoding='utf-8')
        
        # Computer Vision / コンピュータビジョン
        computer_vision = self.knowledge_base_dir / "05_computer_vision.txt"
        computer_vision.write_text("""
# コンピュータビジョン（Computer Vision）

## 概要
コンピュータビジョンは、コンピュータが人間の視覚システムのように
画像や動画から情報を理解・解釈する技術分野です。

## 基本的な画像処理

### 1. 前処理（Preprocessing）
#### 画像の基本操作
- リサイズ（Resize）：画像サイズの変更
- クロッピング（Cropping）：画像の切り出し
- 回転・反転：データ拡張
- 正規化：ピクセル値の標準化

#### ノイズ除去
- ガウシアンフィルタ：滑らかなノイズ除去
- メディアンフィルタ：突発的ノイズ除去
- バイラテラルフィルタ：エッジ保持ノイズ除去

#### エッジ検出
- Sobel フィルタ：勾配ベース
- Canny エッジ検出：多段階処理
- Laplacian：二次微分ベース

### 2. 特徴抽出（Feature Extraction）
#### 従来手法
- SIFT（Scale-Invariant Feature Transform）
- SURF（Speeded Up Robust Features）
- ORB（Oriented FAST and Rotated BRIEF）
- HOG（Histogram of Oriented Gradients）

#### 深層学習ベース
- CNN特徴量：自動的特徴学習
- Transfer Learning：事前学習モデル活用
- Feature Pyramid Networks（FPN）

## 主要なコンピュータビジョンタスク

### 1. 画像分類（Image Classification）

#### 定義と目的
- 画像全体を事前定義されたクラスに分類
- 単一ラベル・マルチラベル分類

#### 代表的データセット
- MNIST：手書き数字（28×28ピクセル）
- CIFAR-10/100：小物体10/100クラス
- ImageNet：1000クラス、120万枚
- Places365：場所・シーン認識

#### 主要モデル
- LeNet-5（1998）：初期CNN
- AlexNet（2012）：ImageNet勝利
- VGG（2014）：深いネットワーク
- GoogLeNet（2014）：Inception構造
- ResNet（2015）：残差接続
- DenseNet（2017）：密接続
- EfficientNet（2019）：効率的設計
- Vision Transformer（2020）：Attention機構

#### 応用分野
- 医療画像診断：X線、CT、MRI解析
- 製造業品質管理：欠陥検出
- 農業：作物の病気診断
- 小売：商品認識・在庫管理

### 2. 物体検出（Object Detection）

#### 定義と目的
- 画像内の物体位置（Bounding Box）と
  クラスの同時予測

#### 手法の分類
- Two-Stage：候補領域→分類
  * R-CNN系列：R-CNN、Fast R-CNN、Faster R-CNN
  * Feature Pyramid Networks（FPN）
  
- One-Stage：直接的検出
  * YOLO系列：YOLOv1-v8
  * SSD（Single Shot MultiBox Detector）
  * RetinaNet：Focal Loss

#### 評価指標
- mAP（mean Average Precision）
- IoU（Intersection over Union）
- 精度・再現率曲線

#### 応用分野
- 自動運転：歩行者・車両検出
- 監視システム：異常行動検知
- ロボティクス：物体認識・把持
- 小売：商品棚分析

### 3. セマンティックセグメンテーション

#### 定義と目的
- 画像の各ピクセルにクラスラベル付与
- 物体の詳細な領域分割

#### 主要アーキテクチャ
- FCN（Fully Convolutional Networks）
- U-Net：医療画像セグメンテーション
- DeepLab系列：Atrous Convolution
- PSPNet（Pyramid Scene Parsing）
- HRNet（High-Resolution Network）

#### データセット
- PASCAL VOC：20クラス
- COCO：80クラス
- Cityscapes：都市風景
- ADE20K：150クラス

#### 応用分野
- 医療：腫瘍・臓器の領域特定
- 自動運転：道路・歩道認識
- 衛星画像：土地利用分析
- 工業：部品の精密測定

### 4. インスタンスセグメンテーション

#### 定義と目的
- 物体検出とセマンティックセグメンテーションの融合
- 同一クラス内の個体識別

#### 主要モデル
- Mask R-CNN：Faster R-CNN + マスク予測
- YOLACT：リアルタイム処理
- SOLOv2：分割による検出

### 5. 顔認識・顔検出

#### 顔検出
- Viola-Jones：Haar特徴量
- MTCNN：Multi-task CNN
- RetinaFace：詳細なランドマーク検出

#### 顔認識
- Eigenfaces：主成分分析
- FaceNet：Triplet Loss
- ArcFace：角度マージン損失
- DeepFace：Facebook開発

#### 応用分野
- セキュリティ：入退室管理
- 決済システム：生体認証
- 写真管理：自動タグ付け
- エンターテインメント：ARフィルタ

### 6. 動画解析（Video Analysis）

#### 行動認識（Action Recognition）
- 3D CNN：時空間特徴学習
- Two-Stream：RGB+光学フロー
- Transformer：時系列注意機構

#### 物体追跡（Object Tracking）
- 単一物体追跡（SOT）：SORT、DeepSORT
- 多物体追跡（MOT）：FairMOT、ByteTrack

#### 動画異常検知
- 正常パターンからの逸脱検出
- 監視カメラでの異常行動検知

## 3D コンピュータビジョン

### 1. 深度推定（Depth Estimation）
- ステレオビジョン：複数カメラ
- 単眼深度推定：単一画像から
- LiDAR：レーザー測距

### 2. 3D物体検出
- Point Cloud処理：PointNet、PointNet++
- RGB-D：色情報+深度情報
- 自動運転での3D物体認識

### 3. SLAM（Simultaneous Localization and Mapping）
- 視覚SLAM：Visual SLAM
- 環境地図構築と自己位置推定
- AR/VRでの空間認識

## 生成系コンピュータビジョン

### 1. 画像生成
- GAN（Generative Adversarial Networks）
  * StyleGAN：高品質顔画像生成
  * BigGAN：高解像度画像生成
  
- 拡散モデル（Diffusion Models）
  * DALL-E 2：テキストから画像生成
  * Stable Diffusion：オープンソース実装
  * Midjourney：アーティスティック生成

### 2. 画像編集
- 超解像（Super Resolution）：SRGAN、ESRGAN
- デノイジング：ノイズ除去
- インペインティング：欠損部分補完
- スタイル変換：Neural Style Transfer

### 3. 画像から画像への変換
- Pix2Pix：ペア画像変換
- CycleGAN：非ペア画像変換
- StarGAN：多ドメイン変換

## 評価指標と課題

### 評価指標
- 分類：Accuracy、Top-k Accuracy
- 検出：mAP、IoU
- セグメンテーション：IoU、Dice係数
- 生成：FID、IS、LPIPS

### 技術的課題
- ドメインギャップ：学習・テストデータの差
- 小さな物体の検出困難
- オクルージョン（遮蔽）への対応
- リアルタイム処理の要求

### 社会的課題
- プライバシー保護
- バイアスと公平性
- 悪用防止：ディープフェイク
- 監視社会への懸念

## 最新動向と今後の展望

### マルチモーダルAI
- CLIP：画像とテキストの統合理解
- DALL-E：テキストから画像生成
- GPT-4V：視覚質問応答

### 効率化技術
- モデル軽量化：知識蒸留、量子化
- Neural Architecture Search（NAS）
- エッジコンピューティング対応

### 新しい応用分野
- メタバース：3D空間理解
- 医療AI：画像診断支援
- 農業AI：作物監視・収穫予測
- 環境監視：衛星画像解析

コンピュータビジョンは深層学習の発展とともに急速に進歩し、
人間の視覚を超える性能を多くのタスクで実現しています。
今後も様々な分野での応用拡大が期待されています。
""", encoding='utf-8')
        
        file_paths = [str(f) for f in self.knowledge_base_dir.glob("*.txt")]
        print(f"✅ Created knowledge base with {len(file_paths)} documents")
        return file_paths
    
    def part1_corpus_creation(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Part 1: Comprehensive corpus creation demonstration
        Part 1: 包括的コーパス作成のデモンストレーション
        """
        
        print("\n" + "="*70)
        print("🚀 PART 1: CORPUS CREATION / パート1: コーパス作成")
        print("="*70)
        
        # Initialize storage / ストレージを初期化
        print("\n📊 Initializing storage components...")
        print("📊 ストレージコンポーネントを初期化中...")
        
        db_path = self.data_dir / "tutorial_corpus.db"
        self.doc_store = SQLiteDocumentStore(str(db_path))
        self.vector_store = InMemoryVectorStore()
        
        print(f"✅ Storage initialized:")
        print(f"   Document store: {db_path}")
        print(f"   Vector store: In-memory")
        
        # Demonstrate different corpus building approaches / 異なるコーパス構築アプローチをデモ
        approaches = [
            ("Simple RAG", "simple_rag"),
            ("Semantic RAG", "semantic_rag"), 
            ("Knowledge RAG", "knowledge_rag")
        ]
        
        results = {}
        
        for approach_name, approach_type in approaches:
            print(f"\n📌 Building corpus with {approach_name}...")
            print(f"📌 {approach_name}でコーパス構築中...")
            
            # Create fresh stores for each approach / 各アプローチ用に新しいストアを作成
            temp_doc_store = SQLiteDocumentStore(":memory:")
            temp_vector_store = InMemoryVectorStore()
            
            # Create corpus manager / コーパスマネージャを作成
            if approach_type == "simple_rag":
                manager = CorpusManager.create_simple_rag(temp_doc_store, temp_vector_store)
            elif approach_type == "semantic_rag":
                manager = CorpusManager.create_semantic_rag(temp_doc_store, temp_vector_store)
            else:  # knowledge_rag
                manager = CorpusManager.create_knowledge_rag(temp_doc_store, temp_vector_store)
            
            # Build corpus / コーパスを構築
            start_time = time.time()
            stats = manager.build_corpus(file_paths)
            build_time = time.time() - start_time
            
            results[approach_type] = {
                'approach_name': approach_name,
                'files_processed': stats.total_files_processed,
                'documents_created': stats.total_documents_created,
                'chunks_created': stats.total_chunks_created,
                'processing_time': stats.total_processing_time,
                'build_time': build_time,
                'stages_executed': stats.pipeline_stages_executed
            }
            
            print(f"✅ {approach_name} completed:")
            print(f"   Files processed: {stats.total_files_processed}")
            print(f"   Documents created: {stats.total_documents_created}")
            print(f"   Chunks created: {stats.total_chunks_created}")
            print(f"   Processing time: {stats.total_processing_time:.3f}s")
            print(f"   Pipeline stages: {stats.pipeline_stages_executed}")
        
        # Use semantic RAG for remaining parts / 残りの部分にはセマンティックRAGを使用
        print(f"\n🎯 Setting up final corpus with Semantic RAG...")
        print(f"🎯 セマンティックRAGで最終コーパスをセットアップ中...")
        
        self.corpus_manager = CorpusManager.create_semantic_rag(self.doc_store, self.vector_store)
        self.corpus_stats = self.corpus_manager.build_corpus(file_paths)
        
        print(f"✅ Final corpus setup completed:")
        print(f"   Total chunks: {self.corpus_stats.total_chunks_created}")
        print(f"   Vector store size: {len(self.vector_store._vectors)}")
        
        # Save corpus comparison results / コーパス比較結果を保存
        comparison_file = self.reports_dir / "corpus_comparison.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Corpus comparison saved to: {comparison_file}")
        
        return results
    
    def part2_query_engine(self) -> Dict[str, Any]:
        """
        Part 2: Query engine demonstration and testing
        Part 2: クエリエンジンのデモとテスト
        """
        
        print("\n" + "="*70)
        print("🔍 PART 2: QUERY ENGINE / パート2: クエリエンジン")
        print("="*70)
        
        # Initialize QueryEngine / QueryEngineを初期化
        print("\n🤖 Initializing QueryEngine...")
        print("🤖 QueryEngineを初期化中...")
        
        self.query_engine = QueryEngine(
            document_store=self.doc_store,
            vector_store=self.vector_store,
            retriever=SimpleRetriever(self.vector_store),
            reranker=SimpleReranker(),
            reader=SimpleReader(),
            config=QueryEngineConfig(
                enable_query_normalization=True,
                include_sources=True,
                include_confidence=True,
                max_response_time=30.0
            )
        )
        
        print(f"✅ QueryEngine initialized successfully")
        
        # Test queries / テストクエリ
        test_queries = [
            "人工知能とは何ですか？",
            "機械学習の主要な種類を教えてください",
            "深層学習のアーキテクチャにはどのようなものがありますか？",
            "自然言語処理の応用分野は？",
            "コンピュータビジョンでできることを教えて",
            "GANとVAEの違いは何ですか？",
            "強化学習の基本概念を説明して",
            "トランスフォーマーの特徴は？"
        ]
        
        print(f"\n💬 Testing QueryEngine with {len(test_queries)} queries...")
        print(f"💬 {len(test_queries)}クエリでQueryEngineをテスト中...")
        
        query_results = []
        total_start_time = time.time()
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            
            try:
                start_time = time.time()
                result = self.query_engine.answer(query)
                end_time = time.time()
                
                query_time = end_time - start_time
                
                query_result = {
                    'query': query,
                    'answer': result.answer,
                    'confidence': result.confidence,
                    'source_count': len(result.sources),
                    'processing_time': query_time,
                    'success': True
                }
                
                print(f"🤖 Answer: {result.answer[:100]}...")
                print(f"📊 Confidence: {result.confidence:.3f}")
                print(f"📚 Sources: {len(result.sources)}")
                print(f"⏱️  Time: {query_time:.3f}s")
                
            except Exception as e:
                query_result = {
                    'query': query,
                    'answer': None,
                    'confidence': 0.0,
                    'source_count': 0,
                    'processing_time': 0.0,
                    'success': False,
                    'error': str(e)
                }
                
                print(f"❌ Query failed: {e}")
            
            query_results.append(query_result)
        
        total_time = time.time() - total_start_time
        
        # Calculate performance metrics / パフォーマンスメトリクスを計算
        successful_queries = [r for r in query_results if r['success']]
        
        if successful_queries:
            avg_time = sum(r['processing_time'] for r in successful_queries) / len(successful_queries)
            avg_confidence = sum(r['confidence'] for r in successful_queries) / len(successful_queries)
            avg_sources = sum(r['source_count'] for r in successful_queries) / len(successful_queries)
            
            performance_metrics = {
                'total_queries': len(test_queries),
                'successful_queries': len(successful_queries),
                'success_rate': len(successful_queries) / len(test_queries) * 100,
                'average_response_time': avg_time,
                'average_confidence': avg_confidence,
                'average_sources': avg_sources,
                'total_time': total_time,
                'throughput': len(test_queries) / total_time
            }
            
            print(f"\n📈 QueryEngine Performance Summary:")
            print(f"   Success rate: {performance_metrics['success_rate']:.1f}%")
            print(f"   Average response time: {avg_time:.3f}s")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   Average sources per query: {avg_sources:.1f}")
            print(f"   Throughput: {performance_metrics['throughput']:.2f} queries/sec")
            
            self.performance_metrics['query_engine'] = performance_metrics
        
        # Save query results / クエリ結果を保存
        query_results_file = self.reports_dir / "query_results.json"
        with open(query_results_file, 'w', encoding='utf-8') as f:
            json.dump(query_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Query results saved to: {query_results_file}")
        
        return {
            'query_results': query_results,
            'performance_metrics': performance_metrics if successful_queries else None
        }
    
    def part3_evaluation(self) -> Dict[str, Any]:
        """
        Part 3: Comprehensive evaluation with QualityLab
        Part 3: QualityLabによる包括的評価
        """
        
        print("\n" + "="*70)
        print("🔬 PART 3: EVALUATION / パート3: 評価")
        print("="*70)
        
        # Initialize QualityLab / QualityLabを初期化
        print("\n🧪 Initializing QualityLab...")
        print("🧪 QualityLabを初期化中...")
        
        quality_lab_config = QualityLabConfig(
            qa_pairs_per_document=2,
            similarity_threshold=0.8,
            question_types=["factual", "conceptual", "analytical"],
            evaluation_metrics=["bleu", "rouge", "llm_judge"],
            include_detailed_analysis=True,
            include_contradiction_detection=True,
            output_format="markdown"
        )
        
        self.quality_lab = QualityLab(
            corpus_name="complete_tutorial",
            config=quality_lab_config
        )
        
        print(f"✅ QualityLab initialized")
        
        # Load documents for evaluation / 評価用文書をロード
        documents = []
        for file_path in self.knowledge_base_dir.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(
                    id=file_path.stem,
                    content=content,
                    metadata={'source': str(file_path), 'topic': file_path.stem}
                ))
        
        # Generate QA pairs / QAペアを生成
        print(f"\n📝 Generating QA pairs from {len(documents)} documents...")
        print(f"📝 {len(documents)}文書からQAペアを生成中...")
        
        qa_pairs = self.quality_lab.generate_qa_pairs(
            documents=documents,
            num_pairs=12,
            question_types=["factual", "conceptual", "analytical"]
        )
        
        print(f"✅ Generated {len(qa_pairs)} QA pairs")
        
        # Analyze QA distribution / QA分布を分析
        type_dist = {}
        for qa_pair in qa_pairs:
            q_type = qa_pair.metadata.get('question_type', 'unknown')
            type_dist[q_type] = type_dist.get(q_type, 0) + 1
        
        print(f"   Question type distribution:")
        for q_type, count in type_dist.items():
            print(f"     {q_type}: {count}")
        
        # Evaluate QueryEngine / QueryEngineを評価
        print(f"\n🔍 Evaluating QueryEngine performance...")
        print(f"🔍 QueryEngineパフォーマンスを評価中...")
        
        start_time = time.time()
        self.evaluation_results = self.quality_lab.evaluate_query_engine(
            query_engine=self.query_engine,
            qa_pairs=qa_pairs,
            evaluation_metrics=["bleu", "rouge"],  # Simplified for demo
            include_contradiction_detection=False,  # Simplified for demo
            detailed_analysis=True
        )
        evaluation_time = time.time() - start_time
        
        print(f"✅ Evaluation completed in {evaluation_time:.2f}s")
        
        # Analyze evaluation results / 評価結果を分析
        test_results = self.evaluation_results.get('test_results', [])
        if test_results:
            passed_tests = sum(1 for result in test_results if result.get('passed', False))
            pass_rate = (passed_tests / len(test_results)) * 100
            
            avg_score = sum(result.get('score', 0) for result in test_results) / len(test_results)
            avg_confidence = sum(result.get('confidence', 0) for result in test_results) / len(test_results)
            
            evaluation_summary = {
                'total_tests': len(test_results),
                'passed_tests': passed_tests,
                'pass_rate': pass_rate,
                'average_score': avg_score,
                'average_confidence': avg_confidence,
                'evaluation_time': evaluation_time
            }
            
            print(f"\n📊 Evaluation Summary:")
            print(f"   Total tests: {len(test_results)}")
            print(f"   Pass rate: {pass_rate:.1f}% ({passed_tests}/{len(test_results)})")
            print(f"   Average score: {avg_score:.3f}")
            print(f"   Average confidence: {avg_confidence:.3f}")
            
            self.performance_metrics['evaluation'] = evaluation_summary
        
        # Generate evaluation report / 評価レポートを生成
        print(f"\n📊 Generating comprehensive evaluation report...")
        print(f"📊 包括的評価レポートを生成中...")
        
        try:
            report_path = self.reports_dir / "evaluation_report.md"
            report = self.quality_lab.generate_evaluation_report(
                evaluation_results=self.evaluation_results,
                output_file=str(report_path),
                include_detailed_analysis=True,
                include_recommendations=True
            )
            
            print(f"✅ Evaluation report generated: {report_path}")
            print(f"   Report length: {len(report)} characters")
            
        except Exception as e:
            print(f"⚠️  Report generation had issues: {e}")
            print(f"   Evaluation data still available for analysis")
        
        # Save evaluation data / 評価データを保存
        eval_data_file = self.reports_dir / "evaluation_data.json"
        with open(eval_data_file, 'w', encoding='utf-8') as f:
            # Prepare JSON-serializable data / JSON化可能データを準備
            json_data = {
                'qa_pairs_count': len(qa_pairs),
                'qa_type_distribution': type_dist,
                'evaluation_summary': evaluation_summary if test_results else None,
                'test_results_count': len(test_results),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Evaluation data saved to: {eval_data_file}")
        
        return {
            'qa_pairs': qa_pairs,
            'evaluation_results': self.evaluation_results,
            'evaluation_summary': evaluation_summary if test_results else None
        }
    
    def generate_final_report(self) -> str:
        """
        Generate comprehensive final report
        包括的最終レポートを生成
        """
        
        print("\n" + "="*70)
        print("📋 GENERATING FINAL REPORT / 最終レポート生成")
        print("="*70)
        
        report_content = f"""# Complete RAG Tutorial Report
# 完全RAGチュートリアルレポート

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
生成日時: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary / 要約

This report presents the results of a comprehensive RAG (Retrieval-Augmented Generation) 
system implementation and evaluation using refinire-rag. The tutorial demonstrated 
the complete workflow from corpus creation to evaluation.

このレポートは、refinire-ragを使用した包括的なRAG（検索拡張生成）システムの
実装と評価の結果を示します。チュートリアルでは、コーパス作成から評価まで
の完全なワークフローを実演しました。

## Part 1: Corpus Creation Results / パート1: コーパス作成結果

### Knowledge Base / 知識ベース
- Documents: {len(list(self.knowledge_base_dir.glob('*.txt')))} files
- Topics: AI Fundamentals, Machine Learning, Deep Learning, NLP, Computer Vision
- Total chunks created: {self.corpus_stats.total_chunks_created if self.corpus_stats else 'N/A'}
- Processing time: {self.corpus_stats.total_processing_time:.3f}s

### Corpus Building Approaches Comparison / コーパス構築アプローチ比較
Multiple corpus building strategies were tested and compared for effectiveness.

"""
        
        # Add performance metrics if available / パフォーマンスメトリクスがあれば追加
        if 'query_engine' in self.performance_metrics:
            qe_metrics = self.performance_metrics['query_engine']
            report_content += f"""
## Part 2: QueryEngine Performance / パート2: QueryEngineパフォーマンス

### Overall Performance / 全体パフォーマンス
- Success Rate: {qe_metrics['success_rate']:.1f}%
- Average Response Time: {qe_metrics['average_response_time']:.3f}s
- Average Confidence: {qe_metrics['average_confidence']:.3f}
- Average Sources per Query: {qe_metrics['average_sources']:.1f}
- Throughput: {qe_metrics['throughput']:.2f} queries/sec

### Query Testing / クエリテスト
The QueryEngine was tested with {qe_metrics['total_queries']} diverse queries covering 
AI fundamentals, machine learning concepts, and technical implementations.
QueryEngineは、AI基礎、機械学習概念、技術実装をカバーする
{qe_metrics['total_queries']}の多様なクエリでテストされました。
"""
        
        if 'evaluation' in self.performance_metrics:
            eval_metrics = self.performance_metrics['evaluation']
            report_content += f"""
## Part 3: Evaluation Results / パート3: 評価結果

### QA Evaluation / QA評価
- Total Test Cases: {eval_metrics['total_tests']}
- Pass Rate: {eval_metrics['pass_rate']:.1f}%
- Average Score: {eval_metrics['average_score']:.3f}
- Average Confidence: {eval_metrics['average_confidence']:.3f}
- Evaluation Time: {eval_metrics['evaluation_time']:.2f}s

### Quality Assessment / 品質評価
The evaluation demonstrated the system's ability to provide accurate and relevant 
answers across diverse question types including factual, conceptual, and analytical queries.
評価では、事実、概念、分析的質問を含む多様な質問タイプにわたって、
正確で関連性の高い回答を提供するシステムの能力が実証されました。
"""
        
        report_content += f"""
## Key Findings / 主要な発見

### Strengths / 強み
1. **Comprehensive Architecture**: Successfully integrated corpus creation, 
   query processing, and evaluation in a unified framework.
   **包括的アーキテクチャ**: コーパス作成、クエリ処理、評価を
   統一フレームワークで正常に統合。

2. **Flexible Configuration**: Multiple corpus building approaches allow 
   optimization for different use cases.
   **柔軟な設定**: 複数のコーパス構築アプローチにより、
   異なるユースケースに最適化可能。

3. **Quality Evaluation**: Automated evaluation system provides 
   comprehensive performance assessment.
   **品質評価**: 自動評価システムが包括的なパフォーマンス評価を提供。

### Areas for Improvement / 改善領域
1. **Response Time Optimization**: Further optimization could improve query response times.
   **応答時間最適化**: さらなる最適化によりクエリ応答時間を改善可能。

2. **Domain Specialization**: Specialized embeddings and models could enhance 
   domain-specific performance.
   **ドメイン特化**: 特化した埋め込みとモデルでドメイン固有の
   パフォーマンスを向上可能。

## Recommendations / 推奨事項

### For Production Deployment / 本番デプロイ用
1. Implement persistent vector storage (e.g., Chroma, FAISS)
2. Add monitoring and logging for production readiness
3. Implement caching for frequently asked questions
4. Consider domain-specific fine-tuning

### For Further Development / さらなる開発用
1. Explore advanced retrieval strategies (hybrid search)
2. Implement custom evaluation metrics for specific domains
3. Add multi-language support
4. Develop specialized processors for different document types

## Conclusion / 結論

The complete RAG tutorial successfully demonstrated the full lifecycle of 
a retrieval-augmented generation system using refinire-rag. The system 
showed strong performance across all evaluated dimensions and provides 
a solid foundation for production RAG applications.

完全RAGチュートリアルは、refinire-ragを使用した検索拡張生成システムの
全ライフサイクルを正常に実演しました。システムは評価されたすべての
次元で強いパフォーマンスを示し、本番RAGアプリケーションのための
堅実な基盤を提供します。

## Generated Files / 生成ファイル
- Knowledge Base: {self.knowledge_base_dir}
- Reports: {self.reports_dir}
- Data: {self.data_dir}

---
*Report generated by refinire-rag Complete Tutorial*
*refinire-rag完全チュートリアルにより生成されたレポート*
"""
        
        # Save final report / 最終レポートを保存
        final_report_path = self.reports_dir / "complete_tutorial_report.md"
        with open(final_report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ Final report generated: {final_report_path}")
        print(f"   Report length: {len(report_content)} characters")
        
        return str(final_report_path)
    
    def run_complete_tutorial(self) -> bool:
        """
        Run the complete RAG tutorial workflow
        完全RAGチュートリアルワークフローを実行
        """
        
        print("🚀 Starting Complete RAG Tutorial")
        print("🚀 完全RAGチュートリアル開始")
        print("="*70)
        print("This tutorial demonstrates the complete RAG workflow:")
        print("  Part 1: Corpus Creation (Document processing & indexing)")
        print("  Part 2: Query Engine (Search & answer generation)")
        print("  Part 3: Evaluation (Performance assessment & reporting)")
        print("")
        print("このチュートリアルは完全なRAGワークフローを実演します：")
        print("  パート1: コーパス作成（文書処理とインデックス）")
        print("  パート2: クエリエンジン（検索と回答生成）")
        print("  パート3: 評価（パフォーマンス評価とレポート）")
        
        tutorial_start_time = time.time()
        
        try:
            # Create knowledge base / 知識ベースを作成
            file_paths = self.create_knowledge_base()
            
            # Part 1: Corpus Creation / パート1: コーパス作成
            corpus_results = self.part1_corpus_creation(file_paths)
            
            # Part 2: Query Engine / パート2: クエリエンジン
            query_results = self.part2_query_engine()
            
            # Part 3: Evaluation / パート3: 評価
            evaluation_results = self.part3_evaluation()
            
            # Generate final report / 最終レポートを生成
            final_report_path = self.generate_final_report()
            
            # Tutorial completion / チュートリアル完了
            tutorial_time = time.time() - tutorial_start_time
            
            print("\n" + "="*70)
            print("🎉 COMPLETE RAG TUTORIAL FINISHED / 完全RAGチュートリアル完了")
            print("="*70)
            print("✅ All parts completed successfully!")
            print("✅ すべてのパートが正常に完了しました！")
            print("")
            print(f"📊 Tutorial Statistics / チュートリアル統計:")
            print(f"   Total time: {tutorial_time:.2f}s")
            print(f"   Knowledge base documents: {len(file_paths)}")
            if self.corpus_stats:
                print(f"   Corpus chunks created: {self.corpus_stats.total_chunks_created}")
            if 'query_engine' in self.performance_metrics:
                qe_metrics = self.performance_metrics['query_engine']
                print(f"   Queries tested: {qe_metrics['total_queries']}")
                print(f"   Query success rate: {qe_metrics['success_rate']:.1f}%")
            if 'evaluation' in self.performance_metrics:
                eval_metrics = self.performance_metrics['evaluation']
                print(f"   Evaluation tests: {eval_metrics['total_tests']}")
                print(f"   Evaluation pass rate: {eval_metrics['pass_rate']:.1f}%")
            
            print(f"\n📁 Generated Files / 生成ファイル:")
            print(f"   Working directory: {self.work_dir}")
            print(f"   Knowledge base: {self.knowledge_base_dir}")
            print(f"   Reports: {self.reports_dir}")
            print(f"   Final report: {final_report_path}")
            
            print(f"\n🎓 Tutorial Learning Outcomes / チュートリアル学習成果:")
            print(f"   ✓ Corpus creation with multiple strategies")
            print(f"     複数戦略でのコーパス作成")
            print(f"   ✓ QueryEngine configuration and optimization")
            print(f"     QueryEngineの設定と最適化")
            print(f"   ✓ Comprehensive RAG system evaluation")
            print(f"     包括的RAGシステム評価")
            print(f"   ✓ End-to-end workflow integration")
            print(f"     エンドツーエンドワークフロー統合")
            
            print(f"\n🚀 Next Steps / 次のステップ:")
            print(f"   • Explore the generated reports for detailed insights")
            print(f"     詳細な洞察のために生成されたレポートを探索")
            print(f"   • Customize the system for your specific domain")
            print(f"     特定ドメイン用にシステムをカスタマイズ")
            print(f"   • Deploy to production with monitoring")
            print(f"     監視付きで本番環境にデプロイ")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Tutorial failed / チュートリアル失敗: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """
    Main function to run the complete RAG tutorial
    完全RAGチュートリアルを実行するメイン関数
    """
    
    # Create working directory / 作業ディレクトリを作成
    temp_dir = Path(tempfile.mkdtemp(prefix="complete_rag_tutorial_"))
    
    try:
        # Initialize and run tutorial / チュートリアルを初期化して実行
        tutorial = CompleteRAGTutorial(temp_dir)
        success = tutorial.run_complete_tutorial()
        
        if success:
            print(f"\n🎊 Tutorial completed successfully!")
            print(f"🎊 チュートリアルが正常に完了しました！")
            print(f"\n📂 All files are available in: {temp_dir}")
            print(f"📂 すべてのファイルが利用可能: {temp_dir}")
        else:
            print(f"\n💥 Tutorial encountered errors")
            print(f"💥 チュートリアルでエラーが発生しました")
        
        return success
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Tutorial interrupted by user")
        print(f"⏹️  ユーザーによりチュートリアルが中断されました")
        return False
    
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(f"💥 予期しないエラー: {e}")
        return False
    
    finally:
        # Note: Keeping temp directory for inspection
        # 注意: 検査のため一時ディレクトリを保持
        print(f"\n💡 Tip: Temporary files kept for inspection at {temp_dir}")
        print(f"💡 ヒント: 検査用の一時ファイルを{temp_dir}に保持")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)