#!/usr/bin/env python3
"""
Part 1: Corpus Creation Tutorial Example
コーパス作成チュートリアル例

This example demonstrates comprehensive corpus creation using refinire-rag's CorpusManager
with different approaches: preset configurations, stage selection, and custom pipelines.

この例では、refinire-ragのCorpusManagerを使用した包括的なコーパス作成を、
プリセット設定、ステージ選択、カスタムパイプラインの異なるアプローチで実演します。
"""

import sys
import tempfile
import shutil
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.processing.dictionary_maker import DictionaryMakerConfig
from refinire_rag.processing.normalizer import NormalizerConfig
from refinire_rag.processing.chunker import ChunkingConfig
from refinire_rag.loader.loader import LoaderConfig


def create_sample_documents(temp_dir: Path) -> List[str]:
    """
    Create sample documents for corpus creation demonstration
    コーパス作成デモ用のサンプル文書を作成
    """
    
    print("📄 Creating sample documents...")
    files = []
    
    # AI Overview document
    ai_doc = temp_dir / "ai_overview.txt"
    ai_doc.write_text("""
人工知能（AI：Artificial Intelligence）とは、人間の知的活動をコンピュータに代行させる技術です。
機械学習（Machine Learning, ML）、深層学習（Deep Learning, DL）、自然言語処理（NLP）、
コンピュータビジョン（Computer Vision）などの技術分野を含みます。

主要な応用分野：
- 自動運転技術（Autonomous Driving）
- 音声認識システム（Speech Recognition）
- 画像認識・分類（Image Classification）
- 機械翻訳（Machine Translation）
- 推薦システム（Recommendation System）

AIの発展により、従来人間が行っていた複雑な判断や創造的な作業も
コンピュータが実行できるようになってきています。
""", encoding='utf-8')
    files.append(str(ai_doc))
    
    # Machine Learning document
    ml_doc = temp_dir / "machine_learning.txt"
    ml_doc.write_text("""
機械学習（ML）は人工知能の一分野で、明示的にプログラムすることなく
コンピュータがデータから学習し、予測や判断を行う技術です。

主要なアルゴリズム：
- 線形回帰（Linear Regression）
- ロジスティック回帰（Logistic Regression）
- 決定木（Decision Tree）
- ランダムフォレスト（Random Forest）
- サポートベクターマシン（SVM: Support Vector Machine）
- ニューラルネットワーク（Neural Network）

学習の種類：
1. 教師あり学習（Supervised Learning）：正解データを使用
2. 教師なし学習（Unsupervised Learning）：正解データなしでパターン発見
3. 強化学習（Reinforcement Learning）：試行錯誤を通じた学習

機械学習は金融、医療、製造業、マーケティングなど
あらゆる分野で活用されています。
""", encoding='utf-8')
    files.append(str(ml_doc))
    
    # Deep Learning document
    dl_doc = temp_dir / "deep_learning.txt"
    dl_doc.write_text("""
深層学習（DL）は機械学習の一手法で、人間の脳の神経回路を模倣した
多層ニューラルネットワークを使用する技術です。

主要なアーキテクチャ：
- 畳み込みニューラルネットワーク（CNN: Convolutional Neural Network）
- 再帰ニューラルネットワーク（RNN: Recurrent Neural Network）
- 長短期記憶（LSTM: Long Short-Term Memory）
- トランスフォーマー（Transformer）
- 生成的敵対ネットワーク（GAN: Generative Adversarial Network）

応用例：
- 画像認識・物体検出
- 自然言語処理・機械翻訳
- 音声認識・合成
- ゲームAI（囲碁、チェス）
- 自動運転

深層学習の発達により、従来困難とされていた
パターン認識や予測タスクが大幅に改善されました。
""", encoding='utf-8')
    files.append(str(dl_doc))
    
    # RAG Technology document
    rag_doc = temp_dir / "rag_technology.txt"
    rag_doc.write_text("""
RAG（Retrieval-Augmented Generation：検索拡張生成）は、
大規模言語モデル（LLM）と外部知識ベースを組み合わせた技術です。

RAGの構成要素：
- 文書データベース（Document Database）
- ベクトル検索エンジン（Vector Search Engine）
- 埋め込みモデル（Embedding Model）
- 大規模言語モデル（Large Language Model）
- 検索・生成パイプライン（Retrieval-Generation Pipeline）

処理フロー：
1. 質問の埋め込み変換
2. ベクトル類似性による文書検索
3. 関連文書の取得
4. LLMによる回答生成

利点：
- 最新情報への対応
- ハルシネーション（幻覚）の減少
- 専門分野への適応性
- 根拠の明示

RAGは企業の文書検索、顧客サポート、研究支援などで
広く活用されています。
""", encoding='utf-8')
    files.append(str(rag_doc))
    
    print(f"✅ Created {len(files)} sample documents")
    return files


def demonstrate_preset_configurations(temp_dir: Path, file_paths: List[str]):
    """
    Demonstrate preset configurations
    プリセット設定のデモンストレーション
    """
    
    print("\n" + "="*60)
    print("🎯 PRESET CONFIGURATIONS DEMONSTRATION")
    print("🎯 プリセット設定のデモンストレーション")
    print("="*60)
    
    # 1. Simple RAG
    print("\n📌 1. Simple RAG (Load → Chunk → Vector)")
    print("📌 1. シンプルRAG（ロード → チャンク → ベクトル）")
    print("-" * 50)
    
    doc_store1 = SQLiteDocumentStore(":memory:")
    vector_store1 = InMemoryVectorStore()
    
    simple_manager = CorpusManager.create_simple_rag(doc_store1, vector_store1)
    simple_stats = simple_manager.build_corpus(file_paths)
    
    print(f"✅ Simple RAG Results / シンプルRAG結果:")
    print(f"   - Files processed / 処理ファイル数: {simple_stats.total_files_processed}")
    print(f"   - Documents created / 作成文書数: {simple_stats.total_documents_created}")
    print(f"   - Chunks created / 作成チャンク数: {simple_stats.total_chunks_created}")
    print(f"   - Processing time / 処理時間: {simple_stats.total_processing_time:.3f}s")
    print(f"   - Pipeline stages / パイプラインステージ: {simple_stats.pipeline_stages_executed}")
    
    # 2. Semantic RAG
    print("\n📌 2. Semantic RAG (Load → Dictionary → Normalize → Chunk → Vector)")
    print("📌 2. セマンティックRAG（ロード → 辞書 → 正規化 → チャンク → ベクトル）")
    print("-" * 50)
    
    doc_store2 = SQLiteDocumentStore(":memory:")
    vector_store2 = InMemoryVectorStore()
    
    semantic_manager = CorpusManager.create_semantic_rag(doc_store2, vector_store2)
    semantic_stats = semantic_manager.build_corpus(file_paths)
    
    print(f"✅ Semantic RAG Results / セマンティックRAG結果:")
    print(f"   - Files processed / 処理ファイル数: {semantic_stats.total_files_processed}")
    print(f"   - Documents created / 作成文書数: {semantic_stats.total_documents_created}")
    print(f"   - Chunks created / 作成チャンク数: {semantic_stats.total_chunks_created}")
    print(f"   - Processing time / 処理時間: {semantic_stats.total_processing_time:.3f}s")
    print(f"   - Enhanced with / 強化機能: Domain-specific dictionary / ドメイン固有辞書")
    
    # 3. Knowledge RAG
    print("\n📌 3. Knowledge RAG (Load → Dictionary → Graph → Normalize → Chunk → Vector)")
    print("📌 3. ナレッジRAG（ロード → 辞書 → グラフ → 正規化 → チャンク → ベクトル）")
    print("-" * 50)
    
    doc_store3 = SQLiteDocumentStore(":memory:")
    vector_store3 = InMemoryVectorStore()
    
    knowledge_manager = CorpusManager.create_knowledge_rag(doc_store3, vector_store3)
    knowledge_stats = knowledge_manager.build_corpus(file_paths)
    
    print(f"✅ Knowledge RAG Results / ナレッジRAG結果:")
    print(f"   - Files processed / 処理ファイル数: {knowledge_stats.total_files_processed}")
    print(f"   - Documents created / 作成文書数: {knowledge_stats.total_documents_created}")
    print(f"   - Chunks created / 作成チャンク数: {knowledge_stats.total_chunks_created}")
    print(f"   - Processing time / 処理時間: {knowledge_stats.total_processing_time:.3f}s")
    print(f"   - Enhanced with / 強化機能: Dictionary + Knowledge Graph / 辞書 + 知識グラフ")


def demonstrate_stage_selection(temp_dir: Path, file_paths: List[str]):
    """
    Demonstrate stage selection approach
    ステージ選択アプローチのデモンストレーション
    """
    
    print("\n" + "="*60)
    print("🎛️  STAGE SELECTION DEMONSTRATION")
    print("🎛️  ステージ選択のデモンストレーション")
    print("="*60)
    
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    corpus_manager = CorpusManager(doc_store, vector_store)
    
    print("\n📌 Custom Stage Selection: Load → Dictionary → Chunk → Vector")
    print("📌 カスタムステージ選択: ロード → 辞書 → チャンク → ベクトル")
    print("-" * 50)
    
    # Configure individual stages / 個々のステージを設定
    stage_configs = {
        "loader_config": LoaderConfig(),
        "dictionary_config": DictionaryMakerConfig(
            dictionary_file_path=str(temp_dir / "tutorial_dictionary.md"),
            focus_on_technical_terms=True,
            extract_abbreviations=True,
            include_definitions=True
        ),
        "chunker_config": ChunkingConfig(
            chunk_size=256,
            overlap=32,
            split_by_sentence=True
        )
    }
    
    # Execute selected stages only / 選択したステージのみを実行
    selected_stages = ["load", "dictionary", "chunk", "vector"]
    stage_stats = corpus_manager.build_corpus(
        file_paths=file_paths,
        stages=selected_stages,
        stage_configs=stage_configs
    )
    
    print(f"✅ Stage Selection Results / ステージ選択結果:")
    print(f"   - Selected stages / 選択ステージ: {selected_stages}")
    print(f"   - Files processed / 処理ファイル数: {stage_stats.total_files_processed}")
    print(f"   - Documents created / 作成文書数: {stage_stats.total_documents_created}")
    print(f"   - Chunks created / 作成チャンク数: {stage_stats.total_chunks_created}")
    print(f"   - Processing time / 処理時間: {stage_stats.total_processing_time:.3f}s")
    print(f"   - Documents by stage / ステージ別文書数: {stage_stats.documents_by_stage}")
    
    # Check generated dictionary / 生成された辞書を確認
    dict_file = temp_dir / "tutorial_dictionary.md"
    if dict_file.exists():
        print(f"\n📖 Generated Dictionary Preview / 生成された辞書のプレビュー:")
        content = dict_file.read_text(encoding='utf-8')
        lines = content.split('\n')[:10]  # Show first 10 lines
        for line in lines:
            if line.strip():
                print(f"   {line}")
        total_lines = len(content.split('\n'))
        if total_lines > 10:
            print(f"   ... ({total_lines - 10} more lines / さらに{total_lines - 10}行)")


def demonstrate_file_format_support(temp_dir: Path):
    """
    Demonstrate support for different file formats
    異なるファイル形式のサポートをデモンストレーション
    """
    
    print("\n" + "="*60)
    print("📁 FILE FORMAT SUPPORT DEMONSTRATION")
    print("📁 ファイル形式サポートのデモンストレーション")
    print("="*60)
    
    # Create different file format samples / 異なるファイル形式のサンプルを作成
    formats_dir = temp_dir / "formats"
    formats_dir.mkdir(exist_ok=True)
    
    # Text file
    (formats_dir / "sample.txt").write_text("This is a text file sample.", encoding='utf-8')
    
    # Markdown file
    (formats_dir / "sample.md").write_text("""
# Markdown Sample
This is a **markdown** file with formatting.
- List item 1
- List item 2
""", encoding='utf-8')
    
    # JSON file
    import json
    json_data = {
        "title": "Sample JSON",
        "content": "This is JSON format data",
        "metadata": {"type": "sample", "version": 1.0}
    }
    (formats_dir / "sample.json").write_text(json.dumps(json_data, ensure_ascii=False), encoding='utf-8')
    
    # CSV file
    (formats_dir / "sample.csv").write_text("""
name,description,category
AI,Artificial Intelligence,Technology
ML,Machine Learning,Technology
DL,Deep Learning,Technology
""", encoding='utf-8')
    
    # HTML file
    (formats_dir / "sample.html").write_text("""
<!DOCTYPE html>
<html>
<head><title>Sample HTML</title></head>
<body>
<h1>HTML Sample</h1>
<p>This is an HTML file example.</p>
</body>
</html>
""", encoding='utf-8')
    
    # Process all formats / すべての形式を処理
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    manager = CorpusManager.create_simple_rag(doc_store, vector_store)
    
    format_files = list(formats_dir.glob("*"))
    stats = manager.build_corpus([str(f) for f in format_files])
    
    print(f"✅ Multi-Format Processing Results / 複数形式処理結果:")
    print(f"   - Supported formats / サポート形式: TXT, MD, JSON, CSV, HTML")
    print(f"   - Files processed / 処理ファイル数: {stats.total_files_processed}")
    print(f"   - Documents created / 作成文書数: {stats.total_documents_created}")
    print(f"   - Chunks created / 作成チャンク数: {stats.total_chunks_created}")
    
    print(f"\n📋 File Details / ファイル詳細:")
    for file_path in format_files:
        print(f"   - {file_path.name}: {file_path.stat().st_size} bytes")


def demonstrate_incremental_loading(temp_dir: Path, initial_files: List[str]):
    """
    Demonstrate incremental loading capabilities
    増分ローディング機能のデモンストレーション
    """
    
    print("\n" + "="*60)
    print("⚡ INCREMENTAL LOADING DEMONSTRATION")
    print("⚡ 増分ローディングのデモンストレーション")
    print("="*60)
    
    # First: Load initial corpus / 最初：初期コーパスをロード
    doc_store = SQLiteDocumentStore(str(temp_dir / "incremental.db"))
    vector_store = InMemoryVectorStore()
    manager = CorpusManager.create_simple_rag(doc_store, vector_store)
    
    print("\n📌 Step 1: Initial corpus loading / ステップ1: 初期コーパスローディング")
    initial_stats = manager.build_corpus(initial_files)
    print(f"   - Initial files / 初期ファイル数: {initial_stats.total_files_processed}")
    print(f"   - Initial chunks / 初期チャンク数: {initial_stats.total_chunks_created}")
    
    # Second: Add new files / 次に：新しいファイルを追加
    new_file = temp_dir / "new_document.txt"
    new_file.write_text("""
新しい文書です。増分ローディングのテストに使用されます。
この文書は初期コーパス作成後に追加されました。

自然言語処理（NLP）の最新技術について：
- BERT（Bidirectional Encoder Representations from Transformers）
- GPT（Generative Pre-trained Transformer）
- T5（Text-to-Text Transfer Transformer）
""", encoding='utf-8')
    
    # Third: Incremental update / 第三：増分更新
    print("\n📌 Step 2: Incremental update / ステップ2: 増分更新")
    all_files = initial_files + [str(new_file)]
    
    # Use incremental loading / 増分ローディングを使用
    incremental_stats = manager.build_corpus(
        file_paths=all_files,
        use_incremental=True
    )
    
    print(f"   - Total files after update / 更新後総ファイル数: {incremental_stats.total_files_processed}")
    print(f"   - New chunks added / 新規追加チャンク数: {incremental_stats.total_chunks_created}")
    print(f"   - Incremental processing time / 増分処理時間: {incremental_stats.total_processing_time:.3f}s")
    print(f"   - Only new/modified files processed / 新規・変更ファイルのみ処理")


def demonstrate_monitoring_and_statistics(temp_dir: Path, file_paths: List[str]):
    """
    Demonstrate monitoring and statistics features
    監視と統計機能のデモンストレーション
    """
    
    print("\n" + "="*60)
    print("📊 MONITORING AND STATISTICS DEMONSTRATION")
    print("📊 監視と統計のデモンストレーション")
    print("="*60)
    
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    manager = CorpusManager.create_semantic_rag(doc_store, vector_store)
    
    # Build corpus with detailed monitoring / 詳細な監視でコーパスを構築
    stats = manager.build_corpus(file_paths)
    
    print(f"\n📈 Detailed Processing Statistics / 詳細な処理統計:")
    print(f"   - Total files processed / 総処理ファイル数: {stats.total_files_processed}")
    print(f"   - Total documents created / 総作成文書数: {stats.total_documents_created}")
    print(f"   - Total chunks created / 総作成チャンク数: {stats.total_chunks_created}")
    print(f"   - Total processing time / 総処理時間: {stats.total_processing_time:.3f}s")
    print(f"   - Pipeline stages executed / 実行パイプラインステージ: {stats.pipeline_stages_executed}")
    
    if hasattr(stats, 'documents_by_stage'):
        print(f"   - Documents by stage / ステージ別文書数:")
        for stage, count in stats.documents_by_stage.items():
            print(f"     * {stage}: {count}")
    
    if hasattr(stats, 'errors_encountered'):
        print(f"   - Errors encountered / 遭遇エラー数: {len(stats.errors_encountered)}")
        if stats.errors_encountered:
            print(f"   - Error details / エラー詳細:")
            for error in stats.errors_encountered[:3]:  # Show first 3 errors
                print(f"     * {error}")
    
    # Storage validation / ストレージ検証
    print(f"\n💾 Storage Validation / ストレージ検証:")
    total_docs = doc_store.count_documents() if hasattr(doc_store, 'count_documents') else "N/A"
    total_vectors = vector_store.count() if hasattr(vector_store, 'count') else len(vector_store._vectors)
    
    print(f"   - Documents in DocumentStore / DocumentStore内文書数: {total_docs}")
    print(f"   - Vectors in VectorStore / VectorStore内ベクトル数: {total_vectors}")
    
    # Performance metrics / パフォーマンス指標
    if stats.total_files_processed > 0:
        avg_time_per_file = stats.total_processing_time / stats.total_files_processed
        avg_chunks_per_file = stats.total_chunks_created / stats.total_files_processed
        
        print(f"\n⚡ Performance Metrics / パフォーマンス指標:")
        print(f"   - Average processing time per file / ファイル当たり平均処理時間: {avg_time_per_file:.3f}s")
        print(f"   - Average chunks per file / ファイル当たり平均チャンク数: {avg_chunks_per_file:.1f}")
        print(f"   - Processing throughput / 処理スループット: {stats.total_files_processed/stats.total_processing_time:.1f} files/sec")


def main():
    """
    Main demonstration function
    メインデモンストレーション関数
    """
    
    print("🚀 Part 1: Corpus Creation Tutorial")
    print("🚀 Part 1: コーパス作成チュートリアル")
    print("="*60)
    print("Comprehensive demonstration of corpus creation with refinire-rag")
    print("refinire-ragを使用したコーパス作成の包括的なデモンストレーション")
    print("")
    print("Features demonstrated / デモ機能:")
    print("✓ Preset configurations / プリセット設定")
    print("✓ Stage selection / ステージ選択") 
    print("✓ File format support / ファイル形式サポート")
    print("✓ Incremental loading / 増分ローディング")
    print("✓ Monitoring & statistics / 監視と統計")
    
    # Create temporary directory / 一時ディレクトリを作成
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create sample documents / サンプル文書を作成
        print(f"\n📁 Setup: Creating sample documents in {temp_dir}")
        print(f"📁 セットアップ: {temp_dir} にサンプル文書を作成中")
        file_paths = create_sample_documents(temp_dir)
        
        # Demonstration sequence / デモシーケンス
        demonstrate_preset_configurations(temp_dir, file_paths)
        demonstrate_stage_selection(temp_dir, file_paths)
        demonstrate_file_format_support(temp_dir)
        demonstrate_incremental_loading(temp_dir, file_paths)
        demonstrate_monitoring_and_statistics(temp_dir, file_paths)
        
        print("\n" + "="*60)
        print("🎉 TUTORIAL COMPLETE / チュートリアル完了")
        print("="*60)
        print("✅ All corpus creation demonstrations completed successfully!")
        print("✅ すべてのコーパス作成デモが正常に完了しました！")
        print("")
        print("📚 What you learned / 学習内容:")
        print("   • Preset configurations for quick setup / クイックセットアップ用プリセット設定")
        print("   • Custom stage selection for flexibility / 柔軟性のためのカスタムステージ選択")
        print("   • Multi-format file support / 複数形式ファイルサポート")
        print("   • Incremental loading for efficiency / 効率性のための増分ローディング")
        print("   • Comprehensive monitoring / 包括的な監視")
        print("")
        print(f"📁 Generated files available in: {temp_dir}")
        print(f"📁 生成ファイルの場所: {temp_dir}")
        print("")
        print("Next step / 次のステップ:")
        print("→ Part 2: Query Engine Tutorial (検索エンジンチュートリアル)")
        
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