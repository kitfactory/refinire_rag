# チュートリアル2: コーパス管理とドキュメント処理

このチュートリアルでは、refinire-ragの高度なコーパス管理機能と、マルチステージドキュメント処理パイプラインを学習します。

## 学習目標

- CorpusManagerの3つのアプローチを理解する
- ドキュメント正規化と辞書作成を体験する
- マルチステージパイプラインを構築する
- カスタムパイプライン設定をマスターする

## CorpusManagerの3つのアプローチ

refinire-ragのCorpusManagerは、異なるレベルの複雑さに対応する3つのアプローチを提供します：

### 1. プリセット設定（Preset Configurations）
事前定義されたパイプライン構成：
- **Simple RAG**: Load → Chunk → Vector
- **Semantic RAG**: Load → Dictionary → Normalize → Chunk → Vector  
- **Knowledge RAG**: Load → Dictionary → Graph → Normalize → Chunk → Vector

### 2. ステージ選択（Stage Selection）
個別ステージを選択してカスタムパイプラインを構築：
- `["load", "dictionary", "chunk", "vector"]`
- 各ステージに個別設定を適用可能

### 3. カスタムパイプライン（Custom Pipelines）
完全にカスタマイズされたパイプライン：
- 複数パイプラインの順次実行
- 任意のDocumentProcessorの組み合わせ

## ステップ1: セットアップとサンプルファイル作成

```python
import tempfile
from pathlib import Path

def create_sample_files_with_variations(temp_dir: Path):
    """表現揺らぎを含むサンプルファイルを作成"""
    
    files = []
    
    # ファイル1: RAGとその表現揺らぎ
    file1 = temp_dir / "rag_overview.txt"
    file1.write_text("""
    RAG（Retrieval-Augmented Generation）は革新的なAI技術です。
    検索拡張生成システムとして、LLMと知識ベースを統合します。
    このRAGシステムは企業で広く使われています。
    検索強化生成とも呼ばれ、情報検索と生成を組み合わせます。
    """, encoding='utf-8')
    files.append(str(file1))
    
    # ファイル2: ベクトル検索の表現揺らぎ
    file2 = temp_dir / "vector_search.txt"
    file2.write_text("""
    ベクトル検索は意味的類似性を基にした検索手法です。
    セマンティック検索とも呼ばれます。
    文書埋め込みを使って意味検索を実現します。
    従来のキーワード検索と異なり、文脈を理解します。
    """, encoding='utf-8')
    files.append(str(file2))
    
    # ファイル3: LLMとその用途
    file3 = temp_dir / "llm_applications.txt"
    file3.write_text("""
    大規模言語モデル（LLM）は自然言語処理の中核です。
    言語モデルとして、文章生成や翻訳を行います。
    LLMモデルは企業でも広く活用されています。
    GPT、Claude、Geminiなどが代表的なLLMです。
    """, encoding='utf-8')
    files.append(str(file3))
    
    return files

def create_test_dictionary(temp_dir: Path):
    """用語統一辞書を作成"""
    
    dict_file = temp_dir / "domain_dictionary.md"
    dict_file.write_text("""# ドメイン用語辞書

## AI・機械学習用語

- **RAG** (Retrieval-Augmented Generation): 検索拡張生成
  - 表現揺らぎ: 検索拡張生成, 検索強化生成, RAGシステム

- **ベクトル検索** (Vector Search): ベクトル検索
  - 表現揺らぎ: ベクトル検索, 意味検索, セマンティック検索

- **LLM** (Large Language Model): 大規模言語モデル
  - 表現揺らぎ: 大規模言語モデル, 言語モデル, LLMモデル

- **埋め込み** (Embedding): 埋め込み
  - 表現揺らぎ: 埋め込み, エンベディング, ベクトル表現
""", encoding='utf-8')
    
    return str(dict_file)
```

## ステップ2: プリセット設定の比較

3つのプリセット設定を比較してみましょう：

```python
def demo_preset_configurations(temp_dir: Path, file_paths: list):
    """プリセット設定のデモ"""
    
    from refinire_rag.use_cases.corpus_manager_new import CorpusManager
    from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
    from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
    
    print("\\n" + "="*60)
    print("🎯 プリセット設定比較デモ")
    print("="*60)
    
    # 1. Simple RAG
    print("\\n📌 Simple RAG: Load → Chunk → Vector")
    doc_store1 = SQLiteDocumentStore(":memory:")
    vector_store1 = InMemoryVectorStore()
    
    simple_manager = CorpusManager.create_simple_rag(doc_store1, vector_store1)
    
    try:
        simple_stats = simple_manager.build_corpus(file_paths)
        print(f"✅ Simple RAG完了:")
        print(f"   - 処理文書数: {simple_stats.total_documents_created}")
        print(f"   - 処理時間: {simple_stats.total_processing_time:.3f}秒")
        print(f"   - 実行ステージ数: {simple_stats.pipeline_stages_executed}")
    except Exception as e:
        print(f"❌ Simple RAG失敗: {e}")
    
    # 2. Semantic RAG
    print("\\n📌 Semantic RAG: Load → Dictionary → Normalize → Chunk → Vector")
    doc_store2 = SQLiteDocumentStore(":memory:")
    vector_store2 = InMemoryVectorStore()
    
    semantic_manager = CorpusManager.create_semantic_rag(doc_store2, vector_store2)
    
    try:
        semantic_stats = semantic_manager.build_corpus(file_paths)
        print(f"✅ Semantic RAG完了:")
        print(f"   - 処理文書数: {semantic_stats.total_documents_created}")
        print(f"   - 処理時間: {semantic_stats.total_processing_time:.3f}秒")
        print(f"   - 実行ステージ数: {semantic_stats.pipeline_stages_executed}")
    except Exception as e:
        print(f"❌ Semantic RAG失敗: {e}")
    
    # 3. Knowledge RAG  
    print("\\n📌 Knowledge RAG: Load → Dictionary → Graph → Normalize → Chunk → Vector")
    doc_store3 = SQLiteDocumentStore(":memory:")
    vector_store3 = InMemoryVectorStore()
    
    knowledge_manager = CorpusManager.create_knowledge_rag(doc_store3, vector_store3)
    
    try:
        knowledge_stats = knowledge_manager.build_corpus(file_paths)
        print(f"✅ Knowledge RAG完了:")
        print(f"   - 処理文書数: {knowledge_stats.total_documents_created}")
        print(f"   - 処理時間: {knowledge_stats.total_processing_time:.3f}秒")
        print(f"   - 実行ステージ数: {knowledge_stats.pipeline_stages_executed}")
    except Exception as e:
        print(f"❌ Knowledge RAG失敗: {e}")
```

## ステップ3: ステージ選択アプローチ

個別ステージを選択してカスタムパイプラインを構築：

```python
def demo_stage_selection(temp_dir: Path, file_paths: list, dict_path: str):
    """ステージ選択アプローチのデモ"""
    
    from refinire_rag.processing.dictionary_maker import DictionaryMakerConfig
    from refinire_rag.processing.normalizer import NormalizerConfig
    from refinire_rag.processing.chunker import ChunkingConfig
    from refinire_rag.loaders.base import LoaderConfig
    
    print("\\n" + "="*60)
    print("🎛️ ステージ選択アプローチデモ")
    print("="*60)
    
    # ストレージ初期化
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    corpus_manager = CorpusManager(doc_store, vector_store)
    
    # カスタムステージ設定
    stage_configs = {
        "loader_config": LoaderConfig(),
        "dictionary_config": DictionaryMakerConfig(
            dictionary_file_path=dict_path,
            focus_on_technical_terms=True,
            extract_abbreviations=True
        ),
        "normalizer_config": NormalizerConfig(
            dictionary_file_path=dict_path,
            normalize_variations=True,
            expand_abbreviations=True,
            whole_word_only=False  # 日本語対応
        ),
        "chunker_config": ChunkingConfig(
            chunk_size=300,
            overlap=50,
            split_by_sentence=True
        )
    }
    
    # 選択するステージ
    selected_stages = ["load", "dictionary", "normalize", "chunk", "vector"]
    
    print(f"📋 選択ステージ: {selected_stages}")
    print("📝 各ステージの設定:")
    for key, config in stage_configs.items():
        print(f"   - {key}: {type(config).__name__}")
    
    try:
        stats = corpus_manager.build_corpus(
            file_paths=file_paths,
            stages=selected_stages,
            stage_configs=stage_configs
        )
        
        print(f"\\n✅ ステージ選択パイプライン完了:")
        print(f"   - 処理文書数: {stats.total_documents_created}")
        print(f"   - チャンク数: {stats.total_chunks_created}")
        print(f"   - 処理時間: {stats.total_processing_time:.3f}秒")
        print(f"   - ステージ別文書数: {stats.documents_by_stage}")
        
        # 生成された辞書の確認
        if Path(dict_path).exists():
            print(f"\\n📖 生成された辞書:")
            with open(dict_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\\n')[:10]
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
                if len(content.split('\\n')) > 10:
                    print(f"   ... (他{len(content.split('\\n')) - 10}行)")
        
    except Exception as e:
        print(f"❌ ステージ選択パイプライン失敗: {e}")
        import traceback
        traceback.print_exc()
```

## ステップ4: カスタムパイプライン

完全にカスタマイズされたマルチステージパイプライン：

```python
def demo_custom_pipelines(temp_dir: Path, file_paths: list, dict_path: str):
    """カスタムパイプラインのデモ"""
    
    from refinire_rag.processing.document_pipeline import DocumentPipeline
    from refinire_rag.processing.document_store_processor import DocumentStoreProcessor
    from refinire_rag.processing.document_store_loader import DocumentStoreLoader, DocumentStoreLoaderConfig
    from refinire_rag.loaders.specialized import TextLoader
    from refinire_rag.processing.dictionary_maker import DictionaryMaker, DictionaryMakerConfig
    from refinire_rag.processing.normalizer import Normalizer, NormalizerConfig
    from refinire_rag.processing.chunker import Chunker, ChunkingConfig
    from refinire_rag.loaders.base import LoaderConfig
    
    print("\\n" + "="*60)
    print("🔧 カスタムパイプラインデモ")
    print("="*60)
    
    # ストレージ初期化
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    
    # マルチステージカスタムパイプラインの定義
    custom_pipelines = [
        # ステージ1: ロードと原文保存
        DocumentPipeline([
            TextLoader(LoaderConfig()),
            DocumentStoreProcessor(doc_store)
        ]),
        
        # ステージ2: 辞書作成（原文から）
        DocumentPipeline([
            DocumentStoreLoader(doc_store, 
                              config=DocumentStoreLoaderConfig(processing_stage="original")),
            DictionaryMaker(DictionaryMakerConfig(
                dictionary_file_path=str(temp_dir / "custom_dictionary.md"),
                focus_on_technical_terms=True
            ))
        ]),
        
        # ステージ3: 正規化と保存
        DocumentPipeline([
            DocumentStoreLoader(doc_store,
                              config=DocumentStoreLoaderConfig(processing_stage="original")),
            Normalizer(NormalizerConfig(
                dictionary_file_path=str(temp_dir / "custom_dictionary.md"),
                normalize_variations=True,
                whole_word_only=False
            )),
            DocumentStoreProcessor(doc_store)
        ]),
        
        # ステージ4: チャンキング
        DocumentPipeline([
            DocumentStoreLoader(doc_store,
                              config=DocumentStoreLoaderConfig(processing_stage="normalized")),
            Chunker(ChunkingConfig(
                chunk_size=200,
                overlap=30,
                split_by_sentence=True
            ))
        ])
    ]
    
    print(f"📋 カスタムパイプライン構成:")
    for i, pipeline in enumerate(custom_pipelines, 1):
        processors = [type(p).__name__ for p in pipeline.processors]
        print(f"   ステージ{i}: {' → '.join(processors)}")
    
    # パイプライン実行
    corpus_manager = CorpusManager(doc_store, vector_store)
    
    try:
        stats = corpus_manager.build_corpus(
            file_paths=file_paths,
            custom_pipelines=custom_pipelines
        )
        
        print(f"\\n✅ カスタムパイプライン完了:")
        print(f"   - パイプライン数: {len(custom_pipelines)}")
        print(f"   - 処理文書数: {stats.total_documents_created}")
        print(f"   - チャンク数: {stats.total_chunks_created}")
        print(f"   - 処理時間: {stats.total_processing_time:.3f}秒")
        print(f"   - エラー数: {stats.errors_encountered}")
        
    except Exception as e:
        print(f"❌ カスタムパイプライン失敗: {e}")
        import traceback
        traceback.print_exc()
```

## ステップ5: 正規化効果の確認

ドキュメント正規化の効果を確認：

```python
def demonstrate_normalization_effects(temp_dir: Path):
    """正規化効果のデモ"""
    
    from refinire_rag.processing.normalizer import Normalizer, NormalizerConfig
    from refinire_rag.models.document import Document
    
    print("\\n" + "="*60)
    print("🔄 正規化効果デモ")
    print("="*60)
    
    # 辞書ファイルパス
    dict_path = create_test_dictionary(temp_dir)
    
    # 正規化設定
    normalizer_config = NormalizerConfig(
        dictionary_file_path=dict_path,
        normalize_variations=True,
        expand_abbreviations=True,
        whole_word_only=False
    )
    
    normalizer = Normalizer(normalizer_config)
    
    # テスト文書
    test_texts = [
        "検索強化生成は革新的な技術です",
        "意味検索の仕組みを説明します", 
        "LLMモデルの特徴について",
        "セマンティック検索とRAGシステム"
    ]
    
    print("📝 正規化前後の比較:")
    print("-" * 50)
    
    for i, text in enumerate(test_texts, 1):
        # 正規化実行
        doc = Document(id=f"test_{i}", content=text, metadata={})
        normalized_docs = normalizer.process(doc)
        
        normalized_text = normalized_docs[0].content if normalized_docs else text
        
        print(f"\\n{i}. 元の文章:")
        print(f"   「{text}」")
        print(f"   正規化後:")
        print(f"   「{normalized_text}」")
        
        if text != normalized_text:
            print(f"   🔄 変更: あり")
        else:
            print(f"   🔄 変更: なし")
```

## ステップ6: 完全なサンプルプログラム

全ての機能を統合したサンプルプログラム：

```python
#!/usr/bin/env python3
"""
チュートリアル2: コーパス管理とドキュメント処理
"""

def main():
    """メイン関数"""
    
    print("🚀 コーパス管理とドキュメント処理 チュートリアル")
    print("="*60)
    print("高度なコーパス管理機能とマルチステージ処理を学習します")
    
    # 一時ディレクトリ作成
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # サンプルファイル作成
        print("\\n📁 サンプルファイル作成")
        file_paths = create_sample_files_with_variations(temp_dir)
        dict_path = create_test_dictionary(temp_dir)
        print(f"✅ {len(file_paths)}個のファイルと辞書を作成")
        
        # プリセット設定のデモ
        demo_preset_configurations(temp_dir, file_paths)
        
        # ステージ選択のデモ
        demo_stage_selection(temp_dir, file_paths, dict_path)
        
        # カスタムパイプラインのデモ
        demo_custom_pipelines(temp_dir, file_paths, dict_path)
        
        # 正規化効果のデモ
        demonstrate_normalization_effects(temp_dir)
        
        print("\\n🎉 チュートリアル2が完了しました！")
        print("\\n📚 学習内容:")
        print("   ✅ 3つのプリセット設定（Simple/Semantic/Knowledge RAG）")
        print("   ✅ ステージ選択による柔軟なパイプライン構築")
        print("   ✅ カスタムパイプラインによる完全制御")
        print("   ✅ ドキュメント正規化と表現揺らぎ統一")
        
        print(f"\\n📁 生成ファイル: {temp_dir}")
        
    except Exception as e:
        print(f"\\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
```

## 実行と結果

このチュートリアルを実行すると、以下のような処理フローが確認できます：

### プリセット設定の比較
- Simple RAG: 基本的なベクトル検索
- Semantic RAG: 用語正規化による検索精度向上
- Knowledge RAG: グラフ情報も活用した高度な処理

### ステージ選択の柔軟性
- 必要なステージのみ選択
- 各ステージの詳細設定
- 段階的な品質向上

### カスタムパイプラインの威力
- 複数パイプラインの順次実行
- DocumentStoreを介したデータ永続化
- 処理段階の完全制御

## 理解度チェック

1. **3つのプリセット設定**の違いは？
2. **ステージ選択**で可能なカスタマイズは？
3. **ドキュメント正規化**の効果は？
4. **マルチステージパイプライン**の利点は？

## 次のステップ

[チュートリアル3: クエリエンジンと回答生成](tutorial_03_query_engine.md)で、構築したコーパスを使った高度なクエリ処理を学習しましょう。