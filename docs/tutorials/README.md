# refinire-rag Tutorials

## Overview / 概要

This directory contains comprehensive tutorials for learning and implementing RAG (Retrieval-Augmented Generation) systems using refinire-rag. The tutorials are designed to guide you through the complete RAG workflow from basic concepts to production deployment.

このディレクトリには、refinire-ragを使用したRAG（検索拡張生成）システムの学習と実装のための包括的なチュートリアルが含まれています。チュートリアルは、基本概念から本番デプロイまでの完全なRAGワークフローをガイドするように設計されています。

## Tutorial Structure / チュートリアル構造

### Core Tutorials / コアチュートリアル

The main tutorial sequence is divided into three interconnected parts:
メインチュートリアルシーケンスは、相互に関連する3つのパートに分かれています：

#### [Part 1: Corpus Creation](tutorial_part1_corpus_creation.md)
**コーパス作成（インデックス）**

Learn how to create and manage document corpora with refinire-rag's CorpusManager.

- **Topics Covered / 対象トピック:**
  - Preset configurations (Simple, Semantic, Knowledge RAG)
  - Custom stage selection and pipeline configuration  
  - Multi-format file support (TXT, CSV, JSON, HTML, MD)
  - Incremental loading and file tracking
  - Performance monitoring and optimization

- **Example Code / サンプルコード:** [`tutorial_part1_corpus_creation_example.py`](../examples/tutorial_part1_corpus_creation_example.py)

#### [Part 2: Query Engine](tutorial_part2_query_engine.md) 
**検索とクエリ処理**

Master query processing and answer generation with QueryEngine.

- **Topics Covered / 対象トピック:**
  - QueryEngine architecture and configuration
  - Retrieval, reranking, and answer synthesis
  - Query normalization and processing
  - Performance monitoring and debugging
  - Error handling and optimization

- **Example Code / サンプルコード:** [`tutorial_part2_query_engine_example.py`](../examples/tutorial_part2_query_engine_example.py)

#### [Part 3: Evaluation](tutorial_part3_evaluation.md)
**システム評価と品質管理**  

Evaluate and improve your RAG system with QualityLab.

- **Topics Covered / 対象トピック:**
  - Automated QA pair generation
  - Performance evaluation with multiple metrics
  - Contradiction detection and consistency analysis
  - Advanced evaluation techniques (LLM judge, robustness testing)
  - Comprehensive report generation

- **Example Code / サンプルコード:** [`tutorial_part3_evaluation_example.py`](../examples/tutorial_part3_evaluation_example.py)

### Integrated Tutorial / 統合チュートリアル

#### [Complete RAG Tutorial](../examples/complete_rag_tutorial.py)
**エンドツーエンド統合**

A comprehensive example that demonstrates the complete RAG workflow, integrating all three parts into a single, cohesive system.

- **Features / 機能:**
  - End-to-end workflow demonstration
  - Performance comparison across different approaches
  - Comprehensive evaluation and reporting
  - Production-ready patterns and best practices

### Additional Tutorials / 追加チュートリアル

#### Basic Tutorials / 基本チュートリアル
- [`tutorial_01_basic_rag.md`](tutorial_01_basic_rag.md) - Quick start guide
- [`tutorial_02_corpus_management.md`](tutorial_02_corpus_management.md) - Corpus management basics
- [`tutorial_03_query_engine.md`](tutorial_03_query_engine.md) - QueryEngine fundamentals

#### Advanced Topics / 高度なトピック
- [`tutorial_04_normalization.md`](tutorial_04_normalization.md) - Text normalization
- [`tutorial_05_enterprise_usage.md`](tutorial_05_enterprise_usage.md) - Enterprise deployment
- [`tutorial_06_incremental_loading.md`](tutorial_06_incremental_loading.md) - Incremental data processing
- [`tutorial_07_rag_evaluation.md`](tutorial_07_rag_evaluation.md) - Advanced evaluation techniques

## Getting Started / はじめに

### Prerequisites / 前提条件

```bash
# Install refinire-rag
pip install refinire-rag

# Set up environment variables
export OPENAI_API_KEY="your-api-key"
export REFINIRE_RAG_LLM_MODEL="gpt-4o-mini"
```

### Quick Start / クイックスタート

1. **Start with Part 1** to understand corpus creation:
   ```bash
   python examples/tutorial_part1_corpus_creation_example.py
   ```

2. **Continue with Part 2** for query processing:
   ```bash  
   python examples/tutorial_part2_query_engine_example.py
   ```

3. **Complete with Part 3** for evaluation:
   ```bash
   python examples/tutorial_part3_evaluation_example.py
   ```

4. **Run the integrated tutorial** for the complete workflow:
   ```bash
   python examples/complete_rag_tutorial.py
   ```

### Learning Path / 学習パス

#### For Beginners / 初心者向け
1. Read [`tutorial_01_basic_rag.md`](tutorial_01_basic_rag.md) for concepts
2. Follow Part 1, 2, 3 tutorials in sequence
3. Run the complete tutorial for integration understanding

#### For Experienced Developers / 経験者向け
1. Jump to specific parts based on your needs
2. Focus on advanced tutorials for specialized topics
3. Use the complete tutorial as a reference implementation

#### For Enterprise Users / 企業ユーザー向け
1. Review [`tutorial_05_enterprise_usage.md`](tutorial_05_enterprise_usage.md)
2. Study incremental loading patterns (Part 1)
3. Implement comprehensive evaluation (Part 3)
4. Adapt the complete tutorial for your domain

## Tutorial Features / チュートリアル機能

### Comprehensive Coverage / 包括的カバレッジ
- ✅ All major refinire-rag components
- ✅ Real-world usage patterns  
- ✅ Performance optimization techniques
- ✅ Error handling and debugging
- ✅ Production deployment considerations

### Hands-on Examples / 実践的な例
- ✅ Working code samples for every concept
- ✅ Sample datasets and knowledge bases
- ✅ Step-by-step implementation guides
- ✅ Performance benchmarks and comparisons

### Bilingual Support / バイリンガルサポート
- ✅ English and Japanese documentation
- ✅ Code comments in both languages
- ✅ Cultural context and best practices

## File Organization / ファイル構成

```
docs/tutorials/
├── README.md                              # This file / このファイル
├── tutorial_part1_corpus_creation.md      # Part 1 documentation
├── tutorial_part2_query_engine.md         # Part 2 documentation  
├── tutorial_part3_evaluation.md           # Part 3 documentation
├── tutorial_01_basic_rag.md              # Basic RAG concepts
├── tutorial_02_corpus_management.md       # Corpus management
├── tutorial_03_query_engine.md           # QueryEngine basics
├── tutorial_04_normalization.md          # Text normalization
├── tutorial_05_enterprise_usage.md       # Enterprise patterns
├── tutorial_06_incremental_loading.md    # Incremental processing
├── tutorial_07_rag_evaluation.md         # Advanced evaluation
├── tutorial_overview.md                  # General overview
└── tutorial_overview_ja.md               # Japanese overview

examples/
├── complete_rag_tutorial.py              # Complete integrated tutorial
├── tutorial_part1_corpus_creation_example.py  # Part 1 example
├── tutorial_part2_query_engine_example.py     # Part 2 example
├── tutorial_part3_evaluation_example.py       # Part 3 example
└── [other example files...]
```

## Best Practices / ベストプラクティス

### Development Workflow / 開発ワークフロー
1. **Start Small**: Begin with simple examples before building complex systems
2. **Iterate**: Use evaluation results to improve your implementation
3. **Monitor**: Track performance metrics throughout development
4. **Test**: Validate your system with diverse queries and edge cases

### Production Considerations / 本番考慮事項
1. **Scalability**: Design for growth from the beginning
2. **Monitoring**: Implement comprehensive logging and metrics
3. **Security**: Protect sensitive data and API keys
4. **Maintenance**: Plan for regular updates and improvements

## Troubleshooting / トラブルシューティング

### Common Issues / よくある問題

#### Environment Setup / 環境設定
```bash
# Missing API key
export OPENAI_API_KEY="your-key-here"

# Virtual environment activation
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

#### Dependencies / 依存関係
```bash
# Install development dependencies
uv pip install -e .

# Update to latest version
uv pip install -U refinire-rag
```

#### Performance Issues / パフォーマンス問題
- Check corpus size and chunking strategy
- Monitor memory usage during vector operations
- Optimize query processing timeouts
- Review evaluation metrics for bottlenecks

### Getting Help / ヘルプの取得
- Review the specific tutorial documentation
- Check the [API documentation](../api/)
- Run examples with debug output enabled
- Consult the [troubleshooting guide](../development/troubleshooting.md)

## Contributing / 貢献

### Improving Tutorials / チュートリアル改善
We welcome contributions to improve these tutorials:

- Report issues or suggest improvements
- Add new example use cases
- Translate content to additional languages
- Create domain-specific tutorials

### Feedback / フィードバック
Please provide feedback on:
- Tutorial clarity and completeness
- Code example effectiveness  
- Missing topics or use cases
- Performance improvement suggestions

## Resources / リソース

### Documentation / ドキュメント
- [API Reference](../api/) - Complete API documentation
- [Architecture Guide](../design/architecture.md) - System design principles
- [Development Guide](../development/) - Development best practices

### Examples / 例
- [Example Scripts](../../examples/) - Additional code examples
- [Sample Data](../../examples/data/) - Test datasets
- [Configuration Examples](../development/processor_config_example.md) - Configuration patterns

### External Resources / 外部リソース
- [RAG Evaluation Best Practices](https://example.com/rag-evaluation)
- [Vector Database Comparison](https://example.com/vector-db-comparison)
- [LLM Integration Patterns](https://example.com/llm-patterns)

---

**Happy Learning! / 楽しい学習を！**

These tutorials will guide you through building production-ready RAG systems with refinire-rag. Start with the basics and work your way up to advanced implementations.

これらのチュートリアルは、refinire-ragを使用した本番対応RAGシステムの構築をガイドします。基本から始めて、高度な実装まで段階的に学習してください。