# refinire-rag Tutorial Guide

## Introduction

This comprehensive tutorial guide provides step-by-step learning paths for building RAG (Retrieval-Augmented Generation) systems using the refinire-rag library. We offer structured learning paths suitable for beginners through advanced users.

## Prerequisites

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install refinire-rag
pip install refinire-rag

# Set required environment variables
export OPENAI_API_KEY="your-api-key-here"
export REFINIRE_RAG_LLM_MODEL="gpt-4o-mini"
```

### 2. Prerequisites Knowledge
- Python fundamentals
- Basic understanding of RAG (Retrieval-Augmented Generation)
- Basic understanding of LLMs (Large Language Models)

## Learning Roadmap

### 🎯 Level 1: Foundation Understanding

**Goal**: Understand basic RAG concepts and build a simple RAG system

#### 1.1 RAG Fundamentals
- **Learning Content**: What is RAG and why is it needed
- **Resources**: [tutorial_01_basic_rag.md](tutorial_01_basic_rag.md)
- **Practice**: Basic RAG system verification

#### 1.2 refinire-rag Overview
- **Learning Content**: Library architecture and key components
- **Resources**: [tutorial_overview.md](tutorial_overview.md)
- **Practice**: Run sample code

#### 1.3 Quick Start
- **Learning Content**: Get RAG system running quickly
- **Resources**: [quickstart_guide.py](../../examples/quickstart_guide.py)
- **Practice**: Build RAG system in 10 minutes

**✅ Level 1 Completion Checklist**
- [ ] Can explain basic RAG concepts
- [ ] Understands refinire-rag key components
- [ ] Can run a basic RAG system

### 🚀 Level 2: Core Functionality Mastery

**Goal**: Master the three main components of refinire-rag

#### 2.1 Part 1: Corpus Creation & Management
- **Learning Content**: Document loading, processing, and indexing using CorpusManager
- **Resources**: [tutorial_part1_corpus_creation.md](tutorial_part1_corpus_creation.md)
- **Practice**: [tutorial_part1_corpus_creation_example.py](../../examples/tutorial_part1_corpus_creation_example.py)

**Key Points**:
- Document loading and preprocessing
- Chunking strategy selection
- Vectorization and index building
- Incremental loading

#### 2.2 Part 2: Query Engine
- **Learning Content**: Search, reranking, and answer generation using QueryEngine
- **Resources**: [tutorial_part2_query_engine.md](tutorial_part2_query_engine.md)
- **Practice**: [tutorial_part2_query_engine_example.py](../../examples/tutorial_part2_query_engine_example.py)

**Key Points**:
- Search strategy configuration
- Reranking method selection
- Answer generation optimization
- Performance tuning

#### 2.3 Part 3: Quality Evaluation
- **Learning Content**: RAG system evaluation and improvement using QualityLab
- **Resources**: [tutorial_part3_evaluation.md](tutorial_part3_evaluation.md)
- **Practice**: [tutorial_part3_evaluation_example.py](../../examples/tutorial_part3_evaluation_example.py)

**Key Points**:
- Evaluation data creation
- Multi-faceted quality assessment
- Contradiction detection
- Improvement recommendation interpretation

**✅ Level 2 Completion Checklist**
- [ ] Can create corpus with custom data
- [ ] Can configure effective search & answer generation
- [ ] Can quantitatively evaluate RAG system quality

### 🎯 Level 3: Integration & Practical Application

**Goal**: Integrate the three components and build practical RAG systems

#### 3.1 End-to-End Integration
- **Learning Content**: Complete RAG Tutorial system integration
- **Resources**: [complete_rag_tutorial.py](../../examples/complete_rag_tutorial.py)
- **Practice**: Implement complete RAG workflow

#### 3.2 Practical Corpus Management
- **Learning Content**: Corpus management with real data
- **Resources**: [tutorial_02_corpus_management.md](tutorial_02_corpus_management.md)
- **Practice**: [corpus_manager_demo.py](../../examples/corpus_manager_demo.py)

#### 3.3 Performance Optimization
- **Learning Content**: System performance monitoring and optimization
- **Practice**: Run performance tests

**✅ Level 3 Completion Checklist**
- [ ] Can build end-to-end RAG systems
- [ ] Can optimize performance with real data
- [ ] Can continuously monitor system quality

### 🏆 Level 4: Advanced Features

**Goal**: Leverage advanced features to build enterprise-grade RAG systems

#### 4.1 Normalization & Knowledge Graph Utilization
- **Learning Content**: Advanced RAG with text normalization and knowledge graphs
- **Resources**: [tutorial_04_normalization.md](tutorial_04_normalization.md)
- **Practice**: Implement normalization features

#### 4.2 Enterprise-Level Usage
- **Learning Content**: Practical RAG system construction in enterprise environments
- **Resources**: [tutorial_05_enterprise_usage.md](tutorial_05_enterprise_usage.md)
- **Practice**: [tutorial_05_enterprise_usage.py](../../examples/tutorial_05_enterprise_usage.py)

#### 4.3 Incremental Data Processing
- **Learning Content**: Efficient processing of large-scale data
- **Resources**: [tutorial_06_incremental_loading.md](tutorial_06_incremental_loading.md)
- **Practice**: [incremental_loading_demo.py](../../examples/incremental_loading_demo.py)

#### 4.4 Advanced Evaluation Methods
- **Learning Content**: Detailed evaluation metrics and improvement methods
- **Resources**: [tutorial_07_rag_evaluation.md](tutorial_07_rag_evaluation.md)
- **Practice**: [tutorial_07_evaluation_example.py](../../examples/tutorial_07_evaluation_example.py)

**✅ Level 4 Completion Checklist**
- [ ] Can leverage advanced normalization features
- [ ] Can design enterprise-level RAG systems
- [ ] Can implement efficient large-scale data processing
- [ ] Can apply advanced evaluation methods

## Next Steps

### 🎯 After Completing Level 1
- Proceed to [tutorial_part1_corpus_creation.md](tutorial_part1_corpus_creation.md)
- Try corpus creation with actual data

### 🚀 After Completing Level 2
- Run [complete_rag_tutorial.py](../../examples/complete_rag_tutorial.py)
- Verify integrated system operation

### 🏆 After Completing Level 3
- Progress to advanced feature learning
- Consider enterprise-level usage

### 🌟 After Completing Level 4
- Challenge yourself with custom plugin development
- Consider contributing to the community

## Plugin System Utilization Guide

### 🔌 Plugin Fundamentals

refinire-rag adopts an environment variable-based plugin architecture, allowing various components to be used as independent packages.

#### Available Plugin Types
- **Search & Retrieval**: VectorStore (Chroma, etc.), KeywordSearch (BM25s, etc.), Retriever
- **Document Processing**: Loader, Splitter, Filter, Metadata
- **Evaluation & Quality Management**: Evaluator, ContradictionDetector, TestSuite
- **Storage**: DocumentStore, EvaluationStore

### 🎯 Plugin Switching Methods

#### 1. Basic Switching with Environment Variables

```bash
# Development environment: Lightweight plugins
export REFINIRE_RAG_RETRIEVERS="inmemory_vector"

# Production environment: High-performance plugins
export REFINIRE_RAG_RETRIEVERS="chroma,bm25s"
export REFINIRE_RAG_CHROMA_HOST="localhost"
export REFINIRE_RAG_CHROMA_PORT="8000"
export REFINIRE_RAG_BM25S_INDEX_PATH="./bm25s_index"
```

#### 2. Checking Available Plugins

```python
from refinire_rag.registry import PluginRegistry

# Check available plugin types
available_plugins = PluginRegistry.get_all_plugins_info()

print("=== Available Plugins ===")
for group, plugins in available_plugins.items():
    print(f"\n{group.upper()}:")
    for name, info in plugins.items():
        print(f"  - {name}: {info['description']}")
```

#### 3. Dynamic Plugin Creation

```python
# Create specific plugin dynamically
vector_store = PluginRegistry.create_plugin('vector_stores', 'chroma', 
                                           host='localhost', port=8000)

# Create multiple plugins from environment variables
from refinire_rag.factories import PluginFactory
retrievers = PluginFactory.create_retrievers_from_env()
```

### 🛠 Plugin Development Basics

#### 1. Project Structure
```
my-refinire-rag-plugin/
├── src/
│   └── my_refinire_plugin/
│       ├── __init__.py
│       ├── vector_store.py      # Plugin implementation
│       ├── config.py           # Configuration class
│       └── env_template.py     # Environment variable template
├── tests/
├── pyproject.toml             # Entry point configuration
└── README.md
```

#### 2. Required Implementation Pattern

```python
# Unified configuration pattern example
class CustomVectorStore(VectorStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import os
        # Configuration priority: kwargs > environment variables > default values
        self.host = kwargs.get('host', 
                              os.getenv('REFINIRE_RAG_CUSTOM_HOST', 'localhost'))
        self.port = int(kwargs.get('port', 
                                  os.getenv('REFINIRE_RAG_CUSTOM_PORT', '8080')))
    
    def get_config(self) -> Dict[str, Any]:
        """Return current configuration as dictionary (required)"""
        return {
            'host': self.host,
            'port': self.port,
            'plugin_type': self.__class__.__name__
        }
```

#### 3. Entry Point Configuration

```toml
# pyproject.toml
[project.entry-points."refinire_rag.vector_stores"]
custom = "my_refinire_plugin:CustomVectorStore"

[project.entry-points."refinire_rag.oneenv_templates"]
custom = "my_refinire_plugin.env_template:custom_env_template"
```

### 📚 Plugin-Related Resources

#### Development Guides
- **Detailed Development Guide**: [docs/development/plugin_development_guide.md](../development/plugin_development_guide.md)
- **Design Strategy**: [docs/design/plugin_strategy.md](../design/plugin_strategy.md)
- **Unified Configuration Pattern**: [docs/development/plugin_development.md](../development/plugin_development.md)

#### Implementation Examples
- **Plugin Configuration Examples**: [docs/development/plugin_setup_example.py](../development/plugin_setup_example.py)
- **Integration Tests**: [tests/integration/plugins/](../../tests/integration/plugins/)

#### Environment Variable Management
- **oneenv Integration**: Automatically generate plugin-specific environment variable templates
- **Configuration Priority**: kwargs > environment variables > default values

### 🔧 Advanced Plugin Utilization

#### 1. Combining Multiple Plugins

```python
# Combine multiple VectorStore and KeywordSearch
import os
os.environ["REFINIRE_RAG_RETRIEVERS"] = "chroma,bm25s"

# CorpusManager automatically integrates multiple retrievers
corpus_manager = CorpusManager.from_env()
print(f"Integrated retrievers: {len(corpus_manager.retrievers)}")
```

#### 2. Environment-Based Plugin Switching

```python
import os

# Dynamic switching by environment
if os.getenv("ENVIRONMENT") == "development":
    os.environ["REFINIRE_RAG_RETRIEVERS"] = "inmemory_vector"
elif os.getenv("ENVIRONMENT") == "production": 
    os.environ["REFINIRE_RAG_RETRIEVERS"] = "chroma,bm25s"
elif os.getenv("ENVIRONMENT") == "testing":
    os.environ["REFINIRE_RAG_RETRIEVERS"] = "inmemory_vector"
```

#### 3. Custom Plugin Distribution

```bash
# Build and distribute plugin package
cd my-refinire-rag-plugin
python -m build
twine upload dist/*

# Install and automatically become available
pip install my-refinire-rag-plugin
```

### ⚡ Plugin Development Best Practices

#### Required Checklist
- [ ] Implement unified configuration pattern (`**kwargs` constructor)
- [ ] Implement `get_config()` method
- [ ] Follow environment variable naming convention (`REFINIRE_RAG_{PLUGIN}_{SETTING}`)
- [ ] Implement proper error handling
- [ ] Select appropriate entry point group
- [ ] Provide oneenv template

#### Development Principles
1. **Single Responsibility**: One plugin has one clear responsibility
2. **Environment Variable Support**: Support environment variables for all settings
3. **Lazy Initialization**: Defer heavy processing until actual use
4. **Error Tolerance**: Properly handle connection failures and other errors
5. **Test Ease**: Design for easy testing with configuration injection

### 🚀 Plugin Learning Progression

#### For Beginners
1. Learn how to use existing plugins
2. Practice configuration changes via environment variables
3. Create simple custom plugins

#### For Intermediate Users
1. Utilize multiple plugin combinations
2. Understand and implement unified configuration patterns
3. Environment variable management through oneenv integration

#### For Advanced Users
1. Design complex plugin architectures
2. Manage plugin inter-dependencies
3. Contribute to the plugin ecosystem

The plugin system provides refinire-rag with high extensibility and flexibility. Select and develop plugins according to your needs to build more powerful RAG systems.

## Learning Support Resources

### 📚 Documentation
- **API Reference**: [docs/api/](../api/)
- **Architecture Guide**: [docs/design/architecture.md](../design/architecture.md)
- **Development Guide**: [docs/development/](../development/)

### 💻 Practical Examples
- **Sample Code**: [examples/](../../examples/)
- **Test Data**: [examples/data/](../../examples/data/)
- **Configuration Examples**: [docs/development/processor_config_example.md](../development/processor_config_example.md)

### 🔧 Development Support
- **Plugin Development**: [docs/development/plugin_development.md](../development/plugin_development.md)
- **Customization Guide**: [docs/development/answer_synthesizer_customization.md](../development/answer_synthesizer_customization.md)

## Frequently Asked Questions (FAQ)

### Q1: Which level should I start from?
**A**: If you have no RAG experience, start from **Level 1**. If you have machine learning experience, you can start from **Level 2**.

### Q2: How long does learning take?
**A**: Learning time varies by individual experience and goals. Progress through the levels at your own pace, focusing on understanding and practice rather than speed.

### Q3: What if I want to try with my own data?
**A**: After completing **Level 2**, refer to Part 1 tutorial to create corpus with your custom data.

### Q4: I'm considering enterprise usage. Where should I start?
**A**: After completing **Level 3**, refer to [tutorial_05_enterprise_usage.md](tutorial_05_enterprise_usage.md) for enterprise implementation considerations.

### Q5: What should I do if errors occur?
**A**: 
1. Check environment setup (API keys, dependencies)
2. Compare with sample code
3. Enable debug output
4. Refer to [troubleshooting guide](../development/troubleshooting.md)

## Learning Support

### 📖 Learning Progress Management
Use the checklist for each level to manage your learning progress.

### 💡 Practical Learning Tips
1. **Start Small**: Begin with simple examples and gradually increase complexity
2. **Try with Real Data**: Don't just learn theory; verify with actual data
3. **Emphasize Evaluation**: Regularly evaluate system quality
4. **Leverage Community**: Actively ask questions when in doubt

---

**Ready to begin?**

Start learning refinire-rag today. Begin with tutorials appropriate for your level and gradually develop your RAG system construction skills.

**Happy Learning! 🚀**