"""
Test coverage analysis for refinire-rag project.
refinire-ragプロジェクトのテストカバレッジ分析
"""

def analyze_test_coverage():
    """
    Analyze test coverage and identify priority areas for improvement.
    テストカバレッジを分析し、改善優先領域を特定します。
    """
    
    print("=== Test Coverage Analysis Report ===")
    print("テストカバレッジ分析レポート")
    print("=" * 50)
    
    # Current coverage data from pytest output
    coverage_data = {
        # Application Layer (アプリケーション層)
        "application": {
            "corpus_manager_new": {"coverage": 13, "lines": 359, "missing": 312, "priority": "high"},
            "query_engine_new": {"coverage": 38, "lines": 249, "missing": 155, "priority": "high"},
            "quality_lab": {"coverage": 14, "lines": 475, "missing": 407, "priority": "high"},
            "corpus_manager": {"coverage": 0, "lines": 306, "missing": 306, "priority": "low"},  # Legacy
            "orchestrator_agent": {"coverage": 0, "lines": 67, "missing": 67, "priority": "medium"},
        },
        
        # Retrieval Layer (リトリーバル層)
        "retrieval": {
            "simple_retriever": {"coverage": 60, "lines": 91, "missing": 36, "priority": "medium"},
            "hybrid_retriever": {"coverage": 38, "lines": 165, "missing": 102, "priority": "medium"},
            "simple_reranker": {"coverage": 41, "lines": 108, "missing": 64, "priority": "medium"},
            "simple_reader": {"coverage": 58, "lines": 100, "missing": 42, "priority": "medium"},
            "base": {"coverage": 70, "lines": 135, "missing": 41, "priority": "low"},
        },
        
        # Processing Layer (処理層)
        "processing": {
            "chunker": {"coverage": 24, "lines": 148, "missing": 113, "priority": "high"},
            "normalizer": {"coverage": 22, "lines": 175, "missing": 137, "priority": "high"},
            "dictionary_maker": {"coverage": 0, "lines": 186, "missing": 186, "priority": "medium"},
            "graph_builder": {"coverage": 0, "lines": 228, "missing": 228, "priority": "medium"},
            "document_pipeline": {"coverage": 0, "lines": 113, "missing": 113, "priority": "medium"},
            "test_suite": {"coverage": 28, "lines": 183, "missing": 131, "priority": "high"},
            "evaluator": {"coverage": 22, "lines": 289, "missing": 225, "priority": "high"},
            "contradiction_detector": {"coverage": 26, "lines": 282, "missing": 208, "priority": "high"},
            "insight_reporter": {"coverage": 18, "lines": 365, "missing": 299, "priority": "high"},
        },
        
        # Storage Layer (ストレージ層)
        "storage": {
            "vector_store": {"coverage": 34, "lines": 199, "missing": 132, "priority": "high"},
            "sqlite_store": {"coverage": 13, "lines": 256, "missing": 223, "priority": "high"},
            "evaluation_store": {"coverage": 27, "lines": 277, "missing": 201, "priority": "high"},
            "document_store": {"coverage": 77, "lines": 57, "missing": 13, "priority": "low"},
            "in_memory_vector_store": {"coverage": 16, "lines": 202, "missing": 169, "priority": "medium"},
            "pickle_vector_store": {"coverage": 19, "lines": 142, "missing": 115, "priority": "medium"},
        },
        
        # Embedding Layer (埋め込み層)
        "embedding": {
            "openai_embedder": {"coverage": 28, "lines": 196, "missing": 142, "priority": "high"},
            "tfidf_embedder": {"coverage": 20, "lines": 211, "missing": 168, "priority": "medium"},
            "base": {"coverage": 47, "lines": 103, "missing": 55, "priority": "medium"},
        },
        
        # Loader Layer (ローダー層)
        "loader": {
            "document_store_loader": {"coverage": 24, "lines": 253, "missing": 193, "priority": "high"},
            "file_tracker": {"coverage": 20, "lines": 106, "missing": 85, "priority": "medium"},
            "incremental_directory_loader": {"coverage": 24, "lines": 97, "missing": 74, "priority": "medium"},
            "csv_loader": {"coverage": 24, "lines": 33, "missing": 25, "priority": "low"},
            "json_loader": {"coverage": 30, "lines": 27, "missing": 19, "priority": "low"},
            "html_loader": {"coverage": 29, "lines": 51, "missing": 36, "priority": "low"},
            "text_loader": {"coverage": 39, "lines": 18, "missing": 11, "priority": "low"},
            "directory_loader": {"coverage": 38, "lines": 26, "missing": 16, "priority": "low"},
        },
        
        # Plugin System (プラグインシステム)
        "plugins": {
            "plugin_registry": {"coverage": 74, "lines": 91, "missing": 24, "priority": "low"},
            "plugin_factory": {"coverage": 81, "lines": 62, "missing": 12, "priority": "low"},
            "plugin_loader": {"coverage": 30, "lines": 130, "missing": 91, "priority": "medium"},
            "base": {"coverage": 71, "lines": 49, "missing": 14, "priority": "low"},
        }
    }
    
    print("\n📊 Layer-wise Coverage Summary")
    print("レイヤー別カバレッジサマリー")
    print("-" * 40)
    
    for layer, components in coverage_data.items():
        total_lines = sum(comp["lines"] for comp in components.values())
        total_missing = sum(comp["missing"] for comp in components.values())
        layer_coverage = ((total_lines - total_missing) / total_lines * 100) if total_lines > 0 else 0
        
        print(f"{layer.upper()}: {layer_coverage:.1f}% coverage")
        
        # Show components by priority
        high_priority = [name for name, comp in components.items() if comp["priority"] == "high"]
        if high_priority:
            print(f"  🔴 High Priority: {', '.join(high_priority)}")
    
    print("\n🎯 Priority Test Cases to Implement")
    print("実装優先度の高いテストケース")
    print("-" * 50)
    
    # Identify high-priority test cases
    priority_cases = []
    
    for layer, components in coverage_data.items():
        for comp_name, data in components.items():
            if data["priority"] == "high" and data["coverage"] < 50:
                impact_score = data["missing"] * (1 if data["coverage"] < 20 else 0.5)
                priority_cases.append({
                    "component": f"{layer}.{comp_name}",
                    "coverage": data["coverage"],
                    "missing_lines": data["missing"],
                    "impact_score": impact_score,
                    "importance": get_importance_reason(layer, comp_name)
                })
    
    # Sort by impact score
    priority_cases.sort(key=lambda x: x["impact_score"], reverse=True)
    
    print("\nTop 10 Priority Test Cases:")
    for i, case in enumerate(priority_cases[:10], 1):
        print(f"{i:2d}. {case['component']}")
        print(f"     Coverage: {case['coverage']}% | Missing: {case['missing_lines']} lines")
        print(f"     Reason: {case['importance']}")
        print()
    
    print("\n📈 Recommended Testing Strategy")
    print("推奨テスト戦略")
    print("-" * 40)
    
    strategies = [
        ("1. Application Layer Tests", "アプリケーション層テスト", [
            "Create integration tests for CorpusManager.from_env()",
            "Add comprehensive QueryEngine workflow tests", 
            "Implement QualityLab evaluation pipeline tests"
        ]),
        ("2. Processing Layer Tests", "処理層テスト", [
            "Unit tests for Chunker with various document types",
            "Normalizer tests with different dictionaries",
            "TestSuite execution and result validation tests"
        ]),
        ("3. Storage Layer Tests", "ストレージ層テスト", [
            "VectorStore CRUD operations tests",
            "SQLiteStore transaction and error handling tests",
            "EvaluationStore persistence and retrieval tests"
        ]),
        ("4. Error Handling Tests", "エラーハンドリングテスト", [
            "Invalid configuration handling",
            "Network failure scenarios",
            "Resource exhaustion scenarios"
        ])
    ]
    
    for title, title_jp, items in strategies:
        print(f"\n{title} ({title_jp}):")
        for item in items:
            print(f"  • {item}")
    
    print(f"\n📊 Current Status: 24% coverage | Target: 70% coverage")
    print("現在の状況: 24%カバレッジ | 目標: 70%カバレッジ")
    
    return priority_cases

def get_importance_reason(layer, component):
    """Get importance reasoning for test coverage prioritization"""
    reasons = {
        ("application", "corpus_manager_new"): "Core component for document management",
        ("application", "query_engine_new"): "Main user-facing API for queries", 
        ("application", "quality_lab"): "Critical for evaluation and quality assurance",
        ("processing", "chunker"): "Fundamental text processing component",
        ("processing", "normalizer"): "Important for text standardization",
        ("processing", "test_suite"): "Essential for evaluation framework",
        ("processing", "evaluator"): "Core evaluation logic",
        ("processing", "contradiction_detector"): "Quality control mechanism",
        ("processing", "insight_reporter"): "Reporting and analysis functionality",
        ("storage", "vector_store"): "Primary storage for embeddings",
        ("storage", "sqlite_store"): "Main persistence layer",
        ("storage", "evaluation_store"): "Evaluation result persistence",
        ("embedding", "openai_embedder"): "Primary embedding provider",
        ("loader", "document_store_loader"): "Core document loading functionality",
    }
    
    return reasons.get((layer, component), "General functionality coverage")

if __name__ == "__main__":
    analyze_test_coverage()