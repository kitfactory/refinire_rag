"""
Example demonstrating how to register existing QA pairs to QualityLab

既存のQAペアをQualityLabに登録する方法を示す例
"""

from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.models.qa_pair import QAPair


def main():
    """
    Demonstrate registering existing QA pairs to QualityLab
    
    既存のQAペアをQualityLabに登録するデモンストレーション
    """
    print("=== QA Pairs Registration Example ===")
    print("QAペア登録の例")
    print()
    
    # Create QualityLab instance
    # QualityLabインスタンスを作成
    config = QualityLabConfig(
        qa_pairs_per_document=2,
        output_format="markdown",
        include_detailed_analysis=True
    )
    
    quality_lab = QualityLab(config=config)
    
    # Create existing QA pairs that you want to register
    # 登録したい既存のQAペアを作成
    existing_qa_pairs = [
        QAPair(
            question="What is the purpose of RAG (Retrieval-Augmented Generation)?",
            answer="RAG combines information retrieval with text generation to provide more accurate and contextual answers by retrieving relevant documents and using them to generate responses.",
            document_id="rag_overview_001",
            metadata={
                "qa_id": "rag_001",
                "question_type": "conceptual",
                "topic": "rag_fundamentals",
                "difficulty": "intermediate",
                "source": "expert_review",
                "expected_sources": ["rag_overview_001", "rag_architecture_002"]
            }
        ),
        QAPair(
            question="How does vector search work in RAG systems?",
            answer="Vector search in RAG systems converts both documents and queries into high-dimensional vectors using embedding models, then finds the most similar documents by calculating cosine similarity or other distance metrics.",
            document_id="vector_search_002",
            metadata={
                "qa_id": "rag_002",
                "question_type": "technical",
                "topic": "vector_search",
                "difficulty": "advanced",
                "source": "expert_review",
                "expected_sources": ["vector_search_002", "embeddings_003"]
            }
        ),
        QAPair(
            question="What are the main components of a RAG system?",
            answer="The main components of a RAG system include: 1) Document ingestion and preprocessing, 2) Embedding generation, 3) Vector storage, 4) Retrieval mechanism, 5) Answer generation using LLM, and 6) Post-processing and evaluation.",
            document_id="rag_architecture_003",
            metadata={
                "qa_id": "rag_003",
                "question_type": "structural",
                "topic": "rag_architecture",
                "difficulty": "intermediate",
                "source": "expert_review",
                "expected_sources": ["rag_architecture_003", "rag_components_004"]
            }
        ),
        QAPair(
            question="What evaluation metrics are commonly used for RAG systems?",
            answer="Common RAG evaluation metrics include: retrieval metrics (Hit Rate, MRR, NDCG), generation metrics (BLEU, ROUGE, BERTScore), and end-to-end metrics (Faithfulness, Answer Relevance, Context Precision/Recall).",
            document_id="rag_evaluation_004",
            metadata={
                "qa_id": "rag_004",
                "question_type": "analytical",
                "topic": "rag_evaluation",
                "difficulty": "advanced",
                "source": "expert_review",
                "expected_sources": ["rag_evaluation_004", "metrics_005"]
            }
        )
    ]
    
    print(f"Registering {len(existing_qa_pairs)} existing QA pairs...")
    print(f"既存の{len(existing_qa_pairs)}個のQAペアを登録しています...")
    print()
    
    # Register QA pairs with additional metadata
    # 追加のメタデータとともにQAペアを登録
    registration_metadata = {
        "collection_name": "rag_expert_knowledge",
        "creation_date": "2024-01-15",
        "review_status": "expert_validated",
        "intended_use": "benchmark_evaluation",
        "domain": "information_retrieval",
        "language": "english"
    }
    
    success = quality_lab.register_qa_pairs(
        qa_pairs=existing_qa_pairs,
        qa_set_name="rag_expert_benchmark_v1",
        metadata=registration_metadata
    )
    
    if success:
        print("✅ Successfully registered QA pairs!")
        print("✅ QAペアの登録に成功しました！")
        print()
        
        # Display registration results
        # 登録結果を表示
        print("Registration Results / 登録結果:")
        print(f"- Registered QA pairs: {len(existing_qa_pairs)}")
        print(f"- 登録されたQAペア: {len(existing_qa_pairs)}")
        
        # Show QualityLab statistics
        # QualityLabの統計を表示
        stats = quality_lab.get_lab_stats()
        print(f"- Total QA pairs in QualityLab: {stats['qa_pairs_generated']}")
        print(f"- QualityLab内の総QAペア数: {stats['qa_pairs_generated']}")
        print()
        
        # Show sample registered QA pair with enhanced metadata
        # 拡張メタデータを含む登録されたQAペアのサンプルを表示
        sample_qa = existing_qa_pairs[0]
        print("Sample registered QA pair / 登録されたQAペアのサンプル:")
        print(f"Question: {sample_qa.question}")
        print(f"Answer: {sample_qa.answer[:100]}...")
        print(f"Enhanced metadata: {sample_qa.metadata}")
        print()
        
        # Demonstrate how to use registered QA pairs for evaluation
        # 登録されたQAペアを評価に使用する方法を示す
        print("Next steps / 次のステップ:")
        print("1. Use registered QA pairs for QueryEngine evaluation")
        print("   登録されたQAペアをQueryEngineの評価に使用")
        print("2. Generate evaluation reports with registered data")
        print("   登録されたデータで評価レポートを生成")
        print("3. Compare performance across different QA sets")
        print("   異なるQAセット間でのパフォーマンス比較")
        print()
        
        # Show how to retrieve evaluation history
        # 評価履歴の取得方法を表示
        print("Evaluation history / 評価履歴:")
        history = quality_lab.get_evaluation_history(limit=5)
        if history:
            for run in history:
                print(f"- {run['name']}: {run['status']} ({run['created_at']})")
        else:
            print("- No evaluation history available yet")
            print("- まだ評価履歴がありません")
    else:
        print("❌ Failed to register QA pairs")
        print("❌ QAペアの登録に失敗しました")
    
    print()
    print("=== Registration Complete ===")
    print("=== 登録完了 ===")


if __name__ == "__main__":
    main()