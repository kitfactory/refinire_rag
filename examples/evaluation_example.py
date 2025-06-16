"""
Example: Using Multiple Evaluation Metrics for RAG Assessment

RAG評価における複数評価指標の使用例

This example demonstrates how to use QuestEval, BLEU, ROUGE, and LLM Judge
evaluators together for comprehensive RAG system assessment.
"""

import asyncio
from refinire_rag.evaluation import (
    create_comprehensive_evaluator,
    create_quick_evaluator,
    QuestEvalEvaluator,
    QuestEvalConfig,
    BLEUEvaluator,
    BLEUConfig,
    ROUGEEvaluator,
    ROUGEConfig,
    LLMJudgeEvaluator,
    LLMJudgeConfig,
    CompositeEvaluator
)


def basic_evaluation_example():
    """
    Basic example using individual evaluators
    
    個別評価器を使用した基本例
    """
    print("=== Basic Evaluation Example ===")
    
    # Sample QA data
    question = "What is the capital of France?"
    reference = "The capital of France is Paris. It is the largest city in France."
    candidate = "Paris is the capital city of France."
    context = {
        "question": question,
        "source_documents": [
            "France is a country in Europe. Its capital city is Paris.",
            "Paris is located in northern France and is the political center."
        ]
    }
    
    # 1. QuestEval Evaluation
    print("\n1. QuestEval Evaluation:")
    questeval_config = QuestEvalConfig(
        model_name="gpt-4o-mini",
        enable_consistency=True,
        enable_answerability=True,
        enable_source_support=True,
        enable_fluency=True
    )
    questeval = QuestEvalEvaluator(questeval_config)
    result = questeval.evaluate(reference, candidate, context)
    print(f"   Score: {result.score:.3f}")
    print(f"   Details: {result.details.get('component_scores', {})}")
    
    # 2. BLEU Evaluation
    print("\n2. BLEU Evaluation:")
    bleu_config = BLEUConfig(max_n=4, smoothing_function="epsilon")
    bleu = BLEUEvaluator(bleu_config)
    result = bleu.evaluate(reference, candidate)
    print(f"   Score: {result.score:.3f}")
    print(f"   N-gram precisions: {result.details.get('n_gram_precisions', {})}")
    
    # 3. ROUGE Evaluation
    print("\n3. ROUGE Evaluation:")
    rouge_config = ROUGEConfig(
        rouge_types=["rouge-1", "rouge-2", "rouge-l"],
        case_sensitive=False
    )
    rouge = ROUGEEvaluator(rouge_config)
    result = rouge.evaluate(reference, candidate)
    print(f"   Score: {result.score:.3f}")
    print(f"   ROUGE scores: {result.details.get('rouge_scores', {})}")
    
    # 4. LLM Judge Evaluation
    print("\n4. LLM Judge Evaluation:")
    llm_judge_config = LLMJudgeConfig(
        model_name="gpt-4o-mini",
        evaluation_criteria=["relevance", "accuracy", "completeness"],
        scoring_scale=10
    )
    llm_judge = LLMJudgeEvaluator(llm_judge_config)
    result = llm_judge.evaluate(reference, candidate, context)
    print(f"   Score: {result.score:.3f}")
    print(f"   Individual scores: {result.details.get('individual_scores', {})}")


def comprehensive_evaluation_example():
    """
    Example using CompositeEvaluator for comprehensive assessment
    
    包括的評価のためのCompositeEvaluatorの使用例
    """
    print("\n=== Comprehensive Evaluation Example ===")
    
    # Sample QA data
    qa_pairs = [
        {
            "question": "What is machine learning?",
            "reference": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            "candidate": "Machine learning allows computers to learn from data and make predictions or decisions.",
            "context": {
                "question": "What is machine learning?",
                "source_documents": [
                    "Machine learning is a method of data analysis that automates analytical model building.",
                    "It is a branch of artificial intelligence based on the idea that systems can learn from data."
                ]
            }
        },
        {
            "question": "How does a neural network work?",
            "reference": "Neural networks work by processing information through interconnected nodes that mimic the structure of the human brain.",
            "candidate": "Neural networks use layers of connected neurons to process data and learn patterns.",
            "context": {
                "question": "How does a neural network work?",
                "source_documents": [
                    "Neural networks consist of layers of artificial neurons that process information.",
                    "They learn by adjusting weights between connections based on training data."
                ]
            }
        }
    ]
    
    # Create comprehensive evaluator
    evaluator = create_comprehensive_evaluator(
        enable_questeval=True,
        enable_bleu=True,
        enable_rouge=True,
        enable_llm_judge=True
    )
    
    print(f"Created evaluator with {len(evaluator.evaluators)} metrics:")
    for eval_obj in evaluator.evaluators:
        print(f"  - {eval_obj.config.name}")
    
    # Evaluate each QA pair
    for i, qa_pair in enumerate(qa_pairs):
        print(f"\n--- QA Pair {i+1} ---")
        print(f"Question: {qa_pair['question']}")
        print(f"Candidate: {qa_pair['candidate']}")
        
        result = evaluator.evaluate(
            reference=qa_pair['reference'],
            candidate=qa_pair['candidate'],
            context=qa_pair['context']
        )
        
        print(f"Overall Score: {result.score:.3f}")
        print("Individual Metric Scores:")
        for metric_result in result.details['individual_results']:
            print(f"  {metric_result.metric_name}: {metric_result.score:.3f}")


def quick_evaluator_examples():
    """
    Examples using quick evaluator presets
    
    クイック評価器プリセットの使用例
    """
    print("\n=== Quick Evaluator Examples ===")
    
    question = "What are the benefits of renewable energy?"
    reference = "Renewable energy provides environmental benefits by reducing carbon emissions and offers economic advantages through job creation."
    candidate = "Renewable energy helps the environment and creates jobs."
    context = {"question": question}
    
    # Test different evaluation types
    evaluation_types = ["comprehensive", "lexical", "semantic", "llm_only"]
    
    for eval_type in evaluation_types:
        print(f"\n{eval_type.upper()} Evaluation:")
        evaluator = create_quick_evaluator(eval_type)
        
        result = evaluator.evaluate(reference, candidate, context)
        print(f"  Score: {result.score:.3f}")
        
        if hasattr(evaluator, 'evaluators'):  # CompositeEvaluator
            metrics = [e.config.name for e in evaluator.evaluators]
            print(f"  Metrics: {', '.join(metrics)}")
        else:  # Single evaluator
            print(f"  Metric: {evaluator.config.name}")


def batch_evaluation_example():
    """
    Example of batch evaluation for efficiency
    
    効率性のためのバッチ評価例
    """
    print("\n=== Batch Evaluation Example ===")
    
    # Sample data for batch evaluation
    evaluation_pairs = [
        {
            "reference": "Artificial intelligence is the simulation of human intelligence processes by machines.",
            "candidate": "AI simulates human thinking in machines.",
            "context": {"question": "What is AI?"}
        },
        {
            "reference": "Python is a high-level programming language known for its simplicity and readability.",
            "candidate": "Python is an easy-to-read programming language.",
            "context": {"question": "What is Python?"}
        },
        {
            "reference": "Cloud computing provides on-demand access to computing resources over the internet.",
            "candidate": "Cloud computing offers internet-based computing services.",
            "context": {"question": "What is cloud computing?"}
        }
    ]
    
    # Use LLM Judge for batch evaluation (supports async)
    llm_judge = LLMJudgeEvaluator(LLMJudgeConfig())
    
    print(f"Evaluating {len(evaluation_pairs)} pairs with LLM Judge...")
    
    # Synchronous batch evaluation
    results = []
    for pair in evaluation_pairs:
        result = llm_judge.evaluate(
            reference=pair["reference"],
            candidate=pair["candidate"],
            context=pair["context"]
        )
        results.append(result)
    
    # Display results
    print("\nBatch Evaluation Results:")
    for i, result in enumerate(results):
        print(f"  Pair {i+1}: {result.score:.3f}")
    
    avg_score = sum(r.score for r in results) / len(results)
    print(f"Average Score: {avg_score:.3f}")


def custom_configuration_example():
    """
    Example with custom configurations for each evaluator
    
    各評価器のカスタム設定例
    """
    print("\n=== Custom Configuration Example ===")
    
    # Custom configurations
    custom_questeval_config = QuestEvalConfig(
        model_name="gpt-4o-mini",
        consistency_weight=0.4,
        answerability_weight=0.3,
        source_support_weight=0.2,
        fluency_weight=0.1
    )
    
    custom_bleu_config = BLEUConfig(
        max_n=2,  # Only BLEU-1 and BLEU-2
        weights=[0.6, 0.4],  # Custom weights
        smoothing_function="add_one"
    )
    
    custom_rouge_config = ROUGEConfig(
        rouge_types=["rouge-1", "rouge-l"],  # Only specific ROUGE variants
        case_sensitive=True,
        remove_stopwords=True
    )
    
    custom_llm_judge_config = LLMJudgeConfig(
        model_name="gpt-4o-mini",
        evaluation_criteria=["relevance", "accuracy"],  # Only 2 criteria
        scoring_scale=5  # 1-5 scale instead of 1-10
    )
    
    # Create evaluator with custom configurations
    evaluator = create_comprehensive_evaluator(
        questeval_config=custom_questeval_config,
        bleu_config=custom_bleu_config,
        rouge_config=custom_rouge_config,
        llm_judge_config=custom_llm_judge_config
    )
    
    # Sample evaluation
    question = "Explain photosynthesis."
    reference = "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen."
    candidate = "Plants use sunlight to make food from CO2 and water, producing oxygen."
    context = {"question": question}
    
    result = evaluator.evaluate(reference, candidate, context)
    
    print(f"Custom Configuration Evaluation:")
    print(f"  Overall Score: {result.score:.3f}")
    print(f"  Number of metrics: {len(result.details['individual_results'])}")
    
    for metric_result in result.details['individual_results']:
        print(f"  {metric_result.metric_name}: {metric_result.score:.3f}")


def main():
    """
    Main function to run all examples
    
    すべての例を実行するメイン関数
    """
    print("RAG Evaluation Metrics Examples")
    print("=" * 50)
    
    # Run all examples
    basic_evaluation_example()
    comprehensive_evaluation_example()
    quick_evaluator_examples()
    batch_evaluation_example()
    custom_configuration_example()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()