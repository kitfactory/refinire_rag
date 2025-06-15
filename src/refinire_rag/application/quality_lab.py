"""
QualityLab - RAG System Evaluation and Quality Assessment

A Refinire Step that provides comprehensive evaluation of RAG systems including
QA pair generation, QueryEngine evaluation, and reporting.
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from ..models.qa_pair import QAPair
from ..models.evaluation_result import EvaluationResult
from ..models.document import Document
from ..processing.test_suite import TestSuite, TestCase, TestResult, TestSuiteConfig
from ..processing.evaluator import Evaluator, EvaluatorConfig
from ..processing.contradiction_detector import ContradictionDetector, ContradictionDetectorConfig
from ..processing.insight_reporter import InsightReporter, InsightReporterConfig
from .query_engine import QueryEngine

logger = logging.getLogger(__name__)


@dataclass
class QualityLabConfig:
    """Configuration for QualityLab
    
    QualityLabの設定
    """
    
    # QA Generation settings
    qa_generation_model: str = "gpt-4o-mini"
    qa_pairs_per_document: int = 3
    question_types: List[str] = None
    
    # Evaluation settings  
    evaluation_timeout: float = 30.0
    similarity_threshold: float = 0.7
    
    # Reporting settings
    output_format: str = "markdown"  # "markdown", "json", "html"
    include_detailed_analysis: bool = True
    include_contradiction_detection: bool = True
    
    # Test Suite settings
    test_suite_config: Optional[TestSuiteConfig] = None
    evaluator_config: Optional[EvaluatorConfig] = None
    contradiction_config: Optional[ContradictionDetectorConfig] = None
    reporter_config: Optional[InsightReporterConfig] = None
    
    def __post_init__(self):
        """Initialize default configurations"""
        if self.question_types is None:
            self.question_types = [
                "factual",      # 事実確認質問
                "conceptual",   # 概念理解質問  
                "analytical",   # 分析的質問
                "comparative"   # 比較質問
            ]
        
        if self.test_suite_config is None:
            self.test_suite_config = TestSuiteConfig()
        
        if self.evaluator_config is None:
            self.evaluator_config = EvaluatorConfig()
        
        if self.contradiction_config is None:
            self.contradiction_config = ContradictionDetectorConfig()
        
        if self.reporter_config is None:
            self.reporter_config = InsightReporterConfig()


class QualityLab:
    """RAG System Quality Assessment and Evaluation Lab
    
    RAGシステムの品質評価と評価ラボ
    
    This class provides comprehensive evaluation capabilities for RAG systems:
    1. QA pair generation from corpus metadata
    2. QueryEngine evaluation using generated QA pairs
    3. Detailed evaluation reporting and analysis
    
    このクラスはRAGシステムの包括的な評価機能を提供します：
    1. コーパスメタデータからのQAペア生成
    2. 生成されたQAペアを使用したQueryEngineの評価
    3. 詳細な評価レポートと分析
    """
    
    def __init__(self, 
                 corpus_name: str,
                 config: Optional[QualityLabConfig] = None):
        """Initialize QualityLab
        
        Args:
            corpus_name: Name of the corpus for evaluation
                        評価対象のコーパス名
            config: Configuration for the lab
                   ラボの設定
        """
        self.corpus_name = corpus_name
        self.config = config or QualityLabConfig()
        
        # Initialize processing components
        self.test_suite = TestSuite(self.config.test_suite_config)
        self.evaluator = Evaluator(self.config.evaluator_config)
        self.contradiction_detector = ContradictionDetector(self.config.contradiction_config)
        self.insight_reporter = InsightReporter(self.config.reporter_config)
        
        # Processing statistics
        self.stats = {
            "qa_pairs_generated": 0,
            "evaluations_completed": 0,
            "reports_generated": 0,
            "total_processing_time": 0.0
        }
        
        logger.info(f"Initialized QualityLab for corpus '{corpus_name}'")
    
    def generate_qa_pairs(self, 
                         corpus_documents: List[Document], 
                         num_pairs: Optional[int] = None) -> List[QAPair]:
        """Generate QA pairs from corpus documents
        
        Args:
            corpus_documents: Documents from the corpus
                             コーパスからの文書
            num_pairs: Maximum number of QA pairs to generate
                      生成するQAペアの最大数
                      
        Returns:
            List[QAPair]: Generated QA pairs
                         生成されたQAペア
        """
        start_time = time.time()
        
        try:
            logger.info(f"Generating QA pairs from {len(corpus_documents)} documents")
            
            qa_pairs = []
            target_pairs = num_pairs or (len(corpus_documents) * self.config.qa_pairs_per_document)
            
            for doc in corpus_documents:
                if len(qa_pairs) >= target_pairs:
                    break
                
                # Generate QA pairs for this document
                doc_qa_pairs = self._generate_qa_pairs_for_document(doc)
                qa_pairs.extend(doc_qa_pairs)
            
            # Limit to requested number
            qa_pairs = qa_pairs[:target_pairs]
            
            # Update statistics
            self.stats["qa_pairs_generated"] += len(qa_pairs)
            self.stats["total_processing_time"] += time.time() - start_time
            
            logger.info(f"Generated {len(qa_pairs)} QA pairs in {time.time() - start_time:.2f}s")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"QA pair generation failed: {e}")
            raise
    
    def evaluate_query_engine(self, 
                             query_engine: QueryEngine,
                             qa_pairs: List[QAPair],
                             include_contradiction_detection: Optional[bool] = None) -> Dict[str, Any]:
        """Evaluate QueryEngine using QA pairs
        
        Args:
            query_engine: QueryEngine to evaluate
                         評価するQueryEngine
            qa_pairs: QA pairs for evaluation
                     評価用のQAペア
            include_contradiction_detection: Whether to include contradiction detection
                                           矛盾検出を含めるか
                                           
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
                           包括的な評価結果
        """
        start_time = time.time()
        
        try:
            logger.info(f"Evaluating QueryEngine with {len(qa_pairs)} QA pairs")
            
            # Convert QA pairs to test cases
            test_cases = self._qa_pairs_to_test_cases(qa_pairs)
            
            # Run evaluation using TestSuite
            test_results = []
            for test_case in test_cases:
                result = self._evaluate_single_case(query_engine, test_case)
                test_results.append(result)
            
            # Aggregate evaluation metrics
            evaluation_metrics = self._compute_evaluation_summary(test_results)
            
            # Optional: Contradiction detection
            contradiction_results = None
            if include_contradiction_detection or self.config.include_contradiction_detection:
                contradiction_results = self._detect_contradictions(test_results)
            
            # Convert test results to dictionary format for serialization
            test_results_dict = []
            for result in test_results:
                if hasattr(result, '__dict__'):
                    test_results_dict.append(result.__dict__)
                else:
                    test_results_dict.append(result)
            
            # Build comprehensive evaluation result
            evaluation_result = {
                "corpus_name": self.corpus_name,
                "query_engine_config": self._get_query_engine_config(query_engine),
                "evaluation_summary": evaluation_metrics,
                "test_results": test_results_dict,
                "contradiction_analysis": contradiction_results,
                "processing_time": time.time() - start_time,
                "timestamp": time.time()
            }
            
            # Update statistics
            self.stats["evaluations_completed"] += 1
            self.stats["total_processing_time"] += time.time() - start_time
            
            logger.info(f"Completed evaluation in {time.time() - start_time:.2f}s")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"QueryEngine evaluation failed: {e}")
            raise
    
    def generate_evaluation_report(self, 
                                 evaluation_results: Dict[str, Any],
                                 output_file: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report
        
        Args:
            evaluation_results: Results from evaluate_query_engine
                              evaluate_query_engineからの結果
            output_file: Optional file path to save report
                        レポートを保存するオプションのファイルパス
                        
        Returns:
            str: Generated report content
                生成されたレポート内容
        """
        start_time = time.time()
        
        try:
            logger.info("Generating evaluation report")
            
            # Create evaluation document for InsightReporter
            evaluation_doc = Document(
                id=f"evaluation_{self.corpus_name}_{int(time.time())}",
                content=self._format_evaluation_content(evaluation_results),
                metadata={
                    "processing_stage": "evaluation",
                    "corpus_name": self.corpus_name,
                    "overall_score": evaluation_results.get("evaluation_summary", {}).get("accuracy", 0.0),
                    "success_rate": evaluation_results.get("evaluation_summary", {}).get("pass_rate", 0.0),
                    "processing_time": evaluation_results.get("processing_time", 0.0),
                    "average_confidence": evaluation_results.get("evaluation_summary", {}).get("average_confidence", 0.0),
                    **evaluation_results.get("evaluation_summary", {})
                }
            )
            
            # Use InsightReporter to process evaluation and generate insights
            insight_docs = self.insight_reporter.process(evaluation_doc)
            
            # Get the report content from the insight document
            report_content = insight_docs[0].content if insight_docs else self._create_fallback_report(evaluation_results)
            
            # Save to file if specified
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                logger.info(f"Report saved to {output_file}")
            
            # Update statistics
            self.stats["reports_generated"] += 1
            self.stats["total_processing_time"] += time.time() - start_time
            
            logger.info(f"Generated report in {time.time() - start_time:.2f}s")
            return report_content
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise
    
    def run_full_evaluation(self, 
                           corpus_documents: List[Document],
                           query_engine: QueryEngine,
                           num_qa_pairs: Optional[int] = None,
                           output_file: Optional[str] = None) -> Dict[str, Any]:
        """Run complete evaluation workflow
        
        Args:
            corpus_documents: Documents from corpus
                             コーパスからの文書
            query_engine: QueryEngine to evaluate
                         評価するQueryEngine
            num_qa_pairs: Number of QA pairs to generate
                         生成するQAペア数
            output_file: File to save evaluation report
                        評価レポートを保存するファイル
                        
        Returns:
            Dict[str, Any]: Complete evaluation results with report
                           レポート付きの完全な評価結果
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting full evaluation for corpus '{self.corpus_name}'")
            
            # Step 1: Generate QA pairs
            qa_pairs = self.generate_qa_pairs(corpus_documents, num_qa_pairs)
            
            # Step 2: Evaluate QueryEngine
            evaluation_results = self.evaluate_query_engine(query_engine, qa_pairs)
            
            # Step 3: Generate report
            report = self.generate_evaluation_report(evaluation_results, output_file)
            
            # Complete results
            complete_results = {
                **evaluation_results,
                "qa_pairs": [self._qa_pair_to_dict(qp) for qp in qa_pairs],
                "evaluation_report": report,
                "total_workflow_time": time.time() - start_time
            }
            
            logger.info(f"Completed full evaluation in {time.time() - start_time:.2f}s")
            return complete_results
            
        except Exception as e:
            logger.error(f"Full evaluation failed: {e}")
            raise
    
    def get_lab_stats(self) -> Dict[str, Any]:
        """Get comprehensive lab statistics
        
        Returns:
            Dict[str, Any]: Lab statistics
                           ラボ統計
        """
        base_stats = self.stats.copy()
        
        # Add component statistics
        base_stats.update({
            "corpus_name": self.corpus_name,
            "test_suite_stats": self.test_suite.get_processing_stats(),
            "evaluator_stats": self.evaluator.get_processing_stats(),
            "contradiction_detector_stats": self.contradiction_detector.get_processing_stats(),
            "insight_reporter_stats": self.insight_reporter.get_processing_stats(),
            "config": {
                "qa_pairs_per_document": self.config.qa_pairs_per_document,
                "similarity_threshold": self.config.similarity_threshold,
                "output_format": self.config.output_format
            }
        })
        
        return base_stats
    
    def get_component_performance_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed component performance summary from evaluation results
        
        Args:
            evaluation_results: Results from evaluate_query_engine
            
        Returns:
            Dict with detailed component performance metrics
        """
        summary = evaluation_results.get("evaluation_summary", {})
        
        # Extract component performance
        retriever_performance = summary.get("retriever_performance", {})
        reranker_performance = summary.get("reranker_performance", {})
        
        # Format retriever performance for easy access
        formatted_retriever_perf = {}
        for retriever_id, perf in retriever_performance.items():
            formatted_retriever_perf[retriever_id] = {
                "type": perf.get("retriever_type", "unknown"),
                "recall": perf.get("average_recall", 0.0),
                "precision": perf.get("average_precision", 0.0),
                "f1_score": self._calculate_f1(perf.get("average_precision", 0.0), perf.get("average_recall", 0.0)),
                "avg_documents_found": perf.get("average_documents_found", 0.0),
                "avg_score": perf.get("average_score", 0.0),
                "error_rate": perf.get("error_count", 0) / perf.get("total_queries", 1),
                "total_queries": perf.get("total_queries", 0)
            }
        
        # Format reranker performance
        formatted_reranker_perf = None
        if reranker_performance.get("enabled", False):
            formatted_reranker_perf = {
                "type": reranker_performance.get("reranker_type", "unknown"),
                "recall_after_rerank": reranker_performance.get("average_recall_after_rerank", 0.0),
                "precision_after_rerank": reranker_performance.get("average_precision_after_rerank", 0.0),
                "f1_score_after_rerank": self._calculate_f1(
                    reranker_performance.get("average_precision_after_rerank", 0.0),
                    reranker_performance.get("average_recall_after_rerank", 0.0)
                ),
                "average_score_improvement": reranker_performance.get("average_improvement", 0.0),
                "avg_documents_removed": reranker_performance.get("total_documents_removed", 0) / reranker_performance.get("total_queries", 1),
                "total_queries": reranker_performance.get("total_queries", 0)
            }
        
        return {
            "retriever_performance": formatted_retriever_perf,
            "reranker_performance": formatted_reranker_perf,
            "overall_metrics": {
                "total_tests": summary.get("total_tests", 0),
                "overall_recall": summary.get("source_recall", 0.0),
                "overall_precision": summary.get("source_precision", 0.0),
                "overall_f1_score": summary.get("source_f1_score", 0.0),
                "pass_rate": summary.get("pass_rate", 0.0)
            }
        }
    
    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    # Private helper methods
    
    def _generate_qa_pairs_for_document(self, document: Document) -> List[QAPair]:
        """Generate QA pairs for a single document"""
        # This would use LLM to generate questions and answers
        # For now, return placeholder implementation
        qa_pairs = []
        
        for i in range(self.config.qa_pairs_per_document):
            question_type = self.config.question_types[i % len(self.config.question_types)]
            
            qa_pair = QAPair(
                question=f"What is discussed in document {document.id} regarding {question_type}?",
                answer=f"Based on the document content: {document.content[:100]}...",
                document_id=document.id,
                metadata={
                    "question_type": question_type,
                    "generated_from": document.id,
                    "corpus_name": self.corpus_name
                }
            )
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def _qa_pairs_to_test_cases(self, qa_pairs: List[QAPair]) -> List[TestCase]:
        """Convert QA pairs to test cases"""
        test_cases = []
        
        for i, qa_pair in enumerate(qa_pairs):
            test_case = TestCase(
                id=f"qa_test_{i}",
                query=qa_pair.question,
                expected_answer=qa_pair.answer,
                expected_sources=[qa_pair.document_id],
                metadata=qa_pair.metadata
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def _evaluate_single_case(self, query_engine: QueryEngine, test_case: TestCase) -> TestResult:
        """Evaluate a single test case with detailed retriever and reranker analysis"""
        start_time = time.time()
        
        try:
            # Perform detailed evaluation with component-wise analysis
            detailed_result = self._evaluate_with_component_analysis(query_engine, test_case.query)
            
            # Extract final source document IDs
            sources_found = [src.document_id for src in detailed_result["final_sources"]]
            
            # Enhanced source matching analysis for final result
            expected_sources_set = set(test_case.expected_sources)
            found_sources_set = set(sources_found)
            
            # Calculate different types of source accuracy
            exact_match = expected_sources_set == found_sources_set
            partial_match = len(expected_sources_set & found_sources_set) > 0
            precision = len(expected_sources_set & found_sources_set) / len(found_sources_set) if found_sources_set else 0
            recall = len(expected_sources_set & found_sources_set) / len(expected_sources_set) if expected_sources_set else 0
            
            # Pass/fail logic with enhanced criteria
            passed = partial_match  # Can be configured based on requirements
            
            # Add enhanced source analysis to metadata including component-wise analysis
            enhanced_metadata = test_case.metadata.copy()
            enhanced_metadata.update({
                "source_analysis": {
                    "exact_match": exact_match,
                    "partial_match": partial_match,
                    "precision": precision,
                    "recall": recall,
                    "expected_count": len(expected_sources_set),
                    "found_count": len(found_sources_set),
                    "intersection_count": len(expected_sources_set & found_sources_set)
                },
                "component_analysis": detailed_result["component_analysis"]
            })
            
            test_result = TestResult(
                test_case_id=test_case.id,
                query=test_case.query,
                generated_answer=detailed_result["answer"],
                expected_answer=test_case.expected_answer,
                sources_found=sources_found,
                expected_sources=test_case.expected_sources,
                processing_time=time.time() - start_time,
                confidence=detailed_result["confidence"],
                passed=passed,
                metadata=enhanced_metadata
            )
            
            return test_result
            
        except Exception as e:
            return TestResult(
                test_case_id=test_case.id,
                query=test_case.query,
                generated_answer="",
                expected_answer=test_case.expected_answer,
                sources_found=[],
                expected_sources=test_case.expected_sources,
                processing_time=time.time() - start_time,
                confidence=0.0,
                passed=False,
                error_message=str(e),
                metadata=test_case.metadata
            )
    
    def _detect_contradictions(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Detect contradictions in test results"""
        # Create documents from test results for contradiction detection
        all_contradictions = []
        
        for result in test_results:
            # Handle both dictionary and object formats
            if hasattr(result, 'test_case_id'):
                test_id = result.test_case_id
                content = result.generated_answer
                metadata = result.metadata
            else:
                test_id = result.get("test_case_id", "unknown")
                content = result.get("generated_answer", "")
                metadata = result.get("metadata", {})
            
            doc = Document(
                id=test_id,
                content=content,
                metadata=metadata
            )
            
            # Run contradiction detection on individual document
            try:
                contradictions = self.contradiction_detector.process(doc)
                all_contradictions.extend(contradictions)
            except Exception as e:
                logger.warning(f"Contradiction detection failed for document {doc.id}: {e}")
        
        return {
            "contradictions_found": len(all_contradictions),
            "contradiction_details": [doc.metadata for doc in all_contradictions if "contradictions" in doc.metadata]
        }
    
    def _evaluate_with_component_analysis(self, query_engine: QueryEngine, query: str) -> Dict[str, Any]:
        """Evaluate query with detailed component-wise analysis
        
        Args:
            query_engine: QueryEngine to analyze
            query: Query to process
            
        Returns:
            Dict containing detailed analysis of each component
        """
        try:
            # Step 1: Analyze each retriever individually
            retriever_results = []
            all_retriever_sources = []
            
            for i, retriever in enumerate(query_engine.retrievers):
                try:
                    # Get results from this specific retriever
                    retriever_sources = retriever.retrieve(
                        query=query, 
                        top_k=query_engine.config.retriever_top_k
                    )
                    
                    retriever_doc_ids = [src.document_id for src in retriever_sources]
                    all_retriever_sources.extend(retriever_sources)
                    
                    retriever_analysis = {
                        "retriever_index": i,
                        "retriever_type": type(retriever).__name__,
                        "documents_found": len(retriever_sources),
                        "document_ids": retriever_doc_ids,
                        "scores": [src.score for src in retriever_sources],
                        "average_score": sum(src.score for src in retriever_sources) / len(retriever_sources) if retriever_sources else 0.0
                    }
                    
                    retriever_results.append(retriever_analysis)
                    
                except Exception as e:
                    logger.warning(f"Retriever {i} analysis failed: {e}")
                    retriever_results.append({
                        "retriever_index": i,
                        "retriever_type": type(retriever).__name__,
                        "error": str(e),
                        "documents_found": 0,
                        "document_ids": [],
                        "scores": [],
                        "average_score": 0.0
                    })
            
            # Step 2: Analyze combined retriever results (before reranking)
            try:
                combined_sources = query_engine._combine_retriever_results(all_retriever_sources)
            except AttributeError:
                # Fallback if _combine_retriever_results doesn't exist
                combined_sources = all_retriever_sources
                
            combined_doc_ids = [src.document_id for src in combined_sources]
            
            combined_analysis = {
                "total_documents_before_rerank": len(combined_sources),
                "document_ids_before_rerank": combined_doc_ids,
                "deduplicated_count": len(set(combined_doc_ids)),
                "average_score_before_rerank": sum(src.score for src in combined_sources) / len(combined_sources) if combined_sources else 0.0
            }
            
            # Step 3: Analyze reranker if present
            reranker_analysis = None
            final_sources = combined_sources
            
            if query_engine.reranker and combined_sources:
                try:
                    reranked_sources = query_engine.reranker.rerank(
                        query=query,
                        sources=combined_sources,
                        top_k=query_engine.config.reranker_top_k
                    )
                    
                    final_sources = reranked_sources
                    reranked_doc_ids = [src.document_id for src in reranked_sources]
                    
                    reranker_analysis = {
                        "reranker_type": type(query_engine.reranker).__name__,
                        "documents_after_rerank": len(reranked_sources),
                        "document_ids_after_rerank": reranked_doc_ids,
                        "rerank_scores": [src.score for src in reranked_sources],
                        "average_score_after_rerank": sum(src.score for src in reranked_sources) / len(reranked_sources) if reranked_sources else 0.0,
                        "documents_removed_by_rerank": len(combined_sources) - len(reranked_sources),
                        "score_change": {
                            "before_avg": combined_analysis["average_score_before_rerank"],
                            "after_avg": sum(src.score for src in reranked_sources) / len(reranked_sources) if reranked_sources else 0.0
                        }
                    }
                    
                except Exception as e:
                    logger.warning(f"Reranker analysis failed: {e}")
                    reranker_analysis = {
                        "reranker_type": type(query_engine.reranker).__name__,
                        "error": str(e),
                        "documents_after_rerank": len(combined_sources),
                        "document_ids_after_rerank": combined_doc_ids
                    }
            
            # Step 4: Generate answer using final sources
            answer = ""
            confidence = 0.0
            
            try:
                if final_sources:
                    answer = query_engine.synthesizer.synthesize(
                        query=query,
                        sources=final_sources
                    )
                    confidence = getattr(query_engine.synthesizer, 'last_confidence', 0.8)  # Default confidence
                else:
                    answer = "No relevant sources found."
                    confidence = 0.0
                    
            except Exception as e:
                logger.warning(f"Answer synthesis failed: {e}")
                answer = f"Answer generation failed: {e}"
                confidence = 0.0
            
            # Compile comprehensive component analysis
            component_analysis = {
                "retriever_analysis": retriever_results,
                "combined_retriever_analysis": combined_analysis,
                "reranker_analysis": reranker_analysis,
                "synthesis_analysis": {
                    "final_source_count": len(final_sources),
                    "answer_generated": len(answer) > 0,
                    "answer_length": len(answer),
                    "confidence": confidence
                }
            }
            
            return {
                "answer": answer,
                "confidence": confidence,
                "final_sources": final_sources,
                "component_analysis": component_analysis
            }
            
        except Exception as e:
            logger.error(f"Component analysis failed: {e}")
            # Fallback to standard query if detailed analysis fails
            try:
                result = query_engine.query(query)
                return {
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "final_sources": result.sources,
                    "component_analysis": {
                        "error": f"Detailed analysis failed: {e}",
                        "fallback_used": True
                    }
                }
            except Exception as fallback_error:
                return {
                    "answer": "Analysis and fallback both failed",
                    "confidence": 0.0,
                    "final_sources": [],
                    "component_analysis": {
                        "error": f"Both detailed analysis and fallback failed: {e}, {fallback_error}"
                    }
                }
    
    def _get_query_engine_config(self, query_engine: QueryEngine) -> Dict[str, Any]:
        """Extract QueryEngine configuration"""
        return {
            "corpus_name": query_engine.corpus_name,
            "retriever_count": len(query_engine.retrievers),
            "has_reranker": query_engine.reranker is not None,
            "has_normalizer": query_engine.normalizer is not None
        }
    
    def _qa_pair_to_dict(self, qa_pair: QAPair) -> Dict[str, Any]:
        """Convert QAPair to dictionary"""
        return {
            "question": qa_pair.question,
            "answer": qa_pair.answer,
            "document_id": qa_pair.document_id,
            "metadata": qa_pair.metadata
        }
    
    def _format_evaluation_content(self, evaluation_results: Dict[str, Any]) -> str:
        """Format evaluation results into content for InsightReporter"""
        lines = []
        
        lines.append("# RAG System Evaluation Results")
        lines.append(f"コーパス: {evaluation_results.get('corpus_name', 'Unknown')}")
        lines.append(f"評価時刻: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(evaluation_results.get('timestamp', time.time())))}")
        lines.append("")
        
        # Evaluation summary
        if "evaluation_summary" in evaluation_results:
            summary = evaluation_results["evaluation_summary"]
            lines.append("## 評価サマリー")
            
            for metric, value in summary.items():
                if isinstance(value, float):
                    if "time" in metric.lower():
                        lines.append(f"- {metric}: {value:.3f}秒")
                    elif "rate" in metric.lower() or "accuracy" in metric.lower():
                        lines.append(f"- {metric}: {value:.1%}")
                    else:
                        lines.append(f"- {metric}: {value:.3f}")
                else:
                    lines.append(f"- {metric}: {value}")
            lines.append("")
        
        # Processing time
        if "processing_time" in evaluation_results:
            lines.append(f"処理時間: {evaluation_results['processing_time']:.3f}秒")
            lines.append("")
        
        # Test results summary
        if "test_results" in evaluation_results:
            test_results = evaluation_results["test_results"]
            passed_count = sum(1 for r in test_results if r.get("passed", False))
            total_count = len(test_results)
            
            lines.append("## テスト結果")
            lines.append(f"- 総テスト数: {total_count}")
            lines.append(f"- 成功: {passed_count}")
            lines.append(f"- 失敗: {total_count - passed_count}")
            
            if total_count > 0:
                lines.append(f"- 成功率: {(passed_count/total_count)*100:.1f}%")
                avg_confidence = sum(r.get("confidence", 0.0) for r in test_results) / total_count
                avg_time = sum(r.get("processing_time", 0.0) for r in test_results) / total_count
                lines.append(f"- 平均信頼度: {avg_confidence:.3f}")
                lines.append(f"- 平均応答時間: {avg_time:.3f}秒")
            else:
                lines.append(f"- 成功率: 0.0%")
            
            lines.append("")
        
        # Contradiction analysis
        if "contradiction_analysis" in evaluation_results and evaluation_results["contradiction_analysis"]:
            contradictions = evaluation_results["contradiction_analysis"]
            lines.append("## 矛盾分析")
            lines.append(f"- 検出された矛盾: {contradictions.get('contradictions_found', 0)}件")
            lines.append("")
        
        return "\n".join(lines)
    
    def _create_fallback_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Create a fallback report when InsightReporter fails"""
        lines = []
        
        lines.append("# RAG System Evaluation Report")
        lines.append("=" * 50)
        lines.append("")
        
        lines.append(f"**Corpus**: {evaluation_results.get('corpus_name', 'Unknown')}")
        lines.append(f"**Evaluation Time**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(evaluation_results.get('timestamp', time.time())))}")
        lines.append("")
        
        # Summary
        if "evaluation_summary" in evaluation_results:
            lines.append("## Summary")
            summary = evaluation_results["evaluation_summary"]
            
            for key, value in summary.items():
                if isinstance(value, float):
                    if "time" in key.lower():
                        lines.append(f"- **{key.title()}**: {value:.3f}s")
                    elif "rate" in key.lower() or "accuracy" in key.lower():
                        lines.append(f"- **{key.title()}**: {value:.1%}")
                    else:
                        lines.append(f"- **{key.title()}**: {value:.3f}")
                else:
                    lines.append(f"- **{key.title()}**: {value}")
            lines.append("")
        
        # Test Results
        if "test_results" in evaluation_results:
            test_results = evaluation_results["test_results"]
            lines.append("## Test Results")
            lines.append(f"Total tests: {len(test_results)}")
            
            passed = sum(1 for r in test_results if r.get("passed", False))
            lines.append(f"Passed: {passed}")
            lines.append(f"Failed: {len(test_results) - passed}")
            lines.append(f"Success Rate: {(passed/len(test_results))*100:.1f}%")
            lines.append("")
        
        # Processing Info
        if "processing_time" in evaluation_results:
            lines.append("## Performance")
            lines.append(f"Processing Time: {evaluation_results['processing_time']:.3f}s")
            lines.append("")
        
        lines.append("---")
        lines.append("Report generated by QualityLab")
        
        return "\n".join(lines)
    
    def _compute_evaluation_summary(self, test_results: List) -> Dict[str, Any]:
        """Compute evaluation summary from test results with component-wise analysis"""
        if not test_results:
            return {}
        
        # Convert test results to dictionary format if needed
        results_data = []
        for result in test_results:
            if hasattr(result, '__dict__'):
                results_data.append(result.__dict__)
            else:
                results_data.append(result)
        
        # Basic metrics
        total_tests = len(results_data)
        passed_tests = sum(1 for r in results_data if r.get("passed", False))
        
        # Calculate metrics
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        avg_confidence = sum(r.get("confidence", 0.0) for r in results_data) / total_tests if total_tests > 0 else 0.0
        avg_processing_time = sum(r.get("processing_time", 0.0) for r in results_data) / total_tests if total_tests > 0 else 0.0
        
        # Enhanced source accuracy analysis
        source_matches = 0
        exact_matches = 0
        total_source_checks = 0
        total_precision = 0.0
        total_recall = 0.0
        
        # Component-wise analysis aggregation
        retriever_performance = {}
        reranker_performance = {"enabled": False}
        
        for r in results_data:
            expected_sources = r.get("expected_sources", [])
            found_sources = r.get("sources_found", [])
            
            if expected_sources:
                total_source_checks += 1
                
                # Get enhanced source analysis from metadata if available
                source_analysis = r.get("metadata", {}).get("source_analysis", {})
                component_analysis = r.get("metadata", {}).get("component_analysis", {})
                
                if source_analysis:
                    if source_analysis.get("exact_match", False):
                        exact_matches += 1
                    if source_analysis.get("partial_match", False):
                        source_matches += 1
                    total_precision += source_analysis.get("precision", 0.0)
                    total_recall += source_analysis.get("recall", 0.0)
                else:
                    # Fallback to simple analysis
                    if any(src in found_sources for src in expected_sources):
                        source_matches += 1
                
                # Aggregate component-wise performance
                if component_analysis:
                    self._aggregate_component_performance(
                        component_analysis, 
                        expected_sources, 
                        retriever_performance, 
                        reranker_performance
                    )
        
        source_accuracy = source_matches / total_source_checks if total_source_checks > 0 else 0.0
        exact_match_rate = exact_matches / total_source_checks if total_source_checks > 0 else 0.0
        avg_precision = total_precision / total_source_checks if total_source_checks > 0 else 0.0
        avg_recall = total_recall / total_source_checks if total_source_checks > 0 else 0.0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        # Error analysis
        error_rate = sum(1 for r in results_data if r.get("error_message")) / total_tests if total_tests > 0 else 0.0
        
        # Finalize component performance metrics
        for retriever_id in retriever_performance:
            perf = retriever_performance[retriever_id]
            if perf["total_queries"] > 0:
                perf["average_recall"] = perf["total_recall"] / perf["total_queries"]
                perf["average_precision"] = perf["total_precision"] / perf["total_queries"]
                perf["average_documents_found"] = perf["total_documents_found"] / perf["total_queries"]
                perf["average_score"] = perf["total_score"] / perf["total_documents_found"] if perf["total_documents_found"] > 0 else 0.0
        
        if reranker_performance["enabled"] and reranker_performance.get("total_queries", 0) > 0:
            reranker_performance["average_improvement"] = reranker_performance["total_score_improvement"] / reranker_performance["total_queries"]
            reranker_performance["average_recall_after_rerank"] = reranker_performance["total_recall_after_rerank"] / reranker_performance["total_queries"]
            reranker_performance["average_precision_after_rerank"] = reranker_performance["total_precision_after_rerank"] / reranker_performance["total_queries"]
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": pass_rate,
            "accuracy": pass_rate,  # alias for pass_rate
            "average_confidence": avg_confidence,
            "average_processing_time": avg_processing_time,
            
            # Enhanced source analysis metrics
            "source_accuracy": source_accuracy,
            "exact_match_rate": exact_match_rate,
            "source_precision": avg_precision,
            "source_recall": avg_recall,
            "source_f1_score": f1_score,
            
            # Component-wise performance metrics
            "retriever_performance": retriever_performance,
            "reranker_performance": reranker_performance,
            
            "error_rate": error_rate,
            "overall_score": (pass_rate + avg_confidence + source_accuracy + f1_score) / 4.0
        }
    
    def _aggregate_component_performance(self, component_analysis: Dict, expected_sources: List[str], 
                                       retriever_performance: Dict, reranker_performance: Dict):
        """Aggregate component performance metrics from individual test cases
        
        Args:
            component_analysis: Component analysis from a single test case
            expected_sources: Expected source documents for this test case
            retriever_performance: Aggregated retriever performance metrics
            reranker_performance: Aggregated reranker performance metrics
        """
        expected_sources_set = set(expected_sources)
        
        # Process retriever analysis
        retriever_analysis = component_analysis.get("retriever_analysis", [])
        for retriever_data in retriever_analysis:
            retriever_id = f"retriever_{retriever_data.get('retriever_index', 0)}_{retriever_data.get('retriever_type', 'unknown')}"
            
            if retriever_id not in retriever_performance:
                retriever_performance[retriever_id] = {
                    "retriever_type": retriever_data.get('retriever_type', 'unknown'),
                    "total_queries": 0,
                    "total_documents_found": 0,
                    "total_recall": 0.0,
                    "total_precision": 0.0,
                    "total_score": 0.0,
                    "error_count": 0
                }
            
            perf = retriever_performance[retriever_id]
            perf["total_queries"] += 1
            
            if "error" in retriever_data:
                perf["error_count"] += 1
            else:
                found_docs = set(retriever_data.get("document_ids", []))
                docs_found_count = retriever_data.get("documents_found", 0)
                perf["total_documents_found"] += docs_found_count
                
                # Calculate recall and precision for this retriever
                if expected_sources_set:
                    intersection = expected_sources_set & found_docs
                    recall = len(intersection) / len(expected_sources_set)
                    precision = len(intersection) / len(found_docs) if found_docs else 0.0
                    
                    perf["total_recall"] += recall
                    perf["total_precision"] += precision
                
                # Add average score
                scores = retriever_data.get("scores", [])
                if scores:
                    perf["total_score"] += sum(scores)
        
        # Process reranker analysis
        reranker_analysis = component_analysis.get("reranker_analysis")
        if reranker_analysis and not reranker_analysis.get("error"):
            reranker_performance["enabled"] = True
            
            if "total_queries" not in reranker_performance:
                reranker_performance.update({
                    "total_queries": 0,
                    "total_score_improvement": 0.0,
                    "total_recall_after_rerank": 0.0,
                    "total_precision_after_rerank": 0.0,
                    "total_documents_removed": 0,
                    "reranker_type": reranker_analysis.get("reranker_type", "unknown")
                })
            
            reranker_performance["total_queries"] += 1
            
            # Calculate score improvement
            score_change = reranker_analysis.get("score_change", {})
            before_avg = score_change.get("before_avg", 0.0)
            after_avg = score_change.get("after_avg", 0.0)
            reranker_performance["total_score_improvement"] += (after_avg - before_avg)
            
            # Calculate recall and precision after reranking
            reranked_docs = set(reranker_analysis.get("document_ids_after_rerank", []))
            if expected_sources_set and reranked_docs:
                intersection = expected_sources_set & reranked_docs
                recall_after = len(intersection) / len(expected_sources_set)
                precision_after = len(intersection) / len(reranked_docs)
                
                reranker_performance["total_recall_after_rerank"] += recall_after
                reranker_performance["total_precision_after_rerank"] += precision_after
            
            # Track documents removed
            reranker_performance["total_documents_removed"] += reranker_analysis.get("documents_removed_by_rerank", 0)