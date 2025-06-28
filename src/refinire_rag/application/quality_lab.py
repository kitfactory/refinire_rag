"""
QualityLab - RAG System Evaluation and Quality Assessment

A Refinire Step that provides comprehensive evaluation of RAG systems including
QA pair generation, QueryEngine evaluation, and reporting.
"""

import logging
import time
import json
import os
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
from .query_engine_new import QueryEngine
from .corpus_manager_new import CorpusManager
from ..storage.evaluation_store import SQLiteEvaluationStore, EvaluationRun
from refinire import RefinireAgent

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

    @classmethod
    def from_env(cls) -> 'QualityLabConfig':
        """Create QualityLabConfig from environment variables
        環境変数からQualityLabConfigを作成
        
        Returns:
            QualityLabConfig: Configuration loaded from environment
                            環境変数から読み込まれた設定
        """
        # QA Generation settings
        qa_generation_model = os.getenv("REFINIRE_RAG_QA_GENERATION_MODEL", "gpt-4o-mini")
        qa_pairs_per_document = int(os.getenv("REFINIRE_RAG_QA_PAIRS_PER_DOCUMENT", "3"))
        
        question_types_str = os.getenv("REFINIRE_RAG_QUESTION_TYPES", "factual,conceptual,analytical,comparative")
        question_types = [t.strip() for t in question_types_str.split(",") if t.strip()]
        
        # Evaluation settings
        evaluation_timeout = float(os.getenv("REFINIRE_RAG_EVALUATION_TIMEOUT", "30.0"))
        similarity_threshold = float(os.getenv("REFINIRE_RAG_SIMILARITY_THRESHOLD", "0.7"))
        
        # Reporting settings
        output_format = os.getenv("REFINIRE_RAG_OUTPUT_FORMAT", "markdown")
        include_detailed_analysis = os.getenv("REFINIRE_RAG_INCLUDE_DETAILED_ANALYSIS", "true").lower() == "true"
        include_contradiction_detection = os.getenv("REFINIRE_RAG_INCLUDE_CONTRADICTION_DETECTION", "true").lower() == "true"
        
        # Create nested configs with defaults
        test_suite_config = TestSuiteConfig()
        evaluator_config = EvaluatorConfig()
        contradiction_config = ContradictionDetectorConfig()
        reporter_config = InsightReporterConfig()
        
        return cls(
            qa_generation_model=qa_generation_model,
            qa_pairs_per_document=qa_pairs_per_document,
            question_types=question_types,
            evaluation_timeout=evaluation_timeout,
            similarity_threshold=similarity_threshold,
            output_format=output_format,
            include_detailed_analysis=include_detailed_analysis,
            include_contradiction_detection=include_contradiction_detection,
            test_suite_config=test_suite_config,
            evaluator_config=evaluator_config,
            contradiction_config=contradiction_config,
            reporter_config=reporter_config
        )


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
    
    def __init__(self, **kwargs):
        """Initialize QualityLab
        
        Args:
            **kwargs: Configuration parameters including:
                - corpus_manager: CorpusManager instance for document retrieval
                - config: QualityLabConfig instance or None for environment loading
                - evaluation_store: Optional evaluation data store for persistence
                - qa_generation_model: Model for QA generation
                - qa_pairs_per_document: Number of QA pairs per document
                - question_types: List of question types to generate
                - evaluation_timeout: Timeout for evaluation operations
                - similarity_threshold: Similarity threshold for evaluation
                - output_format: Output format for reports
                - include_detailed_analysis: Whether to include detailed analysis
                - include_contradiction_detection: Whether to include contradiction detection
        """
        # Extract components from kwargs
        corpus_manager = kwargs.pop('corpus_manager', None)
        config = kwargs.pop('config', None)
        evaluation_store = kwargs.pop('evaluation_store', None)
        
        # Initialize from environment if components are None
        if corpus_manager is None:
            self.corpus_manager = CorpusManager()
        else:
            self.corpus_manager = corpus_manager
            
        # Create config from environment if not provided
        if config is None:
            self.config = QualityLabConfig.from_env()
            # Override with any kwargs parameters
            if kwargs:
                for key, value in kwargs.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
        else:
            self.config = config
        
        if evaluation_store is None:
            db_path = os.getenv("REFINIRE_RAG_EVALUATION_DB_PATH", "./data/evaluation.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.evaluation_store = SQLiteEvaluationStore(db_path)
        else:
            self.evaluation_store = evaluation_store
        
        # Initialize processing components using PluginFactory
        from ..factories.plugin_factory import PluginFactory
        
        # Create plugins from environment variables with fallbacks
        self.test_suite = kwargs.get('test_suite') or PluginFactory.create_test_suites_from_env() or TestSuite(self.config.test_suite_config)
        self.evaluator = kwargs.get('evaluator') or PluginFactory.create_evaluators_from_env() or Evaluator(self.config.evaluator_config)
        self.contradiction_detector = kwargs.get('contradiction_detector') or PluginFactory.create_contradiction_detectors_from_env() or ContradictionDetector(self.config.contradiction_config)
        self.insight_reporter = kwargs.get('insight_reporter') or PluginFactory.create_insight_reporters_from_env() or InsightReporter(self.config.reporter_config)
        
        # Statistics tracking
        self.stats = {
            "qa_pairs_generated": 0,
            "evaluations_completed": 0,
            "reports_generated": 0,
            "total_processing_time": 0.0
        }
        
        logger.info(f"Initialized QualityLab with CorpusManager")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary
        
        Returns:
            Current configuration settings
        """
        return {
            'qa_generation_model': self.config.qa_generation_model,
            'qa_pairs_per_document': self.config.qa_pairs_per_document,
            'question_types': self.config.question_types,
            'evaluation_timeout': self.config.evaluation_timeout,
            'similarity_threshold': self.config.similarity_threshold,
            'output_format': self.config.output_format,
            'include_detailed_analysis': self.config.include_detailed_analysis,
            'include_contradiction_detection': self.config.include_contradiction_detection
        }
    
    def generate_qa_pairs(self, 
                         qa_set_name: str,
                         corpus_name: str,
                         document_filters: Optional[Dict[str, Any]] = None,
                         generation_metadata: Optional[Dict[str, Any]] = None,
                         num_pairs: Optional[int] = None,
                         use_original_documents: bool = True) -> List[QAPair]:
        """Generate QA pairs from corpus documents with identification
        
        Args:
            qa_set_name: Name/ID for the QA pair set for identification
                        QAペアセットの識別用名前/ID
            corpus_name: Name of the source corpus
                        元となるコーパス名
            document_filters: Metadata filters to select documents from corpus
                            コーパスから文書を選択するメタデータフィルタ
            generation_metadata: Additional metadata for generation conditions
                                生成条件の追加メタデータ
            num_pairs: Maximum number of QA pairs to generate
                      生成するQAペアの最大数
            use_original_documents: Use original documents instead of processed ones
                                  処理済み文書ではなく元の文書を使用
                      
        Returns:
            List[QAPair]: Generated QA pairs with enhanced metadata
                         拡張メタデータ付きの生成されたQAペア
        """
        start_time = time.time()
        
        try:
            logger.info(f"Generating QA set '{qa_set_name}' from corpus '{corpus_name}' with filters: {document_filters}")
            
            # Retrieve documents from CorpusManager
            corpus_documents = self._retrieve_corpus_documents(corpus_name, document_filters, use_original_documents)
            logger.info(f"Retrieved {len(corpus_documents)} documents from corpus '{corpus_name}'")
            
            if not corpus_documents:
                logger.warning(f"No documents found in corpus '{corpus_name}' with filters: {document_filters}")
                return []
            
            # Initialize generation metadata
            if generation_metadata is None:
                generation_metadata = {}
            
            # Add generation context to metadata
            base_metadata = {
                "qa_set_name": qa_set_name,
                "corpus_name": corpus_name,
                "document_filters": document_filters,
                "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_document_count": len(corpus_documents),
                "use_original_documents": use_original_documents,
                "generation_config": {
                    "qa_pairs_per_document": self.config.qa_pairs_per_document,
                    "question_types": self.config.question_types,
                    "qa_generation_model": self.config.qa_generation_model
                },
                **generation_metadata
            }
            
            qa_pairs = []
            target_pairs = num_pairs or (len(corpus_documents) * self.config.qa_pairs_per_document)
            
            for doc in corpus_documents:
                if len(qa_pairs) >= target_pairs:
                    break
                
                # Generate QA pairs for this document with enhanced metadata
                doc_qa_pairs = self._generate_qa_pairs_for_document(doc, base_metadata)
                qa_pairs.extend(doc_qa_pairs)
            
            # Limit to requested number
            qa_pairs = qa_pairs[:target_pairs]
            
            # Update statistics
            self.stats["qa_pairs_generated"] += len(qa_pairs)
            self.stats["total_processing_time"] += time.time() - start_time
            
            logger.info(f"Generated {len(qa_pairs)} QA pairs for set '{qa_set_name}' in {time.time() - start_time:.2f}s")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"QA pair generation failed: {e}")
            raise
    
    def _retrieve_corpus_documents(self, 
                                 corpus_name: str, 
                                 document_filters: Optional[Dict[str, Any]] = None, 
                                 use_original_documents: bool = True) -> List[Document]:
        """
        Retrieve documents from CorpusManager based on filters
        
        CorpusManagerからフィルタ条件で文書を取得
        
        Args:
            corpus_name: Name of the corpus to search
                        検索するコーパス名
            document_filters: Metadata filters for document selection
                            文書選択用のメタデータフィルタ
            use_original_documents: Whether to filter for original documents
                                  元文書のみフィルタするかどうか
                                  
        Returns:
            List[Document]: Retrieved documents
                           取得された文書リスト
        """
        try:
            if not self.corpus_manager:
                logger.warning("No corpus manager available")
                return []
            
            # Get documents by stage
            stage = "original" if use_original_documents else "processed"
            documents = self.corpus_manager._get_documents_by_stage(stage)
            
            # Apply additional metadata filters if provided
            if document_filters:
                filtered_documents = []
                for doc in documents:
                    if self._matches_filters(doc, document_filters):
                        filtered_documents.append(doc)
                return filtered_documents
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents from corpus '{corpus_name}': {e}")
            return []
    
    def _matches_filters(self, document: Document, filters: Optional[Dict[str, Any]]) -> bool:
        """
        Check if a document matches the given filters
        
        文書が指定されたフィルタに一致するかチェック
        
        Args:
            document: Document to check
                     チェックする文書
            filters: Filter conditions
                    フィルタ条件
                    
        Returns:
            bool: True if document matches all filters
                 すべてのフィルタに一致する場合True
        """
        if not filters:
            return True
            
        for key, value in filters.items():
            doc_value = document.metadata.get(key)
            
            if isinstance(value, dict):
                # Handle operators like {"$in": [...], "$gte": ...}
                for op, op_value in value.items():
                    if op == "$in" and doc_value not in op_value:
                        return False
                    elif op == "$gte" and (doc_value is None or doc_value < op_value):
                        return False
                    elif op == "$lte" and (doc_value is None or doc_value > op_value):
                        return False
                    elif op == "$gt" and (doc_value is None or doc_value <= op_value):
                        return False
                    elif op == "$lt" and (doc_value is None or doc_value >= op_value):
                        return False
                    elif op == "$contains" and op_value.lower() not in str(doc_value).lower():
                        return False
            else:
                # Direct equality check
                if doc_value != value:
                    return False
        
        return True
    
    def _generate_qa_pairs_for_document(self, document: Document, base_metadata: Dict[str, Any]) -> List[QAPair]:
        """Generate QA pairs for a single document with enhanced metadata using RefinireAgent
        
        RefinireAgentを使用して単一ドキュメントから拡張メタデータ付きのQAペアを生成
        """
        try:
            # Create RefinireAgent for QA generation
            agent = RefinireAgent(
                name="qa_generator",
                generation_instructions=f"""
                You are an expert QA pair generator. Generate high-quality question-answer pairs from the provided document content.
                
                Generate exactly {self.config.qa_pairs_per_document} QA pairs with the following question types: {', '.join(self.config.question_types)}
                
                Requirements:
                - Questions should be clear, specific, and answerable from the document
                - Answers should be accurate and directly based on the document content
                - Vary question types across the specified types
                - Return response in valid JSON format with "qa_pairs" array
                - Each QA pair should have: question, answer, question_type
                
                Document content:
                {document.content}
                """,
                model=self.config.qa_generation_model,
                temperature=0.3,
                max_tokens=2000
            )
            
            # Generate QA pairs using RefinireAgent
            prompt = f"Generate {self.config.qa_pairs_per_document} QA pairs from the document content above."
            result = agent.run(prompt)
            
            # Parse the generated response
            try:
                # Strip markdown code blocks if present
                content = result.content.strip()
                if content.startswith('```json'):
                    content = content[7:]  # Remove ```json
                if content.endswith('```'):
                    content = content[:-3]  # Remove ```
                content = content.strip()
                
                qa_data = json.loads(content)
                generated_pairs = qa_data.get("qa_pairs", [])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse QA generation response as JSON: {result.content}")
                generated_pairs = []
            
            # Convert to QAPair objects with enhanced metadata
            qa_pairs = []
            for i, pair_data in enumerate(generated_pairs):
                question_type = pair_data.get("question_type", self.config.question_types[i % len(self.config.question_types)])
                
                # Combine base metadata with document-specific metadata
                enhanced_metadata = {
                    **base_metadata,
                    "question_type": question_type,
                    "generated_from": document.id,
                    "document_metadata": document.metadata,
                    "pair_index": i,
                    "generation_method": "refinire_agent"
                }
                
                qa_pair = QAPair(
                    question=pair_data.get("question", f"Generated question {i+1}"),
                    answer=pair_data.get("answer", f"Generated answer {i+1}"),
                    document_id=document.id,
                    metadata=enhanced_metadata
                )
                qa_pairs.append(qa_pair)
            
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error generating QA pairs for document {document.id}: {e}")
            return []  # Return empty list on error
    
    def evaluate_query_engine(self, 
                             query_engine: QueryEngine,
                             qa_pairs: List[QAPair],
                             save_results: bool = True) -> Dict[str, Any]:
        """Evaluate QueryEngine using QA pairs with detailed component analysis"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting evaluation with {len(qa_pairs)} QA pairs")
            
            # Convert QA pairs to test cases
            test_cases = self._qa_pairs_to_test_cases(qa_pairs)
            
            # Execute tests with detailed analysis
            test_results = []
            
            for test_case in test_cases:
                result = self._evaluate_single_case(query_engine, test_case)
                test_results.append(result)
            
            # Process results through evaluation pipeline
            evaluation_docs = []
            for result in test_results:
                eval_doc = Document(
                    id=f"eval_{result.test_case_id}",
                    content=self._format_test_result(result),
                    metadata={
                        "processing_stage": "test_results",
                        "test_case_id": result.test_case_id,
                        "passed": result.passed,
                        "confidence": result.confidence,
                        "processing_time": result.processing_time
                    }
                )
                evaluation_docs.append(eval_doc)
            
            # Run evaluation analysis
            evaluation_results = {}
            for eval_doc in evaluation_docs:
                evaluator_results = self.evaluator.process(eval_doc)
                for eval_result in evaluator_results:
                    evaluation_results.update(eval_result.metadata)
            
            # Contradiction detection if enabled
            contradiction_analysis = {}
            if self.config.include_contradiction_detection:
                contradiction_analysis = self._detect_contradictions(test_results)
            
            # Compile comprehensive results
            results = {
                "evaluation_summary": self._compile_evaluation_summary(test_results),
                "test_results": [self._test_result_to_dict(tr) for tr in test_results],
                "contradiction_analysis": contradiction_analysis,
                "evaluation_time": time.time() - start_time,
                "corpus_name": qa_pairs[0].metadata.get("corpus_name", "unknown") if qa_pairs else "unknown",
                "qa_set_name": qa_pairs[0].metadata.get("qa_set_name", "unknown") if qa_pairs else "unknown",
                "timestamp": time.time()
            }
            
            # Update statistics
            self.stats["evaluations_completed"] += 1
            self.stats["total_processing_time"] += time.time() - start_time
            
            logger.info(f"Completed evaluation in {time.time() - start_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

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

    def _evaluate_with_component_analysis(self, query_engine: QueryEngine, query: str) -> Dict[str, Any]:
        """Placeholder for detailed component analysis"""
        # This would perform detailed evaluation of retriever, reranker, and reader
        # For now, return mock result
        return {
            "answer": f"Generated answer for: {query}",
            "confidence": 0.8,
            "final_sources": [{"document_id": "doc_001"}, {"document_id": "doc_002"}],
            "component_analysis": {
                "retriever_performance": {"precision": 0.8, "recall": 0.7},
                "reranker_performance": {"improvement": 0.1},
                "reader_performance": {"confidence": 0.8}
            }
        }

    def _format_test_result(self, result: TestResult) -> str:
        """Format test result as string"""
        status = "✅ PASS" if result.passed else "❌ FAIL"
        return f"""
{status} {result.test_case_id}
**Query**: {result.query}
**Generated Answer**: {result.generated_answer}
**Confidence**: {result.confidence:.3f}
**Processing Time**: {result.processing_time:.3f}s
**Sources Found**: {len(result.sources_found)}
"""

    def _compile_evaluation_summary(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Compile summary statistics from test results"""
        if not test_results:
            return {}
        
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.passed)
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests,
            "average_confidence": sum(r.confidence for r in test_results) / total_tests,
            "average_processing_time": sum(r.processing_time for r in test_results) / total_tests
        }

    def _test_result_to_dict(self, result: TestResult) -> Dict[str, Any]:
        """Convert TestResult to dictionary"""
        return {
            "test_case_id": result.test_case_id,
            "query": result.query,
            "generated_answer": result.generated_answer,
            "expected_answer": result.expected_answer,
            "sources_found": result.sources_found,
            "expected_sources": result.expected_sources,
            "processing_time": result.processing_time,
            "confidence": result.confidence,
            "passed": result.passed,
            "error_message": result.error_message,
            "metadata": result.metadata
        }

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

    def generate_evaluation_report(self, 
                                 evaluation_results: Dict[str, Any], 
                                 output_file: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report"""
        try:
            # Use InsightReporter to generate detailed report
            report_doc = Document(
                id="evaluation_report",
                content=json.dumps(evaluation_results, indent=2),
                metadata={
                    "processing_stage": "evaluation_results",
                    "report_type": "comprehensive",
                    "timestamp": evaluation_results.get("timestamp", time.time())
                }
            )
            
            # Generate insights
            insight_docs = self.insight_reporter.process(report_doc)
            
            if insight_docs:
                report_content = insight_docs[0].content
            else:
                # Fallback to simple report
                report_content = self._create_fallback_report(evaluation_results)
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"Evaluation report saved to {output_file}")
            
            # Update statistics
            self.stats["reports_generated"] += 1
            
            return report_content
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return self._create_fallback_report(evaluation_results)

    def _create_fallback_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Create a fallback report when InsightReporter fails"""
        lines = []
        
        lines.append("# RAG System Evaluation Report")
        lines.append("=" * 50)
        lines.append("")
        
        lines.append(f"**Corpus**: {evaluation_results.get('corpus_name', 'Unknown')}")
        lines.append(f"**QA Set**: {evaluation_results.get('qa_set_name', 'Unknown')}")
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
        
        return "\n".join(lines)

    def run_full_evaluation(self, 
                           qa_set_name: str,
                           corpus_name: str,
                           query_engine: QueryEngine,
                           document_filters: Optional[Dict[str, Any]] = None,
                           generation_metadata: Optional[Dict[str, Any]] = None,
                           num_qa_pairs: Optional[int] = None,
                           output_file: Optional[str] = None) -> Dict[str, Any]:
        """Run complete evaluation workflow with CorpusManager integration"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting full evaluation for QA set '{qa_set_name}' from corpus '{corpus_name}'")
            
            # Step 1: Generate QA pairs from corpus
            qa_pairs = self.generate_qa_pairs(
                qa_set_name=qa_set_name,
                corpus_name=corpus_name,
                document_filters=document_filters,
                generation_metadata=generation_metadata,
                num_pairs=num_qa_pairs
            )
            
            if not qa_pairs:
                logger.warning("No QA pairs generated, skipping evaluation")
                return {"error": "No QA pairs generated"}
            
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

    def _qa_pair_to_dict(self, qa_pair: QAPair) -> Dict[str, Any]:
        """Convert QAPair to dictionary"""
        return {
            "question": qa_pair.question,
            "answer": qa_pair.answer,
            "document_id": qa_pair.document_id,
            "metadata": qa_pair.metadata
        }

    def get_lab_stats(self) -> Dict[str, Any]:
        """Get comprehensive lab statistics"""
        base_stats = self.stats.copy()
        
        # Add component statistics
        base_stats.update({
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
    
    def evaluate_with_existing_qa_pairs(self,
                                       evaluation_name: str,
                                       qa_set_id: str,
                                       query_engine: QueryEngine,
                                       save_results: bool = True,
                                       evaluation_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate QueryEngine using existing QA pairs from storage
        
        既存のQAペアを使用してQueryEngineを評価
        
        Args:
            evaluation_name: Name for this evaluation run
                           この評価実行の名前
            qa_set_id: ID of the QA pair set to use
                      使用するQAペアセットのID
            query_engine: QueryEngine to evaluate
                         評価するQueryEngine
            save_results: Whether to save results to evaluation store
                         結果を評価ストアに保存するかどうか
            evaluation_metadata: Additional metadata for this evaluation
                                この評価の追加メタデータ
                                
        Returns:
            Dict[str, Any]: Evaluation results with metrics
                           メトリクス付きの評価結果
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting evaluation '{evaluation_name}' with QA set '{qa_set_id}'")
            
            # Retrieve QA pairs from storage
            if not self.evaluation_store:
                raise ValueError("EvaluationStore is required for evaluating with existing QA pairs")
            
            qa_pairs = self.evaluation_store.get_qa_pairs_by_set_id(qa_set_id)
            
            if not qa_pairs:
                raise ValueError(f"No QA pairs found for set ID: {qa_set_id}")
            
            logger.info(f"Retrieved {len(qa_pairs)} QA pairs from set '{qa_set_id}'")
            
            # Create evaluation run if saving results
            run_id = None
            if save_results:
                run_id = self._create_evaluation_run(evaluation_name, qa_set_id, evaluation_metadata)
            
            # Execute evaluation
            evaluation_results = self.evaluate_query_engine(query_engine, qa_pairs, save_results=False)
            
            # Save results if requested
            if save_results and run_id:
                self._save_evaluation_results(run_id, qa_pairs, evaluation_results)
            
            # Add evaluation metadata
            evaluation_results.update({
                "evaluation_name": evaluation_name,
                "qa_set_id": qa_set_id,
                "run_id": run_id,
                "qa_pairs_count": len(qa_pairs),
                "evaluation_metadata": evaluation_metadata or {}
            })
            
            logger.info(f"Completed evaluation '{evaluation_name}' in {time.time() - start_time:.2f}s")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation with existing QA pairs failed: {e}")
            raise
    
    def _create_evaluation_run(self,
                              evaluation_name: str,
                              qa_set_id: str,
                              evaluation_metadata: Optional[Dict[str, Any]]) -> str:
        """
        Create a new evaluation run record
        
        新しい評価実行記録を作成
        
        Args:
            evaluation_name: Name of the evaluation
                           評価の名前
            qa_set_id: ID of the QA set being used
                      使用するQAセットのID
            evaluation_metadata: Additional metadata
                                追加メタデータ
                                
        Returns:
            str: Created run ID
                作成された実行ID
        """
        import uuid
        from datetime import datetime
        
        run_id = f"eval_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        run = EvaluationRun(
            id=run_id,
            name=evaluation_name,
            description=f"Evaluation using QA set {qa_set_id}",
            created_at=datetime.now(),
            status="running",
            config={
                "qa_set_id": qa_set_id,
                "evaluation_config": {
                    "similarity_threshold": self.config.similarity_threshold,
                    "evaluation_timeout": self.config.evaluation_timeout,
                    "include_contradiction_detection": self.config.include_contradiction_detection
                },
                **(evaluation_metadata or {})
            },
            tags=["qa_evaluation", qa_set_id]
        )
        
        self.evaluation_store.create_evaluation_run(run)
        logger.info(f"Created evaluation run '{run_id}' for evaluation '{evaluation_name}'")
        
        return run_id
    
    def _save_evaluation_results(self,
                                run_id: str,
                                qa_pairs: List[QAPair],
                                evaluation_results: Dict[str, Any]) -> None:
        """
        Save evaluation results to the evaluation store
        
        評価結果を評価ストアに保存
        
        Args:
            run_id: Evaluation run ID
                   評価実行ID
            qa_pairs: QA pairs used in evaluation
                     評価に使用したQAペア
            evaluation_results: Results from evaluation
                              評価結果
        """
        try:
            # Save test results
            test_results = []
            for result_dict in evaluation_results.get("test_results", []):
                test_result = TestResult(
                    test_case_id=result_dict["test_case_id"],
                    query=result_dict["query"],
                    generated_answer=result_dict["generated_answer"],
                    expected_answer=result_dict["expected_answer"],
                    sources_found=result_dict["sources_found"],
                    expected_sources=result_dict["expected_sources"],
                    processing_time=result_dict["processing_time"],
                    confidence=result_dict["confidence"],
                    passed=result_dict["passed"],
                    error_message=result_dict.get("error_message"),
                    metadata=result_dict.get("metadata", {})
                )
                test_results.append(test_result)
            
            if test_results:
                self.evaluation_store.save_test_results(run_id, test_results)
                logger.info(f"Saved {len(test_results)} test results for run '{run_id}'")
            
            # Update evaluation run with completion status and metrics
            completion_data = {
                "status": "completed",
                "completed_at": datetime.now(),
                "metrics_summary": evaluation_results.get("evaluation_summary", {})
            }
            
            self.evaluation_store.update_evaluation_run(run_id, completion_data)
            logger.info(f"Updated evaluation run '{run_id}' with completion status")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results for run '{run_id}': {e}")
            # Update run status to failed
            try:
                self.evaluation_store.update_evaluation_run(run_id, {"status": "failed"})
            except:
                pass
            raise
    
    def compute_evaluation_metrics(self,
                                 run_ids: List[str],
                                 metric_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute various evaluation metrics from stored results
        
        保存された結果から各種評価メトリクスを計算
        
        Args:
            run_ids: List of evaluation run IDs to analyze
                    分析する評価実行IDのリスト
            metric_types: List of metric types to compute
                         計算するメトリクスタイプのリスト
                         
        Returns:
            Dict[str, Any]: Computed metrics and analysis
                           計算されたメトリクスと分析
        """
        if not self.evaluation_store:
            raise ValueError("EvaluationStore is required for computing metrics")
        
        if metric_types is None:
            metric_types = ["accuracy", "precision", "recall", "f1", "confidence", "response_time", "comparison"]
        
        logger.info(f"Computing metrics for {len(run_ids)} evaluation runs")
        
        # Collect all test results
        all_results = []
        run_metadata = {}
        
        for run_id in run_ids:
            run = self.evaluation_store.get_evaluation_run(run_id)
            if run:
                run_metadata[run_id] = {
                    "name": run.name,
                    "created_at": run.created_at.isoformat(),
                    "config": run.config,
                    "metrics_summary": run.metrics_summary
                }
                
                test_results = self.evaluation_store.get_test_results(run_id)
                for result in test_results:
                    result_dict = {
                        "run_id": run_id,
                        "test_case_id": result.test_case_id,
                        "query": result.query,
                        "generated_answer": result.generated_answer,
                        "expected_answer": result.expected_answer,
                        "sources_found": result.sources_found,
                        "expected_sources": result.expected_sources,
                        "processing_time": result.processing_time,
                        "confidence": result.confidence,
                        "passed": result.passed,
                        "metadata": result.metadata
                    }
                    all_results.append(result_dict)
        
        if not all_results:
            logger.warning("No test results found for the specified runs")
            return {"error": "No test results found"}
        
        # Compute metrics
        computed_metrics = {}
        
        if "accuracy" in metric_types:
            computed_metrics["accuracy"] = self._compute_accuracy_metrics(all_results)
        
        if "precision" in metric_types:
            computed_metrics["precision"] = self._compute_precision_metrics(all_results)
        
        if "recall" in metric_types:
            computed_metrics["recall"] = self._compute_recall_metrics(all_results)
        
        if "f1" in metric_types:
            computed_metrics["f1"] = self._compute_f1_metrics(all_results)
        
        if "confidence" in metric_types:
            computed_metrics["confidence"] = self._compute_confidence_metrics(all_results)
        
        if "response_time" in metric_types:
            computed_metrics["response_time"] = self._compute_response_time_metrics(all_results)
        
        if "comparison" in metric_types:
            computed_metrics["comparison"] = self._compute_comparison_metrics(all_results, run_metadata)
        
        # Overall summary
        computed_metrics["summary"] = {
            "total_runs": len(run_ids),
            "total_test_cases": len(all_results),
            "runs_analyzed": list(run_metadata.keys()),
            "metric_types_computed": metric_types,
            "computation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Computed {len(metric_types)} metric types for {len(all_results)} test results")
        return computed_metrics
    
    def _compute_accuracy_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute accuracy-related metrics"""
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        
        # Per-run accuracy
        run_accuracy = {}
        for result in results:
            run_id = result["run_id"]
            if run_id not in run_accuracy:
                run_accuracy[run_id] = {"total": 0, "passed": 0}
            run_accuracy[run_id]["total"] += 1
            if result["passed"]:
                run_accuracy[run_id]["passed"] += 1
        
        # Calculate accuracy per run
        for run_id in run_accuracy:
            data = run_accuracy[run_id]
            data["accuracy"] = data["passed"] / data["total"] if data["total"] > 0 else 0.0
        
        return {
            "overall_accuracy": passed / total if total > 0 else 0.0,
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "per_run_accuracy": run_accuracy
        }
    
    def _compute_precision_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute precision-related metrics based on source matching"""
        precision_scores = []
        
        for result in results:
            expected_sources = set(result["expected_sources"])
            found_sources = set(result["sources_found"])
            
            if found_sources:
                precision = len(expected_sources & found_sources) / len(found_sources)
            else:
                precision = 0.0
            
            precision_scores.append(precision)
        
        import statistics
        
        return {
            "average_precision": statistics.mean(precision_scores) if precision_scores else 0.0,
            "median_precision": statistics.median(precision_scores) if precision_scores else 0.0,
            "min_precision": min(precision_scores) if precision_scores else 0.0,
            "max_precision": max(precision_scores) if precision_scores else 0.0,
            "std_precision": statistics.stdev(precision_scores) if len(precision_scores) > 1 else 0.0
        }
    
    def _compute_recall_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute recall-related metrics based on source matching"""
        recall_scores = []
        
        for result in results:
            expected_sources = set(result["expected_sources"])
            found_sources = set(result["sources_found"])
            
            if expected_sources:
                recall = len(expected_sources & found_sources) / len(expected_sources)
            else:
                recall = 0.0
            
            recall_scores.append(recall)
        
        import statistics
        
        return {
            "average_recall": statistics.mean(recall_scores) if recall_scores else 0.0,
            "median_recall": statistics.median(recall_scores) if recall_scores else 0.0,
            "min_recall": min(recall_scores) if recall_scores else 0.0,
            "max_recall": max(recall_scores) if recall_scores else 0.0,
            "std_recall": statistics.stdev(recall_scores) if len(recall_scores) > 1 else 0.0
        }
    
    def _compute_f1_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute F1-related metrics"""
        f1_scores = []
        
        for result in results:
            expected_sources = set(result["expected_sources"])
            found_sources = set(result["sources_found"])
            
            if found_sources:
                precision = len(expected_sources & found_sources) / len(found_sources)
            else:
                precision = 0.0
            
            if expected_sources:
                recall = len(expected_sources & found_sources) / len(expected_sources)
            else:
                recall = 0.0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            f1_scores.append(f1)
        
        import statistics
        
        return {
            "average_f1": statistics.mean(f1_scores) if f1_scores else 0.0,
            "median_f1": statistics.median(f1_scores) if f1_scores else 0.0,
            "min_f1": min(f1_scores) if f1_scores else 0.0,
            "max_f1": max(f1_scores) if f1_scores else 0.0,
            "std_f1": statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0
        }
    
    def _compute_confidence_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute confidence-related metrics"""
        confidence_scores = [r["confidence"] for r in results]
        
        import statistics
        
        return {
            "average_confidence": statistics.mean(confidence_scores) if confidence_scores else 0.0,
            "median_confidence": statistics.median(confidence_scores) if confidence_scores else 0.0,
            "min_confidence": min(confidence_scores) if confidence_scores else 0.0,
            "max_confidence": max(confidence_scores) if confidence_scores else 0.0,
            "std_confidence": statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0,
            "high_confidence_rate": sum(1 for c in confidence_scores if c >= 0.8) / len(confidence_scores) if confidence_scores else 0.0,
            "low_confidence_rate": sum(1 for c in confidence_scores if c <= 0.3) / len(confidence_scores) if confidence_scores else 0.0
        }
    
    def _compute_response_time_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute response time metrics"""
        response_times = [r["processing_time"] for r in results]
        
        import statistics
        
        return {
            "average_response_time": statistics.mean(response_times) if response_times else 0.0,
            "median_response_time": statistics.median(response_times) if response_times else 0.0,
            "min_response_time": min(response_times) if response_times else 0.0,
            "max_response_time": max(response_times) if response_times else 0.0,
            "std_response_time": statistics.stdev(response_times) if len(response_times) > 1 else 0.0,
            "fast_response_rate": sum(1 for t in response_times if t <= 1.0) / len(response_times) if response_times else 0.0,
            "slow_response_rate": sum(1 for t in response_times if t >= 5.0) / len(response_times) if response_times else 0.0
        }
    
    def _compute_comparison_metrics(self, results: List[Dict[str, Any]], run_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comparison metrics across runs"""
        run_performance = {}
        
        # Group results by run
        for result in results:
            run_id = result["run_id"]
            if run_id not in run_performance:
                run_performance[run_id] = {
                    "total": 0,
                    "passed": 0,
                    "confidence_sum": 0.0,
                    "response_time_sum": 0.0
                }
            
            perf = run_performance[run_id]
            perf["total"] += 1
            if result["passed"]:
                perf["passed"] += 1
            perf["confidence_sum"] += result["confidence"]
            perf["response_time_sum"] += result["processing_time"]
        
        # Calculate run-level metrics
        run_summary = {}
        for run_id, perf in run_performance.items():
            run_summary[run_id] = {
                "name": run_metadata.get(run_id, {}).get("name", "Unknown"),
                "accuracy": perf["passed"] / perf["total"] if perf["total"] > 0 else 0.0,
                "average_confidence": perf["confidence_sum"] / perf["total"] if perf["total"] > 0 else 0.0,
                "average_response_time": perf["response_time_sum"] / perf["total"] if perf["total"] > 0 else 0.0,
                "total_tests": perf["total"]
            }
        
        # Find best/worst performing runs
        if run_summary:
            best_accuracy_run = max(run_summary.items(), key=lambda x: x[1]["accuracy"])
            worst_accuracy_run = min(run_summary.items(), key=lambda x: x[1]["accuracy"])
            fastest_run = min(run_summary.items(), key=lambda x: x[1]["average_response_time"])
            slowest_run = max(run_summary.items(), key=lambda x: x[1]["average_response_time"])
        else:
            best_accuracy_run = worst_accuracy_run = fastest_run = slowest_run = None
        
        return {
            "run_summary": run_summary,
            "best_accuracy_run": {
                "run_id": best_accuracy_run[0],
                "accuracy": best_accuracy_run[1]["accuracy"]
            } if best_accuracy_run else None,
            "worst_accuracy_run": {
                "run_id": worst_accuracy_run[0],
                "accuracy": worst_accuracy_run[1]["accuracy"]
            } if worst_accuracy_run else None,
            "fastest_run": {
                "run_id": fastest_run[0],
                "average_response_time": fastest_run[1]["average_response_time"]
            } if fastest_run else None,
            "slowest_run": {
                "run_id": slowest_run[0],
                "average_response_time": slowest_run[1]["average_response_time"]
            } if slowest_run else None
        }
    
    def get_evaluation_history(self,
                             limit: int = 50,
                             status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get evaluation history from the store
        
        ストアから評価履歴を取得
        
        Args:
            limit: Maximum number of runs to return
                  返す実行の最大数
            status: Filter by status (completed, failed, running)
                   ステータスでフィルタ (completed, failed, running)
                   
        Returns:
            List[Dict[str, Any]]: List of evaluation runs
                                 評価実行のリスト
        """
        if not self.evaluation_store:
            logger.warning("No evaluation store configured")
            return []
        
        runs = self.evaluation_store.list_evaluation_runs(status=status, limit=limit)
        
        history = []
        for run in runs:
            history.append({
                "run_id": run.id,
                "name": run.name,
                "description": run.description,
                "created_at": run.created_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "status": run.status,
                "metrics_summary": run.metrics_summary,
                "tags": run.tags
            })
        
        return history