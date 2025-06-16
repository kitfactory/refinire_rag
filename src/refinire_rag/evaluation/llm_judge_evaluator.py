"""
LLM Judge Evaluator Implementation

LLM Judge評価器の実装
LLM Judgeは大規模言語モデルを審査員として使用し、回答の品質を人間の判断に近い形で評価します。
"""

from typing import List, Dict, Any, Optional, Union
import asyncio
import json
import logging

from .base_evaluator import ReferenceBasedEvaluator, BaseEvaluatorConfig, EvaluationScore
from refinire import create_simple_gen_agent, Context

logger = logging.getLogger(__name__)


class LLMJudgeConfig(BaseEvaluatorConfig):
    """
    Configuration for LLM Judge evaluator
    
    LLM Judge評価器の設定
    """
    name: str = "LLM_Judge"
    model_name: str = "gpt-4o-mini"  # LLM model for judging
    judge_model_name: str = "gpt-4o-mini"  # Alternative model name
    evaluation_criteria: List[str] = None  # Evaluation criteria
    scoring_scale: int = 10  # Scoring scale (1-10, 1-5, etc.)
    include_reasoning: bool = True  # Include reasoning in output
    temperature: float = 0.1  # Low temperature for consistent judgments
    max_tokens: int = 1000  # Maximum tokens for judge response
    enable_relevance: bool = True  # 関連性評価の有効化
    enable_accuracy: bool = True  # 正確性評価の有効化
    enable_completeness: bool = True  # 完全性評価の有効化
    enable_coherence: bool = True  # 一貫性評価の有効化
    enable_helpfulness: bool = True  # 有用性評価の有効化
    
    def __post_init__(self):
        super().__post_init__()
        if self.evaluation_criteria is None:
            self.evaluation_criteria = [
                "relevance",      # 関連性: 質問に対する回答の関連性
                "accuracy",       # 正確性: 事実の正確性
                "completeness",   # 完全性: 回答の完全性
                "coherence",      # 一貫性: 論理的一貫性
                "helpfulness"     # 有用性: ユーザーにとっての有用性
            ]


class LLMJudgeEvaluator(ReferenceBasedEvaluator):
    """
    LLM Judge evaluator using large language models as judges
    
    大規模言語モデルを審査員として使用するLLM Judge評価器
    
    LLM Judgeは以下の特徴を持ちます：
    1. 人間の判断に近い評価が可能
    2. 複数の評価基準を同時に評価
    3. 評価理由の説明を提供
    4. 柔軟な評価スケール
    """
    
    def __init__(self, config: LLMJudgeConfig):
        super().__init__(config)
        self.config: LLMJudgeConfig = config
        
        # Initialize the LLM agent for judging
        self.judge_agent = create_simple_gen_agent(
            name="llm_judge",
            instructions=self._get_judge_instructions(),
            model=config.judge_model_name or config.model_name
        )
    
    def evaluate(self,
                reference: Union[str, List[str]],
                candidate: str,
                context: Optional[Dict[str, Any]] = None) -> EvaluationScore:
        """
        Evaluate candidate answer using LLM as judge
        
        LLMを審査員として使用して候補回答を評価
        
        Args:
            reference: Reference answer(s)
                      参照回答
            candidate: Candidate answer to evaluate
                      評価する候補回答
            context: Context including question and source documents
                    質問とソース文書を含むコンテキスト
                    
        Returns:
            EvaluationScore: LLM Judge evaluation result
                           LLM Judge評価結果
        """
        self.validate_inputs(reference, candidate)
        
        # Preprocess inputs
        candidate = self.preprocess_text(candidate)
        if isinstance(reference, str):
            reference_text = self.preprocess_text(reference)
        else:
            reference_text = " | ".join([self.preprocess_text(ref) for ref in reference])
        
        # Extract context information
        question = context.get("question", "") if context else ""
        source_documents = context.get("source_documents", []) if context else []
        
        # Create evaluation prompt
        eval_prompt = self._create_evaluation_prompt(
            question=question,
            reference=reference_text,
            candidate=candidate,
            source_documents=source_documents
        )
        
        try:
            # Run LLM evaluation
            result = asyncio.run(self._run_llm_evaluation(eval_prompt))
            
            # Parse LLM response
            parsed_result = self._parse_llm_response(result)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(parsed_result["scores"])
            
            # Detailed breakdown
            details = {
                "individual_scores": parsed_result["scores"],
                "reasoning": parsed_result["reasoning"],
                "evaluation_criteria": self.config.evaluation_criteria,
                "scoring_scale": f"1-{self.config.scoring_scale}",
                "llm_model": self.config.judge_model_name or self.config.model_name,
                "raw_response": result,
                "overall_computation": f"Average of {len(parsed_result['scores'])} criteria scores"
            }
            
            return EvaluationScore(
                metric_name=self.config.name,
                score=overall_score,
                details=details,
                confidence=0.8  # LLM judgments have good but not perfect confidence
            )
            
        except Exception as e:
            logger.error(f"LLM Judge evaluation failed: {e}")
            
            # Return fallback score
            return EvaluationScore(
                metric_name=self.config.name,
                score=0.0,
                details={
                    "error": str(e),
                    "evaluation_criteria": self.config.evaluation_criteria,
                    "scoring_scale": f"1-{self.config.scoring_scale}",
                    "status": "evaluation_failed"
                },
                confidence=0.0
            )
    
    async def _run_llm_evaluation(self, eval_prompt: str) -> str:
        """
        Run LLM evaluation asynchronously
        
        LLM評価を非同期で実行
        """
        context = Context()
        result = await self.judge_agent.run(eval_prompt, context)
        return result.shared_state.get("llm_judge_result", "")
    
    def _create_evaluation_prompt(self,
                                question: str,
                                reference: str,
                                candidate: str,
                                source_documents: List[str]) -> str:
        """
        Create evaluation prompt for LLM judge
        
        LLM審査員用の評価プロンプトを作成
        """
        # Prepare source context
        source_context = ""
        if source_documents:
            source_context = f"\n\nSource Documents:\n{chr(10).join(f'{i+1}. {doc[:500]}...' for i, doc in enumerate(source_documents[:3]))}"
        
        # Create comprehensive evaluation prompt
        prompt = f"""You are an expert evaluator for question-answering systems. Your task is to evaluate the quality of an answer based on multiple criteria.

**Question:** {question}

**Reference Answer:** {reference}

**Candidate Answer:** {candidate}
{source_context}

**Evaluation Criteria:**
Please evaluate the candidate answer on a scale of 1-{self.config.scoring_scale} for each criterion:

"""
        
        # Add enabled criteria descriptions
        criteria_descriptions = {
            "relevance": "How well does the answer address the specific question asked?",
            "accuracy": "How factually correct is the information provided?",
            "completeness": "How thoroughly does the answer cover the question?",
            "coherence": "How logically structured and consistent is the answer?",
            "helpfulness": "How useful is this answer for someone seeking information?"
        }
        
        for criterion in self.config.evaluation_criteria:
            if hasattr(self.config, f"enable_{criterion}") and getattr(self.config, f"enable_{criterion}"):
                description = criteria_descriptions.get(criterion, f"Evaluate the {criterion} of the answer")
                prompt += f"- **{criterion.capitalize()}**: {description}\n"
        
        # Add output format instructions
        prompt += f"""
**Output Format:**
Please respond with a JSON object containing:
1. Individual scores for each criterion (1-{self.config.scoring_scale})
2. Brief reasoning for each score
3. Overall assessment

Example format:
```json
{{
    "scores": {{
        "relevance": 8,
        "accuracy": 7,
        "completeness": 6,
        "coherence": 9,
        "helpfulness": 7
    }},
    "reasoning": {{
        "relevance": "The answer directly addresses the question about...",
        "accuracy": "Most facts are correct, but there's a minor error regarding...",
        "completeness": "The answer covers main points but misses...",
        "coherence": "Well-structured with clear logical flow...",
        "helpfulness": "Provides practical information but could be more specific..."
    }},
    "overall_assessment": "The answer is generally good but could be improved by..."
}}
```

Evaluate carefully and provide constructive feedback.
"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM judge response
        
        LLM審査員の応答を解析
        """
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                # Validate required fields
                if "scores" in parsed and isinstance(parsed["scores"], dict):
                    return {
                        "scores": parsed["scores"],
                        "reasoning": parsed.get("reasoning", {}),
                        "overall_assessment": parsed.get("overall_assessment", "")
                    }
            
            # Fallback parsing
            return self._fallback_parse_response(response)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return self._fallback_parse_response(response)
    
    def _fallback_parse_response(self, response: str) -> Dict[str, Any]:
        """
        Fallback parsing when JSON parsing fails
        
        JSON解析が失敗した場合のフォールバック解析
        """
        # Extract scores using pattern matching
        scores = {}
        reasoning = {}
        
        # Simple pattern matching for scores
        import re
        for criterion in self.config.evaluation_criteria:
            # Look for pattern like "relevance: 8" or "Relevance Score: 8"
            pattern = rf"{criterion}[:\s]*(\d+)"
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                # Clamp score to valid range
                scores[criterion] = max(1, min(self.config.scoring_scale, score))
                reasoning[criterion] = f"Extracted score from response text"
            else:
                # Default score if not found
                scores[criterion] = self.config.scoring_scale // 2
                reasoning[criterion] = "Score not found in response, using default"
        
        return {
            "scores": scores,
            "reasoning": reasoning,
            "overall_assessment": "Response parsing required fallback method"
        }
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate overall score from individual criteria scores
        
        個別基準スコアから総合スコアを計算
        """
        if not scores:
            return 0.0
        
        # Calculate average score
        total_score = sum(scores.values())
        average_score = total_score / len(scores)
        
        # Normalize to 0-1 scale
        normalized_score = (average_score - 1) / (self.config.scoring_scale - 1)
        
        return max(0.0, min(1.0, normalized_score))
    
    def _get_judge_instructions(self) -> str:
        """
        Get instructions for the LLM judge agent
        
        LLM審査員エージェントの指示を取得
        """
        return f"""You are an expert evaluator for question-answering systems. Your role is to:

1. Carefully analyze question-answer pairs
2. Evaluate answers based on multiple quality criteria
3. Provide fair, consistent, and constructive assessments
4. Use a {self.config.scoring_scale}-point scale for scoring
5. Always provide reasoning for your evaluations

Be objective, thorough, and helpful in your evaluations. Focus on the quality of the answer in relation to the question asked."""
    
    def evaluate_batch_async(self,
                           evaluation_pairs: List[Dict[str, Any]]) -> List[EvaluationScore]:
        """
        Evaluate multiple QA pairs asynchronously for better performance
        
        複数のQAペアを非同期で評価してパフォーマンスを向上
        """
        return asyncio.run(self._evaluate_batch_async_impl(evaluation_pairs))
    
    async def _evaluate_batch_async_impl(self,
                                       evaluation_pairs: List[Dict[str, Any]]) -> List[EvaluationScore]:
        """
        Implementation of asynchronous batch evaluation
        
        非同期バッチ評価の実装
        """
        tasks = []
        for pair in evaluation_pairs:
            # Create evaluation task
            task = asyncio.create_task(
                self._evaluate_single_async(
                    reference=pair.get("reference", ""),
                    candidate=pair.get("candidate", ""),
                    context=pair.get("context", {})
                )
            )
            tasks.append(task)
        
        # Wait for all evaluations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        evaluation_scores = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Evaluation {i} failed: {result}")
                evaluation_scores.append(EvaluationScore(
                    metric_name=self.config.name,
                    score=0.0,
                    details={"error": str(result), "status": "batch_evaluation_failed"},
                    confidence=0.0
                ))
            else:
                evaluation_scores.append(result)
        
        return evaluation_scores
    
    async def _evaluate_single_async(self,
                                   reference: Union[str, List[str]],
                                   candidate: str,
                                   context: Optional[Dict[str, Any]] = None) -> EvaluationScore:
        """
        Evaluate single QA pair asynchronously
        
        単一のQAペアを非同期で評価
        """
        # This is essentially the same as evaluate() but async
        # For brevity, calling the sync version wrapped in async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.evaluate, 
            reference, 
            candidate, 
            context
        )