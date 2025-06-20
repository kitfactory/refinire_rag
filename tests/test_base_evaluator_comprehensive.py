"""
Comprehensive test suite for BaseEvaluator module
BaseEvaluatorモジュールの包括的テストスイート

Coverage targets:
- EvaluationScore dataclass and __post_init__ method
- BaseEvaluatorConfig Pydantic model validation
- BaseEvaluator abstract base class with all utility methods
- ReferenceBasedEvaluator batch evaluation implementation
- ReferenceFreeEvaluator reference-free evaluation
- CompositeEvaluator multiple evaluator combination
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Optional, Union
from abc import ABC

from refinire_rag.evaluation.base_evaluator import (
    EvaluationScore,
    BaseEvaluatorConfig,
    BaseEvaluator,
    ReferenceBasedEvaluator,
    ReferenceFreeEvaluator,
    CompositeEvaluator
)


class TestEvaluationScore:
    """Test EvaluationScore dataclass functionality"""
    
    def test_evaluation_score_basic_initialization(self):
        """Test EvaluationScore basic initialization"""
        score = EvaluationScore(
            metric_name="test_metric",
            score=0.85
        )
        
        assert score.metric_name == "test_metric"
        assert score.score == 0.85
        assert score.details == {}  # Should be initialized by __post_init__
        assert score.confidence is None
    
    def test_evaluation_score_full_initialization(self):
        """Test EvaluationScore with all fields"""
        details = {"precision": 0.9, "recall": 0.8}
        score = EvaluationScore(
            metric_name="f1_score",
            score=0.84,
            details=details,
            confidence=0.95
        )
        
        assert score.metric_name == "f1_score"
        assert score.score == 0.84
        assert score.details == details
        assert score.confidence == 0.95
    
    def test_evaluation_score_post_init_details_none(self):
        """Test __post_init__ when details is None"""
        score = EvaluationScore(
            metric_name="test",
            score=0.5,
            details=None
        )
        
        # __post_init__ should initialize details to empty dict
        assert score.details == {}
    
    def test_evaluation_score_post_init_details_provided(self):
        """Test __post_init__ when details is provided"""
        original_details = {"key": "value"}
        score = EvaluationScore(
            metric_name="test",
            score=0.5,
            details=original_details
        )
        
        # __post_init__ should not override provided details
        assert score.details == original_details


class TestBaseEvaluatorConfig:
    """Test BaseEvaluatorConfig Pydantic model"""
    
    def test_base_evaluator_config_required_fields(self):
        """Test BaseEvaluatorConfig with required name field"""
        config = BaseEvaluatorConfig(name="test_evaluator")
        
        assert config.name == "test_evaluator"
        assert config.enabled is True  # Default value
        assert config.weight == 1.0  # Default value
        assert config.threshold is None  # Default value
    
    def test_base_evaluator_config_all_fields(self):
        """Test BaseEvaluatorConfig with all fields"""
        config = BaseEvaluatorConfig(
            name="custom_evaluator",
            enabled=False,
            weight=2.5,
            threshold=0.8
        )
        
        assert config.name == "custom_evaluator"
        assert config.enabled is False
        assert config.weight == 2.5
        assert config.threshold == 0.8
    
    def test_base_evaluator_config_validation_name_required(self):
        """Test that name field is required"""
        with pytest.raises(ValueError):
            BaseEvaluatorConfig()  # Missing required name field
    
    def test_base_evaluator_config_field_types(self):
        """Test field type validation"""
        config = BaseEvaluatorConfig(
            name="test",
            enabled=True,
            weight=1.5,
            threshold=0.75
        )
        
        assert isinstance(config.name, str)
        assert isinstance(config.enabled, bool)
        assert isinstance(config.weight, float)
        assert isinstance(config.threshold, float)


# Concrete test implementations for abstract classes
class ConcreteEvaluator(BaseEvaluator):
    """Concrete implementation for testing BaseEvaluator"""
    
    def evaluate(self, reference, candidate, context=None):
        return EvaluationScore(
            metric_name=self.name,
            score=0.8,
            details={"test": "data"}
        )
    
    def batch_evaluate(self, references, candidates, contexts=None):
        return [self.evaluate(ref, cand, ctx) 
                for ref, cand, ctx in zip(references, candidates, contexts or [None] * len(candidates))]


class ConcreteReferenceBasedEvaluator(ReferenceBasedEvaluator):
    """Concrete implementation for testing ReferenceBasedEvaluator"""
    
    def evaluate(self, reference, candidate, context=None):
        # Simple mock evaluation based on text similarity
        if isinstance(reference, list):
            ref_text = reference[0] if reference else ""
        else:
            ref_text = reference
        
        similarity = 1.0 if ref_text.lower() == candidate.lower() else 0.5
        return EvaluationScore(
            metric_name=self.name,
            score=similarity,
            details={"reference": ref_text, "candidate": candidate}
        )


class ConcreteReferenceFreeEvaluator(ReferenceFreeEvaluator):
    """Concrete implementation for testing ReferenceFreeEvaluator"""
    
    def evaluate_without_reference(self, candidate, context=None):
        # Simple mock evaluation based on text length
        score = min(len(candidate) / 100.0, 1.0)  # Normalize by length
        return EvaluationScore(
            metric_name=self.name,
            score=score,
            details={"length": len(candidate)}
        )


class TestBaseEvaluator:
    """Test BaseEvaluator abstract base class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = BaseEvaluatorConfig(
            name="test_evaluator",
            enabled=True,
            weight=1.0,
            threshold=0.7
        )
        self.evaluator = ConcreteEvaluator(self.config)
    
    def test_base_evaluator_initialization(self):
        """Test BaseEvaluator initialization"""
        assert self.evaluator.config == self.config
        assert self.evaluator.name == "test_evaluator"
        assert self.evaluator.enabled is True
        assert self.evaluator.weight == 1.0
        assert self.evaluator.threshold == 0.7
    
    def test_base_evaluator_abstract_methods(self):
        """Test that BaseEvaluator is abstract"""
        # BaseEvaluator should not be instantiable directly
        with pytest.raises(TypeError):
            BaseEvaluator(self.config)
    
    def test_preprocess_text_default(self):
        """Test default text preprocessing"""
        text = "  Hello World  "
        processed = self.evaluator.preprocess_text(text)
        assert processed == "hello world"
    
    def test_preprocess_text_edge_cases(self):
        """Test text preprocessing edge cases"""
        # Empty string
        assert self.evaluator.preprocess_text("") == ""
        
        # Only whitespace
        assert self.evaluator.preprocess_text("   ") == ""
        
        # Mixed case with special characters
        text = "  Hello, World! 123  "
        processed = self.evaluator.preprocess_text(text)
        assert processed == "hello, world! 123"
    
    def test_validate_inputs_valid_string_reference(self):
        """Test input validation with valid string reference"""
        # Should not raise any exception
        self.evaluator.validate_inputs("reference answer", "candidate answer")
    
    def test_validate_inputs_valid_list_reference(self):
        """Test input validation with valid list reference"""
        # Should not raise any exception
        self.evaluator.validate_inputs(["ref1", "ref2"], "candidate answer")
    
    def test_validate_inputs_empty_candidate(self):
        """Test input validation with empty candidate"""
        with pytest.raises(ValueError, match="Candidate answer cannot be empty"):
            self.evaluator.validate_inputs("reference", "")
        
        with pytest.raises(ValueError, match="Candidate answer cannot be empty"):
            self.evaluator.validate_inputs("reference", "   ")
    
    def test_validate_inputs_empty_string_reference(self):
        """Test input validation with empty string reference"""
        with pytest.raises(ValueError, match="Reference answer cannot be empty"):
            self.evaluator.validate_inputs("", "candidate")
        
        with pytest.raises(ValueError, match="Reference answer cannot be empty"):
            self.evaluator.validate_inputs("   ", "candidate")
    
    def test_validate_inputs_empty_list_reference(self):
        """Test input validation with empty list reference"""
        with pytest.raises(ValueError, match="At least one reference answer must be non-empty"):
            self.evaluator.validate_inputs([], "candidate")
        
        with pytest.raises(ValueError, match="At least one reference answer must be non-empty"):
            self.evaluator.validate_inputs(["", "   "], "candidate")
    
    def test_validate_inputs_invalid_reference_type(self):
        """Test input validation with invalid reference type"""
        with pytest.raises(ValueError, match="Reference must be string or list of strings"):
            self.evaluator.validate_inputs(123, "candidate")
        
        with pytest.raises(ValueError, match="Reference must be string or list of strings"):
            self.evaluator.validate_inputs(None, "candidate")
    
    def test_apply_threshold_no_threshold(self):
        """Test threshold application when no threshold is set"""
        evaluator = ConcreteEvaluator(BaseEvaluatorConfig(name="test"))
        assert evaluator.apply_threshold(0.5) is True
        assert evaluator.apply_threshold(0.0) is True
        assert evaluator.apply_threshold(1.0) is True
    
    def test_apply_threshold_with_threshold(self):
        """Test threshold application with threshold set"""
        assert self.evaluator.apply_threshold(0.8) is True  # Above threshold (0.7)
        assert self.evaluator.apply_threshold(0.7) is True  # Equal to threshold
        assert self.evaluator.apply_threshold(0.6) is False  # Below threshold
    
    def test_get_metric_info(self):
        """Test metric information retrieval"""
        info = self.evaluator.get_metric_info()
        
        expected_info = {
            "name": "test_evaluator",
            "enabled": True,
            "weight": 1.0,
            "threshold": 0.7,
            "description": "Concrete implementation for testing BaseEvaluator"
        }
        
        assert info == expected_info


class TestReferenceBasedEvaluator:
    """Test ReferenceBasedEvaluator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = BaseEvaluatorConfig(name="reference_evaluator")
        self.evaluator = ConcreteReferenceBasedEvaluator(self.config)
    
    def test_batch_evaluate_basic(self):
        """Test basic batch evaluation"""
        references = ["ref1", "ref2", "ref3"]
        candidates = ["ref1", "cand2", "ref3"]
        
        results = self.evaluator.batch_evaluate(references, candidates)
        
        assert len(results) == 3
        assert all(isinstance(result, EvaluationScore) for result in results)
        assert results[0].score == 1.0  # Exact match
        assert results[1].score == 0.5  # No match
        assert results[2].score == 1.0  # Exact match
    
    def test_batch_evaluate_with_contexts(self):
        """Test batch evaluation with contexts"""
        references = ["ref1", "ref2"]
        candidates = ["cand1", "cand2"]
        contexts = [{"key1": "value1"}, {"key2": "value2"}]
        
        results = self.evaluator.batch_evaluate(references, candidates, contexts)
        
        assert len(results) == 2
        assert all(isinstance(result, EvaluationScore) for result in results)
    
    def test_batch_evaluate_mismatched_lengths(self):
        """Test batch evaluation with mismatched input lengths"""
        references = ["ref1", "ref2"]
        candidates = ["cand1"]  # Different length
        
        with pytest.raises(ValueError, match="Length of references, candidates, and contexts must match"):
            self.evaluator.batch_evaluate(references, candidates)
    
    def test_batch_evaluate_list_references(self):
        """Test batch evaluation with list references"""
        references = [["ref1a", "ref1b"], ["ref2a", "ref2b"]]
        candidates = ["ref1a", "other"]
        
        results = self.evaluator.batch_evaluate(references, candidates)
        
        assert len(results) == 2
        assert results[0].score == 1.0  # Matches first reference
        assert results[1].score == 0.5  # No match


class TestReferenceFreeEvaluator:
    """Test ReferenceFreeEvaluator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = BaseEvaluatorConfig(name="reference_free_evaluator")
        self.evaluator = ConcreteReferenceFreeEvaluator(self.config)
    
    def test_evaluate_ignores_reference(self):
        """Test that evaluate method ignores reference parameter"""
        result1 = self.evaluator.evaluate("any reference", "test candidate")
        result2 = self.evaluator.evaluate_without_reference("test candidate")
        
        # Both should give same result since reference is ignored
        assert result1.score == result2.score
        assert result1.details == result2.details
    
    def test_evaluate_without_reference_basic(self):
        """Test basic reference-free evaluation"""
        candidate = "This is a test candidate answer"
        result = self.evaluator.evaluate_without_reference(candidate)
        
        assert isinstance(result, EvaluationScore)
        assert result.metric_name == "reference_free_evaluator"
        assert 0.0 <= result.score <= 1.0
        assert "length" in result.details
        assert result.details["length"] == len(candidate)
    
    def test_batch_evaluate_ignores_references(self):
        """Test that batch evaluation ignores references"""
        references = ["ref1", "ref2", "ref3"]
        candidates = ["short", "medium length text", "very long candidate answer text"]
        
        results = self.evaluator.batch_evaluate(references, candidates)
        
        assert len(results) == 3
        # Scores should increase with length (up to limit)
        assert results[0].score < results[1].score
        assert results[1].score < results[2].score
    
    def test_abstract_method_enforcement(self):
        """Test that evaluate_without_reference is abstract"""
        class IncompleteReferenceFreeEvaluator(ReferenceFreeEvaluator):
            pass
        
        # Should not be instantiable without implementing abstract method
        with pytest.raises(TypeError):
            IncompleteReferenceFreeEvaluator(self.config)


class TestCompositeEvaluator:
    """Test CompositeEvaluator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create mock evaluators
        self.eval1 = ConcreteEvaluator(BaseEvaluatorConfig(name="eval1", weight=2.0))
        self.eval2 = ConcreteEvaluator(BaseEvaluatorConfig(name="eval2", weight=1.0))
        self.eval3 = ConcreteEvaluator(BaseEvaluatorConfig(name="eval3", weight=1.0, enabled=False))
        
        # Mock their evaluate methods
        self.eval1.evaluate = Mock(return_value=EvaluationScore("eval1", 0.8))
        self.eval2.evaluate = Mock(return_value=EvaluationScore("eval2", 0.6))
        self.eval3.evaluate = Mock(return_value=EvaluationScore("eval3", 0.9))
        
        self.config = BaseEvaluatorConfig(name="composite")
        self.evaluators = [self.eval1, self.eval2, self.eval3]
        self.composite = CompositeEvaluator(self.config, self.evaluators)
    
    def test_composite_evaluator_initialization(self):
        """Test CompositeEvaluator initialization"""
        # Should only include enabled evaluators
        assert len(self.composite.evaluators) == 2  # eval3 is disabled
        assert self.eval1 in self.composite.evaluators
        assert self.eval2 in self.composite.evaluators
        assert self.eval3 not in self.composite.evaluators
    
    def test_composite_evaluator_weight_normalization(self):
        """Test that weights are normalized"""
        # Original weights: eval1=2.0, eval2=1.0, total=3.0
        # Normalized: eval1=2/3≈0.667, eval2=1/3≈0.333
        assert abs(self.eval1.weight - 2.0/3.0) < 0.001
        assert abs(self.eval2.weight - 1.0/3.0) < 0.001
    
    def test_composite_evaluate_weighted_average(self):
        """Test composite evaluation with weighted average"""
        result = self.composite.evaluate("reference", "candidate")
        
        # Expected weighted average: (0.8 * 2/3) + (0.6 * 1/3) = 0.533 + 0.2 = 0.733
        expected_score = (0.8 * 2.0/3.0) + (0.6 * 1.0/3.0)
        assert abs(result.score - expected_score) < 0.001
        assert result.metric_name == "composite"
        
        # Check that both evaluators were called
        self.eval1.evaluate.assert_called_once_with("reference", "candidate", None)
        self.eval2.evaluate.assert_called_once_with("reference", "candidate", None)
        self.eval3.evaluate.assert_not_called()  # Disabled evaluator
    
    def test_composite_evaluate_no_evaluators(self):
        """Test composite evaluation with no enabled evaluators"""
        # Create composite with all disabled evaluators
        disabled_evaluators = [
            ConcreteEvaluator(BaseEvaluatorConfig(name="disabled1", enabled=False)),
            ConcreteEvaluator(BaseEvaluatorConfig(name="disabled2", enabled=False))
        ]
        composite = CompositeEvaluator(self.config, disabled_evaluators)
        
        result = composite.evaluate("reference", "candidate")
        
        assert result.score == 0.0
        assert "error" in result.details
        assert "No evaluators enabled" in result.details["error"]
    
    def test_composite_evaluate_with_exception(self):
        """Test composite evaluation when sub-evaluator raises exception"""
        # Make eval2 raise an exception
        self.eval2.evaluate.side_effect = ValueError("Test error")
        
        result = self.composite.evaluate("reference", "candidate")
        
        # Should still compute score from eval1 only
        expected_score = 0.8 * 2.0/3.0  # Only eval1 contributes
        assert abs(result.score - expected_score) < 0.001
        
        # Check error is recorded in details
        assert "eval2" in result.details
        assert "error" in result.details["eval2"]
        assert "Test error" in result.details["eval2"]["error"]
    
    def test_composite_batch_evaluate(self):
        """Test composite batch evaluation"""
        references = ["ref1", "ref2"]
        candidates = ["cand1", "cand2"]
        
        results = self.composite.batch_evaluate(references, candidates)
        
        assert len(results) == 2
        assert all(isinstance(result, EvaluationScore) for result in results)
        
        # Each sub-evaluator should be called twice
        assert self.eval1.evaluate.call_count == 2
        assert self.eval2.evaluate.call_count == 2


class TestEdgeCasesAndIntegration:
    """Test edge cases and integration scenarios"""
    
    def test_evaluation_score_with_negative_score(self):
        """Test EvaluationScore with negative score"""
        score = EvaluationScore("test", -0.5)
        assert score.score == -0.5  # Should allow negative scores
    
    def test_evaluation_score_with_large_score(self):
        """Test EvaluationScore with score > 1.0"""
        score = EvaluationScore("test", 2.5)
        assert score.score == 2.5  # Should allow scores > 1.0
    
    def test_base_evaluator_config_zero_weight(self):
        """Test BaseEvaluatorConfig with zero weight"""
        config = BaseEvaluatorConfig(name="test", weight=0.0)
        assert config.weight == 0.0
    
    def test_base_evaluator_config_negative_threshold(self):
        """Test BaseEvaluatorConfig with negative threshold"""
        config = BaseEvaluatorConfig(name="test", threshold=-0.5)
        assert config.threshold == -0.5
    
    def test_composite_evaluator_all_zero_weights(self):
        """Test CompositeEvaluator with all zero weights"""
        evaluators = [
            ConcreteEvaluator(BaseEvaluatorConfig(name="eval1", weight=0.0)),
            ConcreteEvaluator(BaseEvaluatorConfig(name="eval2", weight=0.0))
        ]
        
        config = BaseEvaluatorConfig(name="composite_zero")
        composite = CompositeEvaluator(config, evaluators)
        
        # Should handle division by zero gracefully
        # Weights should remain 0.0 after normalization attempt
        assert all(e.weight == 0.0 for e in composite.evaluators)
    
    def test_unicode_text_handling(self):
        """Test handling of Unicode text"""
        evaluator = ConcreteEvaluator(BaseEvaluatorConfig(name="unicode_test"))
        
        unicode_text = "こんにちは世界 🌍 émojis"
        processed = evaluator.preprocess_text(unicode_text)
        
        # Should handle Unicode characters properly
        assert "こんにちは世界" in processed
        assert "🌍" in processed
        assert "émojis" in processed
    
    def test_very_long_text_handling(self):
        """Test handling of very long text"""
        evaluator = ConcreteEvaluator(BaseEvaluatorConfig(name="long_text_test"))
        
        # Create very long text (10,000 characters)
        long_text = "a" * 10000
        
        # Should not raise any exceptions
        processed = evaluator.preprocess_text(long_text)
        assert len(processed) == 10000
        
        # Validation should work
        evaluator.validate_inputs("short reference", long_text)
    
    def test_mixed_reference_types_in_batch(self):
        """Test batch evaluation with mixed reference types"""
        evaluator = ConcreteReferenceBasedEvaluator(BaseEvaluatorConfig(name="mixed_test"))
        
        references = [
            "single reference",  # String
            ["ref1", "ref2"],    # List
            "another single"     # String
        ]
        candidates = ["cand1", "cand2", "cand3"]
        
        results = evaluator.batch_evaluate(references, candidates)
        
        assert len(results) == 3
        assert all(isinstance(result, EvaluationScore) for result in results)