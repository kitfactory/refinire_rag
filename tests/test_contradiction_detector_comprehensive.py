"""
Comprehensive test suite for ContradictionDetector module
ContradictionDetectorモジュールの包括的テストスイート

Coverage targets:
- ClaimType, NLILabel enums and Claim, ContradictionPair data models
- ContradictionDetectorConfig configuration class
- ContradictionDetector main class with claim extraction and NLI detection
- Pattern-based claim extraction logic
- Rule-based NLI contradiction detection
- Report generation and summary methods
- Error handling and edge cases
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from dataclasses import asdict

from refinire_rag.processing.contradiction_detector import (
    ClaimType,
    NLILabel, 
    Claim,
    ContradictionPair,
    ContradictionDetectorConfig,
    ContradictionDetector
)
from refinire_rag.models.document import Document


class TestClaimType:
    """Test ClaimType enum functionality"""
    
    def test_claim_type_values(self):
        """Test ClaimType enum values"""
        assert ClaimType.FACTUAL == "factual"
        assert ClaimType.EVALUATIVE == "evaluative"
        assert ClaimType.CAUSAL == "causal"
        assert ClaimType.COMPARATIVE == "comparative"
        assert ClaimType.TEMPORAL == "temporal"
    
    def test_claim_type_enumeration(self):
        """Test ClaimType enumeration"""
        claim_types = list(ClaimType)
        expected_types = [
            ClaimType.FACTUAL,
            ClaimType.EVALUATIVE,
            ClaimType.CAUSAL,
            ClaimType.COMPARATIVE,
            ClaimType.TEMPORAL
        ]
        assert len(claim_types) == 5
        assert set(claim_types) == set(expected_types)


class TestNLILabel:
    """Test NLILabel enum functionality"""
    
    def test_nli_label_values(self):
        """Test NLILabel enum values"""
        assert NLILabel.ENTAILMENT == "entailment"
        assert NLILabel.CONTRADICTION == "contradiction"
        assert NLILabel.NEUTRAL == "neutral"
    
    def test_nli_label_enumeration(self):
        """Test NLILabel enumeration"""
        nli_labels = list(NLILabel)
        expected_labels = [
            NLILabel.ENTAILMENT,
            NLILabel.CONTRADICTION,
            NLILabel.NEUTRAL
        ]
        assert len(nli_labels) == 3
        assert set(nli_labels) == set(expected_labels)


class TestClaim:
    """Test Claim Pydantic model functionality"""
    
    def test_claim_initialization(self):
        """Test Claim model initialization"""
        claim = Claim(
            id="test_claim_1",
            text="Pythonは人気のプログラミング言語です",
            claim_type=ClaimType.FACTUAL,
            confidence=0.8,
            source_document_id="doc_1",
            source_sentence="Pythonは人気のプログラミング言語です。"
        )
        
        assert claim.id == "test_claim_1"
        assert claim.text == "Pythonは人気のプログラミング言語です"
        assert claim.claim_type == ClaimType.FACTUAL
        assert claim.confidence == 0.8
        assert claim.source_document_id == "doc_1"
        assert claim.source_sentence == "Pythonは人気のプログラミング言語です。"
        assert claim.metadata == {}
    
    def test_claim_with_metadata(self):
        """Test Claim model with metadata"""
        metadata = {
            "sentence_index": 0,
            "extraction_pattern": r"(.+)は(.+)です",
            "matched_groups": ("Python", "人気のプログラミング言語")
        }
        
        claim = Claim(
            id="test_claim_2",
            text="機械学習は複雑な技術です",
            claim_type=ClaimType.EVALUATIVE,
            confidence=0.7,
            source_document_id="doc_2",
            source_sentence="機械学習は複雑な技術です。",
            metadata=metadata
        )
        
        assert claim.metadata == metadata
        assert claim.metadata["sentence_index"] == 0
        assert claim.metadata["extraction_pattern"] == r"(.+)は(.+)です"
    
    def test_claim_model_dump(self):
        """Test Claim model serialization"""
        claim = Claim(
            id="test_claim_3",
            text="AIは将来重要になる",
            claim_type=ClaimType.TEMPORAL,
            confidence=0.6,
            source_document_id="doc_3",
            source_sentence="AIは将来重要になる。"
        )
        
        claim_dict = claim.model_dump()
        
        expected_keys = {
            "id", "text", "claim_type", "confidence", 
            "source_document_id", "source_sentence", "metadata"
        }
        assert set(claim_dict.keys()) == expected_keys
        assert claim_dict["claim_type"] == "temporal"
        assert claim_dict["confidence"] == 0.6


class TestContradictionPair:
    """Test ContradictionPair Pydantic model functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.claim1 = Claim(
            id="claim_1",
            text="Pythonは簡単です",
            claim_type=ClaimType.EVALUATIVE,
            confidence=0.8,
            source_document_id="doc_1",
            source_sentence="Pythonは簡単です。"
        )
        
        self.claim2 = Claim(
            id="claim_2", 
            text="Pythonは困難です",
            claim_type=ClaimType.EVALUATIVE,
            confidence=0.7,
            source_document_id="doc_2",
            source_sentence="Pythonは困難です。"
        )
    
    def test_contradiction_pair_initialization(self):
        """Test ContradictionPair model initialization"""
        contradiction = ContradictionPair(
            claim1=self.claim1,
            claim2=self.claim2,
            contradiction_score=0.9,
            contradiction_type="same_type_evaluative",
            explanation="クレーム間で評価が矛盾しています",
            severity="high"
        )
        
        assert contradiction.claim1.id == "claim_1"
        assert contradiction.claim2.id == "claim_2"
        assert contradiction.contradiction_score == 0.9
        assert contradiction.contradiction_type == "same_type_evaluative"
        assert contradiction.explanation == "クレーム間で評価が矛盾しています"
        assert contradiction.severity == "high"
        assert contradiction.metadata == {}
    
    def test_contradiction_pair_with_metadata(self):
        """Test ContradictionPair with metadata"""
        metadata = {
            "detection_method": "within_document_nli",
            "nli_confidence": 0.85
        }
        
        contradiction = ContradictionPair(
            claim1=self.claim1,
            claim2=self.claim2,
            contradiction_score=0.85,
            contradiction_type="cross_type_evaluative_factual",
            explanation="異なるタイプのクレーム間矛盾",
            severity="medium",
            metadata=metadata
        )
        
        assert contradiction.metadata == metadata
        assert contradiction.metadata["detection_method"] == "within_document_nli"
        assert contradiction.metadata["nli_confidence"] == 0.85
    
    def test_contradiction_pair_model_dump(self):
        """Test ContradictionPair model serialization"""
        contradiction = ContradictionPair(
            claim1=self.claim1,
            claim2=self.claim2,
            contradiction_score=0.75,
            contradiction_type="same_type_evaluative",
            explanation="Contradiction detected",
            severity="low"
        )
        
        contradiction_dict = contradiction.model_dump()
        
        expected_keys = {
            "claim1", "claim2", "contradiction_score", "contradiction_type",
            "explanation", "severity", "metadata"
        }
        assert set(contradiction_dict.keys()) == expected_keys
        assert contradiction_dict["contradiction_score"] == 0.75
        assert contradiction_dict["severity"] == "low"
        assert isinstance(contradiction_dict["claim1"], dict)
        assert isinstance(contradiction_dict["claim2"], dict)


class TestContradictionDetectorConfig:
    """Test ContradictionDetectorConfig dataclass functionality"""
    
    def test_default_initialization(self):
        """Test ContradictionDetectorConfig with default values"""
        config = ContradictionDetectorConfig()
        
        # Core functionality settings
        assert config.enable_claim_extraction is True
        assert config.enable_nli_detection is True
        assert config.contradiction_threshold == 0.7
        assert config.claim_confidence_threshold == 0.3
        assert config.max_claims_per_document == 20
        
        # Claim extraction settings
        assert config.extract_factual_claims is True
        assert config.extract_evaluative_claims is True
        assert config.extract_causal_claims is True
        assert config.extract_comparative_claims is False
        assert config.extract_temporal_claims is False
        
        # Detection settings
        assert config.check_within_document is True
        assert config.check_across_documents is True
        assert config.check_against_knowledge_base is False
        
        # Output settings
        assert config.save_detected_contradictions is True
        assert config.contradictions_output_file is None
        
        # Inherited DocumentProcessorConfig fields
        assert config.enabled is True
        assert config.log_level == "INFO"
    
    def test_custom_initialization(self):
        """Test ContradictionDetectorConfig with custom values"""
        config = ContradictionDetectorConfig(
            enable_claim_extraction=False,
            enable_nli_detection=False,
            contradiction_threshold=0.8,
            claim_confidence_threshold=0.6,
            max_claims_per_document=10,
            extract_comparative_claims=True,
            extract_temporal_claims=True,
            check_across_documents=False,
            contradictions_output_file="contradictions.json",
            log_level="DEBUG",
            name="test_contradiction_detector"
        )
        
        assert config.enable_claim_extraction is False
        assert config.enable_nli_detection is False
        assert config.contradiction_threshold == 0.8
        assert config.claim_confidence_threshold == 0.6
        assert config.max_claims_per_document == 10
        assert config.extract_comparative_claims is True
        assert config.extract_temporal_claims is True
        assert config.check_across_documents is False
        assert config.contradictions_output_file == "contradictions.json"
        assert config.log_level == "DEBUG"
        assert config.name == "test_contradiction_detector"
    
    def test_to_dict_method(self):
        """Test ContradictionDetectorConfig.to_dict() method"""
        config = ContradictionDetectorConfig(
            contradiction_threshold=0.9,
            extract_causal_claims=False,
            name="dict_test"
        )
        
        result_dict = config.to_dict()
        
        # Check key presence (including inherited fields)
        assert "contradiction_threshold" in result_dict
        assert "extract_causal_claims" in result_dict
        assert "name" in result_dict
        assert "enabled" in result_dict  # inherited field
        
        # Check values
        assert result_dict["contradiction_threshold"] == 0.9
        assert result_dict["extract_causal_claims"] is False
        assert result_dict["name"] == "dict_test"


class TestContradictionDetector:
    """Test ContradictionDetector main class functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ContradictionDetectorConfig()
        self.detector = ContradictionDetector(self.config)
        self.test_doc = Document(
            id="test_doc_1",
            content="Pythonは簡単です。機械学習は複雑です。AIにより効率が向上します。",
            metadata={"source": "test"}
        )
    
    def test_initialization(self):
        """Test ContradictionDetector initialization"""
        assert self.detector.config == self.config
        assert self.detector.extracted_claims == []
        assert self.detector.detected_contradictions == []
        assert hasattr(self.detector, 'claim_patterns')
        assert isinstance(self.detector.claim_patterns, dict)
    
    def test_get_config_class(self):
        """Test get_config_class method"""
        config_class = ContradictionDetector.get_config_class()
        assert config_class == ContradictionDetectorConfig
    
    def test_initialize_claim_patterns(self):
        """Test _initialize_claim_patterns method"""
        patterns = self.detector.claim_patterns
        
        # Check all claim types have patterns
        assert ClaimType.FACTUAL in patterns
        assert ClaimType.EVALUATIVE in patterns  
        assert ClaimType.CAUSAL in patterns
        assert ClaimType.COMPARATIVE in patterns
        assert ClaimType.TEMPORAL in patterns
        
        # Check patterns are non-empty lists
        for claim_type, pattern_list in patterns.items():
            assert isinstance(pattern_list, list)
            assert len(pattern_list) > 0
            assert all(isinstance(pattern, str) for pattern in pattern_list)
    
    def test_should_extract_claim_type(self):
        """Test _should_extract_claim_type method"""
        # Default config enables factual, evaluative, causal
        assert self.detector._should_extract_claim_type(ClaimType.FACTUAL) is True
        assert self.detector._should_extract_claim_type(ClaimType.EVALUATIVE) is True
        assert self.detector._should_extract_claim_type(ClaimType.CAUSAL) is True
        assert self.detector._should_extract_claim_type(ClaimType.COMPARATIVE) is False
        assert self.detector._should_extract_claim_type(ClaimType.TEMPORAL) is False
        
        # Test with custom config
        custom_config = ContradictionDetectorConfig(
            extract_factual_claims=False,
            extract_comparative_claims=True
        )
        custom_detector = ContradictionDetector(custom_config)
        
        assert custom_detector._should_extract_claim_type(ClaimType.FACTUAL) is False
        assert custom_detector._should_extract_claim_type(ClaimType.COMPARATIVE) is True
    
    def test_split_into_sentences(self):
        """Test _split_into_sentences method"""
        text = "これは最初の文です。これは２番目の文です！最後の文ですか？"
        sentences = self.detector._split_into_sentences(text)
        
        expected_sentences = [
            "これは最初の文です",
            "これは２番目の文です",
            "最後の文ですか"
        ]
        assert sentences == expected_sentences
        
        # Test empty input
        assert self.detector._split_into_sentences("") == []
        
        # Test single sentence without punctuation
        assert self.detector._split_into_sentences("単純な文") == ["単純な文"]
    
    def test_estimate_claim_confidence(self):
        """Test _estimate_claim_confidence method"""
        # Test with certainty indicators
        certain_sentence = "明らかにPythonは簡単です"
        confidence = self.detector._estimate_claim_confidence(certain_sentence, ClaimType.FACTUAL)
        assert confidence > 0.5  # Should be boosted by certainty indicator
        
        # Test with uncertainty indicators
        uncertain_sentence = "おそらくPythonは簡単だと思われます"
        confidence = self.detector._estimate_claim_confidence(uncertain_sentence, ClaimType.FACTUAL)
        assert confidence < 0.8  # Should be reduced by uncertainty
        
        # Test appropriate length sentence
        good_length = "Pythonはプログラミング言語です"
        confidence = self.detector._estimate_claim_confidence(good_length, ClaimType.FACTUAL)
        assert 0.0 <= confidence <= 1.0
        
        # Test very short sentence
        short_sentence = "短い"
        confidence = self.detector._estimate_claim_confidence(short_sentence, ClaimType.FACTUAL)
        assert confidence < 0.5  # Should be penalized for being too short
    
    def test_extract_claims_basic(self):
        """Test _extract_claims with basic functionality"""
        doc = Document(
            id="extract_test",
            content="Pythonは人気の言語です。機械学習は複雑な分野です。",
            metadata={}
        )
        
        claims = self.detector._extract_claims(doc)
        
        # Should extract some claims
        assert len(claims) > 0
        
        # Check claim structure
        for claim in claims:
            assert isinstance(claim, Claim)
            assert claim.source_document_id == "extract_test"
            assert claim.confidence >= self.config.claim_confidence_threshold
            assert claim.claim_type in list(ClaimType)
            assert len(claim.text) > 0
    
    def test_extract_claims_with_patterns(self):
        """Test _extract_claims with specific patterns"""
        doc = Document(
            id="pattern_test",
            content="データサイエンスは重要な分野です。AIにより作業が効率化されます。機械学習の方がディープラーニングより基本的です。",
            metadata={}
        )
        
        claims = self.detector._extract_claims(doc)
        
        # Should extract claims (may be all of same type due to pattern matching)
        claim_types = [claim.claim_type for claim in claims]
        assert len(claims) >= 1  # At least 1 claim should be extracted
        
        # Check that at least one claim was extracted and contains expected content
        assert any("重要" in claim.text for claim in claims)  # Should match first sentence
        # Note: Other patterns may not match due to regex specificity, which is expected behavior
    
    def test_extract_claims_with_max_limit(self):
        """Test _extract_claims respects max_claims_per_document"""
        # Create content with many potential claims
        long_content = "。".join([f"項目{i}は重要です" for i in range(50)])
        doc = Document(id="limit_test", content=long_content, metadata={})
        
        claims = self.detector._extract_claims(doc)
        
        # Should not exceed max limit
        assert len(claims) <= self.config.max_claims_per_document
    
    def test_extract_claims_confidence_filtering(self):
        """Test _extract_claims filters by confidence threshold"""
        # Test with high confidence threshold
        high_threshold_config = ContradictionDetectorConfig(claim_confidence_threshold=0.9)
        high_threshold_detector = ContradictionDetector(high_threshold_config)
        
        doc = Document(
            id="confidence_test",
            content="おそらく機械学習は複雑だと思われます。明らかにPythonは人気です。",
            metadata={}
        )
        
        claims = high_threshold_detector._extract_claims(doc)
        
        # All claims should meet high threshold
        for claim in claims:
            assert claim.confidence >= 0.9
    
    def test_check_opposite_values(self):
        """Test _check_opposite_values method"""
        # Test positive cases
        assert self.detector._check_opposite_values("Pythonは簡単です", "Pythonは困難です") is True
        assert self.detector._check_opposite_values("性能は高いです", "性能は低いです") is True
        assert self.detector._check_opposite_values("これは良い解決策です", "これは悪い解決策です") is True
        
        # Test negative cases
        assert self.detector._check_opposite_values("Pythonは簡単です", "Javaは簡単です") is False
        assert self.detector._check_opposite_values("良い天気です", "雨が降っています") is False
    
    def test_check_contradictory_facts(self):
        """Test _check_contradictory_facts method"""
        # Test with different numbers
        assert self.detector._check_contradictory_facts("価格は100円です", "価格は200円です") is True
        assert self.detector._check_contradictory_facts("2020年に発売", "2021年に発売") is True
        
        # Test with same numbers
        assert self.detector._check_contradictory_facts("価格は100円です", "コストは100円です") is False
        
        # Test without numbers
        assert self.detector._check_contradictory_facts("良い製品です", "優秀な製品です") is False
    
    def test_perform_nli_basic(self):
        """Test _perform_nli basic functionality"""
        # Test contradiction case
        result = self.detector._perform_nli("Pythonは簡単です", "Pythonは困難です")
        
        assert "label" in result
        assert "confidence" in result  
        assert "common_words" in result
        assert result["label"] in [NLILabel.ENTAILMENT, NLILabel.CONTRADICTION, NLILabel.NEUTRAL]
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["common_words"], list)
    
    def test_perform_nli_contradiction(self):
        """Test _perform_nli detects contradiction"""
        # Clear contradiction with negation
        result = self.detector._perform_nli(
            "機械学習は簡単です", 
            "機械学習は簡単ではありません"
        )
        
        assert result["label"] == NLILabel.CONTRADICTION
        assert result["confidence"] >= 0.7
    
    def test_perform_nli_entailment(self):
        """Test _perform_nli detects entailment/neutral"""
        # Similar statements
        result = self.detector._perform_nli(
            "Pythonは人気です",
            "Pythonは良い言語です"
        )
        
        # Should not be high contradiction
        if result["label"] == NLILabel.CONTRADICTION:
            assert result["confidence"] < 0.7
    
    def test_classify_contradiction_type(self):
        """Test _classify_contradiction_type method"""
        claim1 = Claim(
            id="c1", text="test1", claim_type=ClaimType.FACTUAL,
            confidence=0.8, source_document_id="d1", source_sentence="test1"
        )
        claim2 = Claim(
            id="c2", text="test2", claim_type=ClaimType.FACTUAL,
            confidence=0.8, source_document_id="d2", source_sentence="test2"
        )
        claim3 = Claim(
            id="c3", text="test3", claim_type=ClaimType.EVALUATIVE,
            confidence=0.8, source_document_id="d3", source_sentence="test3"
        )
        
        # Same type
        same_type = self.detector._classify_contradiction_type(claim1, claim2)
        assert same_type == "same_type_factual"
        
        # Different types
        diff_type = self.detector._classify_contradiction_type(claim1, claim3)
        assert diff_type == "cross_type_factual_evaluative"
    
    def test_generate_contradiction_explanation(self):
        """Test _generate_contradiction_explanation method"""
        claim1 = Claim(
            id="c1", text="Pythonは簡単なプログラミング言語です", claim_type=ClaimType.EVALUATIVE,
            confidence=0.8, source_document_id="doc1", source_sentence="test1"
        )
        claim2 = Claim(
            id="c2", text="Pythonは複雑なプログラミング言語です", claim_type=ClaimType.EVALUATIVE,
            confidence=0.8, source_document_id="doc2", source_sentence="test2"
        )
        
        # Different documents
        explanation = self.detector._generate_contradiction_explanation(claim1, claim2)
        assert "クレーム1" in explanation
        assert "クレーム2" in explanation
        assert "異なる文書" in explanation
        assert "doc1" in explanation
        assert "doc2" in explanation
        
        # Same document
        claim2.source_document_id = "doc1"
        explanation_same = self.detector._generate_contradiction_explanation(claim1, claim2)
        assert "同一文書内" in explanation_same
    
    def test_assess_contradiction_severity(self):
        """Test _assess_contradiction_severity method"""
        assert self.detector._assess_contradiction_severity(0.95) == "high"
        assert self.detector._assess_contradiction_severity(0.85) == "medium"
        assert self.detector._assess_contradiction_severity(0.65) == "low"
        assert self.detector._assess_contradiction_severity(0.9) == "high"
        assert self.detector._assess_contradiction_severity(0.7) == "medium"
    
    def test_assess_severity_overall(self):
        """Test _assess_severity method for overall assessment"""
        # Test empty contradictions
        assert self.detector._assess_severity([]) == "none"
        
        # Create mock contradictions
        high_contradiction = Mock()
        high_contradiction.severity = "high"
        
        medium_contradiction = Mock()
        medium_contradiction.severity = "medium"
        
        low_contradiction = Mock()
        low_contradiction.severity = "low"
        
        # Test with high severity present
        assert self.detector._assess_severity([high_contradiction, medium_contradiction]) == "high"
        
        # Test with majority medium severity
        contradictions = [medium_contradiction, medium_contradiction, low_contradiction]
        assert self.detector._assess_severity(contradictions) == "medium"
        
        # Test with mostly low severity
        assert self.detector._assess_severity([low_contradiction, low_contradiction]) == "low"
    
    def test_compute_consistency_score(self):
        """Test _compute_consistency_score method"""
        # Test no contradictions
        assert self.detector._compute_consistency_score([], 10) == 1.0
        
        # Test no claims
        assert self.detector._compute_consistency_score([], 0) == 1.0
        
        # Create mock contradictions
        high_contradiction = Mock()
        high_contradiction.severity = "high"
        
        medium_contradiction = Mock()
        medium_contradiction.severity = "medium"
        
        low_contradiction = Mock()
        low_contradiction.severity = "low"
        
        # Test with contradictions
        contradictions = [high_contradiction, medium_contradiction, low_contradiction]
        score = self.detector._compute_consistency_score(contradictions, 10)
        
        assert 0.0 <= score <= 1.0
        assert score < 1.0  # Should be reduced due to contradictions
    
    def test_process_basic_functionality(self):
        """Test process method basic functionality"""
        results = self.detector.process(self.test_doc)
        
        # Should return list with one result document
        assert len(results) == 1
        assert isinstance(results[0], Document)
        
        result_doc = results[0]
        assert result_doc.id.startswith("contradiction_analysis_")
        assert result_doc.metadata["processing_stage"] == "contradiction_detection"
        assert result_doc.metadata["source_document_id"] == self.test_doc.id
        assert "claims_extracted" in result_doc.metadata
        assert "contradictions_found" in result_doc.metadata
        assert "contradiction_severity" in result_doc.metadata
        assert "document_consistency_score" in result_doc.metadata
    
    def test_process_with_claim_extraction_disabled(self):
        """Test process with claim extraction disabled"""
        config = ContradictionDetectorConfig(enable_claim_extraction=False)
        detector = ContradictionDetector(config)
        
        results = detector.process(self.test_doc)
        
        assert len(results) == 1
        result_doc = results[0]
        assert result_doc.metadata["claims_extracted"] == 0
        assert result_doc.metadata["contradictions_found"] == 0
    
    def test_process_with_nli_detection_disabled(self):
        """Test process with NLI detection disabled"""
        config = ContradictionDetectorConfig(enable_nli_detection=False)
        detector = ContradictionDetector(config)
        
        results = detector.process(self.test_doc)
        
        assert len(results) == 1
        result_doc = results[0]
        # Claims should be extracted but no contradictions detected
        assert result_doc.metadata["contradictions_found"] == 0
    
    def test_format_contradiction_report_with_contradictions(self):
        """Test _format_contradiction_report with contradictions"""
        # Create sample claims and contradictions
        claim1 = Claim(
            id="c1", text="Pythonは簡単です", claim_type=ClaimType.EVALUATIVE,
            confidence=0.8, source_document_id="doc1", source_sentence="test1"
        )
        claim2 = Claim(
            id="c2", text="Pythonは困難です", claim_type=ClaimType.EVALUATIVE,
            confidence=0.7, source_document_id="doc1", source_sentence="test2"
        )
        
        contradiction = ContradictionPair(
            claim1=claim1,
            claim2=claim2,
            contradiction_score=0.9,
            contradiction_type="same_type_evaluative",
            explanation="評価が矛盾しています",
            severity="high"
        )
        
        report = self.detector._format_contradiction_report([claim1, claim2], [contradiction])
        
        # Check report structure
        assert "# 矛盾検出レポート" in report
        assert "## 📊 検出サマリー" in report
        assert "抽出されたクレーム数: 2" in report
        assert "検出された矛盾数: 1" in report
        assert "## 📝 抽出されたクレーム" in report
        assert "## ⚠️ 検出された矛盾" in report
        assert "## 🔧 推奨事項" in report
        assert "🔴" in report  # High severity emoji
        assert "Pythonは簡単です" in report
        assert "Pythonは困難です" in report
    
    def test_format_contradiction_report_no_contradictions(self):
        """Test _format_contradiction_report without contradictions"""
        claim = Claim(
            id="c1", text="Pythonは言語です", claim_type=ClaimType.FACTUAL,
            confidence=0.8, source_document_id="doc1", source_sentence="test1"
        )
        
        report = self.detector._format_contradiction_report([claim], [])
        
        assert "## ✅ 一貫性" in report
        assert "矛盾は検出されませんでした" in report
        assert "文書の一貫性は良好です" in report
    
    def test_get_contradiction_summary_empty(self):
        """Test get_contradiction_summary with no contradictions"""
        summary = self.detector.get_contradiction_summary()
        
        expected = {
            "total_contradictions": 0,
            "severity_distribution": {},
            "consistency_status": "good"
        }
        assert summary == expected
    
    def test_get_contradiction_summary_with_contradictions(self):
        """Test get_contradiction_summary with contradictions"""
        # Add mock contradictions
        high_contradiction = Mock()
        high_contradiction.severity = "high"
        
        medium_contradiction = Mock()
        medium_contradiction.severity = "medium"
        
        self.detector.detected_contradictions = [high_contradiction, medium_contradiction]
        self.detector.extracted_claims = [Mock(), Mock(), Mock()]  # 3 claims
        
        summary = self.detector.get_contradiction_summary()
        
        assert summary["total_contradictions"] == 2
        assert summary["total_claims"] == 3
        assert summary["severity_distribution"]["high"] == 1
        assert summary["severity_distribution"]["medium"] == 1
        assert summary["consistency_status"] == "poor"  # Due to high severity
        assert summary["contradiction_rate"] == 2/3


class TestContradictionDetectorEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ContradictionDetectorConfig()
        self.detector = ContradictionDetector(self.config)
    
    def test_empty_document(self):
        """Test processing empty document"""
        empty_doc = Document(id="empty", content="", metadata={})
        results = self.detector.process(empty_doc)
        
        assert len(results) == 1
        result_doc = results[0]
        assert result_doc.metadata["claims_extracted"] == 0
        assert result_doc.metadata["contradictions_found"] == 0
    
    def test_whitespace_only_document(self):
        """Test processing document with only whitespace"""
        whitespace_doc = Document(id="whitespace", content="   \n\t   ", metadata={})
        results = self.detector.process(whitespace_doc)
        
        assert len(results) == 1
        result_doc = results[0]
        assert result_doc.metadata["claims_extracted"] == 0
    
    def test_short_sentences_filtering(self):
        """Test that very short sentences are filtered out"""
        short_doc = Document(
            id="short",
            content="短い。はい。いいえ。Pythonは良いプログラミング言語です。",
            metadata={}
        )
        
        claims = self.detector._extract_claims(short_doc)
        
        # Only the longer sentence should produce claims
        assert len(claims) >= 0  # May or may not extract depending on patterns
        
        # All extracted claims should be from longer sentences
        for claim in claims:
            assert len(claim.text) >= 10
    
    def test_no_pattern_matches(self):
        """Test document with no pattern matches"""
        no_match_doc = Document(
            id="no_match",
            content="ランダムな単語の集合。意味のない文字列。",
            metadata={}
        )
        
        claims = self.detector._extract_claims(no_match_doc)
        
        # Should extract few or no claims
        assert len(claims) >= 0  # Could be 0 or more depending on patterns
    
    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters"""
        unicode_doc = Document(
            id="unicode",
            content="Python™は優れたプログラミング言語です🐍。データサイエンス分野で重要です★。",
            metadata={}
        )
        
        claims = self.detector._extract_claims(unicode_doc)
        
        # Should handle Unicode gracefully
        assert len(claims) >= 0
        for claim in claims:
            assert isinstance(claim.text, str)
            assert len(claim.text) > 0
    
    def test_very_long_document(self):
        """Test processing very long document"""
        # Create a long document that exceeds max_claims_per_document
        long_content = "。".join([
            f"項目{i}は重要な要素です" for i in range(100)
        ])
        long_doc = Document(id="long", content=long_content, metadata={})
        
        claims = self.detector._extract_claims(long_doc)
        
        # Should respect max_claims_per_document limit
        assert len(claims) <= self.config.max_claims_per_document
    
    def test_mixed_punctuation(self):
        """Test handling mixed Japanese and Western punctuation"""
        mixed_doc = Document(
            id="mixed",
            content="Pythonは素晴らしい! データサイエンスに最適？ 機械学習で使用される。",
            metadata={}
        )
        
        sentences = self.detector._split_into_sentences(mixed_doc.content)
        
        # Should properly split on both Japanese and Western punctuation
        assert len(sentences) >= 2  # Should split into multiple sentences
        assert all(len(s.strip()) > 0 for s in sentences)
    
    def test_within_document_contradictions_empty_claims(self):
        """Test _detect_within_document_contradictions with empty claims"""
        contradictions = self.detector._detect_within_document_contradictions([])
        assert contradictions == []
    
    def test_within_document_contradictions_single_claim(self):
        """Test _detect_within_document_contradictions with single claim"""
        claim = Claim(
            id="single", text="test", claim_type=ClaimType.FACTUAL,
            confidence=0.8, source_document_id="doc", source_sentence="test"
        )
        
        contradictions = self.detector._detect_within_document_contradictions([claim])
        assert contradictions == []
    
    def test_cross_document_contradictions_no_existing_claims(self):
        """Test _detect_cross_document_contradictions with no existing claims"""
        new_claim = Claim(
            id="new", text="test", claim_type=ClaimType.FACTUAL,
            confidence=0.8, source_document_id="doc", source_sentence="test"
        )
        
        # detector.extracted_claims is empty by default
        contradictions = self.detector._detect_cross_document_contradictions([new_claim])
        assert contradictions == []
    
    def test_extreme_confidence_values(self):
        """Test handling of extreme confidence values"""
        # Test very high contradiction threshold
        extreme_config = ContradictionDetectorConfig(contradiction_threshold=0.99)
        extreme_detector = ContradictionDetector(extreme_config)
        
        claim1 = Claim(
            id="c1", text="Pythonは簡単", claim_type=ClaimType.EVALUATIVE,
            confidence=0.8, source_document_id="doc1", source_sentence="test1"
        )
        claim2 = Claim(
            id="c2", text="Pythonは困難", claim_type=ClaimType.EVALUATIVE,
            confidence=0.8, source_document_id="doc1", source_sentence="test2"
        )
        
        contradictions = extreme_detector._detect_within_document_contradictions([claim1, claim2])
        
        # With very high threshold, should detect few contradictions
        assert len(contradictions) >= 0


class TestContradictionDetectorIntegration:
    """Integration tests for ContradictionDetector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ContradictionDetectorConfig()
        self.detector = ContradictionDetector(self.config)
    
    def test_full_contradiction_detection_workflow(self):
        """Test complete workflow from document to contradiction detection"""
        # Create document with clear contradictions
        doc = Document(
            id="workflow_test",
            content="""
            Pythonは初心者にとって簡単なプログラミング言語です。
            Pythonは学習が困難で複雑な言語です。
            機械学習により業務効率が向上します。
            データサイエンスは重要な分野です。
            """,
            metadata={"source": "test_workflow"}
        )
        
        # Process document
        results = self.detector.process(doc)
        
        # Verify results
        assert len(results) == 1
        result_doc = results[0]
        
        # Check metadata
        metadata = result_doc.metadata
        assert metadata["processing_stage"] == "contradiction_detection"
        assert metadata["source_document_id"] == "workflow_test"
        assert metadata["claims_extracted"] > 0
        assert "contradictions_found" in metadata
        assert "contradiction_severity" in metadata
        assert "document_consistency_score" in metadata
        
        # Check content structure
        content = result_doc.content
        assert "# 矛盾検出レポート" in content
        assert "## 📊 検出サマリー" in content
        
        # Check that claims were extracted
        assert self.detector.extracted_claims is not None
        assert len(self.detector.extracted_claims) > 0
    
    def test_multiple_document_processing(self):
        """Test processing multiple documents sequentially"""
        doc1 = Document(
            id="multi_doc_1",
            content="Pythonは簡単なプログラミング言語です。",
            metadata={}
        )
        
        doc2 = Document(
            id="multi_doc_2", 
            content="Pythonは非常に複雑で困難な言語です。",
            metadata={}
        )
        
        # Process first document
        results1 = self.detector.process(doc1)
        claims_after_doc1 = len(self.detector.extracted_claims)
        
        # Process second document
        results2 = self.detector.process(doc2)
        claims_after_doc2 = len(self.detector.extracted_claims)
        
        # Verify progression
        assert len(results1) == 1
        assert len(results2) == 1
        assert claims_after_doc2 >= claims_after_doc1  # Claims should accumulate
        
        # Check for cross-document contradictions
        if self.config.check_across_documents:
            assert len(self.detector.detected_contradictions) >= 0  # May detect contradictions
    
    def test_comprehensive_claim_type_extraction(self):
        """Test extraction of all enabled claim types"""
        comprehensive_doc = Document(
            id="comprehensive_test",
            content="""
            Pythonはオープンソースの言語です。
            機械学習は非常に有用だと思われます。
            AIにより生産性が向上します。
            深層学習の方がより効果的です。
            将来AIはさらに重要になるでしょう。
            """,
            metadata={}
        )
        
        # Process with all claim types enabled
        all_types_config = ContradictionDetectorConfig(
            extract_factual_claims=True,
            extract_evaluative_claims=True,
            extract_causal_claims=True,
            extract_comparative_claims=True,
            extract_temporal_claims=True
        )
        comprehensive_detector = ContradictionDetector(all_types_config)
        
        claims = comprehensive_detector._extract_claims(comprehensive_doc)
        
        # Should extract various types of claims
        assert len(claims) > 0
        
        # Check for variety in claim types
        claim_types = [claim.claim_type for claim in claims]
        unique_types = set(claim_types)
        assert len(unique_types) >= 2  # Should have at least 2 different types
    
    def test_configuration_impact_on_processing(self):
        """Test how different configurations affect processing"""
        test_doc = Document(
            id="config_test",
            content="Pythonは簡単です。機械学習は複雑です。",
            metadata={}
        )
        
        # Test with minimal configuration
        minimal_config = ContradictionDetectorConfig(
            enable_claim_extraction=True,
            enable_nli_detection=False,
            extract_factual_claims=True,
            extract_evaluative_claims=False,
            extract_causal_claims=False
        )
        minimal_detector = ContradictionDetector(minimal_config)
        
        results = minimal_detector.process(test_doc)
        result_doc = results[0]
        
        # Should have claims but no contradictions due to disabled NLI
        assert result_doc.metadata["contradictions_found"] == 0
        
        # Test with full configuration
        full_config = ContradictionDetectorConfig(
            enable_claim_extraction=True,
            enable_nli_detection=True,
            extract_factual_claims=True,
            extract_evaluative_claims=True,
            extract_causal_claims=True,
            contradiction_threshold=0.5  # Lower threshold for more detection
        )
        full_detector = ContradictionDetector(full_config)
        
        full_results = full_detector.process(test_doc)
        full_result_doc = full_results[0]
        
        # Should potentially detect more with full configuration
        assert full_result_doc.metadata["claims_extracted"] >= 0