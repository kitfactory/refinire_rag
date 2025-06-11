"""
ContradictionDetector - Claim Extraction + NLI Detection

ドキュメントからクレーム（主張）を抽出し、自然言語推論（NLI）を使用して
矛盾を検出するDocumentProcessor。
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field
from dataclasses import field
import re
import json
from pathlib import Path
from enum import Enum

from .document_processor import DocumentProcessor, DocumentProcessorConfig
from ..models.document import Document


class ClaimType(str, Enum):
    """クレームタイプ"""
    FACTUAL = "factual"          # 事実に関する主張
    EVALUATIVE = "evaluative"    # 評価に関する主張
    CAUSAL = "causal"           # 因果関係に関する主張
    COMPARATIVE = "comparative"  # 比較に関する主張
    TEMPORAL = "temporal"       # 時間に関する主張


class NLILabel(str, Enum):
    """自然言語推論ラベル"""
    ENTAILMENT = "entailment"       # 含意
    CONTRADICTION = "contradiction"  # 矛盾
    NEUTRAL = "neutral"             # 中立


class Claim(BaseModel):
    """クレーム（主張）モデル"""
    
    id: str
    text: str
    claim_type: ClaimType
    confidence: float
    source_document_id: str
    source_sentence: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContradictionPair(BaseModel):
    """矛盾ペアモデル"""
    
    claim1: Claim
    claim2: Claim
    contradiction_score: float
    contradiction_type: str
    explanation: str
    severity: str  # "high", "medium", "low"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContradictionDetectorConfig(DocumentProcessorConfig):
    """ContradictionDetector設定"""
    
    enable_claim_extraction: bool = True
    enable_nli_detection: bool = True
    contradiction_threshold: float = 0.7
    claim_confidence_threshold: float = 0.5
    max_claims_per_document: int = 20
    
    # クレーム抽出設定
    extract_factual_claims: bool = True
    extract_evaluative_claims: bool = True
    extract_causal_claims: bool = True
    extract_comparative_claims: bool = False
    extract_temporal_claims: bool = False
    
    # 矛盾検出設定
    check_within_document: bool = True
    check_across_documents: bool = True
    check_against_knowledge_base: bool = False
    
    # 出力設定
    save_detected_contradictions: bool = True
    contradictions_output_file: Optional[str] = None


class ContradictionDetector(DocumentProcessor):
    """
    矛盾検出器
    
    ドキュメントからクレームを抽出し、NLIを使用して矛盾を検出します。
    """
    
    @classmethod
    def get_config_class(cls):
        return ContradictionDetectorConfig
    
    def __init__(self, config: ContradictionDetectorConfig):
        super().__init__(config)
        self.extracted_claims: List[Claim] = []
        self.detected_contradictions: List[ContradictionPair] = []
        self.claim_patterns = self._initialize_claim_patterns()
    
    def process(self, document: Document) -> List[Document]:
        """
        ドキュメントから矛盾を検出
        
        Args:
            document: 処理対象ドキュメント
            
        Returns:
            List[Document]: 矛盾検出結果ドキュメント
        """
        # クレームを抽出
        claims = []
        if self.config.enable_claim_extraction:
            claims = self._extract_claims(document)
            self.extracted_claims.extend(claims)
        
        # 矛盾を検出
        contradictions = []
        if self.config.enable_nli_detection and claims:
            contradictions = self._detect_contradictions(claims, document)
            self.detected_contradictions.extend(contradictions)
        
        # 矛盾検出結果ドキュメントを作成
        result_doc = Document(
            id=f"contradiction_analysis_{document.id}",
            content=self._format_contradiction_report(claims, contradictions),
            metadata={
                "processing_stage": "contradiction_detection",
                "source_document_id": document.id,
                "claims_extracted": len(claims),
                "contradictions_found": len(contradictions),
                "contradiction_severity": self._assess_severity(contradictions),
                "document_consistency_score": self._compute_consistency_score(contradictions, len(claims))
            }
        )
        
        return [result_doc]
    
    def _initialize_claim_patterns(self) -> Dict[ClaimType, List[str]]:
        """クレーム抽出パターンを初期化"""
        
        return {
            ClaimType.FACTUAL: [
                r"(.+)は(.+)です",
                r"(.+)が(.+)である",
                r"(.+)には(.+)が含まれている",
                r"(.+)は(.+)を持つ",
                r"(.+)によると(.+)",
            ],
            ClaimType.EVALUATIVE: [
                r"(.+)は(.+)的である",
                r"(.+)は(.+)だと思われる",
                r"(.+)は(.+)と評価される",
                r"(.+)は(.+)価値がある",
                r"(.+)は(.+)重要である",
            ],
            ClaimType.CAUSAL: [
                r"(.+)により(.+)が生じる",
                r"(.+)が原因で(.+)",
                r"(.+)のため(.+)になる",
                r"(.+)すると(.+)する",
                r"(.+)の結果(.+)",
            ],
            ClaimType.COMPARATIVE: [
                r"(.+)は(.+)より(.+)",
                r"(.+)と(.+)を比較すると",
                r"(.+)の方が(.+)である",
                r"(.+)に対して(.+)は(.+)",
            ],
            ClaimType.TEMPORAL: [
                r"(.+)の前に(.+)",
                r"(.+)の後(.+)",
                r"(.+)年に(.+)",
                r"将来(.+)になる",
                r"過去に(.+)だった",
            ]
        }
    
    def _extract_claims(self, document: Document) -> List[Claim]:
        """ドキュメントからクレームを抽出"""
        
        claims = []
        content = document.content
        sentences = self._split_into_sentences(content)
        
        for i, sentence in enumerate(sentences[:self.config.max_claims_per_document]):
            sentence = sentence.strip()
            if len(sentence) < 10:  # 短すぎる文は除外
                continue
            
            # 各タイプのクレームパターンをチェック
            for claim_type, patterns in self.claim_patterns.items():
                if not self._should_extract_claim_type(claim_type):
                    continue
                
                for pattern in patterns:
                    matches = re.findall(pattern, sentence)
                    if matches:
                        claim = Claim(
                            id=f"{document.id}_claim_{len(claims)+1}",
                            text=sentence,
                            claim_type=claim_type,
                            confidence=self._estimate_claim_confidence(sentence, claim_type),
                            source_document_id=document.id,
                            source_sentence=sentence,
                            metadata={
                                "sentence_index": i,
                                "extraction_pattern": pattern,
                                "matched_groups": matches[0] if matches else None
                            }
                        )
                        
                        if claim.confidence >= self.config.claim_confidence_threshold:
                            claims.append(claim)
                        break  # 一つのパターンがマッチしたら次の文へ
        
        return claims
    
    def _should_extract_claim_type(self, claim_type: ClaimType) -> bool:
        """クレームタイプを抽出すべきかチェック"""
        
        config_map = {
            ClaimType.FACTUAL: self.config.extract_factual_claims,
            ClaimType.EVALUATIVE: self.config.extract_evaluative_claims,
            ClaimType.CAUSAL: self.config.extract_causal_claims,
            ClaimType.COMPARATIVE: self.config.extract_comparative_claims,
            ClaimType.TEMPORAL: self.config.extract_temporal_claims,
        }
        
        return config_map.get(claim_type, True)
    
    def _estimate_claim_confidence(self, sentence: str, claim_type: ClaimType) -> float:
        """クレームの信頼度を推定"""
        
        confidence = 0.5  # ベース信頼度
        
        # 確実性を示すキーワード
        certainty_indicators = ["明らかに", "確実に", "間違いなく", "必ず", "常に"]
        uncertainty_indicators = ["おそらく", "多分", "かもしれない", "と思われる", "推測される"]
        
        certainty_count = sum(1 for indicator in certainty_indicators if indicator in sentence)
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in sentence)
        
        confidence += certainty_count * 0.2
        confidence -= uncertainty_count * 0.2
        
        # 文の長さによる調整（適度な長さが望ましい）
        if 20 <= len(sentence) <= 100:
            confidence += 0.1
        elif len(sentence) < 10 or len(sentence) > 200:
            confidence -= 0.2
        
        # クレームタイプによる調整
        type_confidence_map = {
            ClaimType.FACTUAL: 0.8,
            ClaimType.EVALUATIVE: 0.6,
            ClaimType.CAUSAL: 0.7,
            ClaimType.COMPARATIVE: 0.6,
            ClaimType.TEMPORAL: 0.7,
        }
        
        confidence *= type_confidence_map.get(claim_type, 0.5)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """テキストを文に分割"""
        
        # 日本語の句読点で分割
        sentences = re.split(r'[。！？]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _detect_contradictions(self, claims: List[Claim], document: Document) -> List[ContradictionPair]:
        """クレーム間の矛盾を検出"""
        
        contradictions = []
        
        if self.config.check_within_document:
            # 同一文書内の矛盾をチェック
            doc_contradictions = self._detect_within_document_contradictions(claims)
            contradictions.extend(doc_contradictions)
        
        if self.config.check_across_documents:
            # 既存のクレームとの矛盾をチェック
            cross_contradictions = self._detect_cross_document_contradictions(claims)
            contradictions.extend(cross_contradictions)
        
        return contradictions
    
    def _detect_within_document_contradictions(self, claims: List[Claim]) -> List[ContradictionPair]:
        """同一文書内のクレーム間矛盾を検出"""
        
        contradictions = []
        
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                claim1 = claims[i]
                claim2 = claims[j]
                
                # NLI判定を実行
                nli_result = self._perform_nli(claim1.text, claim2.text)
                
                if nli_result["label"] == NLILabel.CONTRADICTION:
                    if nli_result["confidence"] >= self.config.contradiction_threshold:
                        contradiction = ContradictionPair(
                            claim1=claim1,
                            claim2=claim2,
                            contradiction_score=nli_result["confidence"],
                            contradiction_type=self._classify_contradiction_type(claim1, claim2),
                            explanation=self._generate_contradiction_explanation(claim1, claim2),
                            severity=self._assess_contradiction_severity(nli_result["confidence"]),
                            metadata={
                                "detection_method": "within_document_nli",
                                "nli_confidence": nli_result["confidence"]
                            }
                        )
                        contradictions.append(contradiction)
        
        return contradictions
    
    def _detect_cross_document_contradictions(self, new_claims: List[Claim]) -> List[ContradictionPair]:
        """異なる文書間のクレーム矛盾を検出"""
        
        contradictions = []
        
        # 既存のクレームと新しいクレームを比較
        for new_claim in new_claims:
            for existing_claim in self.extracted_claims:
                # 同一文書のクレームはスキップ
                if new_claim.source_document_id == existing_claim.source_document_id:
                    continue
                
                nli_result = self._perform_nli(new_claim.text, existing_claim.text)
                
                if nli_result["label"] == NLILabel.CONTRADICTION:
                    if nli_result["confidence"] >= self.config.contradiction_threshold:
                        contradiction = ContradictionPair(
                            claim1=new_claim,
                            claim2=existing_claim,
                            contradiction_score=nli_result["confidence"],
                            contradiction_type=self._classify_contradiction_type(new_claim, existing_claim),
                            explanation=self._generate_contradiction_explanation(new_claim, existing_claim),
                            severity=self._assess_contradiction_severity(nli_result["confidence"]),
                            metadata={
                                "detection_method": "cross_document_nli",
                                "nli_confidence": nli_result["confidence"]
                            }
                        )
                        contradictions.append(contradiction)
        
        return contradictions
    
    def _perform_nli(self, text1: str, text2: str) -> Dict[str, Any]:
        """自然言語推論を実行（簡易実装）"""
        
        # 実際の実装では、BERT-based NLIモデルを使用
        # ここでは簡易的なルールベース判定
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # 否定語の検出
        negation_words = ["ない", "ではない", "でない", "しない", "できない", "いけない"]
        
        text1_negated = any(neg in text1_lower for neg in negation_words)
        text2_negated = any(neg in text2_lower for neg in negation_words)
        
        # 共通キーワードの抽出
        words1 = set(re.findall(r'\w+', text1_lower))
        words2 = set(re.findall(r'\w+', text2_lower))
        common_words = words1.intersection(words2)
        
        # 矛盾判定ロジック
        contradiction_score = 0.0
        
        if len(common_words) > 2:  # 十分な共通語彙がある
            if text1_negated != text2_negated:  # 一方が否定、他方が肯定
                contradiction_score = 0.8
            elif self._check_opposite_values(text1, text2):
                contradiction_score = 0.9
            elif self._check_contradictory_facts(text1, text2):
                contradiction_score = 0.7
        
        # ラベル決定
        if contradiction_score >= 0.7:
            label = NLILabel.CONTRADICTION
        elif contradiction_score >= 0.3:
            label = NLILabel.NEUTRAL
        else:
            label = NLILabel.ENTAILMENT
        
        return {
            "label": label,
            "confidence": contradiction_score if label == NLILabel.CONTRADICTION else 1.0 - contradiction_score,
            "common_words": list(common_words)
        }
    
    def _check_opposite_values(self, text1: str, text2: str) -> bool:
        """対立する値をチェック"""
        
        opposite_pairs = [
            ("高い", "低い"), ("大きい", "小さい"), ("速い", "遅い"),
            ("良い", "悪い"), ("正しい", "間違っている"), ("有効", "無効"),
            ("可能", "不可能"), ("安全", "危険"), ("簡単", "困難")
        ]
        
        for pos, neg in opposite_pairs:
            if (pos in text1 and neg in text2) or (neg in text1 and pos in text2):
                return True
        
        return False
    
    def _check_contradictory_facts(self, text1: str, text2: str) -> bool:
        """矛盾する事実をチェック"""
        
        # 数値の矛盾をチェック
        numbers1 = re.findall(r'\d+', text1)
        numbers2 = re.findall(r'\d+', text2)
        
        if numbers1 and numbers2:
            # 同じ概念について異なる数値が言及されている場合
            if len(set(numbers1).intersection(set(numbers2))) == 0:
                return True
        
        return False
    
    def _classify_contradiction_type(self, claim1: Claim, claim2: Claim) -> str:
        """矛盾のタイプを分類"""
        
        if claim1.claim_type == claim2.claim_type:
            return f"same_type_{claim1.claim_type.value}"
        else:
            return f"cross_type_{claim1.claim_type.value}_{claim2.claim_type.value}"
    
    def _generate_contradiction_explanation(self, claim1: Claim, claim2: Claim) -> str:
        """矛盾の説明を生成"""
        
        explanation = f"クレーム1「{claim1.text[:50]}...」とクレーム2「{claim2.text[:50]}...」"
        
        if claim1.source_document_id == claim2.source_document_id:
            explanation += "は同一文書内で矛盾しています。"
        else:
            explanation += f"は異なる文書（{claim1.source_document_id}と{claim2.source_document_id}）間で矛盾しています。"
        
        return explanation
    
    def _assess_contradiction_severity(self, contradiction_score: float) -> str:
        """矛盾の深刻度を評価"""
        
        if contradiction_score >= 0.9:
            return "high"
        elif contradiction_score >= 0.7:
            return "medium"
        else:
            return "low"
    
    def _assess_severity(self, contradictions: List[ContradictionPair]) -> str:
        """全体的な深刻度を評価"""
        
        if not contradictions:
            return "none"
        
        high_severity_count = sum(1 for c in contradictions if c.severity == "high")
        medium_severity_count = sum(1 for c in contradictions if c.severity == "medium")
        
        if high_severity_count > 0:
            return "high"
        elif medium_severity_count > len(contradictions) * 0.5:
            return "medium"
        else:
            return "low"
    
    def _compute_consistency_score(self, contradictions: List[ContradictionPair], total_claims: int) -> float:
        """一貫性スコアを計算"""
        
        if total_claims == 0:
            return 1.0
        
        # 矛盾の重み付きペナルティ
        penalty = 0.0
        for contradiction in contradictions:
            if contradiction.severity == "high":
                penalty += 0.3
            elif contradiction.severity == "medium":
                penalty += 0.2
            else:
                penalty += 0.1
        
        # クレーム数で正規化
        normalized_penalty = penalty / total_claims
        
        # 一貫性スコア（1.0 - ペナルティ）
        consistency_score = max(0.0, 1.0 - normalized_penalty)
        
        return consistency_score
    
    def _format_contradiction_report(
        self, 
        claims: List[Claim], 
        contradictions: List[ContradictionPair]
    ) -> str:
        """矛盾検出レポートをフォーマット"""
        
        lines = ["# 矛盾検出レポート\n"]
        
        # サマリー
        lines.append(f"## 📊 検出サマリー")
        lines.append(f"- 抽出されたクレーム数: {len(claims)}")
        lines.append(f"- 検出された矛盾数: {len(contradictions)}")
        
        if contradictions:
            severity_counts = {}
            for contradiction in contradictions:
                severity = contradiction.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            lines.append(f"- 深刻度別分布:")
            for severity, count in severity_counts.items():
                lines.append(f"  - {severity}: {count}")
        
        lines.append("")
        
        # 抽出されたクレーム
        if claims:
            lines.append(f"## 📝 抽出されたクレーム")
            for i, claim in enumerate(claims[:10], 1):  # 最初の10件のみ表示
                lines.append(f"### クレーム {i}: {claim.claim_type.value}")
                lines.append(f"**内容**: {claim.text}")
                lines.append(f"**信頼度**: {claim.confidence:.3f}")
                lines.append("")
        
        # 検出された矛盾
        if contradictions:
            lines.append(f"## ⚠️ 検出された矛盾")
            for i, contradiction in enumerate(contradictions, 1):
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                emoji = severity_emoji.get(contradiction.severity, "⚪")
                
                lines.append(f"### {emoji} 矛盾 {i}: {contradiction.contradiction_type}")
                lines.append(f"**深刻度**: {contradiction.severity}")
                lines.append(f"**矛盾スコア**: {contradiction.contradiction_score:.3f}")
                lines.append(f"**説明**: {contradiction.explanation}")
                lines.append(f"**クレーム1**: {contradiction.claim1.text}")
                lines.append(f"**クレーム2**: {contradiction.claim2.text}")
                lines.append("")
        
        # 推奨事項
        if contradictions:
            lines.append(f"## 🔧 推奨事項")
            
            high_severity = [c for c in contradictions if c.severity == "high"]
            if high_severity:
                lines.append("- ⚠️ **重要**: 高深刻度の矛盾が検出されました。該当箇所の検証が必要です。")
            
            same_doc_contradictions = [c for c in contradictions 
                                     if c.claim1.source_document_id == c.claim2.source_document_id]
            if same_doc_contradictions:
                lines.append("- 📄 同一文書内の矛盾があります。文書の論理構造を見直してください。")
            
            cross_doc_contradictions = [c for c in contradictions 
                                      if c.claim1.source_document_id != c.claim2.source_document_id]
            if cross_doc_contradictions:
                lines.append("- 📚 異なる文書間の矛盾があります。情報源の整合性を確認してください。")
        else:
            lines.append(f"## ✅ 一貫性")
            lines.append("矛盾は検出されませんでした。文書の一貫性は良好です。")
        
        return "\n".join(lines)
    
    def get_contradiction_summary(self) -> Dict[str, Any]:
        """矛盾検出の要約を取得"""
        
        total_contradictions = len(self.detected_contradictions)
        
        if total_contradictions == 0:
            return {
                "total_contradictions": 0,
                "severity_distribution": {},
                "consistency_status": "good"
            }
        
        severity_counts = {}
        for contradiction in self.detected_contradictions:
            severity = contradiction.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # 全体的な一貫性ステータス
        if severity_counts.get("high", 0) > 0:
            consistency_status = "poor"
        elif severity_counts.get("medium", 0) > total_contradictions * 0.3:
            consistency_status = "moderate"
        else:
            consistency_status = "good"
        
        return {
            "total_contradictions": total_contradictions,
            "total_claims": len(self.extracted_claims),
            "severity_distribution": severity_counts,
            "consistency_status": consistency_status,
            "contradiction_rate": total_contradictions / len(self.extracted_claims) if self.extracted_claims else 0.0
        }