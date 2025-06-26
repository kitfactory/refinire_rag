"""
Evaluator - Metrics Aggregation

テスト結果を集約し、システム性能のメトリクスを計算するDocumentProcessor。
複数の評価指標を統合して包括的な評価レポートを生成します。
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import json
import statistics
from pathlib import Path

from ..document_processor import DocumentProcessor, DocumentProcessorConfig
from ..models.document import Document


@dataclass
class MetricResult:
    """メトリクス計算結果"""
    name: str
    value: float
    unit: str = ""
    description: str = ""
    threshold: Optional[float] = None
    passed: Optional[bool] = None


class EvaluationMetrics(BaseModel):
    """評価メトリクス定義"""
    
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    average_confidence: float = 0.0
    average_response_time: float = 0.0
    source_accuracy: float = 0.0
    coverage: float = 0.0
    consistency: float = 0.0
    user_satisfaction: float = 0.0


class CategoryMetrics(BaseModel):
    """カテゴリ別メトリクス"""
    
    category: str
    total_queries: int
    successful_queries: int
    success_rate: float
    average_confidence: float
    average_response_time: float
    common_failures: List[str] = field(default_factory=list)


@dataclass
class EvaluatorConfig(DocumentProcessorConfig):
    """Evaluator設定"""
    
    include_category_analysis: bool = True
    include_temporal_analysis: bool = False
    include_failure_analysis: bool = True
    confidence_threshold: float = 0.7
    response_time_threshold: float = 2.0
    accuracy_threshold: float = 0.8
    output_format: str = "markdown"  # "markdown", "json", "html"
    
    # メトリクス重み
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.3,
        "response_time": 0.2,
        "confidence": 0.2,
        "source_accuracy": 0.15,
        "coverage": 0.15
    })


class Evaluator(DocumentProcessor):
    """
    メトリクス集約者
    
    テスト結果ドキュメントを処理し、システム性能の包括的な評価を行います。
    """
    
    @classmethod
    def get_config_class(cls):
        return EvaluatorConfig
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.evaluation_results: List[Dict[str, Any]] = []
        self.computed_metrics: Optional[EvaluationMetrics] = None
        self.category_metrics: Dict[str, CategoryMetrics] = {}
    
    def process(self, document: Document) -> List[Document]:
        """
        テスト結果ドキュメントを評価メトリクスに変換
        
        Args:
            document: TestSuiteからの結果ドキュメント
            
        Returns:
            List[Document]: 評価メトリクスドキュメント
        """
        # ドキュメントがテスト結果かチェック
        if not self._is_test_result_document(document):
            return [document]  # テスト結果でなければそのまま通す
        
        # テスト結果を解析
        test_results = self._parse_test_results(document)
        self.evaluation_results.extend(test_results)
        
        # メトリクスを計算（累積データ使用）
        metrics = self._compute_metrics(self.evaluation_results)
        
        # カテゴリ別分析
        category_analysis = {}
        if self.config.include_category_analysis:
            category_analysis = self._analyze_by_category(test_results)
            self.category_metrics.update(category_analysis)
        
        # 失敗分析
        failure_analysis = {}
        if self.config.include_failure_analysis:
            failure_analysis = self._analyze_failures(test_results)
        
        # 評価レポートドキュメントを作成
        report_doc = Document(
            id=f"evaluation_report_{document.id}",
            content=self._format_evaluation_report(metrics, category_analysis, failure_analysis),
            metadata={
                "processing_stage": "evaluation",
                "source_document_id": document.id,
                "metrics_computed": len(metrics.__dict__),
                "categories_analyzed": len(category_analysis) if category_analysis else 0,
                "overall_score": self._compute_overall_score(metrics),
                "recommendations": self._generate_recommendations(metrics)
            }
        )
        
        return [report_doc]
    
    def _is_test_result_document(self, document: Document) -> bool:
        """ドキュメントがテスト結果かチェック"""
        
        processing_stage = document.metadata.get("processing_stage", "")
        return processing_stage in ["test_execution", "test_results"]
    
    def _parse_test_results(self, document: Document) -> List[Dict[str, Any]]:
        """テスト結果ドキュメントから構造化データを抽出"""
        
        results = []
        content = document.content
        
        # コンテンツからテスト結果を解析（簡易実装）
        lines = content.split('\n')
        current_test = {}
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("## ✅ PASS") or line.startswith("## ❌ FAIL"):
                if current_test:
                    results.append(current_test)
                    current_test = {}
                
                current_test["passed"] = "✅ PASS" in line
                current_test["test_id"] = line.split()[-1] if len(line.split()) > 2 else "unknown"
            
            elif line.startswith("**Query**:"):
                current_test["query"] = line.replace("**Query**:", "").strip()
            
            elif line.startswith("**Confidence**:"):
                try:
                    current_test["confidence"] = float(line.replace("**Confidence**:", "").strip())
                except:
                    current_test["confidence"] = 0.0
            
            elif line.startswith("**Processing Time**:"):
                try:
                    time_str = line.replace("**Processing Time**:", "").replace("s", "").strip()
                    current_test["processing_time"] = float(time_str)
                except:
                    current_test["processing_time"] = 0.0
            
            elif line.startswith("**Sources Found**:"):
                try:
                    current_test["sources_found"] = int(line.replace("**Sources Found**:", "").strip())
                except:
                    current_test["sources_found"] = 0
        
        # 最後のテストを追加
        if current_test:
            results.append(current_test)
        
        # メタデータから統計情報も抽出
        doc_metadata = document.metadata
        
        # 常にdocument_idを追加
        for result in results:
            result["document_id"] = doc_metadata.get("source_document_id", "unknown")
            
        if "tests_run" in doc_metadata:
            # メタデータベースの統計を結果に統合
            for result in results:
                result["success_rate"] = doc_metadata.get("success_rate", 0.0)
        
        return results
    
    def _compute_metrics(self, test_results: List[Dict[str, Any]]) -> EvaluationMetrics:
        """テスト結果からメトリクスを計算"""
        
        if not test_results:
            return EvaluationMetrics()
        
        # 基本統計
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.get("passed", False))
        
        # 各メトリクスを計算
        accuracy = passed_tests / total_tests if total_tests > 0 else 0.0
        
        confidences = [r.get("confidence", 0.0) for r in test_results]
        average_confidence = statistics.mean(confidences) if confidences else 0.0
        
        response_times = [r.get("processing_time", 0.0) for r in test_results]
        average_response_time = statistics.mean(response_times) if response_times else 0.0
        
        # 適合率・再現率の計算（簡易版）
        true_positives = sum(1 for r in test_results 
                           if r.get("passed", False) and r.get("sources_found", 0) > 0)
        false_positives = sum(1 for r in test_results 
                            if not r.get("passed", False) and r.get("sources_found", 0) > 0)
        false_negatives = sum(1 for r in test_results 
                            if r.get("passed", False) and r.get("sources_found", 0) == 0)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # ソース精度
        source_accuracy = sum(1 for r in test_results if r.get("sources_found", 0) > 0) / total_tests
        
        # カバレッジ（テストの多様性）
        unique_queries = len(set(r.get("query", "") for r in test_results))
        coverage = unique_queries / total_tests if total_tests > 0 else 0.0
        
        # 一貫性（同様のクエリでの性能ばらつき）
        confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        consistency = max(0.0, 1.0 - (confidence_std / average_confidence)) if average_confidence > 0 else 0.0
        
        metrics = EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            average_confidence=average_confidence,
            average_response_time=average_response_time,
            source_accuracy=source_accuracy,
            coverage=coverage,
            consistency=consistency,
            user_satisfaction=self._estimate_user_satisfaction(test_results)
        )
        
        self.computed_metrics = metrics
        return metrics
    
    def _estimate_user_satisfaction(self, test_results: List[Dict[str, Any]]) -> float:
        """ユーザー満足度を推定"""
        
        # 複数要因から満足度を推定
        satisfaction_factors = []
        
        for result in test_results:
            # 基本的な成功
            success_factor = 1.0 if result.get("passed", False) else 0.0
            
            # 信頼度要因
            confidence = result.get("confidence", 0.0)
            confidence_factor = min(confidence / self.config.confidence_threshold, 1.0)
            
            # 応答時間要因
            response_time = result.get("processing_time", 0.0)
            time_factor = max(0.0, 1.0 - (response_time / self.config.response_time_threshold))
            
            # 重み付き平均
            overall_satisfaction = (
                success_factor * 0.5 + 
                confidence_factor * 0.3 + 
                time_factor * 0.2
            )
            
            satisfaction_factors.append(overall_satisfaction)
        
        return statistics.mean(satisfaction_factors) if satisfaction_factors else 0.0
    
    def _analyze_by_category(self, test_results: List[Dict[str, Any]]) -> Dict[str, CategoryMetrics]:
        """カテゴリ別の分析を実行"""
        
        category_data = {}
        
        # カテゴリ別にテスト結果をグループ化
        for result in test_results:
            # カテゴリの推定（クエリ内容から）
            category = self._categorize_result(result)
            
            if category not in category_data:
                category_data[category] = []
            
            category_data[category].append(result)
        
        # 各カテゴリのメトリクスを計算
        category_metrics = {}
        
        for category, results in category_data.items():
            total = len(results)
            successful = sum(1 for r in results if r.get("passed", False))
            
            confidences = [r.get("confidence", 0.0) for r in results]
            times = [r.get("processing_time", 0.0) for r in results]
            
            # 共通的な失敗パターンを特定
            failed_results = [r for r in results if not r.get("passed", False)]
            common_failures = self._identify_common_failures(failed_results)
            
            metrics = CategoryMetrics(
                category=category,
                total_queries=total,
                successful_queries=successful,
                success_rate=successful / total if total > 0 else 0.0,
                average_confidence=statistics.mean(confidences) if confidences else 0.0,
                average_response_time=statistics.mean(times) if times else 0.0,
                common_failures=common_failures
            )
            
            category_metrics[category] = metrics
        
        return category_metrics
    
    def _categorize_result(self, result: Dict[str, Any]) -> str:
        """テスト結果をカテゴリ分類"""
        
        query = result.get("query", "").lower()
        
        if any(word in query for word in ["とは", "何ですか", "どのような"]):
            return "definition"
        elif any(word in query for word in ["方法", "手順", "やり方"]):
            return "how_to"
        elif any(word in query for word in ["なぜ", "理由", "原因"]):
            return "why"
        elif any(word in query for word in ["比較", "違い", "差"]):
            return "comparison"
        elif any(word in query for word in ["天気", "料理", "スポーツ"]):
            return "negative"
        else:
            return "general"
    
    def _identify_common_failures(self, failed_results: List[Dict[str, Any]]) -> List[str]:
        """失敗パターンの共通点を特定"""
        
        failure_patterns = []
        
        if not failed_results:
            return failure_patterns
        
        # 低信頼度
        low_confidence_count = sum(1 for r in failed_results if r.get("confidence", 1.0) < 0.3)
        if low_confidence_count > len(failed_results) * 0.5:
            failure_patterns.append("低信頼度")
        
        # 長い応答時間
        slow_response_count = sum(1 for r in failed_results if r.get("processing_time", 0.0) > 2.0)
        if slow_response_count > len(failed_results) * 0.3:
            failure_patterns.append("応答時間遅延")
        
        # ソース不足
        no_source_count = sum(1 for r in failed_results if r.get("sources_found", 1) == 0)
        if no_source_count > len(failed_results) * 0.4:
            failure_patterns.append("関連ソース不足")
        
        return failure_patterns
    
    def _analyze_failures(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """詳細な失敗分析"""
        
        failed_results = [r for r in test_results if not r.get("passed", False)]
        
        if not failed_results:
            return {"total_failures": 0, "failure_rate": 0.0}
        
        failure_analysis = {
            "total_failures": len(failed_results),
            "failure_rate": len(failed_results) / len(test_results),
            "common_patterns": self._identify_common_failures(failed_results),
            "failure_categories": {},
            "improvement_suggestions": []
        }
        
        # カテゴリ別失敗率
        for category in ["definition", "how_to", "why", "comparison", "negative", "general"]:
            category_results = [r for r in test_results if self._categorize_result(r) == category]
            category_failures = [r for r in category_results if not r.get("passed", False)]
            
            if category_results:
                failure_analysis["failure_categories"][category] = {
                    "total": len(category_results),
                    "failures": len(category_failures),
                    "failure_rate": len(category_failures) / len(category_results)
                }
        
        # 改善提案
        if "低信頼度" in failure_analysis["common_patterns"]:
            failure_analysis["improvement_suggestions"].append("モデルの信頼度校正を改善")
        
        if "応答時間遅延" in failure_analysis["common_patterns"]:
            failure_analysis["improvement_suggestions"].append("検索インデックスの最適化")
        
        if "関連ソース不足" in failure_analysis["common_patterns"]:
            failure_analysis["improvement_suggestions"].append("コーパスの拡充または検索戦略の改善")
        
        return failure_analysis
    
    def _compute_overall_score(self, metrics: EvaluationMetrics) -> float:
        """重み付きメトリクスから総合スコアを計算"""
        
        weights = self.config.metric_weights
        
        score = (
            metrics.accuracy * weights.get("accuracy", 0.3) +
            (1.0 - min(metrics.average_response_time / self.config.response_time_threshold, 1.0)) * weights.get("response_time", 0.2) +
            metrics.average_confidence * weights.get("confidence", 0.2) +
            metrics.source_accuracy * weights.get("source_accuracy", 0.15) +
            metrics.coverage * weights.get("coverage", 0.15)
        )
        
        return min(max(score, 0.0), 1.0)  # 0-1の範囲に正規化
    
    def _generate_recommendations(self, metrics: EvaluationMetrics) -> List[str]:
        """メトリクスに基づく改善提案を生成"""
        
        recommendations = []
        
        if metrics.accuracy < self.config.accuracy_threshold:
            recommendations.append("システム精度の改善が必要です")
        
        if metrics.average_response_time > self.config.response_time_threshold:
            recommendations.append("応答時間の最適化を検討してください")
        
        if metrics.average_confidence < self.config.confidence_threshold:
            recommendations.append("モデルの信頼度校正を改善してください")
        
        if metrics.coverage < 0.7:
            recommendations.append("テストケースの多様性を増やしてください")
        
        if metrics.consistency < 0.8:
            recommendations.append("結果の一貫性向上が必要です")
        
        return recommendations
    
    def _format_evaluation_report(
        self, 
        metrics: EvaluationMetrics, 
        category_analysis: Dict[str, CategoryMetrics],
        failure_analysis: Dict[str, Any]
    ) -> str:
        """評価レポートをフォーマット"""
        
        if self.config.output_format == "json":
            return self._format_json_report(metrics, category_analysis, failure_analysis)
        else:
            return self._format_markdown_report(metrics, category_analysis, failure_analysis)
    
    def _format_markdown_report(
        self, 
        metrics: EvaluationMetrics, 
        category_analysis: Dict[str, CategoryMetrics],
        failure_analysis: Dict[str, Any]
    ) -> str:
        """Markdown形式の評価レポート"""
        
        lines = ["# RAGシステム評価レポート\n"]
        
        # 総合スコア
        overall_score = self._compute_overall_score(metrics)
        lines.append(f"## 📊 総合評価: {overall_score:.2f}/1.00")
        
        if overall_score >= 0.8:
            lines.append("🌟 **評価: 優秀** - システムは高品質で安定しています")
        elif overall_score >= 0.6:
            lines.append("👍 **評価: 良好** - 一部改善の余地があります")
        elif overall_score >= 0.4:
            lines.append("📈 **評価: 改善必要** - 重要な問題があります")
        else:
            lines.append("🔧 **評価: 要大幅改善** - システムの根本的な見直しが必要")
        
        lines.append("")
        
        # 主要メトリクス
        lines.append("## 📈 主要メトリクス")
        lines.append(f"- **精度 (Accuracy)**: {metrics.accuracy:.1%}")
        lines.append(f"- **適合率 (Precision)**: {metrics.precision:.1%}")
        lines.append(f"- **再現率 (Recall)**: {metrics.recall:.1%}")
        lines.append(f"- **F1スコア**: {metrics.f1_score:.3f}")
        lines.append(f"- **平均信頼度**: {metrics.average_confidence:.3f}")
        lines.append(f"- **平均応答時間**: {metrics.average_response_time:.3f}秒")
        lines.append(f"- **ソース精度**: {metrics.source_accuracy:.1%}")
        lines.append(f"- **カバレッジ**: {metrics.coverage:.1%}")
        lines.append(f"- **一貫性**: {metrics.consistency:.1%}")
        lines.append(f"- **推定ユーザー満足度**: {metrics.user_satisfaction:.1%}")
        lines.append("")
        
        # カテゴリ別分析
        if category_analysis:
            lines.append("## 📊 カテゴリ別分析")
            for category, cat_metrics in category_analysis.items():
                lines.append(f"### {category.title()}")
                lines.append(f"- 総クエリ数: {cat_metrics.total_queries}")
                lines.append(f"- 成功率: {cat_metrics.success_rate:.1%}")
                lines.append(f"- 平均信頼度: {cat_metrics.average_confidence:.3f}")
                lines.append(f"- 平均応答時間: {cat_metrics.average_response_time:.3f}秒")
                if cat_metrics.common_failures:
                    lines.append(f"- 共通失敗パターン: {', '.join(cat_metrics.common_failures)}")
                lines.append("")
        
        # 失敗分析
        if failure_analysis.get("total_failures", 0) > 0:
            lines.append("## ⚠️ 失敗分析")
            lines.append(f"- 総失敗数: {failure_analysis['total_failures']}")
            lines.append(f"- 失敗率: {failure_analysis['failure_rate']:.1%}")
            
            if failure_analysis.get("common_patterns"):
                lines.append(f"- 共通パターン: {', '.join(failure_analysis['common_patterns'])}")
            
            if failure_analysis.get("improvement_suggestions"):
                lines.append("### 🔧 改善提案")
                for suggestion in failure_analysis["improvement_suggestions"]:
                    lines.append(f"- {suggestion}")
                lines.append("")
        
        # 閾値との比較
        lines.append("## 🎯 閾値との比較")
        accuracy_status = "✅" if metrics.accuracy >= self.config.accuracy_threshold else "❌"
        response_status = "✅" if metrics.average_response_time <= self.config.response_time_threshold else "❌"
        confidence_status = "✅" if metrics.average_confidence >= self.config.confidence_threshold else "❌"
        
        lines.append(f"- 精度閾値 ({self.config.accuracy_threshold:.1%}): {accuracy_status}")
        lines.append(f"- 応答時間閾値 ({self.config.response_time_threshold}秒): {response_status}")
        lines.append(f"- 信頼度閾値 ({self.config.confidence_threshold:.1f}): {confidence_status}")
        
        return "\n".join(lines)
    
    def _format_json_report(
        self, 
        metrics: EvaluationMetrics, 
        category_analysis: Dict[str, CategoryMetrics],
        failure_analysis: Dict[str, Any]
    ) -> str:
        """JSON形式の評価レポート"""
        
        report = {
            "overall_score": self._compute_overall_score(metrics),
            "metrics": metrics.model_dump(),
            "category_analysis": {k: v.model_dump() for k, v in category_analysis.items()},
            "failure_analysis": failure_analysis,
            "recommendations": self._generate_recommendations(metrics),
            "thresholds": {
                "accuracy": self.config.accuracy_threshold,
                "response_time": self.config.response_time_threshold,
                "confidence": self.config.confidence_threshold
            }
        }
        
        return json.dumps(report, ensure_ascii=False, indent=2)
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """サマリーメトリクスを取得"""
        
        if not self.computed_metrics:
            return {}
        
        return {
            "overall_score": self._compute_overall_score(self.computed_metrics),
            "accuracy": self.computed_metrics.accuracy,
            "f1_score": self.computed_metrics.f1_score,
            "average_confidence": self.computed_metrics.average_confidence,
            "average_response_time": self.computed_metrics.average_response_time,
            "user_satisfaction": self.computed_metrics.user_satisfaction
        }