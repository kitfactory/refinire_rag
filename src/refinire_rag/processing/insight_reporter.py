"""
InsightReporter - Threshold-based Interpretation and Reporting

閾値ベースの解釈とレポート生成を行うDocumentProcessor。
評価結果を分析し、実用的なインサイトと推奨事項を生成します。
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from dataclasses import field
from enum import Enum
import statistics
import json
from pathlib import Path
from datetime import datetime

from .document_processor import DocumentProcessor, DocumentProcessorConfig
from ..models.document import Document


class InsightType(str, Enum):
    """インサイトタイプ"""
    PERFORMANCE = "performance"        # 性能に関するインサイト
    QUALITY = "quality"               # 品質に関するインサイト
    CONSISTENCY = "consistency"       # 一貫性に関するインサイト
    EFFICIENCY = "efficiency"         # 効率性に関するインサイト
    RELIABILITY = "reliability"       # 信頼性に関するインサイト
    SCALABILITY = "scalability"       # スケーラビリティに関するインサイト


class Insight(BaseModel):
    """インサイトモデル"""
    
    id: str
    insight_type: InsightType
    title: str
    description: str
    severity: str  # "critical", "high", "medium", "low", "info"
    confidence: float
    affected_metrics: List[str]
    recommendations: List[str]
    supporting_data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Threshold(BaseModel):
    """閾値設定"""
    
    metric_name: str
    critical_threshold: Optional[float] = None
    warning_threshold: Optional[float] = None
    target_threshold: Optional[float] = None
    comparison_operator: str = "greater_than"  # "greater_than", "less_than", "equals"


class InsightReporterConfig(DocumentProcessorConfig):
    """InsightReporter設定"""
    
    # インサイト生成設定
    enable_trend_analysis: bool = True
    enable_comparative_analysis: bool = True
    enable_root_cause_analysis: bool = True
    min_confidence_for_insight: float = 0.6
    
    # レポート設定
    include_executive_summary: bool = True
    include_detailed_analysis: bool = True
    include_action_items: bool = True
    report_format: str = "markdown"  # "markdown", "json", "html"


class InsightReporter(DocumentProcessor):
    """
    インサイトレポーター
    
    評価結果を分析し、閾値ベースの解釈とインサイトを生成します。
    """
    
    @classmethod
    def get_config_class(cls):
        return InsightReporterConfig
    
    def __init__(self, config: InsightReporterConfig):
        super().__init__(config)
        self.generated_insights: List[Insight] = []
        self.all_thresholds = self._initialize_thresholds()
        self.historical_data: List[Dict[str, Any]] = []
    
    def process(self, document: Document) -> List[Document]:
        """
        評価結果ドキュメントからインサイトを生成
        
        Args:
            document: 評価結果ドキュメント
            
        Returns:
            List[Document]: インサイトレポートドキュメント
        """
        # ドキュメントが評価結果かチェック
        if not self._is_evaluation_document(document):
            return [document]  # 評価結果でなければそのまま通す
        
        # メトリクスを抽出
        metrics = self._extract_metrics(document)
        
        # 履歴データに追加
        self.historical_data.append({
            "timestamp": datetime.now().isoformat(),
            "document_id": document.id,
            "metrics": metrics
        })
        
        # インサイトを生成
        insights = self._generate_insights(metrics, document)
        self.generated_insights.extend(insights)
        
        # インサイトレポートを作成
        report_doc = Document(
            id=f"insight_report_{document.id}",
            content=self._format_insight_report(insights, metrics),
            metadata={
                "processing_stage": "insight_reporting",
                "source_document_id": document.id,
                "insights_generated": len(insights),
                "critical_insights": len([i for i in insights if i.severity == "critical"]),
                "overall_health_score": self._compute_health_score(metrics),
                "recommendation_count": sum(len(i.recommendations) for i in insights)
            }
        )
        
        return [report_doc]
    
    def _initialize_thresholds(self) -> List[Threshold]:
        """閾値設定を初期化"""
        
        # デフォルトの閾値を作成
        thresholds = [
            Threshold(
                metric_name="accuracy",
                critical_threshold=0.5,
                warning_threshold=0.7,
                target_threshold=0.9,
                comparison_operator="greater_than"
            ),
            Threshold(
                metric_name="response_time",
                critical_threshold=5.0,
                warning_threshold=2.0,
                target_threshold=1.0,
                comparison_operator="less_than"
            ),
            Threshold(
                metric_name="confidence",
                critical_threshold=0.3,
                warning_threshold=0.5,
                target_threshold=0.8,
                comparison_operator="greater_than"
            )
        ]
        
        return thresholds
    
    def _is_evaluation_document(self, document: Document) -> bool:
        """ドキュメントが評価結果かチェック"""
        
        processing_stage = document.metadata.get("processing_stage", "")
        return processing_stage in ["evaluation", "test_results", "contradiction_detection"]
    
    def _extract_metrics(self, document: Document) -> Dict[str, float]:
        """ドキュメントからメトリクスを抽出"""
        
        metrics = {}
        
        # メタデータからメトリクスを抽出
        metadata = document.metadata
        
        # 一般的なメトリクス
        if "overall_score" in metadata:
            metrics["overall_score"] = float(metadata["overall_score"])
        
        if "success_rate" in metadata:
            metrics["accuracy"] = float(metadata["success_rate"])
        
        if "processing_time" in metadata:
            metrics["response_time"] = float(metadata["processing_time"])
        
        if "average_confidence" in metadata:
            metrics["confidence"] = float(metadata["average_confidence"])
        
        # コンテンツからメトリクスを解析
        content = document.content
        content_metrics = self._parse_metrics_from_content(content)
        metrics.update(content_metrics)
        
        return metrics
    
    def _parse_metrics_from_content(self, content: str) -> Dict[str, float]:
        """コンテンツからメトリクスを解析"""
        
        metrics = {}
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 精度情報
            if "精度" in line and ":" in line:
                try:
                    value_str = line.split(':')[1].replace('%', '').strip()
                    metrics["accuracy"] = float(value_str) / 100 if '%' in line else float(value_str)
                except:
                    pass
            
            # F1スコア
            elif "F1スコア" in line and ":" in line:
                try:
                    value_str = line.split(':')[1].strip()
                    metrics["f1_score"] = float(value_str)
                except:
                    pass
            
            # 応答時間
            elif "応答時間" in line and ":" in line:
                try:
                    value_str = line.split(':')[1].replace('秒', '').strip()
                    metrics["response_time"] = float(value_str)
                except:
                    pass
            
            # 信頼度
            elif "信頼度" in line and ":" in line:
                try:
                    value_str = line.split(':')[1].strip()
                    metrics["confidence"] = float(value_str)
                except:
                    pass
            
            # 一貫性スコア
            elif "一貫性" in line and ":" in line:
                try:
                    value_str = line.split(':')[1].replace('%', '').strip()
                    metrics["consistency"] = float(value_str) / 100 if '%' in line else float(value_str)
                except:
                    pass
        
        return metrics
    
    def _generate_insights(self, metrics: Dict[str, float], document: Document) -> List[Insight]:
        """メトリクスからインサイトを生成"""
        
        insights = []
        
        # 閾値ベースのインサイト生成
        threshold_insights = self._generate_threshold_insights(metrics)
        insights.extend(threshold_insights)
        
        # トレンド分析インサイト
        if self.config.enable_trend_analysis and len(self.historical_data) > 1:
            trend_insights = self._generate_trend_insights(metrics)
            insights.extend(trend_insights)
        
        # 比較分析インサイト
        if self.config.enable_comparative_analysis:
            comparative_insights = self._generate_comparative_insights(metrics)
            insights.extend(comparative_insights)
        
        # 根本原因分析インサイト
        if self.config.enable_root_cause_analysis:
            root_cause_insights = self._generate_root_cause_insights(metrics, document)
            insights.extend(root_cause_insights)
        
        # 信頼度フィルタリング
        filtered_insights = [i for i in insights if i.confidence >= self.config.min_confidence_for_insight]
        
        return filtered_insights
    
    def _generate_threshold_insights(self, metrics: Dict[str, float]) -> List[Insight]:
        """閾値ベースのインサイトを生成"""
        
        insights = []
        
        for threshold in self.all_thresholds:
            metric_name = threshold.metric_name
            if metric_name not in metrics:
                continue
            
            metric_value = metrics[metric_name]
            insight = self._evaluate_threshold(metric_name, metric_value, threshold)
            
            if insight:
                insights.append(insight)
        
        return insights
    
    def _evaluate_threshold(self, metric_name: str, value: float, threshold: Threshold) -> Optional[Insight]:
        """個別閾値の評価"""
        
        severity = "info"
        title = ""
        description = ""
        recommendations = []
        
        is_greater_than = threshold.comparison_operator == "greater_than"
        
        # 重要度判定
        if threshold.critical_threshold is not None:
            if is_greater_than:
                if value < threshold.critical_threshold:
                    severity = "critical"
            else:
                if value > threshold.critical_threshold:
                    severity = "critical"
        
        if severity != "critical" and threshold.warning_threshold is not None:
            if is_greater_than:
                if value < threshold.warning_threshold:
                    severity = "high"
            else:
                if value > threshold.warning_threshold:
                    severity = "high"
        
        if severity not in ["critical", "high"] and threshold.target_threshold is not None:
            if is_greater_than:
                if value < threshold.target_threshold:
                    severity = "medium"
                else:
                    severity = "low"  # 目標達成
            else:
                if value > threshold.target_threshold:
                    severity = "medium"
                else:
                    severity = "low"  # 目標達成
        
        # インサイト内容の生成
        if severity == "critical":
            title = f"{metric_name}が臨界レベルに達しています"
            description = f"{metric_name}の値({value:.3f})が臨界閾値を下回っており、緊急対応が必要です。"
            recommendations = [
                "緊急対応チームを招集してください",
                "システムの根本的な見直しを実施してください",
                "一時的にサービスを停止することを検討してください"
            ]
        elif severity == "high":
            title = f"{metric_name}に重要な問題があります"
            description = f"{metric_name}の値({value:.3f})が警告レベルにあり、早急な改善が必要です。"
            recommendations = [
                "改善計画を立案してください",
                "関連する設定やパラメータを見直してください",
                "監視を強化してください"
            ]
        elif severity == "medium":
            title = f"{metric_name}の改善余地があります"
            description = f"{metric_name}の値({value:.3f})は許容範囲内ですが、目標値に到達していません。"
            recommendations = [
                "最適化の機会を探してください",
                "ベストプラクティスを適用してください"
            ]
        elif severity == "low":
            title = f"{metric_name}は良好です"
            description = f"{metric_name}の値({value:.3f})は目標を達成しています。"
            recommendations = [
                "現在の設定を維持してください",
                "他のメトリクスの改善に集中してください"
            ]
        
        if severity in ["critical", "high", "medium"]:  # info レベルはスキップ
            insight = Insight(
                id=f"threshold_{metric_name}_{severity}",
                insight_type=self._classify_insight_type(metric_name),
                title=title,
                description=description,
                severity=severity,
                confidence=0.9,  # 閾値ベースは高信頼度
                affected_metrics=[metric_name],
                recommendations=recommendations,
                supporting_data={
                    "current_value": value,
                    "threshold_config": threshold.dict(),
                    "target_met": severity == "low"
                }
            )
            return insight
        
        return None
    
    def _classify_insight_type(self, metric_name: str) -> InsightType:
        """メトリクス名からインサイトタイプを分類"""
        
        type_mapping = {
            "accuracy": InsightType.QUALITY,
            "precision": InsightType.QUALITY,
            "recall": InsightType.QUALITY,
            "f1_score": InsightType.QUALITY,
            "response_time": InsightType.PERFORMANCE,
            "processing_time": InsightType.EFFICIENCY,
            "confidence": InsightType.RELIABILITY,
            "consistency": InsightType.CONSISTENCY,
            "throughput": InsightType.SCALABILITY,
            "error_rate": InsightType.RELIABILITY
        }
        
        return type_mapping.get(metric_name, InsightType.PERFORMANCE)
    
    def _generate_trend_insights(self, current_metrics: Dict[str, float]) -> List[Insight]:
        """トレンド分析インサイトを生成"""
        
        insights = []
        
        if len(self.historical_data) < 2:
            return insights
        
        # 過去のデータと比較
        prev_metrics = self.historical_data[-2]["metrics"]
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in prev_metrics:
                prev_value = prev_metrics[metric_name]
                change_rate = (current_value - prev_value) / prev_value if prev_value != 0 else 0
                
                # 有意な変化があった場合のみインサイト生成
                if abs(change_rate) > 0.1:  # 10%以上の変化
                    insight = self._create_trend_insight(metric_name, current_value, prev_value, change_rate)
                    if insight:
                        insights.append(insight)
        
        return insights
    
    def _create_trend_insight(self, metric_name: str, current: float, previous: float, change_rate: float) -> Optional[Insight]:
        """トレンドインサイトを作成"""
        
        is_improvement = self._is_improvement(metric_name, change_rate)
        direction = "改善" if is_improvement else "悪化"
        severity = "low" if is_improvement else ("high" if abs(change_rate) > 0.3 else "medium")
        
        title = f"{metric_name}が{direction}しています"
        description = f"{metric_name}が前回の{previous:.3f}から{current:.3f}に変化しました（{change_rate:+.1%}）。"
        
        recommendations = []
        if is_improvement:
            recommendations = [
                "良好な傾向を維持してください",
                "成功要因を分析して他の領域に応用してください"
            ]
        else:
            recommendations = [
                "悪化の原因を特定してください",
                "対策を講じて改善を図ってください",
                "モニタリングを強化してください"
            ]
        
        insight = Insight(
            id=f"trend_{metric_name}_{direction}",
            insight_type=self._classify_insight_type(metric_name),
            title=title,
            description=description,
            severity=severity,
            confidence=0.8,
            affected_metrics=[metric_name],
            recommendations=recommendations,
            supporting_data={
                "current_value": current,
                "previous_value": previous,
                "change_rate": change_rate,
                "is_improvement": is_improvement
            }
        )
        
        return insight
    
    def _is_improvement(self, metric_name: str, change_rate: float) -> bool:
        """変化が改善かどうかを判定"""
        
        # 値が大きい方が良いメトリクス
        higher_is_better = ["accuracy", "precision", "recall", "f1_score", "confidence", "consistency", "throughput"]
        
        # 値が小さい方が良いメトリクス
        lower_is_better = ["response_time", "processing_time", "error_rate"]
        
        if metric_name in higher_is_better:
            return change_rate > 0
        elif metric_name in lower_is_better:
            return change_rate < 0
        else:
            return change_rate > 0  # デフォルトは大きい方が良い
    
    def _generate_comparative_insights(self, metrics: Dict[str, float]) -> List[Insight]:
        """比較分析インサイトを生成"""
        
        insights = []
        
        # メトリクス間の相関分析
        if "accuracy" in metrics and "confidence" in metrics:
            accuracy = metrics["accuracy"]
            confidence = metrics["confidence"]
            
            # 精度と信頼度の不一致を検出
            if abs(accuracy - confidence) > 0.3:
                insight = Insight(
                    id="accuracy_confidence_mismatch",
                    insight_type=InsightType.RELIABILITY,
                    title="精度と信頼度に大きな差があります",
                    description=f"精度({accuracy:.3f})と信頼度({confidence:.3f})の間に大きな差があります。",
                    severity="medium",
                    confidence=0.7,
                    affected_metrics=["accuracy", "confidence"],
                    recommendations=[
                        "信頼度校正の見直しを検討してください",
                        "モデルの不確実性推定を改善してください"
                    ],
                    supporting_data={
                        "accuracy": accuracy,
                        "confidence": confidence,
                        "difference": abs(accuracy - confidence)
                    }
                )
                insights.append(insight)
        
        # 性能と品質のトレードオフ分析
        if "response_time" in metrics and "accuracy" in metrics:
            response_time = metrics["response_time"]
            accuracy = metrics["accuracy"]
            
            # 高精度だが遅い場合
            if accuracy > 0.8 and response_time > 2.0:
                insight = Insight(
                    id="accuracy_speed_tradeoff",
                    insight_type=InsightType.EFFICIENCY,
                    title="高精度ですが応答速度が遅いです",
                    description="高い精度を維持していますが、応答時間が目標を上回っています。",
                    severity="medium",
                    confidence=0.8,
                    affected_metrics=["accuracy", "response_time"],
                    recommendations=[
                        "キャッシュの導入を検討してください",
                        "モデルの軽量化を検討してください",
                        "並列処理の最適化を行ってください"
                    ]
                )
                insights.append(insight)
        
        return insights
    
    def _generate_root_cause_insights(self, metrics: Dict[str, float], document: Document) -> List[Insight]:
        """根本原因分析インサイトを生成"""
        
        insights = []
        
        # 低性能の根本原因を分析
        if metrics.get("accuracy", 1.0) < 0.6:
            # ドキュメントメタデータから原因を推測
            metadata = document.metadata
            
            potential_causes = []
            
            # データ品質問題
            if metadata.get("contradictions_found", 0) > 0:
                potential_causes.append("データの矛盾が検出されています")
            
            # 処理問題
            if metadata.get("errors_encountered", 0) > 0:
                potential_causes.append("処理エラーが発生しています")
            
            # カバレッジ問題
            if metrics.get("coverage", 1.0) < 0.5:
                potential_causes.append("テストカバレッジが不十分です")
            
            if potential_causes:
                insight = Insight(
                    id="low_performance_root_cause",
                    insight_type=InsightType.QUALITY,
                    title="低性能の根本原因が特定されました",
                    description="システムの低性能について、以下の根本原因が考えられます。",
                    severity="high",
                    confidence=0.7,
                    affected_metrics=["accuracy"],
                    recommendations=[
                        "データクリーニングを実施してください",
                        "処理パイプラインを見直してください",
                        "テストデータを拡充してください"
                    ],
                    supporting_data={
                        "potential_causes": potential_causes,
                        "evidence": metadata
                    }
                )
                insights.append(insight)
        
        return insights
    
    def _compute_health_score(self, metrics: Dict[str, float]) -> float:
        """システムヘルススコアを計算"""
        
        if not metrics:
            return 0.0
        
        # 重要メトリクスの重み
        weights = {
            "accuracy": 0.3,
            "response_time": 0.2,
            "confidence": 0.2,
            "consistency": 0.15,
            "f1_score": 0.15
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                # 正規化（0-1の範囲に）
                if metric_name == "response_time":
                    # 応答時間は小さい方が良い
                    normalized_value = max(0.0, 1.0 - min(value / 5.0, 1.0))
                else:
                    # その他は大きい方が良い
                    normalized_value = min(value, 1.0)
                
                score += normalized_value * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _format_insight_report(self, insights: List[Insight], metrics: Dict[str, float]) -> str:
        """インサイトレポートをフォーマット"""
        
        if self.config.report_format == "json":
            return self._format_json_report(insights, metrics)
        else:
            return self._format_markdown_report(insights, metrics)
    
    def _format_markdown_report(self, insights: List[Insight], metrics: Dict[str, float]) -> str:
        """Markdown形式のインサイトレポート"""
        
        lines = ["# システムインサイトレポート\n"]
        
        # エグゼクティブサマリー
        if self.config.include_executive_summary:
            lines.append("## 📋 エグゼクティブサマリー")
            
            health_score = self._compute_health_score(metrics)
            lines.append(f"**システムヘルススコア**: {health_score:.2f}/1.00")
            
            if health_score >= 0.8:
                lines.append("🟢 **状態**: 健全 - システムは良好に動作しています")
            elif health_score >= 0.6:
                lines.append("🟡 **状態**: 注意 - いくつかの改善領域があります")
            elif health_score >= 0.4:
                lines.append("🟠 **状態**: 警告 - 重要な問題があります")
            else:
                lines.append("🔴 **状態**: 危険 - 緊急対応が必要です")
            
            # 重要インサイトのサマリー
            critical_insights = [i for i in insights if i.severity == "critical"]
            high_insights = [i for i in insights if i.severity == "high"]
            
            if critical_insights:
                lines.append(f"⚠️ **緊急対応が必要**: {len(critical_insights)}件の臨界的問題")
            if high_insights:
                lines.append(f"🔥 **早急な対応が必要**: {len(high_insights)}件の重要問題")
            
            lines.append("")
        
        # キーメトリクス
        lines.append("## 📊 キーメトリクス")
        for metric_name, value in metrics.items():
            if metric_name == "response_time":
                lines.append(f"- **{metric_name}**: {value:.3f}秒")
            elif "rate" in metric_name or "accuracy" in metric_name:
                lines.append(f"- **{metric_name}**: {value:.1%}")
            else:
                lines.append(f"- **{metric_name}**: {value:.3f}")
        lines.append("")
        
        # インサイト別セクション
        if insights:
            lines.append("## 🔍 インサイト")
            
            # 重要度別に分類
            severity_order = ["critical", "high", "medium", "low"]
            severity_icons = {
                "critical": "🚨",
                "high": "⚠️",
                "medium": "📈",
                "low": "✅"
            }
            
            for severity in severity_order:
                severity_insights = [i for i in insights if i.severity == severity]
                
                if severity_insights:
                    lines.append(f"### {severity_icons[severity]} {severity.upper()} 重要度")
                    
                    for insight in severity_insights:
                        lines.append(f"#### {insight.title}")
                        lines.append(f"**カテゴリ**: {insight.insight_type.value}")
                        lines.append(f"**信頼度**: {insight.confidence:.1%}")
                        lines.append(f"**説明**: {insight.description}")
                        
                        if insight.recommendations:
                            lines.append("**推奨事項**:")
                            for rec in insight.recommendations:
                                lines.append(f"- {rec}")
                        
                        lines.append("")
        
        # アクションアイテム
        if self.config.include_action_items and insights:
            lines.append("## 🎯 アクションアイテム")
            
            # 優先度順にアクションアイテムを整理
            critical_actions = []
            high_actions = []
            medium_actions = []
            
            for insight in insights:
                for rec in insight.recommendations:
                    action_item = f"{rec} (関連: {insight.title})"
                    
                    if insight.severity == "critical":
                        critical_actions.append(action_item)
                    elif insight.severity == "high":
                        high_actions.append(action_item)
                    else:
                        medium_actions.append(action_item)
            
            if critical_actions:
                lines.append("### 🚨 緊急対応（24時間以内）")
                for action in critical_actions[:5]:  # 上位5件
                    lines.append(f"1. {action}")
                lines.append("")
            
            if high_actions:
                lines.append("### ⚠️ 重要対応（1週間以内）")
                for action in high_actions[:5]:  # 上位5件
                    lines.append(f"1. {action}")
                lines.append("")
            
            if medium_actions:
                lines.append("### 📈 改善機会（1か月以内）")
                for action in medium_actions[:5]:  # 上位5件
                    lines.append(f"1. {action}")
        
        return "\n".join(lines)
    
    def _format_json_report(self, insights: List[Insight], metrics: Dict[str, float]) -> str:
        """JSON形式のインサイトレポート"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "health_score": self._compute_health_score(metrics),
            "metrics": metrics,
            "insights": [insight.dict() for insight in insights],
            "summary": {
                "total_insights": len(insights),
                "critical_count": len([i for i in insights if i.severity == "critical"]),
                "high_count": len([i for i in insights if i.severity == "high"]),
                "medium_count": len([i for i in insights if i.severity == "medium"]),
                "low_count": len([i for i in insights if i.severity == "low"])
            }
        }
        
        return json.dumps(report, ensure_ascii=False, indent=2)
    
    def get_insight_summary(self) -> Dict[str, Any]:
        """インサイトサマリーを取得"""
        
        if not self.generated_insights:
            return {"message": "No insights generated yet"}
        
        severity_counts = {}
        type_counts = {}
        
        for insight in self.generated_insights:
            # 重要度別カウント
            severity = insight.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # タイプ別カウント
            insight_type = insight.insight_type.value
            type_counts[insight_type] = type_counts.get(insight_type, 0) + 1
        
        avg_confidence = statistics.mean([i.confidence for i in self.generated_insights])
        
        return {
            "total_insights": len(self.generated_insights),
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "average_confidence": avg_confidence,
            "recommendations_count": sum(len(i.recommendations) for i in self.generated_insights)
        }