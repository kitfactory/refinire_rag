# チュートリアル5: 企業部門別RAGシステム

## 概要

このチュートリアルでは、企業環境での部門別RAGシステムの実装方法を説明します。人事部と営業部に独立したRAGインスタンスを作成し、データの分離と部門特有の知識アクセスを確保します。

## シナリオ

**企業**: TechCorp株式会社
- **人事部**: 従業員ポリシー、福利厚生、研修資料へのアクセスが必要
- **営業部**: 製品情報、営業プロセス、顧客データへのアクセスが必要

**要件**:
- 部門間でのデータ分離
- 部門固有のドキュメントアクセス
- 部門ごとの独立したRAGシステム
- 共有インフラだが分離されたデータ

## システム構成

```
TechCorp RAGシステム
├── 人事部 RAG
│   ├── 人事部ドキュメントストア
│   ├── 人事部ベクトルストア
│   └── 人事部クエリエンジン
└── 営業部 RAG
    ├── 営業部ドキュメントストア
    ├── 営業部ベクトルストア
    └── 営業部クエリエンジン
```

## 実装手順

### ステップ1: 部門別ドキュメントストアの設定

各部門が独自の分離されたドキュメントストアとベクトルストアを持ちます：

```python
# 人事部セットアップ
hr_document_store = SQLiteDocumentStore("data/hr/hr_documents.db")
hr_vector_store = InMemoryVectorStore()

# 営業部セットアップ
sales_document_store = SQLiteDocumentStore("data/sales/sales_documents.db")
sales_vector_store = InMemoryVectorStore()
```

### ステップ2: 部門別コーパスマネージャーの設定

```python
# 人事部コーパスマネージャー
hr_config = CorpusManagerConfig(
    enable_processing=True,
    enable_chunking=True,
    enable_embedding=True,
    document_store=hr_document_store,
    vector_store=hr_vector_store
)
hr_corpus_manager = CorpusManager(config=hr_config)

# 営業部コーパスマネージャー
sales_config = CorpusManagerConfig(
    enable_processing=True,
    enable_chunking=True,
    enable_embedding=True,
    document_store=sales_document_store,
    vector_store=sales_vector_store
)
sales_corpus_manager = CorpusManager(config=sales_config)
```

### ステップ3: 部門別クエリエンジンの設定

```python
# 人事部クエリエンジン
hr_retriever = SimpleRetriever(vector_store=hr_vector_store, embedder=hr_embedder)
hr_query_engine = QueryEngine(
    document_store=hr_document_store,
    vector_store=hr_vector_store,
    retriever=hr_retriever,
    reader=SimpleReader(),
    reranker=SimpleReranker()
)

# 営業部クエリエンジン
sales_retriever = SimpleRetriever(vector_store=sales_vector_store, embedder=sales_embedder)
sales_query_engine = QueryEngine(
    document_store=sales_document_store,
    vector_store=sales_vector_store,
    retriever=sales_retriever,
    reader=SimpleReader(),
    reranker=SimpleReranker()
)
```

## 使用例

### 人事部のクエリ

```python
# 従業員が有給休暇ポリシーについて質問
hr_response = hr_query_engine.answer("有給休暇のポリシーはどうなっていますか？")
print(f"人事部回答: {hr_response.answer}")

# マネージャーが人事評価プロセスについて質問
hr_response = hr_query_engine.answer("人事評価はどのように実施すればよいですか？")
print(f"人事部回答: {hr_response.answer}")
```

### 営業部のクエリ

```python
# 営業担当者が製品機能について質問
sales_response = sales_query_engine.answer("当社の企業向け製品の主要機能は何ですか？")
print(f"営業部回答: {sales_response.answer}")

# 営業マネージャーが価格戦略について質問
sales_response = sales_query_engine.answer("新規顧客に対する価格戦略はどのようなものですか？")
print(f"営業部回答: {sales_response.answer}")
```

## データ分離の確認

```python
# 人事部が営業部データにアクセスできないことを確認
hr_response = hr_query_engine.answer("当社の製品価格はいくらですか？")
# 結果: "関連する情報が見つかりませんでした"

# 営業部が人事部データにアクセスできないことを確認
sales_response = sales_query_engine.answer("有給休暇のポリシーは？")
# 結果: "関連する情報が見つかりませんでした"
```

## 部門別品質監視

```python
# 部門別品質監視の設定
hr_quality_lab = QualityLab(
    test_suite=TestSuite(),
    evaluator=Evaluator(),
    contradiction_detector=ContradictionDetector(),
    insight_reporter=InsightReporter()
)

sales_quality_lab = QualityLab(
    test_suite=TestSuite(),
    evaluator=Evaluator(),
    contradiction_detector=ContradictionDetector(),
    insight_reporter=InsightReporter()
)
```

## ベストプラクティス

### 1. アクセス制御
```python
class DepartmentRAGManager:
    def __init__(self, department: str, user_department: str):
        self.department = department
        self.user_department = user_department
        
    def query(self, question: str):
        if self.user_department != self.department:
            raise PermissionError(f"{self.user_department}のユーザーは{self.department}のデータにアクセスできません")
        return self.query_engine.answer(question)
```

### 2. 監査ログ
```python
def log_query(user_id: str, department: str, query: str, response: str):
    logging.info({
        "timestamp": datetime.now(),
        "user_id": user_id,
        "department": department,
        "query": query,
        "response_confidence": response.confidence,
        "sources_used": len(response.sources)
    })
```

### 3. 定期的な品質チェック
```python
# 部門別週次品質評価
def weekly_quality_check(department_rag: QueryEngine, department_name: str):
    test_queries = load_department_test_queries(department_name)
    results = []
    
    for query in test_queries:
        result = department_rag.answer(query)
        results.append(result)
    
    # 品質レポート生成
    quality_report = generate_quality_report(results, department_name)
    send_to_department_admin(quality_report, department_name)
```

## スケーリングの考慮事項

### マルチテナントアーキテクチャ
```python
class EnterpriseRAGSystem:
    def __init__(self):
        self.departments = {}
    
    def add_department(self, dept_name: str, config: CorpusManagerConfig):
        self.departments[dept_name] = {
            'corpus_manager': CorpusManager(config),
            'query_engine': self._create_query_engine(config),
            'quality_lab': self._create_quality_lab()
        }
    
    def get_department_rag(self, dept_name: str):
        return self.departments.get(dept_name)
```

### リソース管理
```python
# 効率性のための共有エンベッダー、分離されたデータストア
shared_embedder = TFIDFEmbedder()

hr_config = CorpusManagerConfig(embedder=shared_embedder, ...)
sales_config = CorpusManagerConfig(embedder=shared_embedder, ...)
```

## セキュリティ考慮事項

1. **データ暗号化**: 保存時のドキュメントストア暗号化
2. **アクセスログ**: 部門別の詳細なアクセスログ維持
3. **ネットワーク分離**: 機密部門にはVPNまたはネットワークセグメント使用
4. **定期監査**: 部門データアクセスの定期的なセキュリティ監査

## 監視・アラート

```python
# 部門別使用状況監視
def monitor_department_usage(dept_name: str):
    rag_system = get_department_rag(dept_name)
    
    # クエリ量監視
    daily_queries = get_daily_query_count(dept_name)
    if daily_queries > threshold:
        alert_department_admin(f"{dept_name}で高いクエリ量を検出")
    
    # 回答品質監視
    avg_confidence = get_average_confidence(dept_name)
    if avg_confidence < 0.5:
        alert_department_admin(f"{dept_name}で低い回答品質を検出")
```

## 次のステップ

- **チュートリアル6**: 高度なマルチモーダルRAG（画像、文書、動画）
- **チュートリアル7**: リアルタイム学習と適応
- **チュートリアル8**: パフォーマンス最適化とスケーリング
- **チュートリアル9**: 本番デプロイとDevOps

この企業設定により、各部門は共有インフラの恩恵を受けながら、データプライバシーを維持し、一貫したRAG機能を活用できます。