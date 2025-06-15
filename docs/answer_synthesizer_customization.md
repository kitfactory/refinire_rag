# AnswerSynthesizer Instruction Customization

`AnswerSynthesizer`では、LLMに対する生成指示（generation instructions）とシステムプロンプト（system prompt）をユーザー側で完全にカスタマイズできます。これにより、異なるドメインやユースケースに応じて最適な回答生成を実現できます。

## 基本的な使用方法

### Refinire LLMPipeline使用時（推奨）

```python
from refinire_rag.retrieval.simple_reader import SimpleAnswerSynthesizer, SimpleAnswerSynthesizerConfig

# カスタム指示を設定
custom_instructions = """あなたは技術文書の専門家です。質問に回答する際は：
1. 正確で詳細な技術説明を提供する
2. 関連するコード例を含める
3. 見出しや箇条書きで明確に構造化する
4. ソースを具体的に引用する
5. 不完全な情報の場合は明確に制限を述べる"""

config = SimpleAnswerSynthesizerConfig(
    generation_instructions=custom_instructions,
    llm_model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=600
)

synthesizer = SimpleAnswerSynthesizer(config)
```

### OpenAI使用時（フォールバック）

```python
# OpenAI用のシステムプロンプトを設定
custom_system_prompt = """あなたは科学研究のアシスタントです。回答は：
- 学術的に厳密で正確
- 関連する引用や参考文献を含む
- 正式な科学的言語を使用
- 確立された事実と現在の研究を明確に区別
- 適切な場合は信頼度レベルを提供"""

config = SimpleAnswerSynthesizerConfig(
    system_prompt=custom_system_prompt,
    generation_instructions="Refinire用の指示（OpenAIでは使用されない）",
    llm_model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=800
)

synthesizer = SimpleAnswerSynthesizer(config)
```

## 設定可能なパラメータ

### SimpleAnswerSynthesizerConfig

| パラメータ | 型 | デフォルト値 | 説明 |
|-----------|-----|-------------|------|
| `generation_instructions` | str | "You are a helpful assistant..." | Refinire LLMPipeline用の生成指示 |
| `system_prompt` | str | "You are a helpful assistant..." | OpenAI用のシステムプロンプト |
| `llm_model` | str | 環境変数から自動設定 | 使用するLLMモデル |
| `temperature` | float | 0.1 | 生成の創造性レベル（0.0-1.0） |
| `max_tokens` | int | 500 | 生成する最大トークン数 |
| `max_context_length` | int | 2000 | 使用するコンテキストの最大長 |
| `openai_api_key` | str | None | OpenAI APIキー（オプション） |
| `openai_organization` | str | None | OpenAI組織ID（オプション） |

## ドメイン別のカスタマイズ例

### 1. カスタマーサービス

```python
customer_service_config = SimpleAnswerSynthesizerConfig(
    generation_instructions="""あなたは親しみやすく親切なカスタマーサービス担当者です：
    
    **対応スタイル：**
    - 共感的で理解を示す
    - 簡潔で分かりやすい言葉を使用
    - 解決志向のアプローチ
    - 前向きで励ましの姿勢
    - 追加のサポートを常に提供
    
    **回答形式：**
    - 顧客の状況を理解していることを示す
    - 段階的な解決手順を提供
    - 「承知いたしました」「喜んでお手伝いします」などの表現を使用
    - 他にお手伝いできることがないか尋ねて終了""",
    temperature=0.7,  # 会話的で親しみやすい
    max_tokens=300
)
```

### 2. 技術文書

```python
technical_config = SimpleAnswerSynthesizerConfig(
    generation_instructions="""あなたはソフトウェアエンジニアリングの技術文書専門家です：
    
    **回答要件：**
    - 正確で詳細な技術的説明
    - 関連するコード例を含める
    - 明確な構造（見出し、箇条書き、番号付きリスト）
    - 「なぜ」そうなるかの文脈説明
    - 公式文書や標準の参照
    
    **回答形式：**
    - 概要から開始
    - 例を含む詳細説明
    - 重要な警告や考慮事項
    - 参考文献や次のステップで終了""",
    temperature=0.2,  # 事実的で一貫性重視
    max_tokens=600,
    max_context_length=3000
)
```

### 3. 学術研究

```python
academic_config = SimpleAnswerSynthesizerConfig(
    generation_instructions="""あなたは科学文献専門の学術研究アシスタントです：
    
    **学術基準：**
    - すべての記述は証拠に基づく
    - 正式で正確な学術用語の使用
    - 具体的な研究や論文の引用
    - 論争がある場合は複数の観点を提示
    - 主張の確実性レベルを明示
    
    **回答構造：**
    - 研究結果の要約
    - 証拠に基づく詳細分析
    - 現在の知識の限界
    - 今後の研究提案""",
    temperature=0.1,  # 非常に事実的
    max_tokens=800,
    max_context_length=4000
)
```

### 4. 法律相談

```python
legal_config = SimpleAnswerSynthesizerConfig(
    generation_instructions="""あなたは法的概念の情報提供を行う法律調査アシスタントです：
    
    **法的回答要件：**
    - 正確な法律用語と概念の使用
    - 関連する法律、判例、規制の引用
    - 適用される法的システムの明示
    - 法的助言ではないことの明記
    - 論理的な法的推論パターン
    
    **免責事項：**
    - 教育目的のみの情報
    - 専門的法的助言の代替ではない
    - 法律は管轄区域により異なる
    - 具体的な事案は有資格弁護士に相談
    
    **法的文書スタイル：**
    - 正確な法律用語の使用
    - 権威に基づく論証
    - 客観的で分析的な語調""",
    temperature=0.15,  # 非常に正確で一貫性重視
    max_tokens=700,
    max_context_length=3500
)
```

## 実装時の注意点

### 1. Refinire vs OpenAI

- **Refinire使用時**: `generation_instructions`が使用される
- **OpenAI使用時**: `system_prompt`が使用される
- 両方設定することで、どちらの環境でも適切に動作

### 2. 温度設定の指針

| 用途 | 推奨温度 | 説明 |
|------|---------|------|
| 事実確認・法律・学術 | 0.1-0.2 | 高い精度と一貫性 |
| 技術文書・説明 | 0.2-0.3 | 正確性を保ちつつ明確な表現 |
| 一般的な質問応答 | 0.3-0.5 | バランスの取れた回答 |
| カスタマーサービス | 0.5-0.7 | 親しみやすく会話的 |
| 創作・アイデア生成 | 0.7-0.9 | 高い創造性と多様性 |

### 3. トークン数の設定

- **短い回答**: 200-300トークン（FAQ、簡単な説明）
- **標準回答**: 400-600トークン（一般的な質問応答）
- **詳細回答**: 700-1000トークン（技術文書、学術回答）
- **包括的回答**: 1000+トークン（複雑な分析、研究レポート）

## ベストプラクティス

### 1. 指示の構造化

```python
instructions = """役割定義: あなたは[専門分野]の専門家です。

対応方針:
1. [方針1]
2. [方針2]
3. [方針3]

回答形式:
- [要素1]
- [要素2]
- [要素3]

品質基準:
- [基準1]
- [基準2]
- [基準3]"""
```

### 2. 段階的改善

1. **基本指示から開始**: シンプルな役割定義
2. **具体例を追加**: 期待する回答の例を含める
3. **品質基準を明確化**: 避けるべき回答パターンを指定
4. **実際のクエリでテスト**: 代表的な質問で動作確認
5. **反復改善**: フィードバックに基づく調整

### 3. パフォーマンス最適化

- **コンテキスト長の調整**: 必要以上に長くしない
- **温度の微調整**: 用途に応じた最適値の設定
- **指示の簡潔性**: 冗長な指示は避ける
- **一貫性の確保**: 複数のクエリで一貫した品質

## QueryEngineでの統合

```python
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.retrieval.simple_reader import SimpleAnswerSynthesizer, SimpleAnswerSynthesizerConfig

# カスタマイズされたAnswerSynthesizer
synthesizer_config = SimpleAnswerSynthesizerConfig(
    generation_instructions="あなたの専用指示...",
    temperature=0.3,
    max_tokens=500
)
synthesizer = SimpleAnswerSynthesizer(synthesizer_config)

# QueryEngineに統合
query_engine = QueryEngine(
    corpus_name="custom_corpus",    # コーパス名を指定
    retrievers=retriever,           # 新しいインターフェース
    synthesizer=synthesizer,        # カスタマイズされたsynthesizer
    reranker=reranker,
    config=QueryEngineConfig()
)

# 使用
result = query_engine.query("あなたの質問")  # 新しいメソッド名
```

## まとめ

AnswerSynthesizerのinstruction customizationにより：

- **ドメイン特化**: 各分野の専門用語と慣例に対応
- **一貫性確保**: 組織やブランドの声調の統一
- **品質向上**: 用途に最適化された回答生成
- **柔軟性**: RefinireとOpenAIの両環境に対応
- **拡張性**: 新しいユースケースへの容易な適応

適切にカスタマイズされた指示により、RAGシステムの回答品質を大幅に向上させることができます。