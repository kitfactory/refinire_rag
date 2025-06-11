# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

refinire-rag is a Python library that provides RAG (Retrieval-Augmented Generation) functionality as a sub-package of the Refinire library. It follows a modular architecture with use cases implemented as Refinire Step subclasses and single-responsibility backend modules.

## Architecture

### Use Case Classes (Refinire Steps)
- **CorpusManager**: Document loading, normalization, chunking, embedding generation, and storage
- **QueryEngine**: Document retrieval, re-ranking, and answer generation
- **QualityLab**: Evaluation data creation, automatic RAG evaluation, conflict detection, and report generation

### Backend Modules (All implement DocumentProcessor)
- **Loader**: External file → Document conversion
- **DictionaryMaker**: LLM-based domain-specific term extraction with cumulative MD dictionary
- **Normalizer**: MD dictionary-based expression variation normalization
- **GraphBuilder**: LLM-based relationship extraction with cumulative MD knowledge graph
- **Chunker**: Token-based chunking
- **VectorStoreProcessor**: Chunk → vector generation and storage (integrates Embedder)
- **Retriever**: Document search
- **Reranker**: Candidate re-ranking  
- **Reader**: LLM-based answer generation
- **TestSuite**: Evaluation runner
- **Evaluator**: Metrics aggregation
- **ContradictionDetector**: Claim extraction + NLI detection
- **InsightReporter**: Threshold-based interpretation and reporting

## Key Implementation Notes

- Each use case is a subclass of Refinire Step for integration with Refinire agents
- All processing modules inherit from DocumentProcessor for unified pipeline architecture
- Dependency injection is used to swap backend modules (e.g., InMemory vs Chroma/Faiss vector stores)
- LLM integration uses Refinire library (get_llm) for unified multi-provider access
- DictionaryMaker and GraphBuilder use LLM for domain-specific extraction and maintain cumulative Markdown files
- Default embeddings use OpenAI Embeddings API
- Configuration is managed through YAML with pydantic validation
- OpenTelemetry tracing is implemented for each use case
- Async/multiprocessing support planned for Loader to handle large corpora

## Development Commands

### Environment Setup
```bash
# Python version: 3.10
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (when added to pyproject.toml)
uv pip install -e .
```

## Commands

### Environment Setup
```bash
# Install in development mode
uv pip install -e .

# Install dependencies
uv add <package>

# Remove dependencies
uv remove <package>
```

### Development Commands
```bash
# Run tests (when test framework is set up)
pytest

# Run with coverage (when coverage is configured)
pytest --cov=flexify

# Build the package
uv build
```

# プロジェクトの構成
## フォルダの位置
・フォルダの位置は、プロジェクトのトップディレクトリです。
・この場所にある、 .venvディレクトリの仮想環境を利用して開発を進めています。
・ただし、仮想環境を起動する必要はなく、claudeコマンド実行時にすでに仮想環境は有効な状態になっています。
・単にuvを使用してプロジェクト管理を行ってください。venvは再作成の必要はありません。

ルート--+-- src
        +-- docs
        +-- tests
        +-- pyproject.toml
        +-- todo.md
        +-- .venv

## Pythonプロジェクト
・uvを使用したプロジェクト管理を行います。
・ライブラリは uv add コマンドを使って追加します。
・pipではなく、uv pipコマンドを使ってください。
・フォルダ構成は以下を守ってください。
  - src以下にソースコード
    - examples以下に利用例
    -tests以下にテスト
    - docs以下にドキュメントを置いてください

・uvを使ったpyproject.tomlファイルでは、pytestとpytest-covを採用する。
・下記のように設定してください。

```pyproject.toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0"
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = [
    "--import-mode=importlib",
    "--cov=oneenv",
    "--cov-report=term-missing",
]
```

## フロントエンド・TypsScript/JavaScriptプロジェクト
・フロントエンドがある場合、プロジェクトのトップにfrontendフォルダを作成し、そこにソースコードを設置してください。
・TypeScript/JavaScriptの場合はnpmでパッケージ管理を行ってください。
・frontendフォルダ内も構成は以下を守ってください。
  - src以下にソースコード
 - docs以下にドキュメントを置いてください


# クラスの設計、メソッドの設計
## コメント
- 全てのコメントは英語と、それを翻訳した日本語で、２つのコメントを連続で記述する
- クラス全体、各メソッドの入出力には必ず詳細なコメントを生成する
- 修正を行った場合は、それにあわせて既存のコメントを修正する

## 単一責務
- クラスやメソッドは役割を持ち、少ない責務で維持する

## クラスの分類
- レイヤーアーキテクチャを意識して作成するクラスの位置づけを明確にする。
- レイヤーはデータベースやファイルに読み書きするゲートウェイ、UIからの入力を受け付けるコントローラ、処理の流れを実現するユースケース、処理の流れに複雑な処理が入り込まないようにする機能クラスを、ドメインのデータを表現するデータクラス、設定やログなどのユーティリティに分ける
- コメントを抜いて、200行を超える複雑な処理が必要であれば、専用に別クラスを設ける
- 複雑処理をステートメントごとに機能クラスにし、そのクラスを呼び出すようにする

# ドキュメント
## 文書の記載方法 
- クラスやAPIの紹介ではグループごとに表形式で一覧を作成する
- クラスの紹介では表形式に加え、概クラス図をmermaid形式で記載する
- 設計文書ではクラスの一覧、メソッドの説明では表形式を採用してください。
- 表形式が採用できる場所では積極的に表を採用してください。
- 図を極力増やして場合、mermaid形式を使って図を作成してください。
- docsフォルダの文書は日本語のみで書いてください、英語を併記する必要はありません

## README.md
・README.mdは英語で記載し、その日本語訳をREADME_ja.mdを作成してください。
・README_ja.mdはコメントとことなり、日本語のみ記載してください。
・README.mdは絵文字を使って楽しいものにしてください。
・README.mdでは現在サポートしている環境を明確にしてください。
・README.mdではユーザーを想定したうえで、メリット、特に簡単さを強く訴求するようにしてください

## 設計文書
・設計文書とは、要件定義書(requirements.md)、アーキテクチャー設計書(architecture.md)、機能仕様書(function_spec.md)のことです。
・ docsフォルダに要件定義書がない場合は要件定義書(requirements.md)から作成してください。
・要件定義書の次にアーキテクチャ設計書、機能仕様書を作成してください。

# 要件定義書
・下記の見出しを持つ文書を作成する、不明瞭な場合はユーザーに確認をする
## 必要な見出し
・プロジェクトの目的・背景
    ・プロジェクトの目的・背景では、どのようなソフトウェアのイメージであるかを確認してください。
・利用者と困りごと
    ・利用者を利用者の種類、利用者のゴール、利用者の制約、利用者の困りごとを表にしてください。
・採用する技術スタック
    ・採用する技術スタックを表形式で記載してください
・機能（ユースケース）一覧
    ・困りごとを解決する必要な機能（ユースケース）を洗い出し、表に一覧にしてください。
    ・機能のイメージも表に書いてください。

# アーキテクチャ設計書
採用する技術スタック、機能の一覧から基本となるデータ形式やアーキテクチャ構造を定めてください。

## 必要な見出し
・システム構成・アーキテクチャの概要
    ・クリーンアーキテクチャのようなレイヤーアーキテクチャが採用出来たら、そういったレイヤー構造を明確にしてください。
・主要インターフェース
    ・機能の入り口となる主要なクラスを表の一覧にしてください。
・クラスは単一の責務で単一のファイルにできるようにしてください。
・主要データ（データの種類、構造）
    ・主要なデータクラスを一覧表にしてください。クラス名、保持するデータなど。
    ・またER図をmermaidのクラス図の形式で書いて下さい。

# ユースケース仕様書
・ユースケースは要求仕様を記載します。
・要求はユーザーやシステムがすることの範囲を定めます。ユーザーがこうしたら、システムでこうなったら、起きることを記載します。
・仕様はそれを実現するための手順を更に詳細に一意となる粒度で書き下さします。
・仕様はユーザーがxxxxし、xxxxxの場合は、xxxxするといった、条件とそれに対する動作を一意になるように記載します。

# 要求 UC-01-xx : ユーザーがame 文章への指示とコマンド入力したら、文書編集計画を作成する。
|  仕様番号  |  仕様  |   実装完了 |  テスト完了 |
|---|---| ---|---|
|UC-01-xx-01| ユーザーから受け取った入力文を指定されたLLMで計画する| 未実装 | 未テスト |

# 機能仕様書
要件定義書を参照して、機能一覧の各機能を詳細化する
## 各機能に必要な見出し
・ユースケース手順
    ・ユースケースを手順として箇条書きにする。
・ユースケースフロー図
    ・手順化されたユースケースを主要インターフェースでmermaidでシーケンス図を書く

# 計画
## 計画の建て方
・日本語でtodo.mdを記載して、進捗があれば、todo.mdも更新して作業を進める。
・todo.mdのアイテムの粒度としては、文書の作成、文書の検証、実装、テストの作成、テストの検証が１つのアイテムになっている。
・todo.mdの計画の考え方は、
    ・要件定義書
    ・アーキテクチャ設計
    ・ユースケース仕様書
    ・機能仕様書
    ・文書間矛盾検証
    ・機能ごとに
        ・ソースコード
        ・テストコード
        ・テスト検証
・機能の実装順序
    ・他のクラスを必要としない、最小機能
    ・それらを組み合わせた機能クラスから作成
    ・バックエンドのユースケース
    ・API機能
    ・フロントエンド機能

# テスト・検証
## テスト・検証の実施タイミング
・1個流し/TDDの考え方に沿って作成をする。
・作成した対象ができたら、テスト・検証を行う。
・テスト・検証ができたら、その次のアイテムを作成する。

## フロント・バックエンドのあるプロジェクト
・ バックエンド、フロントエンドも含めてテストを実行する。
・ テストの構築はバックエンドのテストから構築し、パスができてから、フロントエンドのテストを作成する。
・ バックエンド、フロントエンドのサーバーは標準のポートからずらして、起動してください。開発ユーザーがバックエンド8000、フロントエンド3000ポートをデフォルトで使用している環境では、 8001、30001ポートを使用。

# 技術スタック
## LLMを使用の場合
・refinireのエージェントをもちいて、LLMにアクセスするようにしてください。
https://github.com/kitfactory/refinire


```python
from refinire import create_simple_gen_agent, Context
import asyncio

agent = create_simple_gen_agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    model="gpt-4o-mini"
)

result = asyncio.run(agent.run("Hello, world!", Context()))
print(result.shared_state["assistant_result"])
```

・サポートプロバイダー:
  - OpenAI
  - Ollama（カスタムURL対応）
  - Google Gemini
  - Anthropic Claude

