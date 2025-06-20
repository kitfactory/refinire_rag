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
- ルートにある .venvディレクトリの仮想環境を利用
- uvを使用してプロジェクト管理

ルート--+-- src
        +-- docs
        +-- tests
        +-- pyproject.toml
        +-- todo.md
        +-- .venv

- src以下にソースコード
- examples以下に利用例
- tests以下にテスト
- tests/dataにテストデータ
- docs以下にドキュメント

## Pythonプロジェクト
- uvを使用したプロジェクト管理を
- uv add <package>、uv pip <package>コマンドを使用
- フォルダ構成は以下を守ってください。

- uvを使ったpyproject.tomlファイルでは、pytestとpytest-covを採用する。
- 下記のように設定してください。

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

## [重要] 開発の進め方
- 以下の手順を必ず守って進める

### プロジェクト初期の進め方
- concept.mdを中心に要件定義書、アーキテクチャ設計書を記載
- 実装のフェーズ一覧表(phase.md)を作成
    - 機能部品や下位レイヤー着手のフェーズを優先し、統合が必要な機能フェーズを後に

### [重要] 設計文書後の開発の進め方
- 以下の手順（フェーズ）で開発
    - 1.実現機能をフェーズに分解し、下記の2-8の手順となるよう、@todo.mdに記載
    - 2.実装する機能ブロックとクラスのインターフェースを明確にしたフェーズでの設計文書を作成
    - 3.作成を止めて、ユーザーに確認を得る
    - 4.ユーザーの確認がもらったら実装
    - 5.実装したコード全てにテストプログラムを作成
    - 6.作成したモジュールのカバレッジを70%かつテストの全てをパス
    - 7.実装内容を設計文書に反映
    - 8.todo.mdに進捗を記載
    - 9.フェーズ一覧表の見直し
    - 10.次の機能へ

## フロントエンド・TypsScript/JavaScriptプロジェクト
- フロントエンドがある場合、プロジェクトのトップにfrontendフォルダを作成
- TypeScript/JavaScriptの場合はnpmでパッケージ管理を行ってください。
- frontendフォルダ内も構成は以下を守ってください。
  - src以下にソースコード
  - docs以下にドキュメントを設置

# クラスの設計、メソッドの設計の方針
## コメント
- 全てのコメントは英語と、それを翻訳した日本語で、２つのコメントを連続で記述
- クラス全体、各メソッドの入出力には必ず詳細なコメントを生成
- 修正を行った場合は、既存のコメントを修正

## 単一責務
- クラスやメソッドは役割を持ち、少ない責務で維持

## クラスの分類
- レイヤーアーキテクチャを意識して作成
- レイヤーはデータベースやファイルに読み書きするゲートウェイ、UIからの入力を受け付けるコントローラ、処理の流れを実現するユースケース、処理の流れに複雑な処理が入り込まないようにする機能クラスを、ドメインのデータを表現するデータクラス、設定やログなどのユーティリティ
- コメントを抜いて、200行を超える複雑な処理が必要であれば、専用に別クラス
- 複雑処理をステートメントごとに機能クラスにし、そのクラスを呼び出す

## テスト
- テストはpytestを使用しカバレッジも取得
- testsフォルダにテストのソースコード及びデータを保存

## デバッグのポイント
- デバッグの際に原因が把握できない場合、デバッグ用のメッセージを入れ、原因分析

## 設計文書
- コンセプト文書(concept.md)、主な要件定義書(requirements.md)、アーキテクチャー設計書(architecture.md)、機能仕様書(function_spec.md)を記載
- 作成する機能フェーズごとに設計文書を作成

# 要件定義書
- 下記の見出しを持つ文書を作成する、不明瞭な場合はユーザーに確認をする

## 必要な見出し
- プロジェクトの目的・背景
    ・プロジェクトの目的・背景では、どのようなソフトウェアのイメージであるかを確認してください。
- 利用者と困りごと
    ・利用者を利用者の種類、利用者のゴール、利用者の制約、利用者の困りごとを表にしてください。
- 採用する技術スタック
    ・採用する技術スタックを表形式で記載してください
- 機能（ユースケース）一覧
    ・困りごとを解決する必要な機能（ユースケース）を洗い出し、表に一覧にしてください。
    ・機能のイメージも表に書いてください。

# アーキテクチャ設計書
- 採用する技術スタック、機能の一覧から基本となるデータ形式やアーキテクチャ構造の概要

## 必要な見出し
- システム構成・アーキテクチャの概要
    ・クリーンアーキテクチャのようなレイヤーアーキテクチャが採用出来たら、そういったレイヤー構造を明確にしてください。
- 主要モジュール構成・アーキテクチャレイヤー
- 各レイヤーに含まれる主要クラス

# ユースケース仕様書
- ユースケースは要求仕様を記載します。
- 要求はユーザーやシステムがすることの範囲を定めます。ユーザーがこうしたら、システムでこうなったら、起きることを記載します。
- 仕様はそれを実現するための手順を更に詳細に一意となる粒度で書き下さします。
- 仕様はユーザーがxxxxし、xxxxxの場合は、xxxxするといった、条件とそれに対する動作を一意になるように記載します。

# 要求 UC-01-xx : ユーザーがame 文章への指示とコマンド入力したら、文書編集計画を作成する。
|  仕様番号  |  仕様  |   実装完了 |  テスト完了 |
|---|---| ---|---|
|UC-01-xx-01| ユーザーから受け取った入力文を指定されたLLMで計画する| 未実装 | 未テスト |

# 機能仕様書
- 実装機能のフェーズごとに記載し、ユーザーに確認

## 各機能に必要な見出し
- ユースケース関連クラス
    - ユースケースで利用される主要クラスの
- ユースケース手順
    - ユースケースを手順として箇条書きにする。
- ユースケースフロー図
    - 手順化されたユースケースを主要インターフェースでmermaidでシーケンス図を書く

