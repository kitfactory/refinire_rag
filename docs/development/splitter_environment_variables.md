# Splitter環境変数ガイド

このドキュメントでは、refinire-ragライブラリの全splitterクラスでサポートされている環境変数の詳細な一覧と使用例を提供します。

## 概要

各splitterクラスは統一された**kwargsコンストラクタパターンにより環境変数からの設定読み込みを自動サポートしています。これにより、コードを変更することなく、環境変数を使用してsplitterの動作を柔軟に設定できます。

## 対応splitterクラス一覧

### 1. CharacterTextSplitter

文字数ベースでテキストを分割するsplitterです。

| 環境変数名 | 説明 | デフォルト値 | 型 |
|-----------|------|------------|-----|
| `REFINIRE_RAG_CHARACTER_CHUNK_SIZE` | 各チャンクの最大文字数 | 1000 | int |
| `REFINIRE_RAG_CHARACTER_OVERLAP` | チャンク間のオーバーラップ文字数 | 0 | int |

**使用例:**
```bash
export REFINIRE_RAG_CHARACTER_CHUNK_SIZE=2000
export REFINIRE_RAG_CHARACTER_OVERLAP=100

python -c "
from refinire_rag.splitter.character_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter()
print(f'Chunk size: {splitter.chunk_size}')
print(f'Overlap size: {splitter.overlap_size}')
"
```

### 2. TokenTextSplitter

トークン数ベースでテキストを分割するsplitterです。

| 環境変数名 | 説明 | デフォルト値 | 型 |
|-----------|------|------------|-----|
| `REFINIRE_RAG_TOKEN_CHUNK_SIZE` | 各チャンクの最大トークン数 | 1000 | int |
| `REFINIRE_RAG_TOKEN_OVERLAP` | チャンク間のオーバーラップトークン数 | 0 | int |
| `REFINIRE_RAG_TOKEN_SEPARATOR` | トークンの区切り文字 | " " (空白) | str |

**使用例:**
```bash
export REFINIRE_RAG_TOKEN_CHUNK_SIZE=1500
export REFINIRE_RAG_TOKEN_OVERLAP=50
export REFINIRE_RAG_TOKEN_SEPARATOR="|"

python -c "
from refinire_rag.splitter.token_splitter import TokenTextSplitter
splitter = TokenTextSplitter()
print(f'Chunk size: {splitter.config[\"chunk_size\"]}')
print(f'Overlap size: {splitter.config[\"overlap_size\"]}')
print(f'Separator: {splitter.config[\"separator\"]}')
"
```

### 3. MarkdownTextSplitter

Markdown構造を保持しながらテキストを分割するsplitterです。

| 環境変数名 | 説明 | デフォルト値 | 型 |
|-----------|------|------------|-----|
| `REFINIRE_RAG_MD_CHUNK_SIZE` | 各チャンクの最大文字数 | 1000 | int |
| `REFINIRE_RAG_MD_OVERLAP` | チャンク間のオーバーラップ文字数 | 200 | int |

**使用例:**
```bash
export REFINIRE_RAG_MD_CHUNK_SIZE=800
export REFINIRE_RAG_MD_OVERLAP=150

python -c "
from refinire_rag.splitter.markdown_splitter import MarkdownTextSplitter
splitter = MarkdownTextSplitter()
print(f'Chunk size: {splitter.chunk_size}')
print(f'Overlap size: {splitter.overlap_size}')
"
```

### 4. RecursiveCharacterTextSplitter

複数レベルのセパレータを使って再帰的にテキストを分割するsplitterです。

| 環境変数名 | 説明 | デフォルト値 | 型 |
|-----------|------|------------|-----|
| `REFINIRE_RAG_RECURSIVE_CHUNK_SIZE` | 各チャンクの最大文字数 | 1000 | int |
| `REFINIRE_RAG_RECURSIVE_OVERLAP` | チャンク間のオーバーラップ文字数 | 0 | int |
| `REFINIRE_RAG_RECURSIVE_SEPARATORS` | カンマ区切りのセパレータリスト | "\\n\\n,\\n,., ," | str |

**注意:** セパレータ内でエスケープシーケンス（`\\n`, `\\t`）を使用できます。

**使用例:**
```bash
export REFINIRE_RAG_RECURSIVE_CHUNK_SIZE=1200
export REFINIRE_RAG_RECURSIVE_OVERLAP=80
export REFINIRE_RAG_RECURSIVE_SEPARATORS="\\n\\n,\\n,;, ,"

python -c "
from refinire_rag.splitter.recursive_character_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter()
print(f'Chunk size: {splitter.config[\"chunk_size\"]}')
print(f'Overlap size: {splitter.config[\"overlap_size\"]}')
print(f'Separators: {splitter.config[\"separators\"]}')
"
```

### 5. HTMLTextSplitter

HTML構造を保持しながらテキストを分割するsplitterです。

| 環境変数名 | 説明 | デフォルト値 | 型 |
|-----------|------|------------|-----|
| `REFINIRE_RAG_HTML_CHUNK_SIZE` | 各チャンクの最大文字数 | 1000 | int |
| `REFINIRE_RAG_HTML_OVERLAP` | チャンク間のオーバーラップ文字数 | 0 | int |

**使用例:**
```bash
export REFINIRE_RAG_HTML_CHUNK_SIZE=1800
export REFINIRE_RAG_HTML_OVERLAP=120

python -c "
from refinire_rag.splitter.html_splitter import HTMLTextSplitter
splitter = HTMLTextSplitter()
print(f'Chunk size: {splitter.chunk_size}')
print(f'Overlap size: {splitter.overlap_size}')
"
```

### 6. CodeTextSplitter

プログラミングコードの構造を保持しながらテキストを分割するsplitterです。

| 環境変数名 | 説明 | デフォルト値 | 型 |
|-----------|------|------------|-----|
| `REFINIRE_RAG_CODE_CHUNK_SIZE` | 各チャンクの最大文字数 | 1000 | int |
| `REFINIRE_RAG_CODE_OVERLAP` | チャンク間のオーバーラップ文字数 | 200 | int |
| `REFINIRE_RAG_CODE_LANGUAGE` | プログラミング言語の指定 | None | str |

**使用例:**
```bash
export REFINIRE_RAG_CODE_CHUNK_SIZE=1500
export REFINIRE_RAG_CODE_OVERLAP=250
export REFINIRE_RAG_CODE_LANGUAGE=python

python -c "
from refinire_rag.splitter.code_splitter import CodeTextSplitter
splitter = CodeTextSplitter()
print(f'Chunk size: {splitter.chunk_size}')
print(f'Overlap size: {splitter.overlap_size}')
print(f'Language: {splitter.language}')
"
```

### 7. SizeSplitter

バイトサイズベースでテキストを分割するsplitterです。

| 環境変数名 | 説明 | デフォルト値 | 型 |
|-----------|------|------------|-----|
| `REFINIRE_RAG_SIZE_CHUNK_SIZE` | 各チャンクの最大バイトサイズ | 1024 | int |
| `REFINIRE_RAG_SIZE_OVERLAP` | チャンク間のオーバーラップバイト数 | 0 | int |

**使用例:**
```bash
export REFINIRE_RAG_SIZE_CHUNK_SIZE=2048
export REFINIRE_RAG_SIZE_OVERLAP=256

python -c "
from refinire_rag.splitter.size_splitter import SizeSplitter
splitter = SizeSplitter()
print(f'Chunk size: {splitter.chunk_size}')
print(f'Overlap size: {splitter.overlap_size}')
"
```

## 全環境変数一覧

| カテゴリ | 環境変数名 | splitter | デフォルト値 | 説明 |
|----------|------------|----------|------------|-----|
| Character | `REFINIRE_RAG_CHARACTER_CHUNK_SIZE` | CharacterTextSplitter | 1000 | 文字数ベースチャンクサイズ |
| Character | `REFINIRE_RAG_CHARACTER_OVERLAP` | CharacterTextSplitter | 0 | 文字数ベースオーバーラップ |
| Token | `REFINIRE_RAG_TOKEN_CHUNK_SIZE` | TokenTextSplitter | 1000 | トークン数ベースチャンクサイズ |
| Token | `REFINIRE_RAG_TOKEN_OVERLAP` | TokenTextSplitter | 0 | トークン数ベースオーバーラップ |
| Token | `REFINIRE_RAG_TOKEN_SEPARATOR` | TokenTextSplitter | " " | トークン区切り文字 |
| Markdown | `REFINIRE_RAG_MD_CHUNK_SIZE` | MarkdownTextSplitter | 1000 | Markdownチャンクサイズ |
| Markdown | `REFINIRE_RAG_MD_OVERLAP` | MarkdownTextSplitter | 200 | Markdownオーバーラップ |
| Recursive | `REFINIRE_RAG_RECURSIVE_CHUNK_SIZE` | RecursiveCharacterTextSplitter | 1000 | 再帰分割チャンクサイズ |
| Recursive | `REFINIRE_RAG_RECURSIVE_OVERLAP` | RecursiveCharacterTextSplitter | 0 | 再帰分割オーバーラップ |
| Recursive | `REFINIRE_RAG_RECURSIVE_SEPARATORS` | RecursiveCharacterTextSplitter | "\\n\\n,\\n,., ," | 再帰分割セパレータリスト |
| HTML | `REFINIRE_RAG_HTML_CHUNK_SIZE` | HTMLTextSplitter | 1000 | HTMLチャンクサイズ |
| HTML | `REFINIRE_RAG_HTML_OVERLAP` | HTMLTextSplitter | 0 | HTMLオーバーラップ |
| Code | `REFINIRE_RAG_CODE_CHUNK_SIZE` | CodeTextSplitter | 1000 | コードチャンクサイズ |
| Code | `REFINIRE_RAG_CODE_OVERLAP` | CodeTextSplitter | 200 | コードオーバーラップ |
| Code | `REFINIRE_RAG_CODE_LANGUAGE` | CodeTextSplitter | None | プログラミング言語 |
| Size | `REFINIRE_RAG_SIZE_CHUNK_SIZE` | SizeSplitter | 1024 | サイズベースチャンクサイズ |
| Size | `REFINIRE_RAG_SIZE_OVERLAP` | SizeSplitter | 0 | サイズベースオーバーラップ |

## 統合使用例

### 複数splitterを組み合わせた使用例

```bash
# 複数のsplitter設定を一度に行う
export REFINIRE_RAG_CHARACTER_CHUNK_SIZE=1500
export REFINIRE_RAG_CHARACTER_OVERLAP=150
export REFINIRE_RAG_TOKEN_CHUNK_SIZE=1200
export REFINIRE_RAG_TOKEN_OVERLAP=100
export REFINIRE_RAG_MD_CHUNK_SIZE=800
export REFINIRE_RAG_MD_OVERLAP=160

python -c "
from refinire_rag.splitter.character_splitter import CharacterTextSplitter
from refinire_rag.splitter.token_splitter import TokenTextSplitter
from refinire_rag.splitter.markdown_splitter import MarkdownTextSplitter

# 環境変数から設定を読み込み
char_splitter = CharacterTextSplitter()
token_splitter = TokenTextSplitter()
md_splitter = MarkdownTextSplitter()

print('Character Splitter:', char_splitter.chunk_size, char_splitter.overlap_size)
print('Token Splitter:', token_splitter.config['chunk_size'], token_splitter.config['overlap_size'])
print('Markdown Splitter:', md_splitter.chunk_size, md_splitter.overlap_size)
"
```

### Docker環境での使用例

```dockerfile
# Dockerfile
FROM python:3.10

# 環境変数を設定
ENV REFINIRE_RAG_CHARACTER_CHUNK_SIZE=2000
ENV REFINIRE_RAG_CHARACTER_OVERLAP=200
ENV REFINIRE_RAG_TOKEN_CHUNK_SIZE=1500
ENV REFINIRE_RAG_TOKEN_SEPARATOR="|"

# アプリケーションコードをコピー
COPY . /app
WORKDIR /app

# アプリケーション実行
CMD ["python", "app.py"]
```

### 設定ファイルからの環境変数読み込み

```bash
# .env ファイル
REFINIRE_RAG_CHARACTER_CHUNK_SIZE=1800
REFINIRE_RAG_CHARACTER_OVERLAP=180
REFINIRE_RAG_TOKEN_CHUNK_SIZE=1400
REFINIRE_RAG_TOKEN_OVERLAP=140
REFINIRE_RAG_MD_CHUNK_SIZE=900
REFINIRE_RAG_MD_OVERLAP=180
```

```python
# Python script with python-dotenv
from dotenv import load_dotenv
from refinire_rag.splitter.character_splitter import CharacterTextSplitter

# .envファイルから環境変数を読み込み
load_dotenv()

# 環境変数を使用してsplitterを作成
splitter = CharacterTextSplitter()
```

## エラーハンドリング

### 無効な環境変数値の処理

環境変数に無効な値が設定された場合、適切なエラーが発生します：

```bash
export REFINIRE_RAG_CHARACTER_CHUNK_SIZE="invalid_value"

python -c "
from refinire_rag.splitter.character_splitter import CharacterTextSplitter
try:
    splitter = CharacterTextSplitter()
except ValueError as e:
    print(f'Error: {e}')
"
```

### 部分的な設定

一部の環境変数のみを設定した場合、残りの設定にはデフォルト値が使用されます：

```bash
export REFINIRE_RAG_TOKEN_CHUNK_SIZE=1500
# REFINIRE_RAG_TOKEN_OVERLAP and REFINIRE_RAG_TOKEN_SEPARATOR は未設定

python -c "
from refinire_rag.splitter.token_splitter import TokenTextSplitter
splitter = TokenTextSplitter()
print(f'Chunk size: {splitter.config[\"chunk_size\"]}')  # 1500 (環境変数より)
print(f'Overlap size: {splitter.config[\"overlap_size\"]}')  # 0 (デフォルト値)
print(f'Separator: {splitter.config[\"separator\"]}')  # \" \" (デフォルト値)
"
```

## ベストプラクティス

1. **環境変数の命名**: すべての環境変数は`REFINIRE_RAG_`プレフィックスで始まります
2. **デフォルト値の確認**: 環境変数を設定しない場合のデフォルト値を確認してください
3. **型安全性**: すべての数値パラメータは整数として解析されるため、適切な値を設定してください
4. **セパレータのエスケープ**: RecursiveCharacterTextSplitterでは、エスケープシーケンス（`\\n`, `\\t`）が正しく処理されます
5. **テスト環境**: 本番環境に適用する前に、テスト環境で設定値を検証してください


このドキュメントを参考に、用途に応じて適切な環境変数を設定してsplitterの動作をカスタマイズしてください。