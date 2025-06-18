# 統一インポートシステム

## 概要

refinire-ragでは、すべてのプラグイン（ベクトルストア、キーワードストア、ローダー）に対して統一されたインポートパスを提供します。これにより、組み込みプラグインか外部プラグインかに関わらず、一貫した方法で利用できます。

## メリット

### **1. 一貫したインポートパス**
```python
# どのプラグインも同じパターン
from refinire.rag.vectorstore import OpenAIVectorStore, ChromaVectorStore
from refinire.rag.keywordstore import TFIDFKeywordStore, BM25KeywordStore  
from refinire.rag.loader import TextLoader, DoclingLoader
```

### **2. プラグインの透明性**
```python
# ユーザーはプラグインが組み込みか外部かを意識しない
from refinire.rag.vectorstore import list_available_stores

# 利用可能なストアの確認
stores = list_available_stores()
if "ChromaVectorStore" in stores:
    from refinire.rag.vectorstore import ChromaVectorStore
    store = ChromaVectorStore()
```

### **3. IDE支援**
- 自動補完が全プラグインで動作
- 型ヒントの提供
- ドキュメント表示

## 使用方法

### **ベクトルストア**

#### **基本的な使用**
```python
from refinire.rag.vectorstore import OpenAIVectorStore

store = OpenAIVectorStore()
```

#### **利用可能なストアの確認**
```python
from refinire.rag.vectorstore import list_available_stores

stores = list_available_stores()
for name, available in stores.items():
    print(f"{name}: {'✓' if available else '✗'}")

# 出力例:
# OpenAIVectorStore: ✓
# ChromaVectorStore: ✗ (パッケージ未インストール)
# FaissVectorStore: ✓ (refinire-rag-faissがインストール済み)
```

#### **動的なストア選択**
```python
from refinire.rag.vectorstore import list_available_stores

# 利用可能な最初のストアを使用
available_stores = {name: store for name, available in list_available_stores().items() if available}

if "ChromaVectorStore" in available_stores:
    from refinire.rag.vectorstore import ChromaVectorStore
    store = ChromaVectorStore()
elif "OpenAIVectorStore" in available_stores:
    from refinire.rag.vectorstore import OpenAIVectorStore
    store = OpenAIVectorStore()
```

### **キーワードストア**

```python
from refinire.rag.keywordstore import TFIDFKeywordStore, list_available_stores

# 利用可能なキーワードストア確認
stores = list_available_stores()
print("利用可能なキーワードストア:", [name for name, available in stores.items() if available])

# 使用
store = TFIDFKeywordStore()
```

### **ローダー**

```python
from refinire.rag.loader import TextLoader, list_available_loaders

# 利用可能なローダー確認
loaders = list_available_loaders()
print("利用可能なローダー:", [name for name, available in loaders.items() if available])

# 使用
loader = TextLoader()
```

## 外部プラグインの追加

### **標準的な外部プラグイン**

#### **ChromaDBベクトルストア**
```bash
pip install refinire-rag-chroma
```

```python
from refinire.rag.vectorstore import ChromaVectorStore
store = ChromaVectorStore()
```

#### **Doclingローダー**
```bash
pip install refinire-rag-docling
```

```python
from refinire.rag.loader import DoclingLoader
loader = DoclingLoader()
```

#### **BM25キーワードストア**
```bash
pip install refinire-rag-bm25
```

```python
from refinire.rag.keywordstore import BM25KeywordStore
store = BM25KeywordStore()
```

### **カスタムプラグインの登録**

独自のプラグインを作成した場合、`pyproject.toml`で登録します：

```toml
[project.entry-points."refinire.rag"]
vectorstore = "my_custom_package.vectorstore:CustomVectorStore"
keywordstore = "my_custom_package.keywordstore:CustomKeywordStore"
loader = "my_custom_package.loader:CustomLoader"
```

## エラーハンドリング

### **プラグイン未インストール時**

```python
try:
    from refinire.rag.vectorstore import ChromaVectorStore
    store = ChromaVectorStore()
except ImportError as e:
    print(f"ChromaVectorStore is not available: {e}")
    # フォールバック
    from refinire.rag.vectorstore import OpenAIVectorStore
    store = OpenAIVectorStore()
```

### **プラグイン可用性の事前チェック**

```python
from refinire.rag.vectorstore import list_available_stores

available = list_available_stores()
if available.get("ChromaVectorStore", False):
    from refinire.rag.vectorstore import ChromaVectorStore
    store = ChromaVectorStore()
else:
    print("ChromaVectorStore is not available")
```

## プラグイン作成者向けガイド

### **パッケージ命名規則**

外部プラグインは以下の命名規則に従ってください：

- **ベクトルストア**: `refinire-rag-<name>`
- **キーワードストア**: `refinire-rag-<name>`  
- **ローダー**: `refinire-rag-<name>`

### **パッケージ構造**

```
refinire-rag-myplugin/
├── pyproject.toml
├── src/
│   └── my_refinire_plugin/
│       ├── __init__.py
│       ├── vectorstore.py
│       └── keywordstore.py
└── tests/
    ├── __init__.py
    ├── test_vectorstore.py
    └── test_keywordstore.py
```