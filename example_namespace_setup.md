# Namespace Package Setup Guide

## 1. 親パッケージ (refinire) の構造

```
refinire/
├── pyproject.toml
└── src/
    └── refinire/
        ├── __init__.py  # このファイルを削除するか、PEP 420準拠にする
        ├── agents/
        ├── llm.py
        └── ...
```

### refinire の pyproject.toml
```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "refinire"
version = "0.0.2"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["refinire*"]
# namespace packageとして明示
namespace = ["refinire"]
```

## 2. サブパッケージ (refinire-rag) の構造

```
refinire-rag/
├── pyproject.toml
└── src/
    └── refinire/
        ├── # __init__.py なし（重要！）
        └── rag/
            ├── __init__.py
            └── ...
```

### refinire-rag の pyproject.toml
```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "refinire-rag"
version = "0.1.0"
dependencies = [
    "refinire>=0.0.2",  # 親パッケージに依存
]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["refinire.rag*"]
# namespace packageとして明示
namespace = ["refinire"]
```

## 3. PEP 420方式（推奨）

### 重要なポイント
1. **refinire/__init__.py を削除する**（両方のパッケージで）
2. またはpkgutil-style namespace packageを使用

### pkgutil-style の場合の __init__.py
```python
# refinire/__init__.py (両方のパッケージで同じ内容)
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
```

## 4. 実装例

### Option A: PEP 420 (暗黙的namespace package)
- `src/refinire/__init__.py` を削除
- Python 3.3以降で自動的にnamespace packageとして認識

### Option B: pkgutil-style (明示的namespace package)
```python
# src/refinire/__init__.py
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)
```

## 5. インストールと使用

```bash
# 親パッケージをインストール
pip install refinire

# サブパッケージをインストール
pip install refinire-rag

# 使用例
from refinire import create_simple_gen_agent  # 親パッケージから
from refinire.rag import Document  # サブパッケージから
```

## 6. 注意事項

1. **バージョン管理**: 各パッケージは独立してバージョン管理される
2. **依存関係**: refinire-ragはrefinireに依存するよう設定
3. **__init__.py**: namespace packageのルートには__init__.pyを置かない（PEP 420）
4. **互換性**: 古いPythonバージョンではpkgutil-styleを使用

## 7. 現在のrefinire-ragでの実装

現在の構造を維持しながらnamespace package化する場合：

1. `src/refinire/__init__.py` を削除または上記のnamespace package用コードに置換
2. pyproject.tomlは現状のままでOK（setuptools.packages.findが自動認識）
3. 親のrefinireパッケージも同様の変更が必要