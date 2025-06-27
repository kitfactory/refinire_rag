# プラグイン設定方針 / Plugin Configuration Policy

## 概要 / Overview

refinire-ragライブラリのプラグインシステムでは、統一的な設定パターンを採用しています。
The refinire-rag library plugin system adopts a unified configuration pattern.

## 設定方針 / Configuration Policy

### ✅ 推奨パターン / Recommended Patterns

#### 1. 引数なしコンストラクタ（環境変数自動取得）
```python
# 環境変数から自動設定
loader = CSVLoader()
embedder = OpenAIEmbedder()
store = SQLiteDocumentStore()
```

#### 2. キーワード引数コンストラクタ（明示的設定）
```python
# 明示的設定
loader = CSVLoader(encoding='utf-16', delimiter=';')
embedder = OpenAIEmbedder(model_name='text-embedding-3-large', batch_size=50)
store = SQLiteDocumentStore(db_path='/custom/path.db', timeout=60.0)
```

#### 3. 混合パターン（環境変数 + 引数オーバーライド）
```python
# 環境変数をベースに一部引数でオーバーライド
# REFINIRE_RAG_CSV_ENCODING=iso-8859-1 が設定されている場合
loader = CSVLoader(delimiter='|')  # encoding は環境変数、delimiter は引数
```

### ❌ 非推奨パターン / Deprecated Patterns

```python
# ❌ from_env() メソッドは削除済み（冗長なため）
loader = CSVLoader.from_env()  # 利用不可
```

## 設定優先順位 / Configuration Priority

```
kwargs > 環境変数 > デフォルト値
kwargs > environment variables > default values
```

## 環境変数命名規則 / Environment Variable Naming Convention

```bash
REFINIRE_RAG_{COMPONENT_TYPE}_{SETTING_NAME}

例 / Examples:
REFINIRE_RAG_CSV_ENCODING=utf-16
REFINIRE_RAG_CSV_DELIMITER=;
REFINIRE_RAG_OPENAI_MODEL=text-embedding-3-large
REFINIRE_RAG_OPENAI_API_KEY=sk-...
REFINIRE_RAG_SQLITE_PATH=/custom/db.sqlite
REFINIRE_RAG_SQLITE_TIMEOUT=60.0
```

## 設定取得 / Configuration Access

```python
# 現在の設定を辞書として取得
config = loader.get_config()
print(f"Encoding: {config['encoding']}")
print(f"Delimiter: {config['delimiter']}")
```

## 実装ガイドライン / Implementation Guidelines

### プラグインクラス実装時
When implementing plugin classes:

1. **コンストラクタは `**kwargs` パターンを使用**
   ```python
   def __init__(self, **kwargs):
       # 環境変数サポート付きの設定処理
   ```

2. **`get_config()` メソッドを実装**
   ```python
   def get_config(self) -> Dict[str, Any]:
       return {'setting1': self.setting1, 'setting2': self.setting2}
   ```

3. **`from_env()` メソッドは実装しない**
   - コンストラクタが環境変数を自動処理するため不要

### Configクラス
Config classes may retain `from_env()` methods for backward compatibility and explicit config creation:

```python
class SomePluginConfig:
    @classmethod
    def from_env(cls) -> 'SomePluginConfig':
        # Config-specific environment loading
        pass
```

## 移行ガイド / Migration Guide

### Before (旧パターン)
```python
# 旧方式
loader = CSVLoader.from_env()
config = SomePluginConfig.from_env()
plugin = SomePlugin(config)
```

### After (新パターン)
```python
# 新方式
loader = CSVLoader()  # 環境変数自動取得
plugin = SomePlugin()  # 環境変数自動取得
# または明示的設定
plugin = SomePlugin(setting1='value1', setting2='value2')
```

## 利点 / Benefits

1. **API簡素化**: 1つの方法で設定を管理
2. **柔軟性**: 引数と環境変数の組み合わせが可能
3. **一貫性**: 全プラグインで統一的なパターン
4. **保守性**: コードの重複削減
5. **使いやすさ**: 引数なしでの簡単な利用が可能

---

This policy ensures consistent, flexible, and maintainable plugin configuration across the entire refinire-rag library.