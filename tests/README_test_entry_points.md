# エントリーポイントベース発見システムのテスト

## 概要

このテストスイートは、refinire-ragのエントリーポイントベースプラグイン発見システムの包括的なテストを提供します。

## テストファイル

### test_entry_point_discovery.py

プラグイン自動発見システムと統一インポートシステムの完全なテストスイートです。

#### テストクラス

| テストクラス | 目的 | テスト数 |
|---|---|---|
| `TestPluginAutoDiscovery` | プラグイン自動発見システムのテスト | 7 |
| `TestVectorStoreRegistry` | ベクトルストアレジストリのテスト | 6 |
| `TestUnifiedImportSystem` | 統一インポートシステムのテスト | 4 |
| `TestIntegration` | 統合テスト | 2 |

#### 主要テストケース

**プラグイン発見テスト**
- エントリーポイントスキャン（空、正常、失敗ケース）
- インストール済みパッケージスキャン
- 包括的発見プロセス
- タイプ別プラグイン取得
- 発見リフレッシュ

**レジストリテスト**
- 組み込みストアクラス取得
- エントリーポイントからのクラス取得
- 未知のストアクラス処理
- 利用可能ストア一覧
- 外部ストア登録
- キャッシング機能

**統一インポートテスト**
- 動的`__getattr__`機能
- 存在しないストアの処理
- IDE自動補完サポート（`__dir__`）
- ユーティリティ関数アクセス

**統合テスト**
- 完全プラグインライフサイクル
- エラーハンドリング統合

## 実行方法

```bash
# 仮想環境をアクティベート
source .venv/bin/activate

# 単一テストクラス実行
python -m pytest tests/test_entry_point_discovery.py::TestPluginAutoDiscovery -v

# 全テスト実行
python -m pytest tests/test_entry_point_discovery.py -v

# カバレッジ付き実行
python -m pytest tests/test_entry_point_discovery.py --cov=refinire_rag.plugins --cov=refinire_rag.vectorstore -v
```

## モック戦略

### pkg_resources モック
- エントリーポイント発見の模擬
- インストール済みパッケージリストの制御
- エラー条件のシミュレーション

### importlib モック
- 動的インポートの制御
- 欠落パッケージのシミュレーション
- クラス読み込み失敗のテスト

### レジストリモック
- キャッシュ動作の検証
- エラーハンドリングの確認
- 状態管理のテスト

## 依存関係

- `pytest>=7.0.0` - テストフレームワーク
- `setuptools>=65.0.0` - `pkg_resources`サポート
- `pyyaml>=6.0.0` - YAML設定サポート

## 注意事項

1. **pkg_resources 非推奨警告**: Python 3.12以降で`pkg_resources`は非推奨です。将来的には`importlib.metadata`に移行予定です。

2. **仮想環境**: すべてのテストは仮想環境内で実行してください。

3. **モック使用**: 実際のプラグインパッケージをインストールせずにテストが可能です。

## テスト結果

最新実行結果: **19/19 テストパス** ✅

```
=================== 19 passed, 2 warnings in 15.56s =================
```

警告は既知の非推奨警告であり、機能に影響はありません。