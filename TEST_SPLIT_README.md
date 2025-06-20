# Test Split Runner - テスト分割実行ツール

大量のテストをタイムアウトなしで実行し、カバレッジと結果を集計するツールです。

## 🚀 クイックスタート

### 全テストを実行（推奨）
```bash
python run_all_tests.py
```

これだけで全テストが最適な設定で実行され、結果が `test_results/` に保存されます。

## 機能

- ✅ テストを指定サイズのチャンクに自動分割
- ✅ 各チャンクを独立して実行（タイムアウト対応）
- ✅ カバレッジデータの集計
- ✅ テスト結果の統合（成功/失敗/スキップ/エラー数）
- ✅ 実行時間の計測と分析
- ✅ 詳細なレポート出力（JSON + テキスト）
- ✅ 失敗したテストの一覧表示

## 基本的な使用方法

### 全テストを実行
```bash
python run_tests_split.py
```

### 特定のテストディレクトリを実行
```bash
python run_tests_split.py --pattern tests/unit/
```

### 小さなチャンクサイズで実行（より細かく分割）
```bash
python run_tests_split.py --chunk-size 20
```

### タイムアウト時間を延長
```bash
python run_tests_split.py --timeout 600
```

## 高度な使用例

### 統合テストのみを実行
```bash
python run_tests_split.py --pattern tests/integration/
```

### 特定のテストファイルを実行
```bash
python run_tests_split.py --pattern tests/test_document_pipeline_comprehensive.py
```

### 大きなプロジェクト用の設定
```bash
python run_tests_split.py --chunk-size 30 --timeout 900 --results-dir large_test_results
```

## 出力ファイル

実行後、`test_results/`ディレクトリ（または指定したディレクトリ）に以下のファイルが生成されます：

- `aggregated_results.json` - 詳細な実行結果（JSON形式）
- `test_summary.txt` - 人間が読みやすい概要レポート
- `coverage_chunk_*.xml` - 各チャンクのカバレッジデータ
- `intermediate_results_*.json` - 中間結果（長時間実行時の保険）

## パラメーター詳細

| パラメーター | デフォルト | 説明 |
|------------|-----------|------|
| `--pattern` | `tests/` | 実行するテストのパターン |
| `--chunk-size` | `50` | 1チャンクあたりの最大テストファイル数 |
| `--timeout` | `300` | 1チャンクあたりのタイムアウト（秒） |
| `--results-dir` | `test_results` | 結果を保存するディレクトリ |

## 推奨設定

### 通常のプロジェクト
```bash
python run_tests_split.py --chunk-size 50 --timeout 300
```

### 大規模プロジェクト
```bash
python run_tests_split.py --chunk-size 30 --timeout 600
```

### CI/CD環境
```bash
python run_tests_split.py --chunk-size 25 --timeout 180
```

### 重いテストがある場合
```bash
python run_tests_split.py --chunk-size 20 --timeout 900
```

## 出力例

```
=== Running test chunk 1 (50 files) ===
Chunk 1: 45 passed, 2 failed, 3 skipped, 0 errors
Execution time: 125.34s

=== Running test chunk 2 (50 files) ===
Chunk 2: 48 passed, 0 failed, 2 skipped, 0 errors
Execution time: 98.76s

============================================================
TEST EXECUTION SUMMARY
============================================================
Total chunks executed: 33
Total execution time: 3247.82s
Average time per chunk: 98.42s

Test Results:
  Passed:  1523 (92.9%)
  Failed:    45 (2.7%)
  Skipped:   71 (4.3%)
  Errors:     0 (0.0%)
  Total:   1639

Coverage: 21.4%

Overall Status: 🟡 MOSTLY SUCCESSFUL
============================================================
```

## トラブルシューティング

### メモリエラーが発生する場合
チャンクサイズを小さくしてください：
```bash
python run_tests_split.py --chunk-size 20
```

### タイムアウトが頻発する場合
タイムアウト時間を延長してください：
```bash
python run_tests_split.py --timeout 600
```

### カバレッジが計算されない場合
`pytest-cov`がインストールされているか確認してください：
```bash
uv add pytest-cov
```

## 終了コード

- `0`: すべてのテストが成功
- `1`: 失敗またはエラーのテストが存在
- `130`: ユーザーによる中断（Ctrl+C）

## 注意事項

- 各チャンクは独立して実行されるため、テスト間の依存関係がある場合は注意が必要です
- カバレッジデータは各チャンクで個別に収集され、最後に集計されます
- 中断された場合、中間結果が保存されているため部分的な分析が可能です