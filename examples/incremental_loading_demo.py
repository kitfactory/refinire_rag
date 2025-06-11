#!/usr/bin/env python3
"""
増分ローディングのデモ

フォルダ内の文書変更を検出し、新規・更新された文書のみを処理する
実用的なサンプルです。
"""

import sys
import os
import time
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.loaders.incremental_loader import IncrementalLoader
from refinire_rag.storage import SQLiteDocumentStore
from refinire_rag.models.document import Document


def create_sample_documents(demo_dir: Path):
    """デモ用のサンプル文書を作成"""
    
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # 初期文書セット
    documents = {
        "product_manual.txt": """
# 製品マニュアル v1.0

## 概要
当社の主力製品について説明します。

## 機能
- 基本機能A
- 基本機能B  
- 基本機能C

## 使用方法
1. 電源を入れる
2. 設定を行う
3. 操作を開始する

最終更新: 2024年1月
        """,
        
        "company_policy.txt": """
# 会社ポリシー

## 勤務時間
- 平日: 9:00-18:00
- 昼休み: 12:00-13:00

## 休暇制度
- 年次有給休暇: 20日
- 夏季休暇: 3日
- 年末年始: 5日

## 服装規定
ビジネスカジュアル推奨

発行日: 2024年1月
        """,
        
        "sales_report.txt": """
# 営業レポート Q1

## 売上実績
- 1月: 1,000万円
- 2月: 1,200万円  
- 3月: 1,100万円

## 主要顧客
- A社: 300万円
- B社: 250万円
- C社: 200万円

## 課題
- 新規開拓の強化が必要
- 既存顧客のフォロー改善

作成日: 2024年4月1日
        """
    }
    
    for filename, content in documents.items():
        with open(demo_dir / filename, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    print(f"✅ {len(documents)}件のサンプル文書を作成: {demo_dir}")


def update_sample_documents(demo_dir: Path):
    """既存文書を更新し、新規文書を追加"""
    
    # 既存文書の更新
    updated_manual = """
# 製品マニュアル v2.0

## 概要
当社の主力製品について説明します。

## 新機能
- 新機能X（追加）
- 新機能Y（追加）
- 改良された基本機能A
- 基本機能B  
- 基本機能C

## 使用方法
1. 電源を入れる
2. 新しい設定ウィザードを実行
3. 高度な設定を行う
4. 操作を開始する

## トラブルシューティング
- よくある問題とその解決方法

最終更新: 2024年6月（大幅更新）
    """
    
    with open(demo_dir / "product_manual.txt", 'w', encoding='utf-8') as f:
        f.write(updated_manual.strip())
    
    # 新規文書の追加
    new_document = """
# 技術仕様書

## システム要件
- OS: Windows 10以降
- メモリ: 8GB以上
- ストレージ: 500GB以上

## API仕様
- REST API v2.0
- 認証: OAuth 2.0
- レスポンス形式: JSON

## セキュリティ
- データ暗号化: AES-256
- 通信暗号化: TLS 1.3

作成日: 2024年6月
    """
    
    with open(demo_dir / "technical_spec.txt", 'w', encoding='utf-8') as f:
        f.write(new_document.strip())
    
    print("✅ 文書を更新・追加しました")
    print("   - product_manual.txt: 更新（v1.0 → v2.0）")
    print("   - technical_spec.txt: 新規追加")


def demo_incremental_loading():
    """増分ローディングのデモンストレーション"""
    
    print("📁 増分ローディング デモンストレーション")
    print("=" * 60)
    
    # デモ用ディレクトリ設定
    demo_dir = Path("demo_documents")
    db_path = "demo_documents.db"
    cache_path = ".demo_cache.json"
    
    # クリーンアップ（前回のテスト結果）
    if Path(db_path).exists():
        os.remove(db_path)
    if Path(cache_path).exists():
        os.remove(cache_path)
    
    # ドキュメントストア初期化
    document_store = SQLiteDocumentStore(db_path)
    
    # 増分ローダー初期化
    incremental_loader = IncrementalLoader(
        document_store=document_store,
        cache_file=cache_path
    )
    
    print("\n🔍 Step 1: 初回処理（すべて新規文書）")
    print("-" * 40)
    
    # 初期文書作成
    create_sample_documents(demo_dir)
    
    # 初回処理
    results1 = incremental_loader.process_incremental(demo_dir)
    
    print(f"処理結果:")
    print(f"  新規: {len(results1['new'])}件")
    print(f"  更新: {len(results1['updated'])}件")
    print(f"  スキップ: {len(results1['skipped'])}件")
    print(f"  エラー: {len(results1['errors'])}件")
    
    # キャッシュ統計表示
    cache_stats = incremental_loader.get_cache_stats()
    print(f"  キャッシュ: {cache_stats['total_files']}ファイル")
    
    print("\n⏱️ Step 2: 再処理（変更なし - すべてスキップ）")
    print("-" * 40)
    
    # 変更なしで再処理
    results2 = incremental_loader.process_incremental(demo_dir)
    
    print(f"処理結果:")
    print(f"  新規: {len(results2['new'])}件")
    print(f"  更新: {len(results2['updated'])}件")
    print(f"  スキップ: {len(results2['skipped'])}件")
    print(f"  エラー: {len(results2['errors'])}件")
    
    print("\n📝 Step 3: 文書更新・追加後の増分処理")
    print("-" * 40)
    
    # 少し待機（ファイル更新時刻の差を確実にするため）
    time.sleep(1)
    
    # 文書を更新・追加
    update_sample_documents(demo_dir)
    
    # 増分処理実行
    results3 = incremental_loader.process_incremental(demo_dir)
    
    print(f"処理結果:")
    print(f"  新規: {len(results3['new'])}件")
    print(f"  更新: {len(results3['updated'])}件")
    print(f"  スキップ: {len(results3['skipped'])}件")
    print(f"  エラー: {len(results3['errors'])}件")
    
    # 処理された文書の詳細表示
    if results3['new']:
        print(f"\n新規追加された文書:")
        for doc in results3['new']:
            print(f"  - {doc.id}: {doc.metadata.get('path', 'Unknown')}")
    
    if results3['updated']:
        print(f"\n更新された文書:")
        for doc in results3['updated']:
            print(f"  - {doc.id}: {doc.metadata.get('path', 'Unknown')}")
    
    print("\n🗂️ Step 4: ディレクトリスキャン詳細")
    print("-" * 40)
    
    # ディレクトリスキャンの詳細情報
    new_files, updated_files, unchanged_files = incremental_loader.scan_directory(demo_dir)
    
    print(f"ディレクトリスキャン結果:")
    print(f"  新規ファイル: {len(new_files)}件")
    for f in new_files:
        print(f"    - {f}")
    
    print(f"  更新ファイル: {len(updated_files)}件")  
    for f in updated_files:
        print(f"    - {f}")
    
    print(f"  未変更ファイル: {len(unchanged_files)}件")
    for f in unchanged_files:
        print(f"    - {f}")
    
    print("\n🔄 Step 5: 強制再処理")
    print("-" * 40)
    
    # 特定ファイルを強制再処理
    force_files = {str(demo_dir / "company_policy.txt")}
    results4 = incremental_loader.process_incremental(demo_dir, force_reload=force_files)
    
    print(f"強制再処理結果:")
    print(f"  新規: {len(results4['new'])}件")
    print(f"  更新: {len(results4['updated'])}件")
    print(f"  スキップ: {len(results4['skipped'])}件")
    print(f"  エラー: {len(results4['errors'])}件")
    
    print("\n🗑️ Step 6: ファイル削除対応")
    print("-" * 40)
    
    # ファイルを削除
    deleted_file = demo_dir / "sales_report.txt"
    if deleted_file.exists():
        deleted_file.unlink()
        print(f"削除: {deleted_file}")
    
    # クリーンアップ実行
    deleted_docs = incremental_loader.cleanup_deleted_files([demo_dir])
    print(f"クリーンアップ: {len(deleted_docs)}件の文書を削除")
    
    print("\n📊 Step 7: 最終統計")
    print("-" * 40)
    
    # 最終的なキャッシュ統計
    final_stats = incremental_loader.get_cache_stats()
    print(f"最終キャッシュ統計:")
    print(f"  総ファイル数: {final_stats['total_files']}")
    print(f"  総サイズ: {final_stats['total_size_bytes']:,} bytes")
    print(f"  キャッシュファイル: {final_stats['cache_file']}")
    print(f"  最新処理: {final_stats.get('latest_processed', 'N/A')}")
    
    # ドキュメントストア統計
    try:
        store_stats = document_store.get_stats()
        print(f"\nドキュメントストア統計:")
        print(f"  総文書数: {store_stats.total_documents}")
    except Exception as e:
        print(f"ストア統計取得エラー: {e}")
    
    print("\n🎯 増分ローディングの利点")
    print("-" * 40)
    print("✅ 処理済み文書の自動スキップ")
    print("✅ ファイル変更の高速検出")
    print("✅ 大規模ディレクトリでの効率的処理")
    print("✅ 削除ファイルの自動クリーンアップ")
    print("✅ 強制再処理オプション")
    print("✅ 詳細な処理統計とログ")
    
    print("\n🚀 実運用での使用例")
    print("-" * 40)
    print("1. 定期バッチ処理（毎日深夜実行）")
    print("2. ファイル監視システムとの連携")
    print("3. Git hook による自動更新")
    print("4. Webアプリでの動的コンテンツ更新")
    print("5. 大規模ドキュメント管理システム")
    
    # クリーンアップ
    print("\n🧹 クリーンアップ")
    print("-" * 40)
    
    try:
        document_store.close()
        
        # デモファイルの削除（オプション）
        cleanup_demo = input("\nデモファイルを削除しますか？ (y/N): ").lower().strip()
        if cleanup_demo == 'y':
            import shutil
            if demo_dir.exists():
                shutil.rmtree(demo_dir)
            if Path(db_path).exists():
                os.remove(db_path)
            if Path(cache_path).exists():
                os.remove(cache_path)
            print("✅ デモファイルを削除しました")
        else:
            print(f"📁 デモファイルを保持: {demo_dir}")
            print(f"💾 データベース: {db_path}")
            print(f"🗂️ キャッシュ: {cache_path}")
    
    except Exception as e:
        print(f"クリーンアップエラー: {e}")
    
    print("\n🎉 増分ローディングデモ完了！")


if __name__ == "__main__":
    demo_incremental_loading()