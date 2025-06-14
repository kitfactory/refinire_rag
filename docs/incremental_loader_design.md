# インクリメンタルローダー設計書

## 概要

ディレクトリ内の文書の変更（追加・更新・削除）を検出し、CorpusStoreと同期するためのインクリメンタルローダー機能の設計。

## 要件

1. **差分検出**: ディレクトリ内のファイルとCorpusStore内の文書の差分を検出
2. **変更タイプの識別**: 追加・更新・削除の3つの変更タイプを識別
3. **メタデータベースの比較**: ファイルの最終更新日時、サイズ、ハッシュ値を使用
4. **効率的な処理**: 変更されたファイルのみを処理

## アーキテクチャ

### 1. コアクラス

#### IncrementalDirectoryLoader
```python
class IncrementalDirectoryLoader:
    """
    ディレクトリの変更を検出し、インクリメンタルにロードする
    """
    def __init__(self, directory_path: str, corpus_store: CorpusStore):
        self.directory_path = Path(directory_path)
        self.corpus_store = corpus_store
        self.file_tracker = FileTracker()
        self.directory_loader = DirectoryLoader(directory_path)
    
    def detect_changes(self) -> ChangeSet:
        """ディレクトリとCorpusStoreの差分を検出"""
        pass
    
    def sync(self) -> SyncResult:
        """差分に基づいてCorpusStoreを同期"""
        pass
```

#### FileTracker
```python
class FileTracker:
    """
    ファイルの状態を追跡・比較する
    """
    def scan_directory(self, path: Path) -> Dict[str, FileInfo]:
        """ディレクトリ内のファイル情報を収集"""
        pass
    
    def get_corpus_files(self, corpus_store: CorpusStore) -> Dict[str, FileInfo]:
        """CorpusStore内のファイル情報を取得"""
        pass
    
    def compare(self, current: Dict[str, FileInfo], stored: Dict[str, FileInfo]) -> ChangeSet:
        """ファイル情報を比較して変更を検出"""
        pass
```

### 2. データモデル

#### FileInfo
```python
@dataclass
class FileInfo:
    """
    ファイルの状態情報
    """
    path: str
    size: int
    modified_at: datetime
    hash_md5: str
    file_type: str
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'FileInfo':
        """ファイルからFileInfoを生成"""
        pass
    
    @classmethod
    def from_document(cls, document: Document) -> 'FileInfo':
        """DocumentのメタデータからFileInfoを生成"""
        pass
```

#### ChangeSet
```python
@dataclass
class ChangeSet:
    """
    検出された変更の集合
    """
    added: List[str]       # 新規追加されたファイル
    modified: List[str]    # 更新されたファイル
    deleted: List[str]     # 削除されたファイル
    unchanged: List[str]   # 変更されていないファイル
    
    def has_changes(self) -> bool:
        """変更があるかどうか"""
        return bool(self.added or self.modified or self.deleted)
```

#### SyncResult
```python
@dataclass
class SyncResult:
    """
    同期処理の結果
    """
    added_documents: List[Document]
    updated_documents: List[Document]
    deleted_document_ids: List[str]
    errors: List[str]
    
    @property
    def total_processed(self) -> int:
        return len(self.added_documents) + len(self.updated_documents) + len(self.deleted_document_ids)
```

## 実装戦略

### 1. 変更検出アルゴリズム

```python
def detect_changes(self) -> ChangeSet:
    # 1. ディレクトリ内の現在のファイル情報を収集
    current_files = self.file_tracker.scan_directory(self.directory_path)
    
    # 2. CorpusStore内のファイル情報を取得
    stored_files = self.file_tracker.get_corpus_files(self.corpus_store)
    
    # 3. 差分を計算
    changes = self.file_tracker.compare(current_files, stored_files)
    
    return changes
```

### 2. 同期処理アルゴリズム

```python
def sync(self) -> SyncResult:
    changes = self.detect_changes()
    result = SyncResult()
    
    # 新規追加ファイルの処理
    for file_path in changes.added:
        try:
            docs = self.directory_loader.load_file(file_path)
            self.corpus_store.add_documents(docs)
            result.added_documents.extend(docs)
        except Exception as e:
            result.errors.append(f"Error adding {file_path}: {e}")
    
    # 更新ファイルの処理
    for file_path in changes.modified:
        try:
            # 既存文書を削除
            old_docs = self.corpus_store.get_documents_by_path(file_path)
            self.corpus_store.delete_documents([doc.id for doc in old_docs])
            
            # 新しい文書を追加
            new_docs = self.directory_loader.load_file(file_path)
            self.corpus_store.add_documents(new_docs)
            result.updated_documents.extend(new_docs)
        except Exception as e:
            result.errors.append(f"Error updating {file_path}: {e}")
    
    # 削除ファイルの処理
    for file_path in changes.deleted:
        try:
            docs = self.corpus_store.get_documents_by_path(file_path)
            doc_ids = [doc.id for doc in docs]
            self.corpus_store.delete_documents(doc_ids)
            result.deleted_document_ids.extend(doc_ids)
        except Exception as e:
            result.errors.append(f"Error deleting {file_path}: {e}")
    
    return result
```

### 3. CorpusStoreの拡張

CorpusStoreに以下のメソッドを追加する必要があります：

```python
class CorpusStore:
    def get_documents_by_path(self, file_path: str) -> List[Document]:
        """指定されたファイルパスの文書を取得"""
        pass
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """指定されたIDの文書を削除"""
        pass
    
    def get_file_metadata(self) -> Dict[str, FileInfo]:
        """保存されている全ファイルのメタデータを取得"""
        pass
```

## 使用例

```python
from refinire_rag.loader.incremental_directory_loader import IncrementalDirectoryLoader
from refinire_rag.corpus_store.sqlite_corpus_store import SQLiteCorpusStore

# 初期化
corpus_store = SQLiteCorpusStore("documents.db")
loader = IncrementalDirectoryLoader("/path/to/documents", corpus_store)

# 初回ロード
initial_result = loader.sync()
print(f"初回ロード: {initial_result.total_processed}個の文書を処理")

# 後で変更を同期
changes = loader.detect_changes()
if changes.has_changes():
    print(f"変更検出: 追加{len(changes.added)}, 更新{len(changes.modified)}, 削除{len(changes.deleted)}")
    
    sync_result = loader.sync()
    print(f"同期完了: {sync_result.total_processed}個の文書を処理")
    
    if sync_result.errors:
        print(f"エラー: {len(sync_result.errors)}件")
        for error in sync_result.errors:
            print(f"  - {error}")
```

## メタデータ設計

各文書に以下のメタデータを追加：

```python
metadata = {
    'file_path': '/path/to/file.txt',
    'file_size': 1024,
    'modified_at': '2025-06-14T18:30:00',
    'file_hash': 'abc123...',
    'loaded_at': '2025-06-14T18:30:05',
    'loader_version': '1.0.0'
}
```

## 利点

1. **効率性**: 変更されたファイルのみを処理
2. **追跡可能性**: 各文書の由来と更新履歴を追跡
3. **一貫性**: ディレクトリとCorpusStoreの状態を同期
4. **拡張性**: 他のローダー（S3、データベースなど）にも適用可能

## 今後の拡張

1. **バックアップ機能**: 更新前の文書をバックアップ
2. **並列処理**: 大量ファイルの並列処理
3. **フィルタリング**: ファイルタイプや更新日時によるフィルタリング
4. **通知機能**: 変更検出時の通知機能