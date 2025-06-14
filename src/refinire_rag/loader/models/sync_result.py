"""
SyncResult model for representing synchronization results
同期結果を表現するSyncResultモデル
"""

from dataclasses import dataclass, field
from typing import List
from refinire_rag.models.document import Document

@dataclass 
class SyncResult:
    """
    Results of a synchronization operation between directory and corpus store
    ディレクトリとコーパスストア間の同期操作の結果
    
    This class provides detailed information about what was processed
    during an incremental sync operation.
    このクラスは、インクリメンタル同期操作中に処理された内容の
    詳細情報を提供します。
    """
    added_documents: List[Document] = field(default_factory=list)    # Successfully added documents / 正常に追加された文書
    updated_documents: List[Document] = field(default_factory=list)  # Successfully updated documents / 正常に更新された文書
    deleted_document_ids: List[str] = field(default_factory=list)    # Successfully deleted document IDs / 正常に削除された文書ID
    errors: List[str] = field(default_factory=list)                  # Error messages / エラーメッセージ
    
    @property
    def total_processed(self) -> int:
        """
        Get the total number of documents processed successfully
        正常に処理された文書の総数を取得
        
        Returns:
            Total count of successfully processed documents
            正常に処理された文書の総数
        """
        return len(self.added_documents) + len(self.updated_documents) + len(self.deleted_document_ids)
    
    @property
    def has_errors(self) -> bool:
        """
        Check if there were any errors during sync
        同期中にエラーがあったかどうかをチェック
        
        Returns:
            True if there were errors
            エラーがあった場合True
        """
        return bool(self.errors)
    
    @property
    def success_rate(self) -> float:
        """
        Calculate the success rate of the sync operation
        同期操作の成功率を計算
        
        Returns:
            Success rate as a float between 0.0 and 1.0
            0.0から1.0の間の浮動小数点数としての成功率
        """
        total_attempted = self.total_processed + len(self.errors)
        if total_attempted == 0:
            return 1.0
        return self.total_processed / total_attempted
    
    def get_summary(self) -> dict:
        """
        Get a summary of the sync results
        同期結果のサマリーを取得
        
        Returns:
            Dictionary with sync statistics
            同期統計を含む辞書
        """
        return {
            'added': len(self.added_documents),
            'updated': len(self.updated_documents), 
            'deleted': len(self.deleted_document_ids),
            'total_processed': self.total_processed,
            'errors': len(self.errors),
            'has_errors': self.has_errors,
            'success_rate': self.success_rate
        }
    
    def add_error(self, error_message: str):
        """
        Add an error message to the result
        結果にエラーメッセージを追加
        
        Args:
            error_message: Description of the error
            error_message: エラーの説明
        """
        self.errors.append(error_message)
    
    def merge(self, other: 'SyncResult') -> 'SyncResult':
        """
        Merge this result with another SyncResult
        この結果を別のSyncResultとマージ
        
        Args:
            other: Another SyncResult to merge
            other: マージする別のSyncResult
            
        Returns:
            New SyncResult with combined results
            結合された結果を含む新しいSyncResult
        """
        return SyncResult(
            added_documents=self.added_documents + other.added_documents,
            updated_documents=self.updated_documents + other.updated_documents,
            deleted_document_ids=self.deleted_document_ids + other.deleted_document_ids,
            errors=self.errors + other.errors
        )
    
    def __str__(self) -> str:
        """
        String representation of the sync result
        同期結果の文字列表現
        """
        summary = self.get_summary()
        return (
            f"SyncResult(added={summary['added']}, "
            f"updated={summary['updated']}, "
            f"deleted={summary['deleted']}, "
            f"errors={summary['errors']})"
        )