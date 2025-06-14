from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
from .models import Document

class CorpusStore(ABC):
    """
    Abstract base class for storing documents with metadata.
    メタデータを持つ文書を保存するための抽象基底クラス
    """
    @abstractmethod
    def add_document(self, document: Document) -> str:
        """
        Add a document to the store.
        ストアに文書を追加する
        
        Args:
            document (Document): Document to add
            追加する文書
            
        Returns:
            str: Document ID
            文書ID
        """
        pass

    @abstractmethod
    def update_document(self, document_id: str, document: Document) -> bool:
        """
        Update an existing document in the store.
        ストア内の既存の文書を更新する
        
        Args:
            document_id (str): ID of the document to update
            更新する文書のID
            document (Document): Updated document
            更新された文書
            
        Returns:
            bool: True if update was successful, False otherwise
            更新が成功した場合はTrue、それ以外はFalse
        """
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Get a document by its ID.
        IDによって文書を取得する
        
        Args:
            document_id (str): Document ID
            文書ID
            
        Returns:
            Optional[Document]: Document if found, None otherwise
            見つかった場合は文書、それ以外はNone
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the store.
        ストアから文書を削除する
        
        Args:
            document_id (str): Document ID
            文書ID
            
        Returns:
            bool: True if deletion was successful, False otherwise
            削除が成功した場合はTrue、それ以外はFalse
        """
        pass

    @abstractmethod
    def list_documents(self, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        List documents with optional metadata filtering.
        オプションのメタデータフィルタリングを使用して文書をリストする
        
        Args:
            metadata_filter (Optional[Dict[str, Any]]): Filter criteria for metadata
            メタデータのフィルタ条件
            
        Returns:
            List[Document]: List of matching documents
            条件に一致する文書のリスト
        """
        pass

    def export_documents(self, 
                        export_dir: Path,
                        metadata_filter: Optional[Dict[str, Any]] = None,
                        file_extension: str = ".txt",
                        include_metadata: bool = True) -> List[Path]:
        """
        Export documents to files in the specified directory.
        指定されたディレクトリに文書をファイルとしてエクスポートする
        
        Args:
            export_dir (Path): Directory to export documents to
            エクスポート先のディレクトリ
            metadata_filter (Optional[Dict[str, Any]]): Filter criteria for metadata
            メタデータのフィルタ条件
            file_extension (str): Extension for exported files (default: .txt)
            エクスポートするファイルの拡張子（デフォルト: .txt）
            include_metadata (bool): Whether to include metadata in the exported files (default: True)
            エクスポートするファイルにメタデータを含めるかどうか（デフォルト: True）
            
        Returns:
            List[Path]: List of paths to exported files
            エクスポートされたファイルのパスのリスト
            
        Raises:
            OSError: If the export directory cannot be created or written to
            エクスポートディレクトリが作成できない、または書き込みできない場合
        """
        # Create export directory if it doesn't exist
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Get documents matching the filter
        documents = self.list_documents(metadata_filter)
        exported_paths = []
        
        for doc in documents:
            # Create filename from document ID
            filename = f"{doc.id}{file_extension}"
            file_path = export_dir / filename
            
            # Prepare content
            content = doc.content
            if include_metadata:
                metadata_str = "\n".join(f"{k}: {v}" for k, v in doc.metadata.items())
                content = f"---\n{metadata_str}\n---\n\n{content}"
            
            # Write to file
            file_path.write_text(content, encoding='utf-8')
            exported_paths.append(file_path)
        
        return exported_paths 