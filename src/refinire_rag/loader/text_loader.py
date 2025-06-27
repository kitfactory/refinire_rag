from typing import Iterable, Iterator, Optional, Any, Dict
from pathlib import Path
from refinire_rag.loader.loader import Loader
from refinire_rag.models.document import Document

class TextLoader(Loader):
    """
    Loader for text files.
    テキストファイルを読み込むためのローダー。

    Args:
        encoding (str): File encoding (default: 'utf-8')
        encoding (str): ファイルのエンコーディング（デフォルト: 'utf-8'）
    """
    def __init__(self, **kwargs):
        """
        Initialize TextLoader.
        TextLoaderを初期化する。

        Args:
            **kwargs: Configuration parameters, environment variables used as fallback
                     設定パラメータ、環境変数をフォールバックとして使用
                encoding (str): File encoding
                               ファイルのエンコーディング
                strip_whitespace (bool): Strip leading/trailing whitespace
                                       前後の空白を削除するか
                autodetect_encoding (bool): Attempt to auto-detect encoding
                                          エンコーディングの自動検出を試行するか
        """
        super().__init__()
        
        # Environment variable support with priority: kwargs > env vars > defaults
        import os
        
        self.encoding = kwargs.get('encoding', os.getenv('REFINIRE_RAG_TEXT_ENCODING', 'utf-8'))
        self.strip_whitespace = kwargs.get('strip_whitespace', os.getenv('REFINIRE_RAG_TEXT_STRIP_WHITESPACE', 'true').lower() == 'true')
        self.autodetect_encoding = kwargs.get('autodetect_encoding', os.getenv('REFINIRE_RAG_TEXT_AUTODETECT_ENCODING', 'false').lower() == 'true')
        self.max_file_size = int(kwargs.get('max_file_size', os.getenv('REFINIRE_RAG_TEXT_MAX_FILE_SIZE', '10485760')))  # 10MB default
        self.skip_empty_files = kwargs.get('skip_empty_files', os.getenv('REFINIRE_RAG_TEXT_SKIP_EMPTY_FILES', 'true').lower() == 'true')

    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """
        Load text files and convert them to Document objects.
        テキストファイルを読み込み、Documentオブジェクトに変換する。

        Args:
            documents: Iterable of Document objects containing file paths
            documents: ファイルパスを含むDocumentオブジェクトのイテラブル
            config: Optional configuration (not used in this implementation)
            config: オプション設定（この実装では使用しない）

        Yields:
            Document: Document object containing the text content
            Document: テキスト内容を含むDocumentオブジェクト

        Raises:
            FileNotFoundError: If the specified file does not exist
            FileNotFoundError: 指定されたファイルが存在しない場合
            UnicodeDecodeError: If the file cannot be decoded with the specified encoding
            UnicodeDecodeError: 指定されたエンコーディングでファイルをデコードできない場合
        """
        for doc in documents:
            file_path = Path(doc.metadata.get('file_path', ''))
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            try:
                with open(file_path, 'r', encoding=self.encoding) as f:
                    content = f.read()
            except UnicodeDecodeError as e:
                raise UnicodeDecodeError(
                    f"Failed to decode file {file_path} with encoding {self.encoding}",
                    e.object,
                    e.start,
                    e.end,
                    e.reason
                )

            # Create new Document with the text content
            # テキスト内容を含む新しいDocumentを作成
            yield Document(
                id=doc.id,
                content=content,
                metadata={
                    **doc.metadata,
                    'file_path': str(file_path),
                    'encoding': self.encoding,
                    'file_type': 'text',
                    'file_size': len(content)
                }
            )
    
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary
        
        Returns:
            Dict[str, Any]: Current configuration parameters
        """
        config = super().get_config()
        config.update({
            'encoding': self.encoding,
            'strip_whitespace': self.strip_whitespace,
            'autodetect_encoding': self.autodetect_encoding,
            'max_file_size': self.max_file_size,
            'skip_empty_files': self.skip_empty_files
        })
        return config 