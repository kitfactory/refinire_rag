from typing import Iterable, Iterator, Optional, Any
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
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize TextLoader.
        TextLoaderを初期化する。

        Args:
            encoding (str): File encoding
            encoding (str): ファイルのエンコーディング
        """
        self.encoding = encoding

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
                    'file_type': 'text'
                }
            ) 