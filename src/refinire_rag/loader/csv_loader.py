import csv
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from refinire_rag.models.document import Document
from refinire_rag.loader.loader import Loader

class CSVLoader(Loader):
    """
    Load CSV files and convert each row into a Document.
    CSVファイルを読み込み、各行をDocumentに変換する。
    """
    def __init__(self, **kwargs):
        """
        Initialize the CSVLoader.
        CSVLoaderを初期化する。

        Args:
            **kwargs: Configuration parameters, environment variables used as fallback
                     設定パラメータ、環境変数をフォールバックとして使用
                encoding (str): The encoding to use when reading the CSV file.
                               CSVファイルを読み込む際に使用するエンコーディング。
                include_header (bool): Whether to include the header row in each document.
                                     各ドキュメントにヘッダー行を含めるかどうか。
                delimiter (str): CSV field delimiter
                               CSVフィールドデリミタ
        """
        super().__init__()
        
        # Environment variable support with priority: kwargs > env vars > defaults
        import os
        
        self.encoding = kwargs.get('encoding', os.getenv('REFINIRE_RAG_CSV_ENCODING', 'utf-8'))
        self.include_header = kwargs.get('include_header', os.getenv('REFINIRE_RAG_CSV_INCLUDE_HEADER', 'false').lower() == 'true')
        self.delimiter = kwargs.get('delimiter', os.getenv('REFINIRE_RAG_CSV_DELIMITER', ','))
        self.quotechar = kwargs.get('quotechar', os.getenv('REFINIRE_RAG_CSV_QUOTECHAR', '"'))
        self.skip_blank_lines = kwargs.get('skip_blank_lines', os.getenv('REFINIRE_RAG_CSV_SKIP_BLANK_LINES', 'true').lower() == 'true')
        self.text_column = kwargs.get('text_column', os.getenv('REFINIRE_RAG_CSV_TEXT_COLUMN', None))
        self.content_columns = kwargs.get('content_columns', None)
        if self.content_columns is None:
            content_cols_env = os.getenv('REFINIRE_RAG_CSV_CONTENT_COLUMNS')
            self.content_columns = content_cols_env.split(',') if content_cols_env else None

    def process(self, documents: List[Document]) -> Iterator[Document]:
        """
        Process the documents by loading CSV files and converting each row into a Document.
        ドキュメントを処理し、CSVファイルを読み込んで各行をDocumentに変換する。

        Args:
            documents (List[Document]): List of documents containing file paths.
                                       ファイルパスを含むドキュメントのリスト。

        Yields:
            Document: A document for each row in the CSV file.
                     CSVファイルの各行に対応するドキュメント。
        """
        for doc in documents:
            file_path = doc.metadata.get('file_path')
            if not file_path:
                continue

            try:
                # First pass: count total rows
                # 最初のパス：総行数をカウント
                with open(file_path, 'r', encoding=self.encoding) as f:
                    total_rows = sum(1 for _ in csv.DictReader(f))

                # Second pass: process rows
                # 二回目のパス：行を処理
                with open(file_path, 'r', encoding=self.encoding) as f:
                    reader = csv.DictReader(f)
                    columns = reader.fieldnames
                    if not columns:
                        continue

                    # ヘッダー行を取得
                    # Get header row
                    header_row = dict(zip(columns, columns))

                    for i, row in enumerate(reader):
                        # Create metadata for the row
                        # 行のメタデータを作成
                        metadata = doc.metadata.copy()
                        metadata.update({
                            'columns': columns,
                            'row_index': i,
                            'total_rows': total_rows
                        })

                        # コンテンツを作成
                        # Create content
                        if self.include_header:
                            # ヘッダー行と該当行を組み合わせる
                            # Combine header row and current row
                            content = f"Header: {header_row}\nRow: {row}"
                        else:
                            content = str(row)

                        # Create a new document for the row
                        # 行の新しいドキュメントを作成
                        yield Document(
                            id=f"{doc.id}_row_{i}",
                            content=content,
                            metadata=metadata
                        )
            except FileNotFoundError:
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            except Exception as e:
                raise Exception(f"Error processing CSV file {file_path}: {str(e)}")
    
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary
        
        Returns:
            Dict[str, Any]: Current configuration parameters
        """
        config = super().get_config()
        config.update({
            'encoding': self.encoding,
            'include_header': self.include_header,
            'delimiter': self.delimiter,
            'quotechar': self.quotechar,
            'skip_blank_lines': self.skip_blank_lines,
            'text_column': self.text_column,
            'content_columns': self.content_columns
        })
        return config 