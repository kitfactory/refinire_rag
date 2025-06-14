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
    def __init__(self, encoding: str = 'utf-8', include_header: bool = False):
        """
        Initialize the CSVLoader.
        CSVLoaderを初期化する。

        Args:
            encoding (str): The encoding to use when reading the CSV file.
                           CSVファイルを読み込む際に使用するエンコーディング。
            include_header (bool): Whether to include the header row in each document.
                                 各ドキュメントにヘッダー行を含めるかどうか。
        """
        super().__init__()
        self.encoding = encoding
        self.include_header = include_header

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
                            'total_rows': sum(1 for _ in csv.DictReader(open(file_path, 'r', encoding=self.encoding)))
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