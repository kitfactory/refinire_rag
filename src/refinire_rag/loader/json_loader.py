import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from refinire_rag.models.document import Document
from refinire_rag.loader.loader import Loader

class JSONLoader(Loader):
    """
    Load JSON files and convert their content into Documents.
    JSONファイルを読み込み、そのコンテンツをDocumentに変換する。
    """
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the JSONLoader.
        JSONLoaderを初期化する。

        Args:
            encoding (str): The encoding to use when reading the JSON file.
                           JSONファイルを読み込む際に使用するエンコーディング。
        """
        super().__init__()
        self.encoding = encoding

    def process(self, documents: List[Document]) -> Iterator[Document]:
        """
        Process the documents by loading JSON files and converting their content into Documents.
        ドキュメントを処理し、JSONファイルを読み込んでそのコンテンツをDocumentに変換する。

        Args:
            documents (List[Document]): List of documents containing file paths.
                                       ファイルパスを含むドキュメントのリスト。

        Yields:
            Document: A document containing the JSON content.
                     JSONコンテンツを含むドキュメント。
        """
        for doc in documents:
            file_path = doc.metadata.get('file_path')
            if not file_path:
                continue

            try:
                with open(file_path, 'r', encoding=self.encoding) as f:
                    json_content = json.load(f)

                # Convert JSON content to string
                # JSONコンテンツを文字列に変換
                content = json.dumps(json_content, ensure_ascii=False, indent=2)

                # Create metadata
                # メタデータを作成
                metadata = doc.metadata.copy()
                metadata.update({
                    'content_type': 'json',
                    'file_encoding': self.encoding
                })

                # Create a new document
                # 新しいドキュメントを作成
                yield Document(
                    id=doc.id,
                    content=content,
                    metadata=metadata
                )

            except FileNotFoundError:
                raise FileNotFoundError(f"JSON file not found: {file_path}")
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON in file {file_path}: {str(e)}")
            except Exception as e:
                raise Exception(f"Error processing JSON file {file_path}: {str(e)}") 