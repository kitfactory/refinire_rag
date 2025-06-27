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
    def __init__(self, **kwargs):
        """
        Initialize the JSONLoader.
        JSONLoaderを初期化する。

        Args:
            **kwargs: Configuration parameters, environment variables used as fallback
                     設定パラメータ、環境変数をフォールバックとして使用
                encoding (str): The encoding to use when reading the JSON file.
                               JSONファイルを読み込む際に使用するエンコーディング。
                json_path (str): JSON path expression to extract specific data
                                特定のデータを抽出するJSONパス式
                flatten_objects (bool): Flatten nested JSON objects
                                      ネストしたJSONオブジェクトを平均化するか
        """
        super().__init__()
        
        # Environment variable support with priority: kwargs > env vars > defaults
        import os
        
        self.encoding = kwargs.get('encoding', os.getenv('REFINIRE_RAG_JSON_ENCODING', 'utf-8'))
        self.json_path = kwargs.get('json_path', os.getenv('REFINIRE_RAG_JSON_PATH', None))
        self.flatten_objects = kwargs.get('flatten_objects', os.getenv('REFINIRE_RAG_JSON_FLATTEN_OBJECTS', 'false').lower() == 'true')
        self.ensure_ascii = kwargs.get('ensure_ascii', os.getenv('REFINIRE_RAG_JSON_ENSURE_ASCII', 'false').lower() == 'true')
        self.indent_size = int(kwargs.get('indent_size', os.getenv('REFINIRE_RAG_JSON_INDENT_SIZE', '2')))
        self.content_field = kwargs.get('content_field', os.getenv('REFINIRE_RAG_JSON_CONTENT_FIELD', None))
        self.metadata_fields = kwargs.get('metadata_fields', None)
        if self.metadata_fields is None:
            meta_fields_env = os.getenv('REFINIRE_RAG_JSON_METADATA_FIELDS')
            self.metadata_fields = meta_fields_env.split(',') if meta_fields_env else []

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
    
    @classmethod
    def from_env(cls) -> "JSONLoader":
        """Create JSONLoader instance from environment variables
        
        環境変数からJSONLoaderインスタンスを作成
        
        Returns:
            JSONLoader instance configured from environment
        """
        return cls()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary
        
        Returns:
            Dict[str, Any]: Current configuration parameters
        """
        config = super().get_config()
        config.update({
            'encoding': self.encoding,
            'json_path': self.json_path,
            'flatten_objects': self.flatten_objects,
            'ensure_ascii': self.ensure_ascii,
            'indent_size': self.indent_size,
            'content_field': self.content_field,
            'metadata_fields': self.metadata_fields
        })
        return config 