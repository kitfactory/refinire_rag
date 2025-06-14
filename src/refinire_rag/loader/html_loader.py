import html.parser
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from refinire_rag.models.document import Document
from refinire_rag.loader.loader import Loader

class HTMLParser(html.parser.HTMLParser):
    """
    Custom HTML parser to extract text content.
    テキストコンテンツを抽出するためのカスタムHTMLパーサー。
    """
    def __init__(self):
        """
        Initialize the HTML parser.
        HTMLパーサーを初期化する。
        """
        super().__init__()
        self.text_parts = []
        self.current_tag = None
        self.ignore_tags = {'script', 'style', 'meta', 'link', 'head'}
        self.block_tags = {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th'}

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        """
        Handle the start of an HTML tag.
        HTMLタグの開始を処理する。

        Args:
            tag (str): The tag name.
                      タグ名。
            attrs (List[tuple[str, Optional[str]]]): The tag attributes.
                                                    タグの属性。
        """
        self.current_tag = tag.lower()
        if self.current_tag in self.block_tags:
            self.text_parts.append('\n')

    def handle_endtag(self, tag: str) -> None:
        """
        Handle the end of an HTML tag.
        HTMLタグの終了を処理する。

        Args:
            tag (str): The tag name.
                      タグ名。
        """
        if tag.lower() in self.block_tags:
            self.text_parts.append('\n')
        self.current_tag = None

    def handle_data(self, data: str) -> None:
        """
        Handle the text content of an HTML tag.
        HTMLタグのテキストコンテンツを処理する。

        Args:
            data (str): The text content.
                       テキストコンテンツ。
        """
        if self.current_tag not in self.ignore_tags:
            # Remove extra whitespace
            # 余分な空白を削除
            data = re.sub(r'\s+', ' ', data.strip())
            if data:
                self.text_parts.append(data)

    def get_text(self) -> str:
        """
        Get the extracted text content.
        抽出されたテキストコンテンツを取得する。

        Returns:
            str: The extracted text content.
                 抽出されたテキストコンテンツ。
        """
        # Join text parts and remove extra whitespace
        # テキストパーツを結合し、余分な空白を削除
        text = ' '.join(self.text_parts)
        return re.sub(r'\s+', ' ', text).strip()

class HTMLLoader(Loader):
    """
    Load HTML files and convert their content into Documents.
    HTMLファイルを読み込み、そのコンテンツをDocumentに変換する。
    """
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the HTMLLoader.
        HTMLLoaderを初期化する。

        Args:
            encoding (str): The encoding to use when reading the HTML file.
                           HTMLファイルを読み込む際に使用するエンコーディング。
        """
        super().__init__()
        self.encoding = encoding

    def process(self, documents: List[Document]) -> Iterator[Document]:
        """
        Process the documents by loading HTML files and converting their content into Documents.
        ドキュメントを処理し、HTMLファイルを読み込んでそのコンテンツをDocumentに変換する。

        Args:
            documents (List[Document]): List of documents containing file paths.
                                       ファイルパスを含むドキュメントのリスト。

        Yields:
            Document: A document containing the extracted text content.
                     抽出されたテキストコンテンツを含むドキュメント。
        """
        for doc in documents:
            file_path = doc.metadata.get('file_path')
            if not file_path:
                continue

            try:
                with open(file_path, 'r', encoding=self.encoding) as f:
                    html_content = f.read()

                # Parse HTML content
                # HTMLコンテンツを解析
                parser = HTMLParser()
                parser.feed(html_content)
                text_content = parser.get_text()

                # Create metadata
                # メタデータを作成
                metadata = doc.metadata.copy()
                metadata.update({
                    'content_type': 'html',
                    'file_encoding': self.encoding
                })

                # Create a new document
                # 新しいドキュメントを作成
                yield Document(
                    id=doc.id,
                    content=text_content,
                    metadata=metadata
                )

            except FileNotFoundError:
                raise FileNotFoundError(f"HTML file not found: {file_path}")
            except Exception as e:
                raise Exception(f"Error processing HTML file {file_path}: {str(e)}") 