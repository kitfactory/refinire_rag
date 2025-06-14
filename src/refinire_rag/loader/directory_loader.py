from typing import Iterable, Iterator, Optional, Any, Dict, Type
from pathlib import Path
from refinire_rag.loader.loader import Loader
from refinire_rag.loader.text_loader import TextLoader
from refinire_rag.loader.csv_loader import CSVLoader  # 仮想的に存在する前提
from refinire_rag.loader.json_loader import JSONLoader  # 仮想的に存在する前提
from refinire_rag.models.document import Document

class DirectoryLoader(Loader):
    """
    DirectoryLoader loads files from a directory, dispatching to the appropriate loader by file extension.
    ディレクトリ内のファイルを拡張子ごとに適切なLoaderで読み込むローダー。

    Args:
        extension_loader_map (dict): 拡張子とLoaderのマッピング
        recursive (bool): サブディレクトリも再帰的に探索するか
    """
    def __init__(self, extension_loader_map: Optional[Dict[str, Loader]] = None, recursive: bool = True):
        """
        Initialize DirectoryLoader.
        DirectoryLoaderを初期化する。

        Args:
            extension_loader_map (dict): 拡張子とLoaderのマッピング
            recursive (bool): サブディレクトリも再帰的に探索するか
        """
        self.recursive = recursive
        # デフォルトの拡張子→Loaderマッピング
        self.extension_loader_map = extension_loader_map or {
            '.txt': TextLoader(),
            '.csv': CSVLoader(),
            '.json': JSONLoader(),
        }

    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """
        Load all files in the given directories, dispatching to the appropriate loader by extension.
        指定ディレクトリ内の全ファイルを拡張子ごとに適切なLoaderで読み込む。

        Args:
            documents: Iterable of Document objects (metadata['dir_path']にディレクトリパス)
            config: Optional configuration
        Yields:
            Document: Loaded Document objects
        """
        for doc in documents:
            dir_path = Path(doc.metadata.get('dir_path', ''))
            if not dir_path.exists() or not dir_path.is_dir():
                continue
            # ファイル探索
            pattern = '**/*' if self.recursive else '*'
            for file_path in dir_path.glob(pattern):
                if not file_path.is_file():
                    continue
                ext = file_path.suffix.lower()
                loader = self.extension_loader_map.get(ext)
                if loader is None:
                    continue  # 未対応拡張子はスキップ
                # ファイルごとにDocumentを生成し、Loaderで処理
                file_doc = Document(
                    id=f"{doc.id}:{file_path.name}",  # 親DocumentのIDとファイル名を組み合わせてIDを生成
                    content="",
                    metadata={
                        **doc.metadata,
                        'file_path': str(file_path),
                        'parent_dir': str(dir_path),
                        'file_ext': ext
                    }
                )
                yield from loader.process([file_doc]) 