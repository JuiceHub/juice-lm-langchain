from typing import List

from langchain.vectorstores import VectorStore
from langchain.document_loaders.base import BaseLoader


class DataConnection:
    text_splitter = None
    embeddings = None
    vector_store: VectorStore = None
    retriever = None

    def add_doc(self, loader: BaseLoader):
        print('开始加载文件')
        documents = loader.load()
        print('开始分割文件')
        docs = self.text_splitter.split_documents(documents)
        print('存入向量存储')

        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        self.vector_store.add_texts(
            metadatas=metadatas,
            texts=texts
        )

    def get_retriever(self):
        return self.vector_store.as_retriever()
