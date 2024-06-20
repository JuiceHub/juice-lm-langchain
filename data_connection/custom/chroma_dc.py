from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from data_connection.custom.dc.data_connection import DataConnection
from langchain.text_splitter import CharacterTextSplitter


class ChromaDC(DataConnection):
    vector_store: Chroma

    def __init__(self,
                 embedding_model_name='/root/.cache/torch/sentence_transformers/GanymedeNil_text2vec-large-chinese',
                 persist=True):
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=embedding_model_name
        )

        collection_name = 'test'
        if persist:
            persist_directory = 'data_connection/persist_vector_store/chroma'
        else:
            persist_directory = None

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
