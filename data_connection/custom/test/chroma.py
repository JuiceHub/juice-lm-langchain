from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name='GanymedeNil/text2vec-large-chinese')

chroma = Chroma(
    collection_name="test",
    persist_directory='../chroma',
    embedding_function=embeddings
)

chroma.add_texts(ids=["1"], texts=["a"], metadatas=[{'source': '1'}])

print(chroma.get(where={"source": "1"}))
