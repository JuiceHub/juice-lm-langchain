import os

from langchain.document_loaders import UnstructuredAPIFileLoader

from data_connection.custom.chroma_dc import ChromaDC

filename = 'A Survey on Multimodal Large Language Models.pdf'
save_path = 'data_connection/source'
save_file = os.path.join(save_path, filename)
loader = UnstructuredAPIFileLoader(
    save_file,
)
chroma = ChromaDC()
chroma.add_doc(loader)

doc = chroma.vector_store.get(where={"source": save_file})
print(doc)
print(save_file)
print(chroma.vector_store._collection.count())
