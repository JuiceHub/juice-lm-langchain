import os

from data_connection.custom.chroma_dc import ChromaDC

save_path = 'data_connection/source'
files = ['A Survey on Multimodal Large Language Models.pdf']
full_files = []
for file in files:
    full_files.append(save_path + f'/{file}')

chroma = ChromaDC()
print(full_files[0])
doc = chroma.vector_store.get(where={"source": full_files[0]})
print(doc)
print(chroma.vector_store._collection.count())
