import os

from data_connection.custom.chroma_dc import ChromaDC

save_path = 'data_connection/source'
filename = 'A Survey on Multimodal Large Language Models.pdf'
save_file = os.path.join(save_path, filename)
# os.remove(save_file)

chroma = ChromaDC()
doc = chroma.vector_store.get(where={"source": save_file})
print(doc)
chroma.vector_store.delete(ids=doc.get('ids'))
