import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredAPIFileLoader
from data_connection.custom.chroma_dc import ChromaDC
from model.large_models.language_models.chatglm.chatglm import ChatGLM

human_prompt = '中文概括论文内容'
save_dir = 'data_connection/source'

llm = ChatGLM()

files = os.listdir(save_dir)
full_files = []
loaders = []
for file in files:
    full_files.append(save_dir + f'/{file}')
for file in full_files:
    loader = UnstructuredAPIFileLoader(
        file, mode="elements"
    )
    loaders.append(loader)

chroma = ChromaDC()
for loader in loaders:
    chroma.add_doc(loader)
retriever = chroma.get_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
res = qa.run(human_prompt)
