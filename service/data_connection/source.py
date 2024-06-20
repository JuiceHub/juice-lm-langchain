import logging
import os

from fastapi import APIRouter, UploadFile
from langchain.document_loaders import UnstructuredAPIFileLoader

from data_connection.custom.chroma_dc import ChromaDC

source = APIRouter()
logger = logging.getLogger(__name__)

chroma = ChromaDC()


@source.get('')
def read_all_source():
    global chroma
    save_path = 'data_connection/source'
    files = os.listdir(save_path)
    full_files = []
    for file in files:
        full_files.append({
            'filename': file,
            'source': save_path + '/' + file
        })
    return full_files


@source.post('/upload')
async def upload_source(file: UploadFile):
    global chroma
    filename = file.filename
    save_path = 'data_connection/source'

    save_file = os.path.join(save_path, filename)
    # 本地保存文件
    if not os.path.exists(save_file):
        f = open(save_file, 'wb')
        data = await file.read()
        f.write(data)
        f.close()

        loader = UnstructuredAPIFileLoader(
            save_file,
        )
        chroma.add_doc(loader)
        return f'成功上传文件{filename}'
    else:
        return '文件已存在'


@source.delete('')
async def delete_source(filename: str):
    global chroma
    save_path = 'data_connection/source'
    save_file = os.path.join(save_path, filename)

    doc = chroma.vector_store.get(where={"source": save_file})
    print(doc)
    chroma.vector_store.delete(ids=doc.get('ids'))
    os.remove(save_file)
    return f'成功删除文件{filename}'


def get_data_connection():
    return chroma
