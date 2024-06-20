# 模型
import logging

from fastapi import APIRouter
from langchain.llms import BaseLLM

from model.large_models.language_models import ChatGLM, Vicuna, Llama

model = APIRouter()
logger = logging.getLogger(__name__)

llm: BaseLLM
model_name: str = 'None'
llm_map = {
    'chatglm': ChatGLM,
    'vicuna': Vicuna,
    'llama': Llama
}

llm_name_map = {
    'chatglm': 'ChatGLM',
    'vicuna': 'Vicuna',
    'llama': 'Llama'
}


@model.post('/load_model')
def load_model(data: dict):
    global llm
    global model_name
    model_name = data.get('model_name')
    llm = llm_map[model_name]()


@model.get('/model_name')
def get_model_name():
    global model_name
    return model_name


@model.get('/models')
def get_models():
    global llm_name_map
    return llm_name_map


def get_llm():
    return llm
