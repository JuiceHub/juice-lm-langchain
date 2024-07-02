import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_google_genai import ChatGoogleGenerativeAI
# from model.language_models.llama.llama3 import Llama3

import torch
from langchain_core.messages import HumanMessage, SystemMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

os.environ["GOOGLE_API_KEY"] = "AIzaSyAr_chv_fdwm2uJWBQ8tFH2KiKei6wmSxI"
model = ChatGoogleGenerativeAI(model="gemini-pro")

# model = Llama3()

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

parser = StrOutputParser()

chain = prompt_template | model | parser

result = chain.invoke({"language": "italian", "text": "hi"})

print(result)
