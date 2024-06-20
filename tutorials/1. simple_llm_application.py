import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_google_vertexai import ChatVertexAI

from langchain_core.messages import HumanMessage, SystemMessage

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

os.environ["GOOGLE_API_KEY"] = "AIzaSyAr_chv_fdwm2uJWBQ8tFH2KiKei6wmSxI"

model = ChatVertexAI(model="gemini-pro")

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

model.invoke(messages)

parser = StrOutputParser()

result = model.invoke(messages)

chain = prompt_template | model | parser

chain.invoke({"language": "italian", "text": "hi"})