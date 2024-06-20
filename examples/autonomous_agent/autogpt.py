from langchain.experimental import AutoGPT
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

from data_connection.custom.chroma_dc import ChromaDC
from model.large_models.language_models.chatglm.chatglm_chat_model import ChatGLMChatModel

search = SerpAPIWrapper(serpapi_api_key='b1c6f89c612b9fba33e06b87ab00d475b2535f235950234d706ef7ffc4ffba98')
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    WriteFileTool(),
    ReadFileTool(),
]

chroma = ChromaDC(persist=False)

agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="AI",
    tools=tools,
    llm=ChatGLMChatModel(),
    memory=chroma.get_retriever(),
)
# Set verbose to be true
agent.chain.verbose = True

agent.run(["Please write a report on the top 10 nba players of all time"])
