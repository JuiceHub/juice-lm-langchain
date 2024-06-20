import logging

from fastapi import APIRouter
from langchain import LLMChain, PromptTemplate, Wikipedia
from langchain.agents import initialize_agent, AgentType
from langchain.agents.react.base import DocstoreExplorer
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

from service.chat.model import get_llm
from service.chat.session import get_message_history
from service.data_connection.source import get_data_connection

chains = APIRouter()
logger = logging.getLogger(__name__)


@chains.post('/llm_chain')
def llm_chain(data: dict):
    human_prompt = data.get('prompt')
    llm = get_llm()
    # model_args = data.get('model_args')
    # llm.load_model_args(**model_args)

    template = """You are a AI having a conversation with a human.

    {chat_history}
    Human: {human_input}
    AI:"""
    message_history = get_message_history()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history,
                                      # human_prefix='人类', ai_prefix="人工智能"
                                      )

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template
    )
    chain = LLMChain(
        prompt=prompt,
        llm=llm,
        memory=memory
    )
    res = chain.predict(human_input=human_prompt)
    return res


@chains.post('/agent_chain')
def agent_chain(data: dict):
    human_prompt = data.get('prompt')
    llm = get_llm()
    message_history = get_message_history()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)
    doc_store = DocstoreExplorer(Wikipedia())
    tools = [
        Tool(
            name="Search",
            func=doc_store.search,
            description="useful for when you need to ask with search",
        ),
        Tool(
            name="Lookup",
            func=doc_store.lookup,
            description="useful for when you need to ask with lookup",
        ),
    ]
    react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True, memory=memory)
    res = react.run(human_prompt)
    return res


@chains.post('/retrieval_qa_chain')
def retrieval_qa_chain(data: dict):
    print('开始信息检索')
    llm = get_llm()
    human_prompt = data.get('prompt')
    message_history = get_message_history()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)

    chroma = get_data_connection()
    retriever = chroma.get_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, memory=memory)
    res = qa.run(human_prompt)
    return res
