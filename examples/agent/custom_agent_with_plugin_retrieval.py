from langchain.agents.agent_toolkits import NLAToolkit
from langchain.tools.plugin import AIPlugin

from data_connection.custom.chroma_dc import ChromaDC
from model.large_models.language_models.chatglm.chatglm import ChatGLM

llm = ChatGLM()

urls = [
    "https://datasette.io/.well-known/ai-plugin.json",
    "https://api.speak.com/.well-known/ai-plugin.json",
    "https://www.wolframalpha.com/.well-known/ai-plugin.json",
    "https://www.zapier.com/.well-known/ai-plugin.json",
    "https://www.klarna.com/.well-known/ai-plugin.json",
    "https://www.joinmilo.com/.well-known/ai-plugin.json",
    "https://slack.com/.well-known/ai-plugin.json",
    "https://schooldigger.com/.well-known/ai-plugin.json",
]

AI_PLUGINS = [AIPlugin.from_url(url) for url in urls]

chroma = ChromaDC(persist=False)

toolkits_dict = {
    plugin.name_for_model: NLAToolkit.from_llm_and_ai_plugin(llm, plugin)
    for plugin in AI_PLUGINS
}
