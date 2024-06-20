from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from model.large_models.language_models import Qwen

template = """Tell me a {adjective} joke about {subject}."""
prompt = PromptTemplate(template=template, input_variables=["adjective", "subject"])
llm_chain = LLMChain(prompt=prompt, llm=Qwen())

print(llm_chain.predict(adjective="sad", subject="ducks"))
