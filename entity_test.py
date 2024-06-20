import json

from model.large_models.language_models import ChatGLM, Qwen
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

with open(
        "/home/user/ygz/nlp/langchain/juice-lm-langchain/a/system_training/others/data/camrest/data_raw/data_QTOD/test.json",
        "r", encoding="utf-8") as f:
    dataset = json.loads(f.read())
index = 0
example = dataset[index]['dialogue']
kb = dataset[index]['scenario']['kb']['items']
kb = map(str, kb)
kb_txt = '\n'.join(kb)
llm = ChatGLM()
context = []

for turn in example:
    role = turn['turn']
    utterance = turn['utterance']
    if role == 'user':
        context.append('<user>' + utterance + '</user>')
    else:
        context.append('<system>' + utterance + '</system>\n')
        continue
    cxt = '\n'.join(context)
    query_template = \
        "Based on the dialog context, please select entities from the 'Knowledge Base' that best meets the needs of the user's last round of dialog.\n" + \
        "If the user needs to select entities, then 'Entities' is all selected entities' id.\n" + \
        "If no entities are required, 'Entities' is 'None'.\n" + \
        "Each entity includes attributes such as name,area,food,phone,pricerange,location,address,type,id,postcode.\n" + \
        "Dialog Context:\n{context}" + \
        "Knowledge Base:{kb}\n" \
        "Entities:"
    prompt = PromptTemplate(template=query_template, input_variables=["context", 'kb'])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    print(llm_chain.predict(context=cxt, kb=kb_txt))
