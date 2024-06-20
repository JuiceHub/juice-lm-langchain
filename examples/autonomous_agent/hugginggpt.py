from transformers import load_tool

from model.large_models.language_models.chatglm.chatglm import ChatGLM

hf_tools = [
    load_tool(tool_name)
    for tool_name in [
        # "document-question-answering",
        # "image-captioning",
        # "image-question-answering",
        # "image-segmentation",
        # "speech-to-text",
        # "summarization",
        # "text-classification",
        # "text-question-answering",
        # "translation",
        "huggingface-tools/text-to-image",
        # "huggingface-tools/text-to-video",
        # "text-to-speech",
        # "huggingface-tools/text-download",
        # "huggingface-tools/image-transformation",
    ]
]

from langchain_experimental.autonomous_agents import HuggingGPT

llm = ChatGLM()
agent = HuggingGPT(llm, hf_tools)

agent.run("请在本地下载一个图片，内容是星空")
