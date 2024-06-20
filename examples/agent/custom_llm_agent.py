from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

from agent.tools import serp_api

from model.large_models.language_models import ChatGLM

search = serp_api
tools = [
    Tool(
        name="搜索引擎",
        func=search.run,
        description="工具用于搜索未知信息的答案，工具输入为想搜索的内容"
    )
]

# Set up the base template
# template = """请尽可能回答下列问题。您可以使用以下工具：
#
# {tools}
#
# 使用以下格式：
#
# 问题：此项是你需要回答的问题
# 思考：此项是你对下一步该怎么做的思考
# 工具：此项必须是[{tool_names}]中的一个,若不需要使用则为无
# 工具输入：此项是工具的输入
# 工具输出：此项是工具的工具输出
# ...（思考、工具、工具输入、工具输出的过程可以重复多次）
# 思考：我知道最终答案了
# 最终答案：问题的最终答案
#
# 开始回答!
#
# 问题：{input}
# {agent_scratchpad}"""

template = """请尽可能回答以下问题。
必须使用以下工具：

{tools}

必须使用以下格式：

问题:此项是需要回答的问题
思考:此项是对下一步该怎么做的思考
工具:此项是使用的工具，是[{tool_names}]之一,不需要使用工具则直接输出结论
工具输入:此项是针对问题，对工具的输入
工具输出:此项是根据工具输出的结果
结论:此项是根据工具工具输出得到的问题的结论

开始回答:

问题:{input}
{agent_scratchpad}"""


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n工具输出： {observation}\n思考： "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "结论:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("结论:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"工具\s*\d*\s*:(.*?)\n工具\s*\d*\s*输入\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


output_parser = CustomOutputParser()

llm = ChatGLM()

# llm = Qwen()

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\n工具输出:"],
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

print(agent_executor.run("什么是仿真"))
