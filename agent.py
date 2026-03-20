import os
from tools.scipy_tool import scipy_tools
from tools.sympy_tool import sympy_tools
from langchain_groq import ChatGroq
from langchain_classic.agents import create_structured_chat_agent, AgentExecutor
from langchain_classic import hub

tools = []
tools.extend(scipy_tools)
tools.extend(sympy_tools)

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    streaming=True 
)

prompt = hub.pull("hwchase17/structured-chat-agent")

agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

aptitude_agent = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
