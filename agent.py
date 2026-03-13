from tools.scipy_tool import scipy_tools
from tools.sympy_tool import sympy_tools
from langchain_classic.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
import os

tools = []
tools.extend(scipy_tools)
tools.extend(sympy_tools)

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="moonshotai/kimi-k2-instruct-0905",
    temperature=0
)

aptitude_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)