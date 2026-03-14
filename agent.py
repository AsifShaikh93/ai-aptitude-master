import os
from tools.scipy_tool import scipy_tools
from tools.sympy_tool import sympy_tools
from langchain_groq import ChatGroq
from langchain_classic.agents import create_structured_chat_agent, AgentExecutor
from langchain_classic import hub

# 1. Setup Tools
tools = []
tools.extend(scipy_tools)
tools.extend(sympy_tools)

# 2. Initialize LLM with streaming enabled
# The 'streaming=True' is critical for your FastAPI StreamingResponse to work!
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    streaming=True 
)

# 3. Pull the Prompt 
# This is the standard prompt for structured chat agents (supports multi-input tools)
prompt = hub.pull("hwchase17/structured-chat-agent")

# 4. Create the Agent
# We use create_structured_chat_agent instead of the deprecated initialize_agent
agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# 5. Create the Executor
# handle_parsing_errors=True is a lifesaver for math agents
aptitude_agent = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)