from dotenv import load_dotenv
load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm = llm,
    tools = tools,
    prompt = react_prompt
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent = agent,
    tools = tools,
    verbose = True
)
chain = agent_executor

def main():
    print("Hello from search-agent!")
    result = chain.invoke(
            input={
            "input": "search for 1 job posting for ai engineer in Bengaluru India"
            }
        )
    print("Result:", result)
    


if __name__ == "__main__":
    main()
