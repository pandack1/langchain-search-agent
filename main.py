from dotenv import load_dotenv

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent
#from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
structured_llm = llm.with_structured_output(AgentResponse) # creates a aversion of the model which outputs structured data according to the Pydantic model's AgentResponse schema
react_prompt = hub.pull("hwchase17/react")
#output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instruction = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=[
        "tools",
        "tool_names",
        "input",
        "agent_scratchpad",
    ]
).partial(
    format_instructions=""
)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instruction
    )

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

extract_output = RunnableLambda(lambda x: x["output"]) # parsing the output key (dict) from the agent executor's result as string
#parse_output = RunnableLambda(lambda x: output_parser.parse(x)) # parsing the output string into the Pydantic model
#chain = agent_executor | extract_output | parse_output # this is langchain expression language syntax
chain = agent_executor | extract_output | structured_llm # this is langchain expression language syntax

def main():
    #print("Hello from search-agent!")
    result = chain.invoke(
        input={"input": "search for 1 job posting for ai engineer in Bengaluru India"}
    )
    print("Result:", result)


if __name__ == "__main__":
    main()
