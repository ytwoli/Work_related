from dotenv import load_dotenv
from langchain.tools import BaseTool
from math import pi,sqrt, cos, sin
from typing import Union, Optional
import os
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentExecutor
from langchain.agents import load_tools, AgentType

OpenAIKey = os.getenv("OPENAI_API_KEY") 
load_dotenv()
print (OpenAIKey)


class CircumferenceTool(BaseTool):
    name = "Circumference calculator"
    description = "use this tool when you need to calculate a circumference using the radius of a circle"

    def _run(self, radius: Union[int, float]):
        return float(radius)*2.0*pi
    #when a tool is to be used asynchronously
    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")
    
class Hypotenuse(BaseTool):
    name = "Hypotenuse calculator"
    description = ("use this tool when you need to calculate the lengthe of the a hypotenuse"
                  "give two sides or an angle(in degrees) with one side of a triangle"
                  "it must contains at least two of the following parameters"
                  "['adjacent_side','opposite_side','angle']")
    def _run(self, adjacent_side: Optional[Union[int, float]] = None,
             opposite_side: Optional[Union[int, float]] = None,
             angle: Optional[Union[int, float]] = None,
            ):
        if adjacent_side and opposite_side:
            return sqrt(float(adjacent_side)**2 + float(opposite_side)**2)
        elif adjacent_side and angle:
            return adjacent_side / cos(float(angle))
        elif opposite_side and angle:
            return opposite_side / sin(float(angle))
        else:
            return "Could not calculate the hypotenuse of the triangle. Need two or more of `adjacent_side`, `opposite_side`, or `angle`."
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
#prompt = "can you calculate the circumference of a circle that has a radius of 7.81mm"
#print(CircumferenceTool()._run(7.8))




# initialize LLM (we use ChatOpenAI because we'll later define a `chat` agent)
llm = ChatOpenAI(
        openai_api_key= OpenAIKey,
        temperature=0,
        model_name='gpt-3.5-turbo'
)

# initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)


os.environ['SERPAPI_API_KEY'] = "5129d3b10a0ac66e67b0113d878f9e3cc2c150af88170f43cd2b829b0d9f308b"

tool_name = ["serpapi"]
tool1 = load_tools(tool_name)
tools = [CircumferenceTool()]
tools.append(Hypotenuse())
# initialize agent with tools
agent = initialize_agent(
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)
agent.run("can you calculate the lengthe of the a hypotenuse with adjecent_side=3, opposite_side=4?")