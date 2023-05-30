
import os
from typing import Any, List
from dotenv import load_dotenv
from langchain.schema import BaseMessage
from langchain.tools import BaseTool, Tool
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentExecutor
from langchain.agents import load_tools, AgentType
from langchain.prompts import BaseChatPromptTemplate

import Controllor
import Tools

device = Controllor.MongoAPI()

#a base template
template = """
            
            Please act as a Sunsniffer chatbot, Sunsniffer is a solar analytics and installation company, answer the question accordingly, and for the installation. You can find tools in {Tools.py} 
            Use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take should be one of [Tools.create(), Tools.Get_Web()]
            Action Input: the input of the action
            Observation: the result of the action 
            ... (this Thought/Action/Action Input/Observation can repeat N times)

"""
open_AI_Key = os.getenv("OPENAI_API_KEY")
load_dotenv()

#tools = [Tools.Get_Web()]
tools = [Tools.CRUD()]
#set up a prompt template
# class CustomPromptTemplate(BaseChatPromptTemplate):
#     template: str
#     tools: list[tools]

#     def format_messages(self, **kwargs: Any) -> str:
#         # Get the intermediate steps (AgentAction, Observation tuples)
#         # Format them in a particular way
#         return super().format_messages(**kwargs)


llm = ChatOpenAI(
        openai_api_key=open_AI_Key,
        temperature=0,
        model_name='gpt-3.5-turbo'
)
conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)


agent = initialize_agent(
    agent= AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm = llm,
    verbose = True,
    memory = conversational_memory
)

agent.run("edit the voltage of a device with name Device34 into 54")