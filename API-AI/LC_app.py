
import os
from dotenv import load_dotenv
from langchain.tools import BaseTool
import os
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentExecutor
from langchain.agents import load_tools, AgentType
from typing import Optional, Type
import requests
import aiohttp
import Controllor

device = Controllor.MongoAPI()
load_dotenv()
open_AI_Key = os.getenv("OPENAI_API_KEY")
description = ""
class Get_Web(BaseTool):
    name: str = "find a device"
    description: str = """
                use this tool when you need to search/find devices
                Tool that calls GET on <http://localhost:5000/devices*> apis. 
                Valid params can only include one of the followings: "device name": "device name", "parent name": "parent name"
                """
    #args_schema: Type[SearchSchema] = SearchSchema
    base_url: str = "<http://localhost:5000/devices>"
    
    def _run(self, name : Optional[str] = None, 
             parent_name : Optional[str] = None ):
        query = {}
        if name:
            query['name'] = name

        if parent_name:
            query['parentName'] = parent_name

        response = device.read(query)
    
        return response
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


llm = ChatOpenAI(
        openai_api_key= open_AI_Key,
        temperature=0,
        model_name='gpt-3.5-turbo'
)
conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)

tools = [Get_Web()]

agent = initialize_agent(
    agent= AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm = llm,
    verbose = True,
    memory = conversational_memory
)

agent.run("find the device of name Device4")