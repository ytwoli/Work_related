from feast import FeatureStore, Entity
import os
import pyarrow.parquet as pq
import subprocess
from datetime import datetime
import pandas as pd
# from feast.data_source import PushMode
from langchain.agents import load_tools, AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.agents import initialize_agent
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
from getpass import getpass

from langchain.prompts import PromptTemplate, StringPromptTemplate
from dotenv import load_dotenv

load_dotenv()
repo_path = os.getenv("repo_path")
data_path = os.getenv("data_path")
store = FeatureStore(repo_path=repo_path)
entities=pd.read_parquet(path=data_path)

template = """Answer the question {query} by give the information of a certain inverter/device, which can have a search key like device id, panel id or serial number, 


Form the information together and try to reply as polite as possible.

Here are the inverter information:

device id: {did}
no Id: {noId}
panel id:{pid}
serial number:{serial}
message of voltage:{voltage}
power of DC:{Power_DC}
power of AC: {Power_AC},
Energy : {Energy},
message of ar: {ar},
message of rr: {rr},
message of t: {t}. 


Your response:"""

model = SentenceTransformer('all-MiniLM-L6-v2')
prompt = PromptTemplate.from_template(template)

class Semantic_parse():
    def inverter(self,query) -> dict:
        sequence = {}
        phrases_dicts = {
            "search": {
                "device_id",
                "panel_id",
                "serial_number"
            },
            
            "value": {
                "power_of_DC",
                "power_of_AC",
                "panel_id",
                "serial_number",
                "Energy",
                "message_of_ar",
                "message_of_rr",
                "message_of_t"
             }
        }
        
        query_embedding = model.encode([query])
        # For each dictionary
        sequence["timestamp"] = self.is_time(query)

        for dict_key, phrases_dict in phrases_dicts.items():
            # Compute the sentence embeddings
            phrase_embeddings = model.encode(list(phrases_dict))
            # Compute cosine similarity between the query and all phrases
            similarities = cosine_similarity(query_embedding, phrase_embeddings)

            # Find the index of the most similar phrase
            most_similar_idx = np.argmax(similarities)
            sequence[dict_key] = list(phrases_dict)[most_similar_idx]
            

        if sequence["search"] == "device_id":
             sequence["identify"] = self.is_id(query)
        elif sequence["search"] == "serial_number":
             sequence["identify"] = self.is_serial(query)
        elif sequence["search"] == "panel_id":
             sequence["identify"] = self.is_pid(query)
        return sequence
    def is_time(self,s):
        pattern = r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}"
        r = re.search(pattern, s)
        if r:

            return r.group()

        else:
            return None
    def is_id(self,s):
        pattern = r"[a-zA-Z0-9]{24}"
        r = re.search(pattern, s)
        if r:

            return r.group()

        else:
            return None
        


    def is_pid(self, s):
        pattern = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
        match = re.search(pattern, s)
        if match:
            return match.group()
        else:
            return None
        

    def is_serial(self,s):
        pattern = r"^[0-9]{10}"
        match = re.search(pattern, s)
        if match:
            return match.group()
        else:
            return None
class FeastPromptTemplate(StringPromptTemplate):
    def format(self, query) -> str:
        parse = Semantic_parse()
        kwargs = parse.inverter(query)
        search_by = kwargs["search"]
        identify = kwargs["identify"]
        time = kwargs["timestamp"]
        if time:
            timestamp_field = datetime.strptime(kwargs["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
            print({
                search_by:[identify],
                "timestamp_field": [timestamp_field]
            })
            entity= pd.DataFrame.from_dict(
            {
                search_by:[identify],
                "event_timestamp": [timestamp_field]
            })
        else:
            entity= pd.DataFrame.from_dict(
            {
                search_by:[identify]
            })
        feature_vector = store.get_historical_features(
            features=[
                'inverters_information:did',
                'inverters_information:noId',
                'inverters_information:pid',
                'inverters_information:serial',
                "inverters_information:m.v",
                'inverters_information:m.p0',
                'inverters_information:m.pac',
                'inverters_information:m.tw',
                'inverters_information:m.ar',
                'inverters_information:m.rr',
                'inverters_information:m.t',
            ],
           entity_df= entity
        ).to_df()
        kwargs["did"] = feature_vector["did"][0]
        kwargs["noId"] = feature_vector["noId"][0]
        kwargs["pid"] = feature_vector["pid"][0]
        kwargs["serial"] = feature_vector["serial"][0]
        kwargs["voltage"] = feature_vector["m.v"][0]
        kwargs["Power_DC"] = feature_vector["m.p0"][0]
        kwargs["Power_AC"] = feature_vector["m.pac"][0]
        kwargs["Energy"] = feature_vector["m.tw"][0]
        kwargs["ar"] = feature_vector["m.ar"][0]
        kwargs["rr"] = feature_vector["m.rr"][0]
        kwargs["t"] = feature_vector["m.t"][0]
        print(type(**kwargs))
        return prompt.format(kwargs)
    

prompt_template = FeastPromptTemplate(input_variables=["query"])
# print(prompt_template.format(search = "_id", value="bb4f006075043026aedbb42451ef5b3b", timeslot="2023-02-09 00:00:00.000"))

llm = ChatOpenAI(
    openai_api_key="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", 
    temperature=0,
    model_name='gpt-3.5-turbo')

chain = LLMChain(
    llm=llm, prompt=prompt_template)
# data = chain.run(search = "_id", value="23142c5a4fee7ac757a167bc255284ae", timeslot="2023-02-0900:00:00.000")
# print(data)
data = chain.run(query = r"What is information  of my device with device id: 63e4d47cd79747e38aa73526 at time : 2023-02-09 00:00:00.000")
print(data)
conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)

agent = initialize_agent(
    agent= AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=[],
    llm = chain,
    verbose = True,
    memory = conversational_memory
)
# Q: What is   of my device with id = 23142c5a4fee7ac757a167bc255284ae
# A: 
agent.run("What is information  of my device with id = 23142c5a4fee7ac757a167bc255284ae")

# def run_demo(): 
#     store = FeatureStore(repo_path=repo_path)
#     print("\n--- Historical features for training ---")
#     fetch_h_feature(store)

#     print("\n ---")


# def fetch_h_feature(store):
#     #feast uses entity DtaFrmes to join different feature view together
#     #feast matches the entity names and event timestmps in feature views and the entity DataFrame
#     entities=pd.read_parquet(path=data_path)
#     file = pq.ParquetFile(data_path)
#     parquet_schema = file.schema
# # Extract and print the column names
#     column_names = parquet_schema.names
#     print(column_names)
#     #print(entities.head(5))
#     # if for_batch_scoring:
#     #     entities["event_timestamp"] = pd.to_datetime("now", utc=True)
#     training_data = store.get_historical_features(
#         entity_df=entities,
#         features=[
#             "inverters_information:did",
#             "inverters_information:m",
#             "inverters_information:noId",
#             "inverters_information:pid",
#             "inverters_information:serial"
#         ]
#     ).to_df()
#     print(training_data.info())
#     print(training_data.head())

# if __name__ == "__main__":
#     run_demo()
