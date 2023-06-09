from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools, AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
import re 
embeddings = OpenAIEmbeddings(openai_api_key="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")
llm = ChatOpenAI(
    openai_api_key="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", 
    temperature=0,
    model_name='gpt-3.5-turbo'
)
conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')



# Define multiple dictionaries of phrases
phrases_dicts = {
    "device": {
        "battery",
        "panel",
        "module"
    },
    "value": {
        "power",
        "current",
        "voltage"
    },
    # Add more dictionaries as needed...
}

# A query sentence
query = "What's the power of my battery?"

# Compute the embedding for the query
query_embedding = model.encode([query])

best_similarity = -1
best_phrase_key = None
best_dict_key = None
sequence = {}
# For each dictionary
for dict_key, phrases_dict in phrases_dicts.items():
    # Compute the sentence embeddings
    phrase_embeddings = model.encode(list(phrases_dict))
    # Compute cosine similarity between the query and all phrases
    similarities = cosine_similarity(query_embedding, phrase_embeddings)

    # Find the index of the most similar phrase
    most_similar_idx = np.argmax(similarities)
    sequence[dict_key] = list(phrases_dict)[most_similar_idx]
    # # If this is the best similarity we've seen, update the best phrase and dictionary
    # if similarities[0, most_similar_idx] > best_similarity:
    #     best_similarity = similarities[0, most_similar_idx]
    #     best_phrase_key = list(phrases_dict.keys())[most_similar_idx]
    #     best_dict_key = dict_key

# Print the most similar phrase and dictionary
def inverter(query) -> dict:
        sequence = {}
        phrases_dicts = {
            "search": {
                "id",
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
        sequence["timestamp"] = is_time(query)

        for dict_key, phrases_dict in phrases_dicts.items():
            # Compute the sentence embeddings
            phrase_embeddings = model.encode(list(phrases_dict))
            # Compute cosine similarity between the query and all phrases
            similarities = cosine_similarity(query_embedding, phrase_embeddings)

            # Find the index of the most similar phrase
            most_similar_idx = np.argmax(similarities)
            sequence[dict_key] = list(phrases_dict)[most_similar_idx]
            

        if sequence["search"] == "id":
             sequence["identify"] = is_id(query)
        elif sequence["search"] == "serial_number":
             sequence["identify"] = is_serial(query)
        elif sequence["search"] == "panel_id":
             sequence["identify"] = is_pid(query)
        return sequence
        
question = r"what is the AC power of my inverter with id  = 63e4d47cd79747e38aa73526 at 2023-06-07T10:00:00.000Z"



def is_time(s):
    pattern = r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z"
    r = re.search(pattern, s)
    if r:

        return r.group()

    else:
        return None
    


def is_id(s):
    pattern = r"[a-zA-Z0-9]{24}"
    r = re.search(pattern, s)
    if r:

        return r.group()

    else:
        return None
    


def is_pid(s):
     pattern = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
     match = re.search(pattern, s)
     if match:
          return match.group()
     else:
          return None


def is_serial(s):
     pattern = r"^[0-9]{10}"
     match = re.search(pattern, s)
     if match:
          return match.group()
     else:
          return None
     
print(inverter(question))