from feast import FeatureStore
repo_path = "/Users/ooolivia/Desktop/FEAST/inverters/feature_repo/inverters"
store = FeatureStore(repo_path=repo_path)


from langchain.prompts import PromptTemplate, StringPromptTemplate

template = """Given the inverters's up to date information, 
write them note relaying those information to them.

Here are the inverter information:
device id: {did}
m: {m}
noId: {noId}
pId:{pId}
serial:{serial}

Your response:"""

prompt = PromptTemplate.from_template(template)
class FeastPromptTemplate(StringPromptTemplate):
    
    def format(self, **kwargs) -> str:
        inverter_id = kwargs.pop("_id")
        feature_vector = store.get_online_features(
            features=[
                'inverters_information:did',
                'inverters_information:m',
                'inverters_information:noId',
                'inverters_information:pId',
                'inverters_information:serial'
            ],
            entity_rows=[{"_id": inverter_id}]
        ).to_dict()
        kwargs["did"] = feature_vector["did"][0]
        kwargs["m"] = feature_vector["m"][0]
        kwargs["noId"] = feature_vector["noId"][0]
        kwargs["pId"] = feature_vector["noId"][0]
        kwargs["serial"] = feature_vector["serial"][0]
        return prompt.format(**kwargs)
    
prompt_template = FeastPromptTemplate(input_variables=["id"])
print(prompt_template.format(id="23142c5a4fee7ac757a167bc255284ae"))
