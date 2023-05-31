from langchain.tools import BaseTool
from typing import Optional
import Controllor
import json
device = Controllor.MongoAPI()
import requests
class Get_Web(BaseTool):
    name: str = "find a device"
    description: str = """
                use this tool when you need to search/find devices
                Tool that calls GET on <http://localhost:5000/devices*> apis. 
                find me a device (with name is abc) (with parent name is abc)
                Valid params can only include one of the followings: "device name": "device name"
                """
    #args_schema: Type[SearchSchema] = SearchSchema
    base_url: str = "<http://localhost:5000/devices>"
    
    def _run(self, name 
            #  : Optional[str] = None, 
            #  parent_name : Optional[str] = None 
             ):
        query = {}
        if name:
            query['name'] = name

        # if parent_name:
        #     query['parentName'] = parent_name

        response = device.read(query)
    
        return response
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    

class CRUD(BaseTool):
    name: str = "operations of a device"
    description: str = """
                use this tool when you need to edit/update, find/search, delete a device, 
                or create a new device
                Tool that calls http_methods like DELETE/GET/POST/PATCH on <http://localhost:5000/devices/*> apis. 
                path can includes 
                "create": "create",
                "delete": "delete",
                "edit": "edit". 
                valid params can include
                "name": "name",
                "parent_name": "parent_name".
                and form into json format
                """
    #args_schema: Type[SearchSchema] = SearchSchema
    base_url: str = "http://localhost:5000/devices/"
    def _run(self, path: str="",
             query_params: Optional[dict]=None,
             http_method: str=""
             ):
        # data = json.dumps(new_device)
        # print(type(data))
        # if data is None or data == {}:
        #     return json.dumps({"Error": "Please provide connection information"})
        # result = device.write(data)
        # if http_method == "GET":
        #     result = requests.get(self.base_url + path, params=query_params)
        # elif http_method =="POST":
        #     result = requests.post(self.base_url + path, params=query_params)
        # elif http_method == "PATCH":
        #     result = requests.patch(self.base_url + path, params=query_params)
        # elif http_method == "DELETE":
        #     result = requests.delete(self.base_url + path, params=query_params)
        result = requests.request(http_method, self.base_url + path, params=query_params)
        return result.json
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
