import pymongo
import os
from dotenv import load_dotenv
import json
from bson import ObjectId

load_dotenv()
URL = os.getenv('MONGO_URL')
Name = os.getenv('DB_NAME')
Collection_name = os.getenv('COLLECTION')
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return super().default(o)
class MongoAPI:
    def __init__(self):
        self.client = pymongo.MongoClient(URL)
        db = self.client[Name]
        self.collection = db[Collection_name]
        #self.name = name

    def read(self, query):
        results = list(self.collection.find(query))

        return json.dumps(results, cls=JSONEncoder)
            
    def write(self, newdata):
        response = self.collection.insert_one(newdata)
        output = {'Status': 'Successfully inserted',
                  'Device_Name': str(newdata["name"])}
        
        return output
    
    def update(self, filter, updates):
        response = self.collection.update_one(filter, {'$set': updates})
        output = 'Successfully Updated' if response.modified_count > 0 else 'Nothing Changed'
        print (output)
        self.read(filter)
        
    
if __name__ == '__main__':
    mongo_obj = MongoAPI()
    # query = {"name": "Device1"}
    # print(mongo_obj.read(query))


    name = "Device12"
    # newdata= {
    # "name": "Device9",
    # "parentName": "Device4",
    # "Voltage": 30,
    # "Power": 4546
    # }
    # print(mongo_obj.write(newdata))

    update = {"Power": 4521}
    print(mongo_obj.update(name, update))



        



#create a database


# find data in database
# for collection ippn db.list_collection_names():
#     print(collection,'\n')
#     for document in db[collection].find({}):
#         print(document)
# db_list = client.list_database_names()



# def create_one():

# def find_one():
