import quart
import quart_cors
from quart import request,Response,jsonify
import Controllor
import os
import open_ai

app = quart.Quart(__name__)
# app = quart_cors.cors(quart.Quart(__name__), allow_origin="https://chat.openai.com")
device = Controllor.MongoAPI()

@app.route('/devices', methods=['GET'])
async def search_device():
    name = request.args.get('name')
    parent_name = request.args.get('parentName')
    query = {}
    if name:
        query['name'] = name

    if parent_name:
        query['parentName'] = parent_name

    response = device.read(query)
    
    return response

@app.route('/devices/create', methods=['POST'])
async def create():
    data = await request.get_json()
    if data is None or data == {}:
        return jsonify({"Error": "Please provide connection information"})
    result = device.write(data)
    return result


@app.route('/devices/edit',methods=['POST'])
async def update():
    name = request.args.get('name')
    parent_name = request.args.get('parentName')
    query = {}
    if name:
        query['name'] = name

    if parent_name:
        query['parentName'] = parent_name
    data = await request.get_json()
    result = device.update(query, data)
    return result

@app.route('/chatbox', methods=['GET', 'POST'])
async def chat():
    prompt="Please act as a Sunsniffer chatbot, Sunsniffer is a solar analytics and installation company, answer the question accordingly, and for the installation, please request the user's email, phone number, address, and name, and send \"SYSTEM:\" command to the API  and save data collected from user. Also, allow user to get their solar production, faults in the system, and if they ask for such info, send SYSTEM: command to let system know to query data from database and add it to response."
    test = open_ai.API()
    # response = test.send_message(prompt)
    # print("assistant:", response)
    while True:
         prompt, response = test.res(prompt)
         print("AI:", response)      
         prompt += response + "\n"


def main():
    app.run(debug=True, host="localhost", port= 5000)

if __name__ == "__main__":
    main()

    
    

