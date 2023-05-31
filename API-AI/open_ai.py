
import os
import openai
from dotenv import load_dotenv
load_dotenv()
  
class API:
    def __init__(self):
        pass

    def req(self,prompt):
        message = input("User: ")
        prompt += message + "\n"
        return prompt
    
    def res(self,prompt):
        prompt = self.req(prompt)
        response = openai.Completion.create (
            engine="text-davinci-003",
            prompt= prompt,
            max_tokens=256
        )
        return  prompt, response.choices[0].text.strip()
    
    def send_message(self,prompt):
        response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=256
    )
        return response.choices[0].text.strip()
if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")  
    

