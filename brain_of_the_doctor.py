# setup groq api key

import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# print(GROQ_API_KEY)

#step 2- convert image to required format 

import base64

def encode_image(image_path):   
    image_file=open(image_path, "rb")
    return base64.b64encode(image_file.read()).decode('utf-8')


#step 3- Setup Multimodal LLM
from groq import Groq

client = Groq()
model = "llama-3.2-90b-vision-preview"
query="Is there something wrong with my eye?"

def analyze_image_with_query(query, model, encoded_image):
    client=Groq()  
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }]
    chat_completion=client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content