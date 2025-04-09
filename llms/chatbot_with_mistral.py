import os
from mistralai import Mistral

from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    api_key = os.getenv('MISTRAL_API_KEY')
    if api_key is None:
        print('You need to set your MISTRAL_API_KEY environment variable')
        exit(1)

    model = "mistral-large-latest"
    client = Mistral(api_key=api_key)
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content" : "My name is Zubiya"
            },
            {
                "role": "user",
                "content": "What is the capital of Madhya Pradesh?",
            },
            {
                "role": "user",
                "content": "What is my name ?",
            }

        ]
    )
    print(chat_response.choices[0].message.content)