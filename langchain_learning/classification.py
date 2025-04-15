import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from typing import Optional

# Load environment variables from .env file
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Define the schema for the extracted information
class Person(BaseModel):
    name: Optional[str] = Field(default=None, description="The name of the person")
    height: Optional[str] = Field(default=None, description="Convert the height in meters")
    hair_color: Optional[str] = Field(default=None, description="The hair color of the person if known")

def build_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value."
        ),
        ("human", "{input}")
    ])

def extract_and_display_info(input_text: str):
    llm = ChatMistralAI(temperature=0, model="mistral-large-latest")
    prompt = build_prompt().invoke({"input": input_text})
    structured_llm = llm.with_structured_output(schema=Person)

    response = structured_llm.invoke(prompt)

    print("\n--- Extraction Result ---")
    print("Input:", input_text)
    print("Output:", response)

if __name__ == "__main__":
    user_input_text = input("Enter a person's name, height, and hair color: ")
    extract_and_display_info(user_input_text)