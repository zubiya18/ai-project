import os
import getpass
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.mistralai import MistralAI

# Change working directory
os.chdir("C:/Users/HP/Desktop/AI Project/LlamaIndex")
print("Is 'data' a directory?:", os.path.isdir("data"))
print("Current working directory:", os.getcwd())
print("Contents of directory:", os.listdir())

load_dotenv()

if __name__ == '__main__':
    if not os.environ.get("MISTRAL_API_KEY"):
        print("Mistral API key missing")
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")


    documents = SimpleDirectoryReader("data").load_data()
    if not documents:
        print("No documents were loaded. Check your 'data' directory.")
    else:
        print(f"Loaded {len(documents)} documents")

        Settings.embed_model = OpenAIEmbedding()
        Settings.llm = MistralAI(model="mistral-small-latest")
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        response = query_engine.query("What is Nike's annual revenue for 2023?")
        print(response)
