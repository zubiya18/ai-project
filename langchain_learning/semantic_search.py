import getpass
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain

load_dotenv()

if __name__ == '__main__':
    # Set Mistral API key
    if not os.environ.get("MISTRAL_API_KEY"):
        print("mistral")
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

    # Load Pdf--------------------------------------------------------------------------
    file_path = "example_file.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(len(docs))
    print(f"{docs[0].page_content[:200]}\n")
    print(docs[0].metadata)

    # Split into chunks------------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(len(all_splits))

    # Set OpenAI API key--------------------------------------------------------------
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
        print("API key set via prompt")

    # Embedding logic----------------------------------------------------------
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)

    assert len(vector_1) == len(vector_2)
    print(f"Generated vectors of length {len(vector_1)}\n")
    print(vector_1[:10])

    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=all_splits)
    results = vector_store.similarity_search(
        "How many countries does Nike operate in?"
    )
    print(results[0])

    @chain
    def retriever(query: str) -> List[Document]:
        return vector_store.similarity_search(query, k=1)


    queries=[
        "What is Nike's annual revenue for 2023?",
        "Where is Nike's headquarters located?"
    ]
    results = retriever.batch(queries)
    for i, result in enumerate(results):
     print(f"\nResult {i+1}:\n{result[0].page_content}")


