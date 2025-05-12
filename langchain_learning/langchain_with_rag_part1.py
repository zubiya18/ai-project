import os
import bs4
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph


def run():
    # ✅ Load API key
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ You need to set your MISTRAL_API_KEY environment variable")
        return

    # ✅ Init embeddings & vector store
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = InMemoryVectorStore(embeddings)

    # ✅ Load and split web content
    strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": strainer},
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks: List[Document] = splitter.split_documents(docs)
    vector_store.add_documents(chunks)

    # ✅ Define graph step
    def retrieve(state: dict):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    # ✅ Build LangGraph
    graph_builder = StateGraph(dict).add_node("retrieve", retrieve)
    graph_builder.set_entry_point("retrieve")
    graph = graph_builder.compile()

    # ✅ Repeatedly accept input
    while True:
        user_question = input("\nAsk your question (or type 'exit' to quit): ").strip()
        if user_question.lower() == "exit":
            print("👋 Exiting. Goodbye!")
            break

        result = graph.invoke({"question": user_question})

        print("\n📄 Retrieved Context:")
        for i, doc in enumerate(result["context"], 1):
            print(f"\n--- Document {i} ---\n{doc.page_content[:500]}...\n")


# 🔁 Run the app
if __name__ == "__main__":
    run()
