import os
from dotenv import load_dotenv
from bson.json_util import dumps  # ✅ Proper JSON serializer for MongoDB
from pymongo import MongoClient
from langchain_mistralai import MistralAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Main function that handles all vector search operations
def vector_search_app():
    # Validate environment variables
    if not MISTRAL_API_KEY:
        print("❌ MISTRAL_API_KEY not found in environment. Please check your .env file.")

    if not MONGO_URI:
        print("❌ MONGO_URI not found in environment. Please check your .env file.")

    # Initialize components
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Initialize Chroma collection
    collection = Chroma(
        collection_name="blog_vector_db",
        embedding_function=embeddings,
        persist_directory="chroma_storage"
    )

#mongo db connection to retrieve all documents of blog
    try:
            print("🔄 Connecting to MongoDB...")
            client = MongoClient(MONGO_URI)
            mongo_collection = client["app-dev"]["blogs"]
            blog_docs = list(mongo_collection.find({}))
            print(f"📄 Found {len(blog_docs)} blog documents.")

            if not blog_docs:
                print("❌ No documents retrieved from MongoDB.")
                return False

            # Prepare documents
            documents = []
            for blog in blog_docs:
                full_text = dumps(blog, indent=2)
                doc = Document(
                    page_content=full_text,

                )
                documents.append(doc)
            print(f"✅ Prepared {len(documents)} documents for embedding.")

            # Split docs into chunks
            print("✂️ Splitting documents into chunks...")
            chunks = splitter.split_documents(documents)
            print(f"🔨 Split into {len(chunks)} chunks.")

            # Limit chunks if needed
            chunk_limit = 100
            limited_chunks = chunks[:chunk_limit]
            print(f"🔨 Using the first {len(limited_chunks)} chunks.")

            # Add documents to vector store
            print(f"🚀 Adding {len(limited_chunks)} chunks to Chroma...")
            collection.add_documents(limited_chunks)
            collection.persist()
            print(f"✅ Successfully stored {len(limited_chunks)} chunks in Chroma.")

    except Exception as e:
            print(f"❌ Error building vector database: {e}")
            return False


    print("\n🔎 Vector database ready for searching!")

    while True:
        query = input("\n🔎 Enter search query (or type 'exit'): ")
        if query.lower() == "exit":
            print("🚪 Exiting the semantic search...")
            break

        # Perform search
        try:
            print(f"🔍 Performing semantic search for query: '{query}'...")

            # Get results
            results = collection.similarity_search(query, k=1)

            if results:
                print(f"\n📝 Found {len(results)} relevant documents:")
                for doc in results:
                  print(doc.page_content)

            else:
                print("❌ No relevant results found.")

        except Exception as e:
            print(f"❌ Error during search: {e}")

    return True

# Main execution
if __name__ == "__main__":
    vector_search_app()
