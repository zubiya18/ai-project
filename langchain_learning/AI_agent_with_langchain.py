from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.memory import ConversationBufferMemory
from langchain_mistralai import ChatMistralAI

# Load environment variables
load_dotenv()

# Initialize model
model = ChatMistralAI(model="mistral-large-latest")

# Define tools
tools = [
    Tool(
        name="Search",
        func=TavilySearchResults(max_results=1).run,
        description="Useful for answering questions about current events or general world knowledge."
    ),
    Tool(
        name="Python REPL",
        func=PythonREPLTool().run,
        description="Useful for performing Python calculations or code execution."
    )
]

# Set up memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create agent executor
agent_executor = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,

)

# Main interaction loop
if __name__ == "__main__":
    while True:
        user_prompt = input("\n🧠 Enter prompt (type 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            print("👋 Exiting...")
            break

        response = agent_executor.run(user_prompt)
        print(f"\n🤖 Response: {response}")
