from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_mistralai.chat_models import ChatMistralAI

class ChatBot:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in .env file.")
        print("✅ MISTRAL_API_KEY loaded successfully.")

        # Initialize model
        self.model = ChatMistralAI(
            model="mistral-large-latest",
            api_key=self.api_key,
        )

        # Setup workflow with memory
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_node("model", self.call_model)
        workflow.set_entry_point("model")
        memory = MemorySaver()
        self.workflow = workflow.compile(checkpointer=memory)

        # Config for memory (threading)
        self.thread_id = "unique-thread-id"
        self.config_dict = {"configurable": {"thread_id": self.thread_id}}

    def start_conversation(self):
        print("😎 Chatbot is ready! Type 'quit' to exit.")
        while True:
            user_input = input("YOU: ")
            if user_input.lower() == "quit":
                print("👋 Exiting!")
                break

            input_messages = [HumanMessage(content=user_input)]
            response = self.workflow.invoke({"messages": input_messages}, self.config_dict)
            print("Chatbot:", response["messages"][-1].content)

    def call_model(self, state: MessagesState):
        response = self.model.invoke(state["messages"])
        return {"messages": response}

if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.start_conversation()

