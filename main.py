import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph import create_supervisor_graph

load_dotenv()

if __name__ == "__main__":
    app = create_supervisor_graph()

    print("Running Modular Groq Multi-Agent Supervisor System...\n")

    initial_input = {
        "messages": [
            HumanMessage(content="Write a short blog post about the benefits of AI in healthcare.")
        ],
        "next_agent": "",
        "task_complete": False,
        "final_response": "",
    }

    result = app.invoke(initial_input)

    print("\n====================")

    print("FINAL PRODUCED RESPONSE:")
    print("====================\n")
    print(result["final_response"])