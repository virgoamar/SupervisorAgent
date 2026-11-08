from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langgraph.graph import  add_messages
from langgraph.prebuilt import create_react_agent

from pydantic import BaseModel,Field

from tools import web_search, analyze_text_readability
from  langchain_groq import ChatGroq
from typing import TypedDict, Annotated
import os

load_dotenv()

# Initialize Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")
llm = ChatGroq(model=model_name, temperature=0.1, api_key=groq_api_key)


class SupervisorState(TypedDict):
    messages: Annotated[
        list[BaseMessage], add_messages] if "add_messages" in globals() else None  # imported cleanly below
    next_agent: str
    task_complete: bool
    final_response: str

class RouteDecision(BaseModel):
    next: Literal["researcher", "writer", "critic", "FINISH"] = Field(
        description="The next agent to call, or FINISH if the overall task is fully complete."
    )
    reasoning: str = Field(description="Explanation of why this agent was chosen next.")

supervisor_llm = llm.with_structured_output(RouteDecision)

def supervisor_node(state: SupervisorState) -> dict:
    system_prompt = """You are an expert supervisor managing a team of specialist agents:
1. researcher - Gathers information, facts, and external context using search tools.
2. writer - Creates structured, engaging content and text based on research.
3. critic - Reviews work objectively, suggests improvements, or verifies quality.

Review the conversation history and decide which agent should act next. 
If the user's request has been completely fulfilled and polished, respond with FINISH."""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    decision = supervisor_llm.invoke(messages)

    print(f"\n[Supervisor Decision] Next: {decision.next} | Reason: {decision.reasoning}")

    if decision.next == "FINISH":
        return {"next_agent": "FINISH", "task_complete": True}

    return {
        "next_agent": decision.next,
        "messages": [
            AIMessage(content=f"[Supervisor Routing] Assigned task to **{decision.next}**: {decision.reasoning}")
        ],
    }

# Specialist ReAct Agents
research_agent = create_react_agent(
    model=llm,
    tools=[web_search],
    prompt="You are a meticulous research specialist. Use your search tools to gather accurate facts before reporting back."
)

writer_agent = create_react_agent(
    model=llm,
    tools=[analyze_text_readability],
    prompt="You are a professional content writer. Synthesize provided research into engaging, high-quality drafts."
)

critic_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt="You are a constructive quality critic. Evaluate previous work for accuracy, tone, and completeness. Provide actionable feedback."
)

# Wrapper Nodes
def researcher_node(state: SupervisorState) -> dict:
    result = research_agent.invoke({"messages": state["messages"]})
    latest_msg = result["messages"][-1]
    return {"messages": [AIMessage(content=f"[Researcher Output] {latest_msg.content}")]}

def writer_node(state: SupervisorState) -> dict:
    result = writer_agent.invoke({"messages": state["messages"]})
    latest_msg = result["messages"][-1]
    return {"messages": [AIMessage(content=f"[Writer Output] {latest_msg.content}")]}

def critic_node(state: SupervisorState) -> dict:
    result = critic_agent.invoke({"messages": state["messages"]})
    latest_msg = result["messages"][-1]
    return {"messages": [AIMessage(content=f"[Critic Output] {latest_msg.content}")]}

def finalize_node(state: SupervisorState) -> dict:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and "[Writer Output]" in msg.content:
            clean_content = msg.content.replace("[Writer Output] ", "")
            return {"final_response": clean_content}
    return {"final_response": "Task completed successfully."}

def route_to_agent(state: SupervisorState) -> str:
    if state.get("task_complete"):
        return "finalize"
    return state["next_agent"]