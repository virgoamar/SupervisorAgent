from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agents import (
    SupervisorState,
    supervisor_node,
    researcher_node,
    writer_node,
    critic_node,
    finalize_node,
    route_to_agent
)


def create_supervisor_graph():
    """Build and compile the multi-agent supervisor graph."""
    graph = StateGraph(SupervisorState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalize", finalize_node)

    # Define edges and routing workflow
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "researcher": "researcher",
            "writer": "writer",
            "critic": "critic",
            "finalize": "finalize",
        },
    )

    # Loop back to supervisor after specialist execution
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("critic", "supervisor")
    graph.add_edge("finalize", END)

    return graph.compile()