# Multi-Agent Supervisor System (LangGraph & Groq)

A modular, production-ready multi-agent supervisor architecture built using **LangGraph**, **LangChain**, and **Groq (`llama-3.3-70b-versatile`)**. This system coordinates specialized autonomous agents (Researcher, Writer, and Critic) under a centralized supervisor node to handle complex, multi-step tasks efficiently.

---

## 🏗️ Project Architecture

```text
SupervisorAgent/
├── .env                  # API keys and environment configuration
├── requirements.txt      # Dependency list
├── tools.py              # Custom tool definitions (Web search, Text analysis)
├── agents.py             # Agent definitions, states, and supervisor logic
├── graph.py              # Compiled LangGraph state machine workflow
└── main.py               # Application entry point