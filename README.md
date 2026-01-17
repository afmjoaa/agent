# Function Calling and Agents Notebooks
This repository have exercise on the following topics 

1. What makes LLM agent
2. Type of function calling
3. How vanilla agent work
4. What is MCP
5. Type of agents
6. Two main type of LLM call which make agent
7. Agentic frameworks
8. DSPy
9. Langgraph


## 💡 Usage
1. Open the notebooks sequentially (`01` → `06`) for a structured learning flow. 
2. 01, 05 and 06 have tutorial, and 02, 03 & 04 have some very basic exercises.
3. src/solution folder have all the solutions for the exercises.
4. Explore the `assets/` folder for visual aids used within the notebooks.

## Prerequisites for Exercises 05 & 06

Exercises 05 (DSPy MCP Client) and 06 (LangGraph MCP Client) require additional setup:

---
### Step 1: Configure Environment Variables
Create a `.env` file in the project root with required API keys:

See `.env.example` for all required variables.

---
### Step 2: Start the WhatsApp MCP Server
The exercises connect to a local MCP server that provides WhatsApp automation tools.

**To start the server:**

```bash
cd whatsapp-mcp/whatsapp-bridge
go run main.go
```

**Keep this server running** in a separate terminal while working on exercises 05 and 06.