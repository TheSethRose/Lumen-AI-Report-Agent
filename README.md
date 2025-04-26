# Lumen - AI Report Agent

An AI-powered assistant designed to help users review, interpret, and act on their reports and related documentation. Lumen is designed to replicate the core functionality of Claude Projects.

- Search, summarize, and explain the contents of local documentation (especially reports and documents)
- Track ongoing work, accomplishments, blockers, and plans
- Receive actionable suggestions and clear explanations for reporting and project management
- Reference both local knowledge and web search results for comprehensive answers

## Key Features

- **Automated Knowledge Base**: Loads and indexes your reports and documentation from a specified folder.
- **Conversational Agent**: Ask natural language questions about your reports, project progress, or documentation.
- **Actionable Guidance**: Get suggestions for what to include in future reports, how to summarize accomplishments, and how to communicate blockers.
- **Markdown Output**: Responses are formatted for easy copy-paste into your reporting tools.
- **Web Search Integration**: If local knowledge is insufficient, the agent can search the web for credible supplementary information.
- **Customizable Instructions**: Easily configure the agent's name, description, and behavior using environment variables.

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/TheSethRose/Lumen-AI-Report-Agent
cd Lumen-AI-Report-Agent
```

### 2. Create a Virtual Environment
```bash
uv venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
uv pip install -r requirements.txt
cp .env.example .env
```

### 4. Configure Environment Variables
- Edit `.env` and fill in your OpenAI API key and desired settings.
- Set `PROJECT_FOLDER` to the directory containing your reports and documentation.

### 5. Run the Agent
```bash
python agent.py
```

## Configuration
- **OPENAI_API_KEY**: Your OpenAI API key for language model access
- **PROJECT_FOLDER**: Path to your reports/documents
- **VECTOR_DB_FOLDER**: Path for vector database storage
- **AGENT_NAME**: Name of your assistant (default: Lumen - AI Report Agent)
- **AGENT_DESCRIPTION**: Short description of the agent
- **AGENT_INSTRUCTIONS**: Detailed instructions for the agent's behavior

See `.env.example` for a full list and descriptions of configurable options.

## Example Use Cases
- "What should I put in my next report?"
- "Summarize all blockers from March."
- "What accomplishments did I have in February?"
- "List all items mentioned throughout the last 3 documents."

---

For questions or contributions, please open an issue or pull request on GitHub.
