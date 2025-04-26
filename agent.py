from __future__ import annotations  # For future compatibility of type hints

# agent.py
"""
Agent module for document search and reasoning.

This module defines the primary agent and supporting utilities for document search,
reasoning, and tool integration. All code is compatible with Python 3.13.1 and follows
PEP 8, with type hints, variable annotations, and NumPy/SciPy-style docstrings.

Examples
--------
>>> agent = document_agent
>>> custom_print_response(agent, "What is the project status?")
"""

import os
import sys
import logging
import re
from typing import Any, Dict, List, Optional, Set, Union
from dotenv import load_dotenv
from pathlib import Path
from textwrap import dedent
from datetime import datetime

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.tools.reasoning import ReasoningTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.knowledge import KnowledgeTools
from agno.document.chunking.recursive import RecursiveChunking
from agno.embedder.openai import OpenAIEmbedder
from agno.run.response import RunEvent, RunResponse
from agno.utils.timer import Timer
from agno.utils.log import logger, log_error

# Set logging level
os.environ["RUST_LOG"] = "error"
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
logger.setLevel(logging.ERROR)

# Load Environment Variables
load_dotenv()

# Configuration from environment variables
NUM_DOCUMENTS: int = int(os.getenv("NUM_DOCUMENTS", 100))  # Number of documents to use
PROJECT_FOLDER: Path = Path(os.environ.get("PROJECT_FOLDER", "./documents"))  # Project directory
VECTOR_DB_FOLDER: Path = Path(os.environ.get("VECTOR_DB_FOLDER", "./vector_db"))  # Vector DB directory
OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")  # OpenAI API key
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # Model name
FORMATS: list[str] = os.getenv("FORMATS", ".txt").split(",")  # Accepted formats
AGENT_NAME: str = os.getenv("AGENT_NAME", "Agent")  # Agent name
AGENT_DESCRIPTION: str = os.getenv("AGENT_DESCRIPTION", "Help user with their query.")  # Description
AGENT_INSTRUCTIONS: str = os.getenv("AGENT_INSTRUCTIONS", "")  # Instructions

# Prepend current datetime to instructions
CURRENT_DATETIME: str = datetime.now().isoformat()

chunking_strategy: RecursiveChunking = RecursiveChunking(
    chunk_size=2000,
    overlap=200
)

class CustomTools(KnowledgeTools):
    """Custom tools extending KnowledgeTools with search capability.

    Parameters
    ----------
    knowledge : Optional[Any]
        The knowledge base to use.

    Examples
    --------
    >>> tools = CustomTools(knowledge=knowledge_base)
    """
    def __init__(self, knowledge: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(knowledge=knowledge, **kwargs)
        self.register(self.search_knowledge_base)

    def search_knowledge_base(self, agent: Agent, query: str) -> str:
        """Search the knowledge base for information.

        Parameters
        ----------
        agent : Agent
            The agent instance.
        query : str
            The search query.

        Returns
        -------
        str
            Search results or error message.
        """
        return search_knowledge_base(agent, query)

knowledge_base: TextKnowledgeBase = TextKnowledgeBase(
    path=PROJECT_FOLDER,
    formats=FORMATS,
    vector_db=LanceDb(
        uri=str(VECTOR_DB_FOLDER),
        table_name="project_knowledge",
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
    num_documents=NUM_DOCUMENTS,
    chunking_strategy=chunking_strategy,
)

# Initialize the agent with custom tools
document_agent: Agent = Agent(
    name=AGENT_NAME,
    model=OpenAIChat(id=OPENAI_MODEL),
    description=dedent(AGENT_DESCRIPTION),
    instructions=dedent(f"""Current datetime: {CURRENT_DATETIME}\n\nWhen searching for documents, use general keywords rather than specific date formats.\nFor example, instead of searching for \"Weekly Status - 12-18-2024\", search for \"Weekly Status December 2024\".\n\n{AGENT_INSTRUCTIONS}"""),
    knowledge=knowledge_base,
    tools=[
        ReasoningTools(add_instructions=True),
        CustomTools(knowledge=knowledge_base, add_instructions=True),
        DuckDuckGoTools(),
    ],
    search_knowledge=True,
    add_references=True,
    markdown=True,
    show_tool_calls=True,
)

def search_knowledge_base(agent: Agent, query: str) -> str:
    """Search the knowledge base with proper error handling.

    Parameters
    ----------
    agent : Agent
        The agent instance for the search.
    query : str
        The search query string.

    Returns
    -------
    str
        Search results or error message.

    Examples
    --------
    >>> search_knowledge_base(agent, "project timeline")
    'Document: ...\nContent: ...'
    """
    try:
        sanitized_query: str = sanitize_search_query(query)
        if agent.knowledge is not None:
            docs: list[Any] = agent.knowledge.search(query=sanitized_query, num_documents=20)
            if not docs:
                return f"No documents found for query: {query}"
            results: list[str] = []
            for doc in docs:
                results.append(f"Document: {doc.name}\nContent: {doc.content[:500]}...\n")
            return "\n".join(results)
        return "Knowledge base not available"
    except Exception as e:
        error_msg: str = f"Error searching for documents: {str(e)}"
        log_error(error_msg)
        return error_msg

def sanitize_search_query(query: str) -> str:
    """Sanitize search query to prevent syntax errors.

    Parameters
    ----------
    query : str
        The original query string.

    Returns
    -------
    str
        Sanitized query string.

    Examples
    --------
    >>> sanitize_search_query("Weekly Status - 12-18-2024")
    'Weekly Status   12 18 2024'
    """
    sanitized: str = re.sub(r'[-:]', ' ', query)
    return sanitized

def escape_markdown_tags(text: str, tags: Optional[set[str]] = None) -> str:
    """Escape backticks and triple backticks to prevent markdown formatting.

    Parameters
    ----------
    text : str
        The input text to escape.
    tags : Optional[set[str]], optional
        Tags to include in markdown (default is None).

    Returns
    -------
    str
        Escaped text.
    """
    temp_text: str = str(text)
    # Use double backslashes to avoid SyntaxWarning for invalid escape sequences
    temp_text = temp_text.replace('```', '\\`\\`\\`').replace('`', '\\`')
    return temp_text

def get_text_from_message(message: object) -> str:
    """Extract content from a message object or dict.

    Parameters
    ----------
    message : object
        The message object or dict.

    Returns
    -------
    str
        The extracted content as a string.
    """
    if isinstance(message, dict) and 'content' in message:
        return str(message['content'])
    if hasattr(message, 'content'):
        return str(message.content)
    return str(message)

def format_tool_call(tool_call: Any) -> str:
    """Format a tool call for display in a user-friendly way.

    Parameters
    ----------
    tool_call : Any
        The tool call to format.

    Returns
    -------
    str
        A formatted string representation of the tool call.
    """
    if not isinstance(tool_call, dict):
        return str(tool_call)
    function_info: dict[str, Any] = tool_call.get("function", {})
    function_name: str = function_info.get("name", "unknown")
    # Special handling for analyze function
    if function_name == "analyze":
        try:
            arguments: dict[str, Any] = json.loads(function_info.get("arguments", "{}"))
            title: str = arguments.get("title", "Analysis")
            return f"{function_name}(title=\"{title}\")"
        except Exception:
            pass
    # Special handling for search_knowledge_base function
    elif function_name == "search_knowledge_base":
        try:
            arguments: dict[str, Any] = json.loads(function_info.get("arguments", "{}"))
            query: str = arguments.get("query", "")
            return f"{function_name}(query=\"{query}\")"
        except Exception:
            pass
    # Default formatting for other functions
    return f"{function_name}({function_info.get('arguments', '')})"

def custom_print_response(
    agent: Agent,
    message: object,
    stream: bool = True,
    show_full_reasoning: bool = True,
    stream_intermediate_steps: bool = True,
    num_documents: int = 20,
    show_message: bool = True,
    show_reasoning: bool = True,
    tags_to_include_in_markdown: set[str] = {"think", "thinking"},
) -> None:
    """Custom implementation to replace agent.print_response() with more control.

    Parameters
    ----------
    agent : Agent
        The agent instance.
    message : object
        The message to process.
    stream : bool, optional
        Whether to stream the response (default is True).
    show_full_reasoning : bool, optional
        Whether to show detailed reasoning (default is True).
    stream_intermediate_steps : bool, optional
        Whether to stream intermediate steps (default is True).
    num_documents : int, optional
        Number of documents to use in search (default is 20).
    show_message : bool, optional
        Whether to show the input message (default is True).
    show_reasoning : bool, optional
        Whether to show reasoning steps (default is True).
    tags_to_include_in_markdown : set[str], optional
        Tags to include in markdown (default is {"think", "thinking"}).

    Returns
    -------
    None
    """
    # Set markdown formatting
    agent.markdown = True

    if stream:
        _response_content: str = ""
        _response_thinking: str = ""
        reasoning_steps: list[Any] = []
        last_printed_length: int = 0
        processed_tool_calls: set[str] = set()  # Track which tool calls we've already processed

        response_timer: Timer = Timer()
        response_timer.start()

        try:
            for resp in agent.run(
                message=message,
                stream=True,
                stream_intermediate_steps=stream_intermediate_steps,
                num_documents=num_documents,
            ):
                if isinstance(resp, RunResponse):
                    if resp.event == RunEvent.run_response:
                        # Only print the new content that was added
                        if isinstance(resp.content, str):
                            new_content: str = resp.content
                            _response_content += new_content
                            # Print only the new content
                            print(new_content, end="", flush=True)

                        if resp.thinking is not None:
                            _response_thinking += resp.thinking

                    if resp.extra_data is not None and resp.extra_data.reasoning_steps is not None:
                        # Only print reasoning steps that are new
                        new_steps: list[Any] = resp.extra_data.reasoning_steps[len(reasoning_steps):]
                        if new_steps and show_reasoning:
                            for i, step in enumerate(new_steps, len(reasoning_steps) + 1):
                                step_content: str = ""
                                if step.title is not None:
                                    step_content += f"{step.title}\n"
                                if step.action is not None:
                                    step_content += f"Action: {step.action}\n"
                                if step.result is not None:
                                    step_content += f"Result: {step.result}\n"

                                if show_full_reasoning:
                                    # Add detailed reasoning information if available
                                    if step.reasoning is not None:
                                        step_content += f"Reasoning: {step.reasoning}\n"
                                    if step.confidence is not None:
                                        step_content += f"Confidence: {step.confidence}\n"

                                print(f"\n\n============ Reasoning step {i} ============\n{step_content}")

                        reasoning_steps = resp.extra_data.reasoning_steps

                    # Handle tool calls if they're new - format them like reasoning steps
                    if (
                        agent.show_tool_calls
                        and agent.run_response is not None
                        and agent.run_response.formatted_tool_calls
                        and len(agent.run_response.formatted_tool_calls) > 0
                    ):
                        # Process new tool calls
                        for i, tool_call in enumerate(agent.run_response.formatted_tool_calls):
                            # Generate a unique ID for this tool call to avoid duplicates
                            tool_id: str = str(hash(str(tool_call)))
                            if tool_id not in processed_tool_calls:
                                processed_tool_calls.add(tool_id)

                                # Extract information from the tool call
                                tool_content: str = format_tool_call_as_reasoning(tool_call, show_full_reasoning)

                                # Print the tool call as a reasoning step
                                print(f"\n\n============ Tool Call ============\n{tool_content}")

            # Print a newline at the end of the response
            print("\n")

            # Add thinking content if available and not already printed
            if len(_response_thinking) > 0:
                print(f"\n============ Thinking ({response_timer.elapsed:.1f}s) ============\n{_response_thinking}")

            response_timer.stop()

            # Add citations if available
            if (
                agent.run_response
                and agent.run_response.citations
                and agent.run_response.citations.urls
            ):
                md_content: str = "\n".join(
                    f"{i + 1}. [{citation.title or citation.url}]({citation.url})"
                    for i, citation in enumerate(agent.run_response.citations.urls)
                    if citation.url  # Only include citations with valid URLs
                )
                if md_content:  # Only create panel if there are citations
                    print(f"\n============ Citations ============\n{md_content}")

        except KeyboardInterrupt:
            print("\n\n[Interrupted by user]")
        except Exception as e:
            print(f"\n\n[Error: {str(e)}]")

    else:
        # Non-streaming implementation (unchanged)
        print("Thinking...")
        try:
            response: RunResponse = agent.run(message=message, num_documents=num_documents)

            # Add message
            if message and show_message:
                print(f"\n============ Message ============\n{get_text_from_message(message)}\n")

            # Add reasoning steps
            if (
                isinstance(response, RunResponse)
                and response.extra_data is not None
                and response.extra_data.reasoning_steps is not None
                and show_reasoning
            ):
                for i, step in enumerate(response.extra_data.reasoning_steps, 1):
                    # Build step content
                    step_content: str = ""
                    if step.title is not None:
                        step_content += f"{step.title}\n"
                    if step.action is not None:
                        step_content += f"Action: {step.action}\n"
                    if step.result is not None:
                        step_content += f"Result: {step.result}\n"

                    if show_full_reasoning:
                        # Add detailed reasoning information if available
                        if step.reasoning is not None:
                            step_content += f"Reasoning: {step.reasoning}\n"
                        if step.confidence is not None:
                            step_content += f"Confidence: {step.confidence}\n"

                    print(f"\n============ Reasoning step {i} ============\n{step_content}")

            # Add thinking
            if isinstance(response, RunResponse) and response.thinking is not None:
                print(f"\n============ Thinking ============\n{response.thinking}")

            # Add tool calls
            if (
                agent.show_tool_calls
                and isinstance(response, RunResponse)
                and response.formatted_tool_calls
            ):
                tool_calls_content: str = ""
                for tool_call in response.formatted_tool_calls:
                    tool_calls_content += f"• {tool_call}\n"
                print(f"\n============ Tool Calls ============\n{tool_calls_content}")

            # Add response
            if isinstance(response, RunResponse) and response.content:
                content: str = response.content
                if agent.markdown and isinstance(content, str):
                    escaped_content: str = escape_markdown_tags(content, tags_to_include_in_markdown)
                    content = escaped_content

                print(f"\n============ Response ============\n{content}")

            # Add citations
            if (
                isinstance(response, RunResponse)
                and response.citations is not None
                and response.citations.urls is not None
            ):
                md_content: str = "\n".join(
                    f"{i + 1}. [{citation.title or citation.url}]({citation.url})"
                    for i, citation in enumerate(response.citations.urls)
                    if citation.url  # Only include citations with valid URLs
                )
                if md_content:  # Only create panel if there are citations
                    print(f"\n============ Citations ============\n{md_content}")
        except KeyboardInterrupt:
            print("\n\n[Interrupted by user]")
        except Exception as e:
            print(f"\n\n[Error: {str(e)}]")

def format_tool_call_as_reasoning(tool_call: Any, show_full_details: bool = True) -> str:
    """Format a tool call to look like a reasoning step.

    Parameters
    ----------
    tool_call : Any
        The tool call to format.
    show_full_details : bool, optional
        Whether to show full details like confidence (default is True).

    Returns
    -------
    str
        A formatted string representation of the tool call.
    """
    try:
        # If tool_call is a string, just return it as-is
        if isinstance(tool_call, str):
            return tool_call

        # Otherwise, assume it's a dict and proceed as before
        tool_call_dict: dict[str, Any] = tool_call
        function_info: dict[str, Any] = tool_call_dict.get("function", {})
        function_name: str = function_info.get("name", "unknown")

        try:
            arguments: dict[str, Any] = json.loads(function_info.get("arguments", "{}"))
        except Exception:
            arguments = {}

        # Extract common fields that might be in the arguments
        title: str = arguments.get("title", function_name.capitalize())
        thought: str = arguments.get("thought", "")
        action: str = arguments.get("action", "")
        result: str = arguments.get("result", "")
        confidence: Optional[float] = arguments.get("confidence", None)

        # Build the formatted output
        content: str = f"{title}\n"

        if action:
            content += f"Action: {action}\n"

        if result:
            content += f"Result: {result}\n"

        if show_full_details:
            if thought:
                content += f"Reasoning: {thought}\n"

            if confidence is not None:
                content += f"Confidence: {confidence}\n"

        return content
    except Exception as e:
        return f"Error formatting tool call: {str(e)}\nRaw tool call: {tool_call}"

def main() -> None:
    # Set to True only the first time to load the knowledge base
    load_knowledge: bool = True
    recreate: bool = "--recreate" in sys.argv

    if load_knowledge and knowledge_base is not None:
        print("\n============ Startup ============")
        print("Loading knowledge base from project files...")
        knowledge_base.load(recreate=recreate)
        # Count number of loaded documents (show only if > 0)
        try:
            table = getattr(knowledge_base.vector_db, 'table', None)
            actual_count: Optional[int] = None
            if table is not None and hasattr(table, 'count_rows') and callable(table.count_rows):
                actual_count = table.count_rows()
            if actual_count is not None:
                loaded_count: str = f"{actual_count} documents loaded."

            print(f"Knowledge base loaded successfully! {loaded_count}\n")

        except Exception:
            print("Knowledge base loaded successfully!\n")
            logging.exception("Could not determine number of loaded documents from LanceDB.")
    # Welcome message
    print(f"============ Welcome ============")
    print(f"{AGENT_NAME}")
    print()
    print(AGENT_DESCRIPTION + "\n")
    print("Type 'exit' to quit.\n")

    while True:
        query: str = input("Enter your question: ")
        if query.lower() in ('exit', 'quit'):
            break

        # Use custom_print_response instead of agent.print_response
        custom_print_response(
            document_agent,
            query,
            stream=True,
            show_full_reasoning=True,
            stream_intermediate_steps=True,
            num_documents=20
        )

def search_all_documents(agent: Agent, query: str) -> list[Any]:
    """Helper function to search all documents in the knowledge base"""
    if agent.knowledge is not None:
        return agent.knowledge.search(query=query, num_documents=50)
    return []

if __name__ == "__main__":
    main()
