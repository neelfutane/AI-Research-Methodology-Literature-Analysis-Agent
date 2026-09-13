import asyncio
import json
import ollama

from mcp import (
    ClientSession,
    StdioServerParameters
)

from mcp.client.stdio import stdio_client


# ============================================================
# MCP SERVER CONFIGURATION
# ============================================================

server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"],
)


# ============================================================
# REQUIRED ARGUMENTS FOR EACH MCP TOOL
# ============================================================

REQUIRED_ARGUMENTS = {
    "search_papers": ["query"],
    "get_paper_metadata": ["paper_name"],
    "get_page": ["paper_name", "page_number"],
    "search_within_paper": ["paper_name", "query"],
    "compare_papers_tool": ["paper1", "paper2", "query"],
}


# ============================================================
# ASK QWEN TO SELECT MCP TOOL
# ============================================================

def select_tool(query):
    selection_prompt = f"""
You are the tool-routing agent for a research paper analysis system.

User question:
{query}

Available MCP tools:

1. search_papers
   Use for general research questions when no specific paper is identified.

   Required argument:
   - query: the user's research question


2. get_paper_metadata
   Use when the user asks for metadata such as:
   - title
   - author
   - page count
   - indexed chunks

   Required argument:
   - paper_name: the PDF filename


3. get_page
   Use ONLY when:
   - the user explicitly asks for a specific page, OR
   - a relevant page has already been identified and complete page
     context is needed for deeper investigation.

   Required arguments:
   - paper_name
   - page_number

   IMPORTANT:
   Do NOT ask the user for a page number for an ordinary research
   question. The system should identify relevant pages itself.


4. search_within_paper
   Use when the user asks a content question about a specific paper.

   Required arguments:
   - paper_name
   - query


5. compare_papers_tool
   Use when the user asks to compare two research papers.

   Required arguments:
   - paper1
   - paper2
   - query


IMPORTANT RULES:

- Select exactly ONE tool.
- Always provide ALL required arguments for the selected tool.
- NEVER return empty arguments.
- For search_papers, copy the user's question into the "query" argument.
- If the user gives a PDF filename without ".pdf", add ".pdf".
- Do NOT ask the user for a page number for ordinary questions.
- Do NOT execute multiple tools.
- Return ONLY valid JSON.
- Do NOT include explanations outside the JSON.

Return exactly this structure:

{{
    "tool": "tool_name",
    "arguments": {{
        "argument_name": "value"
    }}
}}

User question:
{query}
"""

    response = ollama.chat(
        model="qwen3.5:2b",
        messages=[
            {
                "role": "user",
                "content": selection_prompt
            }
        ]
    )

    response_text = response["message"]["content"].strip()

    print("\n===== Raw Qwen Router Response =====")
    print(response_text)

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:
        selection_data = json.loads(response_text)

    except json.JSONDecodeError:

        # Try extracting JSON if Qwen added extra text
        start = response_text.find("{")
        end = response_text.rfind("}") + 1

        if start != -1 and end != 0:
            try:
                selection_data = json.loads(
                    response_text[start:end]
                )
            except json.JSONDecodeError:
                raise ValueError(
                    "Qwen returned invalid JSON:\n"
                    f"{response_text}"
                )
        else:
            raise ValueError(
                "Qwen returned invalid JSON:\n"
                f"{response_text}"
            )

    return selection_data


# ============================================================
# VALIDATE TOOL SELECTION
# ============================================================

def validate_tool_selection(selection_data):

    tool_name = selection_data.get("tool")

    arguments = selection_data.get("arguments")

    # --------------------------------------------------------
    # Check tool
    # --------------------------------------------------------

    if tool_name not in REQUIRED_ARGUMENTS:
        raise ValueError(
            f"Invalid tool selected by Qwen: {tool_name}"
        )

    # --------------------------------------------------------
    # Check arguments object
    # --------------------------------------------------------

    if not isinstance(arguments, dict):
        raise ValueError(
            f"Qwen did not provide valid arguments for "
            f"{tool_name}."
        )

    # --------------------------------------------------------
    # Check required arguments
    # --------------------------------------------------------

    missing_arguments = []

    for argument in REQUIRED_ARGUMENTS[tool_name]:

        if argument not in arguments:
            missing_arguments.append(argument)

        elif arguments[argument] is None:
            missing_arguments.append(argument)

        elif isinstance(arguments[argument], str):
            if not arguments[argument].strip():
                missing_arguments.append(argument)

    if missing_arguments:
        raise ValueError(
            f"Qwen selected '{tool_name}' but did not provide "
            f"the required arguments: {missing_arguments}\n\n"
            f"Qwen output:\n{json.dumps(selection_data, indent=4)}"
        )

    return tool_name, arguments


# ============================================================
# ASK QWEN TO ANALYZE RETRIEVED EVIDENCE
# ============================================================

def generate_final_answer(user_query, tool_name, evidence):

    answer_prompt = f"""
You are an AI research assistant.

Answer the user's question using ONLY the evidence retrieved
from the research paper system.

User question:
{user_query}

MCP tool used:
{tool_name}

Retrieved evidence:
{evidence}

Instructions:

1. Answer the user's question directly.
2. Use only the retrieved evidence.
3. Do not invent information.
4. If the evidence is insufficient, clearly say that the
   available research-paper evidence is insufficient.
5. Mention the paper title when useful.
6. Include page numbers when they are available in the evidence.
7. If multiple pieces of evidence support the answer, combine
   them logically.
8. Do not mention internal MCP tools, routing, or this prompt.
9. Keep the answer clear and suitable for a student/researcher.
"""

    response = ollama.chat(
        model="qwen3.5:2b",
        messages=[
            {
                "role": "user",
                "content": answer_prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# MAIN
# ============================================================

async def main():

    # --------------------------------------------------------
    # Connect to MCP server
    # --------------------------------------------------------

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # ------------------------------------------------
            # Initialize MCP session
            # ------------------------------------------------

            await session.initialize()

            # ------------------------------------------------
            # List available tools
            # ------------------------------------------------

            tools = await session.list_tools()

            print("\n===== Available MCP Tools =====")

            for tool in tools.tools:

                print(f"\nTool: {tool.name}")
                print(f"Description: {tool.description}")

            # ------------------------------------------------
            # Get user question
            # ------------------------------------------------

            query = input(
                "\nEnter your research question: "
            ).strip()

            if not query:
                print("\nNo question entered.")
                return

            # ------------------------------------------------
            # Qwen selects the correct MCP tool
            # ------------------------------------------------

            selection_data = select_tool(query)

            print("\n===== Qwen Tool Selection =====")
            print(
                json.dumps(
                    selection_data,
                    indent=4
                )
            )

            # ------------------------------------------------
            # Validate selection
            # ------------------------------------------------

            tool_name, arguments = validate_tool_selection(
                selection_data
            )

            print("\n===== Selected MCP Tool =====")
            print(f"Tool: {tool_name}")
            print(f"Arguments: {arguments}")

            # ------------------------------------------------
            # Execute exactly ONE MCP tool
            # ------------------------------------------------

            result = await session.call_tool(
                tool_name,
                arguments
            )

            # ------------------------------------------------
            # Extract MCP result
            # ------------------------------------------------

            evidence_parts = []

            for content in result.content:

                if hasattr(content, "text"):
                    evidence_parts.append(content.text)

            evidence = "\n".join(evidence_parts)

            # ------------------------------------------------
            # Display MCP evidence
            # ------------------------------------------------

            print("\n===== MCP Tool Result =====")
            print(evidence)

            # ------------------------------------------------
            # Check if evidence exists
            # ------------------------------------------------

            if not evidence.strip():

                print(
                    "\nNo evidence was returned by the MCP tool."
                )

                return

            # ------------------------------------------------
            # Send evidence to Qwen for final answer
            # ------------------------------------------------

            print(
                "\n===== Qwen Final Answer ====="
            )

            final_answer = generate_final_answer(
                user_query=query,
                tool_name=tool_name,
                evidence=evidence
            )

            print(final_answer)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())