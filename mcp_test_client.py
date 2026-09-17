import asyncio
import json
import ollama

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"],
)


REQUIRED_ARGUMENTS = {
    "search_papers": ["query"],
    "get_paper_metadata": ["paper_name"],
    "get_page": ["paper_name", "page_number"],
    "search_within_paper": ["paper_name", "query"],
    "compare_papers_tool": ["paper1", "paper2", "query"],
}


def select_tool(query):
    selection_prompt = f"""
You are the tool-routing agent for an AI research paper analysis system.

Your job is to understand the user's question and select EXACTLY ONE
appropriate MCP tool.

USER QUESTION:
{query}


AVAILABLE TOOLS:

1. search_papers

Use this for:
- General questions about research topics
- Questions where no specific paper is identified
- Questions that require searching across all indexed papers

Arguments:
{{
    "query": "..."
}}


2. get_paper_metadata

Use this ONLY when the user asks for paper metadata such as:
- title
- author
- number of pages
- number of indexed chunks
- indexing information

Arguments:
{{
    "paper_name": "..."
}}


3. get_page

Use this when:
- The user explicitly asks for a specific page
- OR the system already knows a relevant page and complete page
  context is required

Arguments:
{{
    "paper_name": "...",
    "page_number": 3
}}

IMPORTANT:
For a normal research question, DO NOT ask the user for a page number.
The system must identify relevant pages itself.


4. search_within_paper

Use this when the question concerns the CONTENT of ONE SPECIFIC PAPER.

This includes:
- Questions about a particular paper
- Finding methods in one paper
- Finding algorithms in one paper
- Finding datasets in one paper
- Finding results in one paper
- Finding limitations in one paper
- Comparing TWO CONCEPTS, METHODS, ALGORITHMS, APPROACHES, etc.
  WITHIN THE SAME PAPER

Examples:
- "What algorithms are discussed in test_paper.pdf?"
- "What dataset does test_paper.pdf use?"
- "Compare supervised and unsupervised learning in test_paper.pdf."
- "Compare CART and Naive Bayes in test_paper.pdf."
- "What is the difference between classification and regression
   in test_paper.pdf?"

Arguments:
{{
    "paper_name": "...",
    "query": "..."
}}

IMPORTANT:
If the user mentions ONE paper and asks to compare things inside it,
use search_within_paper.


5. compare_papers_tool

Use this ONLY when the user explicitly wants to compare
TWO DIFFERENT RESEARCH PAPERS.

Examples:
- "Compare paper2.pdf and test_paper.pdf."
- "Compare the approaches used in paper2.pdf and test_paper.pdf."
- "What are the differences between the methods in these two papers?"

Arguments:
{{
    "paper1": "...",
    "paper2": "...",
    "query": "..."
}}

IMPORTANT:
The two paper names must come from the user's question.

For comparison questions, make the "query" meaningful and specific.
Do NOT use vague queries such as only:
"approaches used"
"comparison"
"difference"

Instead include the actual concepts from the question.

For example:

User:
"Compare the approaches used in paper2.pdf and test_paper.pdf."

Good query:
"learning approaches, algorithms, methodology, perceptron,
supervised machine learning approaches"

User:
"Compare the algorithms used in paper2.pdf and test_paper.pdf."

Good query:
"algorithms, perceptron, learning algorithms, supervised learning,
machine learning methods"

The comparison query should contain useful concepts that can retrieve
relevant evidence from BOTH papers.


GENERAL RULES:

- Select EXACTLY ONE tool.
- Always provide ALL required arguments.
- NEVER return empty arguments.
- NEVER ask the user for a page number.
- NEVER invent a paper name.
- If a PDF filename is given without ".pdf", add ".pdf".
- For same-paper comparisons, use search_within_paper.
- For two-paper comparisons, use compare_papers_tool.
- Do NOT execute multiple tools.
- Return ONLY valid JSON.
- Do NOT include explanations outside JSON.

Return EXACTLY this structure:

{{
    "tool": "tool_name",
    "arguments": {{
        "argument_name": "value"
    }}
}}

USER QUESTION:
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

    try:
        selection_data = json.loads(response_text)

    except json.JSONDecodeError:

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


def validate_tool_selection(selection_data):

    tool_name = selection_data.get("tool")
    arguments = selection_data.get("arguments")

    if tool_name not in REQUIRED_ARGUMENTS:
        raise ValueError(
            f"Invalid tool selected by Qwen: {tool_name}"
        )

    if not isinstance(arguments, dict):
        raise ValueError(
            f"Qwen did not provide valid arguments for {tool_name}."
        )

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
            f"Qwen output:\n"
            f"{json.dumps(selection_data, indent=4)}"
        )

    return tool_name, arguments


def limit_evidence_for_qwen(tool_name, evidence):
    """
    Prevent large MCP responses from exceeding the useful context
    available to the local Qwen model.

    For comparison results, preserve evidence from BOTH papers.
    """

    MAX_TOTAL_CHARS = 8000

    if len(evidence) <= MAX_TOTAL_CHARS:
        return evidence

    if tool_name != "compare_papers_tool":
        print(
            f"\nLimiting evidence from {len(evidence)} "
            f"to {MAX_TOTAL_CHARS} characters for Qwen."
        )

        return evidence[:MAX_TOTAL_CHARS]

    # ---------------------------------------------------------
    # Comparison-specific handling
    # ---------------------------------------------------------

    marker1 = "========== PAPER 1:"
    marker2 = "========== PAPER 2:"

    paper1_start = evidence.find(marker1)
    paper2_start = evidence.find(marker2)

    if paper1_start == -1 or paper2_start == -1:
        print(
            f"\nComparison sections not detected. "
            f"Limiting evidence to {MAX_TOTAL_CHARS} characters."
        )

        return evidence[:MAX_TOTAL_CHARS]

    paper1_section = evidence[paper1_start:paper2_start]
    paper2_section = evidence[paper2_start:]

    # Give approximately half the available context to each paper.
    per_paper_limit = MAX_TOTAL_CHARS // 2

    paper1_section = paper1_section[:per_paper_limit]
    paper2_section = paper2_section[:per_paper_limit]

    limited_evidence = (
        paper1_section
        + "\n\n"
        + paper2_section
    )

    print(
        f"\nComparison evidence was {len(evidence)} characters."
    )

    print(
        "Balanced evidence sent to Qwen: "
        f"{len(limited_evidence)} characters."
    )

    return limited_evidence


def generate_final_answer(
    user_query,
    tool_name,
    evidence
):

    answer_prompt = f"""
You are an AI research assistant.

Answer the user's question using ONLY the evidence retrieved
from the research-paper system.

USER QUESTION:
{user_query}

MCP TOOL USED:
{tool_name}

RETRIEVED EVIDENCE:
{evidence}


STRICT RULES:

1. Answer the user's question directly.

2. Use ONLY information contained in the retrieved evidence.

3. DO NOT invent facts.

4. DO NOT use outside knowledge.

5. If the evidence is insufficient, explicitly say that the
   available research-paper evidence is insufficient.

6. Preserve the EXACT paper filenames/names found in the evidence.

7. NEVER rename a paper.

8. NEVER invent a paper filename.

9. NEVER refer to evidence from one paper as if it came from
   another paper.

10. For a two-paper comparison, keep the evidence from each paper
    separate and compare only what is actually supported.

11. If evidence from one paper is insufficient, say so instead
    of substituting evidence from the other paper.

12. For same-paper comparisons, compare the requested concepts,
    algorithms, methods, or approaches using evidence from that
    same paper.

13. Include page numbers when they are available.

14. A page number may ONLY be cited for a statement if the
    retrieved evidence from that page actually supports that
    statement.

15. Do not attach an unrelated page number merely because it
    appeared somewhere in the retrieved evidence.

16. Do not mention MCP, routing, tools, prompts, Qwen, or
    internal system implementation.

17. Write a clear answer suitable for a student/researcher.

18. Do not claim that the papers cannot be compared simply because
    they study different approaches. If the user asks for a
    comparison and the evidence supports meaningful differences,
    explain those differences.

19. If comparing two papers, explicitly identify each paper before
    discussing its evidence.

20. If the question is a comparison, structure the answer clearly
    so the similarities/differences are easy to understand.
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


async def main():

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            tools = await session.list_tools()

            print("\n===== Available MCP Tools =====")

            for tool in tools.tools:

                print(f"\nTool: {tool.name}")
                print(f"Description: {tool.description}")

            query = input(
                "\nEnter your research question: "
            ).strip()

            if not query:

                print("\nNo question entered.")
                return

            # -------------------------------------------------
            # Qwen selects the MCP tool
            # -------------------------------------------------

            selection_data = select_tool(query)

            print("\n===== Qwen Tool Selection =====")

            print(
                json.dumps(
                    selection_data,
                    indent=4
                )
            )

            # -------------------------------------------------
            # Validate Qwen's decision
            # -------------------------------------------------

            tool_name, arguments = validate_tool_selection(
                selection_data
            )

            print("\n===== Selected MCP Tool =====")
            print(f"Tool: {tool_name}")
            print(f"Arguments: {arguments}")

            # -------------------------------------------------
            # Execute exactly ONE MCP tool
            # -------------------------------------------------

            result = await session.call_tool(
                tool_name,
                arguments
            )

            evidence_parts = []

            for content in result.content:

                if hasattr(content, "text"):

                    evidence_parts.append(
                        content.text
                    )

            evidence = "\n".join(evidence_parts)

            print("\n===== MCP Tool Result =====")
            print(evidence)

            if not evidence.strip():

                print(
                    "\nNo evidence was returned "
                    "by the MCP tool."
                )

                return

            # -------------------------------------------------
            # Prepare evidence for final Qwen call
            # -------------------------------------------------

            evidence_for_qwen = limit_evidence_for_qwen(
                tool_name,
                evidence
            )

            # -------------------------------------------------
            # Generate grounded final answer
            # -------------------------------------------------

            print("\n===== Qwen Final Answer =====")

            final_answer = generate_final_answer(
                user_query=query,
                tool_name=tool_name,
                evidence=evidence_for_qwen
            )

            print(final_answer)


if __name__ == "__main__":
    asyncio.run(main())