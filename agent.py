import json
import sys
import re
import subprocess

import requests


MODEL = "qwen3.5:2b"
OLLAMA_URL = "http://localhost:11434/api/generate"


# ---------------------------------------------------------
# QWEN
# ---------------------------------------------------------

def ask_qwen(prompt, max_tokens=300):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "num_predict": max_tokens
            }
        },
        timeout=300
    )

    response.raise_for_status()
    return response.json()["response"]


# ---------------------------------------------------------
# TOOL SELECTION
# ---------------------------------------------------------

def choose_tool(question, tools):
    tool_names = [tool.name for tool in tools]

    prompt = f"""
You are the tool router for a research paper assistant.

Available tools:
{chr(10).join("- " + name for name in tool_names)}

User question:
{question}

Choose exactly ONE tool.

Rules:

1. If the user explicitly asks for a page number/content of a page,
   use get_page.

2. If the user asks for metadata, title, author, number of pages,
   or indexed chunks of a paper,
   use get_paper_metadata.

3. If the user asks about one specific paper,
   use search_within_paper.

4. If the user asks a general research question without specifying
   a paper, use search_papers.

5. If the user explicitly compares TWO DIFFERENT papers,
   use compare_papers_tool.

6. If the user compares two algorithms/concepts INSIDE ONE paper,
   use search_within_paper.

Return ONLY valid JSON.

Format:
{{
  "tool": "tool_name",
  "arguments": {{}}
}}
"""

    response = ask_qwen(prompt, max_tokens=150)

    # Extract JSON
    match = re.search(r'\{.*\}', response, re.DOTALL)

    if not match:
        return {
            "tool": "search_papers",
            "arguments": {"query": question}
        }

    try:
        decision = json.loads(match.group())
    except Exception:
        return {
            "tool": "search_papers",
            "arguments": {"query": question}
        }

    return decision


# ---------------------------------------------------------
# SIMPLE ROUTING FIXES
# ---------------------------------------------------------

def fix_routing(question, decision):
    q = question.lower()

    # -----------------------------------------------------
    # 1. PAGE REQUEST
    # -----------------------------------------------------

    page_match = re.search(r'\bpage\s+(\d+)\b', q)

    if page_match:
        page_number = int(page_match.group(1))

        pdf_match = re.search(
            r'([\w.-]+\.pdf)',
            question,
            re.IGNORECASE
        )

        if pdf_match:
            paper_name = pdf_match.group(1)
        else:
            paper_name = decision.get(
                "arguments", {}
            ).get("paper_name", "")

        return {
            "tool": "get_page",
            "arguments": {
                "paper_name": paper_name,
                "page_number": page_number
            }
        }

    # -----------------------------------------------------
    # 2. TWO-PAPER COMPARISON
    # -----------------------------------------------------

    pdfs = re.findall(
        r'([\w.-]+\.pdf)',
        question,
        re.IGNORECASE
    )

    comparison_words = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "versus",
        "vs"
    ]

    is_comparison = any(word in q for word in comparison_words)

    if is_comparison and len(pdfs) >= 2:

        return {
            "tool": "compare_papers_tool",
            "arguments": {
                "paper1": pdfs[0],
                "paper2": pdfs[1],
                "query": question
            }
        }

    # -----------------------------------------------------
    # 3. ONE-PAPER COMPARISON / QUESTION
    # -----------------------------------------------------

    if is_comparison and len(pdfs) == 1:

        return {
            "tool": "search_within_paper",
            "arguments": {
                "paper_name": pdfs[0],
                "query": question
            }
        }

    # -----------------------------------------------------
    # 4. SPECIFIC PAPER
    # -----------------------------------------------------

    if len(pdfs) == 1:

        # Metadata request
        metadata_words = [
            "metadata",
            "title",
            "author",
            "number of pages",
            "indexed chunks"
        ]

        if any(word in q for word in metadata_words):

            return {
                "tool": "get_paper_metadata",
                "arguments": {
                    "paper_name": pdfs[0]
                }
            }

        return {
            "tool": "search_within_paper",
            "arguments": {
                "paper_name": pdfs[0],
                "query": question
            }
        }

    # -----------------------------------------------------
    # 5. GENERAL QUESTION
    # -----------------------------------------------------

    return {
        "tool": "search_papers",
        "arguments": {
            "query": question
        }
    }


# ---------------------------------------------------------
# FINAL ANSWER
# ---------------------------------------------------------

def generate_final_answer(question, evidence):
    prompt = f"""
You are an AI research assistant.

Answer the user's question using ONLY the retrieved evidence.

User question:
{question}

Retrieved evidence:
{evidence}

Instructions:

- Give a clear and concise answer.
- Do not invent information.
- Use the paper name and page number from the evidence when useful.
- If comparing algorithms from the same paper, compare only what the
  evidence supports.
- If comparing two papers, keep evidence from each paper separate.
- Do not claim that a paper has no approach/algorithm unless the
  retrieved evidence actually establishes that.
- If the evidence is insufficient, say that clearly.

Answer:
"""

    return ask_qwen(prompt, max_tokens=400)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    # Start MCP server
    process = subprocess.Popen(
        [sys.executable, "mcp_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True
    )

    # Import MCP client components
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"]
    )

    async def run_agent():

        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        async with stdio_client(server_params) as (read, write):

            async with ClientSession(read, write) as session:

                await session.initialize()

                tools_result = await session.list_tools()
                tools = tools_result.tools

                print("\n===== MCP TOOLS =====\n")

                for tool in tools:
                    print(f"- {tool.name}")

                question = input(
                    "\nEnter your research question: "
                ).strip()

                if not question:
                    return

                # Qwen first chooses
                qwen_decision = choose_tool(
                    question,
                    tools
                )

                print("\n===== QWEN TOOL DECISION =====\n")
                print(
                    json.dumps(
                        qwen_decision,
                        indent=4
                    )
                )

                # Small deterministic correction layer
                decision = fix_routing(
                    question,
                    qwen_decision
                )

                print("\n===== FINAL ROUTING DECISION =====\n")
                print(
                    json.dumps(
                        decision,
                        indent=4
                    )
                )

                tool_name = decision["tool"]
                arguments = decision.get(
                    "arguments",
                    {}
                )

                print(
                    f"\nSelected MCP tool: {tool_name}"
                )
                print(
                    f"Arguments: {arguments}"
                )

                print(
                    f"\n===== EXECUTING MCP TOOL: "
                    f"{tool_name} =====\n"
                )

                result = await session.call_tool(
                    tool_name,
                    arguments=arguments
                )

                evidence = ""

                for content in result.content:

                    if hasattr(content, "text"):
                        evidence += content.text

                print("\n===== MCP TOOL RESULT =====\n")
                print(evidence)

                # Final answer
                print("\n===== QWEN FINAL ANSWER =====\n")

                if not evidence.strip():
                    print("No relevant evidence found.")
                    return

                answer = generate_final_answer(
                    question,
                    evidence
                )

                print(answer)

    import asyncio

    try:
        asyncio.run(run_agent())

    except KeyboardInterrupt:
        print("\nStopped.")

    except Exception as e:
        print(f"\nError: {e}")

    finally:
        process.terminate()


if __name__ == "__main__":
    main()

