import json
import sys
import re
import os

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
            "keep_alive": -1,  # Keep model resident in VRAM/RAM between queries
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
You are the tool router for an AI research assistant.

Available tools:
1. search_papers
   - Use for general research questions across all papers, OR general comparisons when NO specific paper is named.
   - Arguments: {{"query": "..."}}

2. get_paper_metadata
   - Use ONLY when the user asks for paper metadata such as title, author, number of pages, or chunk count.
   - Arguments: {{"paper_name": "..."}}

3. get_page
   - Use when the user explicitly asks for a specific page number of a paper.
   - Arguments: {{"paper_name": "...", "page_number": <int>}}

4. search_within_paper
   - Use when the question is about ONE specific paper (e.g. methods, algorithms, findings), OR when comparing concepts/algorithms WITHIN that same paper.
   - Arguments: {{"paper_name": "...", "query": "..."}}

5. compare_papers_tool
   - Use ONLY when comparing TWO different research papers. Both papers must be specified.
   - Arguments: {{"paper1": "...", "paper2": "...", "query": "..."}}

User question:
{question}

Choose exactly ONE tool. Return ONLY valid JSON:
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
    raw_args = decision.get("arguments", {}) if isinstance(decision, dict) else {}

    # -----------------------------------------------------
    # Paper Name Detection & Normalization
    # -----------------------------------------------------
    papers = []

    # 1. Check direct .pdf mentions in query
    for p in re.findall(r'([A-Za-z0-9_\-\.]+\.pdf)', question, re.IGNORECASE):
        norm = p.lower()
        if not any(x.lower() == norm for x in papers):
            papers.append(p)

    # 2. Check known papers in papers/ folder
    known_papers = []
    if os.path.exists("papers"):
        try:
            known_papers = [f for f in os.listdir("papers") if f.lower().endswith(".pdf")]
        except Exception:
            known_papers = []

    for kp in known_papers:
        stem = os.path.splitext(kp)[0].lower()
        if re.search(r'(?:\b|_)' + re.escape(stem) + r'(?:\b|_)', q):
            if not any(x.lower() == kp.lower() for x in papers):
                papers.append(kp)

    # 3. Check generic patterns like "paper 2" or "paper2"
    for m in re.finditer(r'\bpaper\s*(\d+)\b', q):
        cand = f"paper{m.group(1)}.pdf"
        if not any(x.lower() == cand.lower() for x in papers):
            papers.append(cand)

    # 4. Check paper names from decision arguments ONLY if verified against known papers
    # or explicitly named with .pdf in the question (avoids turning algorithms like "CART" into "cart.pdf")
    for key in ["paper1", "paper2", "paper_name", "paper", "paper_id", "paper_file", "file"]:
        val = raw_args.get(key)
        if val and isinstance(val, str):
            val = val.strip()
            if val:
                norm = val if val.lower().endswith(".pdf") else val + ".pdf"
                is_known = any(kp.lower() == norm.lower() for kp in known_papers)
                is_explicit_in_q = norm.lower() in q
                if is_known or is_explicit_in_q:
                    if not any(x.lower() == norm.lower() for x in papers):
                        papers.append(norm)

    # -----------------------------------------------------
    # 1. EXPLICIT PAGE REQUEST
    # -----------------------------------------------------
    page_match = re.search(r'\b(?:page|p\.?)\s*[:#]?\s*(\d+)\b', q)
    if page_match:
        page_number = int(page_match.group(1))

        if papers:
            paper_name = papers[0]
        else:
            raw_paper = (
                raw_args.get("paper_name")
                or raw_args.get("paper")
                or raw_args.get("paper_id")
                or raw_args.get("file")
                or raw_args.get("paper_file")
                or ""
            )
            if raw_paper and not raw_paper.lower().endswith(".pdf"):
                raw_paper += ".pdf"
            paper_name = raw_paper

        return {
            "tool": "get_page",
            "arguments": {
                "paper_name": paper_name,
                "page_number": page_number
            }
        }

    # -----------------------------------------------------
    # 2. METADATA REQUEST
    # -----------------------------------------------------
    metadata_words = [
        "metadata",
        "title",
        "author",
        "number of pages",
        "page count",
        "how many pages",
        "indexed chunks",
        "chunk count"
    ]
    is_metadata = any(word in q for word in metadata_words)

    if is_metadata and len(papers) >= 1:
        return {
            "tool": "get_paper_metadata",
            "arguments": {
                "paper_name": papers[0]
            }
        }

    # -----------------------------------------------------
    # 3. COMPARISON REQUEST
    # -----------------------------------------------------
    comparison_words = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "versus",
        "vs",
        "vs."
    ]
    is_comparison = any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in comparison_words)

    if is_comparison:
        # Comparison across two different papers
        if len(papers) >= 2:
            comp_query = raw_args.get("query", "").strip() if isinstance(raw_args, dict) else ""
            if not comp_query or comp_query.lower() in [papers[0].lower(), papers[1].lower()]:
                comp_query = question

            return {
                "tool": "compare_papers_tool",
                "arguments": {
                    "paper1": papers[0],
                    "paper2": papers[1],
                    "query": comp_query
                }
            }

        # Comparison within one single paper
        if len(papers) == 1:
            return {
                "tool": "search_within_paper",
                "arguments": {
                    "paper_name": papers[0],
                    "query": question
                }
            }

        # General comparison without a paper falls through to search_papers

    # -----------------------------------------------------
    # 4. QUESTION ABOUT ONE SPECIFIC PAPER
    # -----------------------------------------------------
    if len(papers) == 1:
        return {
            "tool": "search_within_paper",
            "arguments": {
                "paper_name": papers[0],
                "query": question
            }
        }

    # -----------------------------------------------------
    # 5. GENERAL QUESTION (NO SPECIFIED PAPER)
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
You are an AI research assistant. Answer the user's question using ONLY the
retrieved evidence below. Do not invent facts, results, or formulas not present
in the evidence.

User question:
{question}

Retrieved evidence:
{evidence}

Instructions:
- Directly answer the question asked. If the retrieved evidence does NOT mention or discuss the topic requested by the user, state clearly and concisely that the paper does not discuss or contain information on that topic, rather than describing unrelated evidence.
- When the evidence is relevant, cover EVERY item that the evidence mentions regarding the question. If the evidence lists multiple algorithms, metrics, or methods, explain ALL of them — do not stop early.
- When the evidence contains a formula, reproduce it in plain-text math
  (e.g. Accuracy = (TP + TN) / (TP + TN + FP + FN)) and define every
  variable (TP, TN, FP, FN).
- Cite the source filename and page number for each key claim.
- Keep each explanation clear and grounded in the evidence.
- Do not invent information or results not present in the evidence.

Answer:
"""

    return ask_qwen(prompt, max_tokens=800)


# ---------------------------------------------------------
# ---------------------------------------------------------
# REUSABLE AGENT QUERY INTERFACE
# ---------------------------------------------------------

async def run_agent_query(question):
    import asyncio
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tools = tools_result.tools

            qwen_decision = choose_tool(question, tools)
            decision = fix_routing(question, qwen_decision)

            tool_name = decision["tool"]
            arguments = decision.get("arguments", {})

            result = await session.call_tool(tool_name, arguments=arguments)

            evidence = ""
            for content in result.content:
                if hasattr(content, "text"):
                    evidence += content.text

            if not evidence.strip():
                answer = "No relevant evidence found."
            else:
                answer = generate_final_answer(question, evidence)

            return {
                "question": question,
                "qwen_decision": qwen_decision,
                "decision": decision,
                "tool_name": tool_name,
                "arguments": arguments,
                "evidence": evidence,
                "answer": answer
            }


def ask_agent(question):
    """
    Synchronous entry point for UI or external callers.
    Safely executes run_agent_query across various event loop environments.
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(run_agent_query(question))).result()
    else:
        return asyncio.run(run_agent_query(question))


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    question = input(
        "\nEnter your research question: "
    ).strip()

    if not question:
        return

    try:
        res = ask_agent(question)
        print("\n===== QWEN TOOL DECISION =====\n")
        print(json.dumps(res["qwen_decision"], indent=4))

        print("\n===== FINAL ROUTING DECISION =====\n")
        print(json.dumps(res["decision"], indent=4))

        print(f"\nSelected MCP tool: {res['tool_name']}")
        print(f"Arguments: {res['arguments']}")

        print(f"\n===== EXECUTING MCP TOOL: {res['tool_name']} =====\n")
        print("\n===== MCP TOOL RESULT =====\n")
        print(res["evidence"])

        print("\n===== QWEN FINAL ANSWER =====\n")
        print(res["answer"])

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()


