import sys
from contextlib import redirect_stdout

# Prevent model-loading logs from corrupting MCP stdout.
with redirect_stdout(sys.stderr):
    from retriever import (
        retrieve_relevant_chunks,
        retrieve_within_paper,
        compare_papers
    )

from mcp.server.mcpserver import MCPServer


# ============================================================
# MCP SERVER
# ============================================================

mcp = MCPServer("Research Paper Server")


# ============================================================
# TOOL 1: SEARCH ACROSS ALL PAPERS
# ============================================================

@mcp.tool(
    description=(
        "Search across all indexed research papers for relevant evidence "
        "related to the user's research question. Use this when the user "
        "asks a general question and the relevant paper is not known."
    )
)
def search_papers(query: str) -> str:

    results = retrieve_relevant_chunks(query)

    if not results:
        return "No relevant evidence found."

    output = []

    for i, result in enumerate(results, start=1):

        metadata = result["metadata"]

        output.append(
            f"""
===== Evidence {i} =====

Text:
{result["document"]}

Title:
{metadata.get("title", "Unknown")}

Author:
{metadata.get("author", "Unknown")}

Source:
{metadata.get("source", "Unknown")}

Page:
{metadata.get("page", "Unknown")}

Reranker Score:
{result["reranker_score"]:.4f}
"""
        )

    return "\n".join(output)


# ============================================================
# TOOL 2: PAPER METADATA
# ============================================================

@mcp.tool(
    description=(
        "Get metadata and indexing information for a specific research "
        "paper. Use this when you need to identify a paper's title, "
        "author, page count, or number of indexed chunks."
    )
)
def get_paper_metadata(paper_name: str) -> str:

    # Normalize paper name
    if not paper_name.lower().endswith(".pdf"):
        paper_name += ".pdf"

    import chromadb

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_collection(
        name="research_papers"
    )

    records = collection.get(
        where={"source": paper_name}
    )

    if not records["ids"]:
        return f"Paper '{paper_name}' was not found."

    metadatas = records["metadatas"]

    first_metadata = metadatas[0]

    title = first_metadata.get(
        "title",
        "Unknown"
    )

    author = first_metadata.get(
        "author",
        "Unknown"
    )

    pages = sorted(
        set(
            metadata.get("page")
            for metadata in metadatas
            if metadata.get("page") is not None
        )
    )

    return (
        f"Paper: {paper_name}\n"
        f"Title: {title}\n"
        f"Author: {author}\n"
        f"Pages: {len(pages)}\n"
        f"Indexed chunks: {len(records['ids'])}"
    )


# ============================================================
# TOOL 3: GET PAGE
# ============================================================

@mcp.tool(
    description=(
        "Retrieve all indexed text chunks from a specific page of a "
        "research paper. Use this when you need to inspect the complete "
        "content of a particular page for deeper investigation."
    )
)
def get_page(
    paper_name: str,
    page_number: int
) -> str:

    if not paper_name.lower().endswith(".pdf"):
        paper_name += ".pdf"

    import chromadb

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_collection(
        name="research_papers"
    )

    records = collection.get(
        where={
            "$and": [
                {"source": paper_name},
                {"page": page_number}
            ]
        }
    )

    if not records["ids"]:
        return (
            f"No indexed content found for "
            f"{paper_name}, page {page_number}."
        )

    # Sort chunks by their IDs so page content appears
    # in the original chunk order.
    combined = []

    for document, metadata in zip(
        records["documents"],
        records["metadatas"]
    ):
        combined.append(
            {
                "document": document,
                "chunk": metadata.get("chunk", 0)
            }
        )

    combined.sort(
        key=lambda x: x["chunk"]
    )

    output = [
        f"Paper: {paper_name}",
        f"Page: {page_number}",
        "",
        "===== Page Content ====="
    ]

    for i, item in enumerate(combined, start=1):

        output.append(
            f"\n--- Chunk {i} ---\n"
            f"{item['document']}"
        )

    return "\n".join(output)


# ============================================================
# TOOL 4: SEARCH WITHIN ONE PAPER
# ============================================================

@mcp.tool(
    description=(
        "Search for relevant evidence within one specific research "
        "paper only. Use this when the user asks about a particular "
        "paper or when evidence must be restricted to one paper."
    )
)
def search_within_paper(
    paper_name: str,
    query: str
) -> str:

    results = retrieve_within_paper(
        paper_name=paper_name,
        query=query
    )

    if not results:
        return (
            f"No relevant evidence found in "
            f"{paper_name}."
        )

    output = []

    for i, result in enumerate(results, start=1):

        metadata = result["metadata"]

        output.append(
            f"""
===== Evidence {i} =====

Text:
{result["document"]}

Title:
{metadata.get("title", "Unknown")}

Author:
{metadata.get("author", "Unknown")}

Source:
{metadata.get("source", "Unknown")}

Page:
{metadata.get("page", "Unknown")}

Reranker Score:
{result["reranker_score"]:.4f}
"""
        )

    return "\n".join(output)


# ============================================================
# TOOL 5: COMPARE TWO PAPERS
# ============================================================

@mcp.tool(
    description=(
        "Retrieve separate relevant evidence from two research papers "
        "for a comparison question. Use this when the user asks to "
        "compare methods, concepts, findings, algorithms, approaches, "
        "or other information across two papers. This tool retrieves "
        "evidence from each paper; the final comparison should be "
        "reasoned from that evidence."
    )
)
def compare_papers_tool(
    paper1: str,
    paper2: str,
    query: str
) -> str:

    comparison = compare_papers(
        paper1=paper1,
        paper2=paper2,
        query=query,
        candidate_k=20,
        final_k=6
    )

    output = []

    # --------------------------------------------------------
    # Paper 1
    # --------------------------------------------------------

    output.append(
        f"\n========== PAPER 1: "
        f"{comparison['paper1']['name']} ==========\n"
    )

    paper1_results = comparison["paper1"]["results"]

    if not paper1_results:

        output.append(
            "No relevant evidence found."
        )

    else:

        for i, result in enumerate(
            paper1_results,
            start=1
        ):

            metadata = result["metadata"]

            output.append(
                f"""
--- Evidence {i} ---

Text:
{result["document"]}

Title:
{metadata.get("title", "Unknown")}

Author:
{metadata.get("author", "Unknown")}

Page:
{metadata.get("page", "Unknown")}

Reranker Score:
{result["reranker_score"]:.4f}
"""
            )

    # --------------------------------------------------------
    # Paper 2
    # --------------------------------------------------------

    output.append(
        f"\n========== PAPER 2: "
        f"{comparison['paper2']['name']} ==========\n"
    )

    paper2_results = comparison["paper2"]["results"]

    if not paper2_results:

        output.append(
            "No relevant evidence found."
        )

    else:

        for i, result in enumerate(
            paper2_results,
            start=1
        ):

            metadata = result["metadata"]

            output.append(
                f"""
--- Evidence {i} ---

Text:
{result["document"]}

Title:
{metadata.get("title", "Unknown")}

Author:
{metadata.get("author", "Unknown")}

Page:
{metadata.get("page", "Unknown")}

Reranker Score:
{result["reranker_score"]:.4f}
"""
            )

    return "\n".join(output)


# ============================================================
# START MCP SERVER
# ============================================================

if __name__ == "__main__":

    mcp.run()