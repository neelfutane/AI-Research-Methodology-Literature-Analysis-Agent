import asyncio

from mcp import (
    ClientSession,
    StdioServerParameters
)

from mcp.client.stdio import stdio_client


# ============================================================
# MCP Server Configuration
# ============================================================

server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"],
)


# ============================================================
# Main
# ============================================================

async def main():

    async with stdio_client(
        server_params
    ) as (read, write):

        async with ClientSession(
            read,
            write
        ) as session:

            # ------------------------------------------------
            # Initialize MCP session
            # ------------------------------------------------

            await session.initialize()


            # ------------------------------------------------
            # List available tools
            # ------------------------------------------------

            tools = await session.list_tools()

            print(
                "\n===== Available MCP Tools ====="
            )

            for tool in tools.tools:

                print(
                    f"\nTool: {tool.name}"
                )

                print(
                    f"Description: "
                    f"{tool.description}"
                )


            # =================================================
            # TOOL 1 - Search Papers
            # =================================================

            query = input(
                "\nEnter your research question: "
            )

            result = await session.call_tool(
                "search_papers",
                {
                    "query": query
                }
            )

            print(
                "\n===== Search Papers Result ====="
            )

            for content in result.content:

                if hasattr(content, "text"):

                    print(
                        content.text
                    )


            # =================================================
            # TOOL 2 - Paper Metadata
            # =================================================

            paper_name = input(
                "\nEnter PDF name for metadata "
                "(example: test_paper.pdf): "
            )

            metadata_result = await session.call_tool(
                "get_paper_metadata",
                {
                    "paper_name": paper_name
                }
            )

            print(
                "\n===== Paper Metadata Result ====="
            )

            for content in metadata_result.content:

                if hasattr(content, "text"):

                    print(
                        content.text
                    )


            # =================================================
            # TOOL 3 - Get Page
            # =================================================

            page_number = int(
                input(
                    "\nEnter page number to retrieve: "
                )
            )

            page_result = await session.call_tool(
                "get_page",
                {
                    "paper_name": paper_name,
                    "page_number": page_number
                }
            )

            print(
                "\n===== Get Page Result ====="
            )

            for content in page_result.content:

                if hasattr(content, "text"):

                    print(
                        content.text
                    )


            # =================================================
            # TOOL 4 - Search Within Paper
            # =================================================

            paper_query = input(
                "\nEnter a question to search "
                "within this paper: "
            )

            paper_search_result = await session.call_tool(
                "search_within_paper",
                {
                    "paper_name": paper_name,
                    "query": paper_query
                }
            )

            print(
                "\n===== Search Within Paper Result ====="
            )

            for content in paper_search_result.content:

                if hasattr(content, "text"):

                    print(
                        content.text
                    )


            # =================================================
            # TOOL 5 - Compare Papers
            # =================================================

            print(
                "\n===== Compare Two Papers ====="
            )

            paper1 = input(
                "\nEnter first PDF name "
                "(example: paper2.pdf): "
            )

            paper2 = input(
                "Enter second PDF name "
                "(example: test_paper.pdf): "
            )

            comparison_query = input(
                "Enter comparison question: "
            )

            comparison_result = await session.call_tool(
                "compare_papers_tool",
                {
                    "paper1": paper1,
                    "paper2": paper2,
                    "query": comparison_query
                }
            )

            print(
                "\n===== Compare Papers Result ====="
            )

            for content in comparison_result.content:

                if hasattr(content, "text"):

                    print(
                        content.text
                    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())