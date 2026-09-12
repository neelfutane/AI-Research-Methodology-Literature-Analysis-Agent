import ollama

# Import the improved retrieval function
from retriever import retrieve_relevant_chunks


# ============================================================
# ASK QWEN
# ============================================================

def ask_qwen(question, retrieved_chunks):

    # --------------------------------------------------------
    # Build research context
    # --------------------------------------------------------

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks):

        # ----------------------------------------------------
        # Get text safely
        # ----------------------------------------------------

        text = (
            chunk.get("text")
            or chunk.get("document")
            or chunk.get("content")
            or ""
        )

        # ----------------------------------------------------
        # Get metadata safely
        # ----------------------------------------------------

        metadata = chunk.get(
            "metadata",
            {}
        )

        title = metadata.get(
            "title",
            "Unknown"
        )

        author = metadata.get(
            "author",
            "Unknown"
        )

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        reranker_score = chunk.get(
            "reranker_score",
            0
        )


        # ----------------------------------------------------
        # Build evidence block
        # ----------------------------------------------------

        context_parts.append(
            f"""
EVIDENCE {i + 1}

Title: {title}
Author: {author}
Source: {source}
Page: {page}
Relevance Score: {reranker_score:.4f}

Text:
{text}
"""
        )


    context_text = "\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # Prompt Qwen
    # --------------------------------------------------------

    prompt = f"""
You are an AI Research Paper Assistant.

Your task is to answer the user's research question using
ONLY the evidence retrieved from the uploaded research papers.

IMPORTANT RULES:

1. Use ONLY the provided research evidence.

2. Do NOT use outside knowledge.

3. Do NOT invent facts, explanations, authors, papers,
   or page numbers.

4. If the answer cannot be determined from the evidence,
   respond exactly with:

"Information not found in the provided papers."

5. Give a clear and concise research-oriented answer.

6. Combine information from multiple evidence chunks
   when necessary.

7. Do not mention relevance scores.

8. When making an important claim, mention the paper
   page that supports it.

9. Do not copy large portions of the research paper.
   Summarize the evidence in your own words.

10. If multiple papers are relevant, distinguish between
    their findings.

------------------------------------------------------------

RESEARCH EVIDENCE
------------------------------------------------------------

{context_text}

------------------------------------------------------------

USER QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------

ANSWER
------------------------------------------------------------
"""


    # --------------------------------------------------------
    # Send request to Qwen
    # --------------------------------------------------------

    response = ollama.chat(

        model="qwen3.5:2b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        think=False,

        options={
            "num_predict": 400
        }
    )


    return response["message"]["content"]


# ============================================================
# MAIN RAG PIPELINE
# ============================================================

if __name__ == "__main__":

    print("\n==========================================")
    print("       AI RESEARCH PAPER ASSISTANT")
    print("==========================================")


    # --------------------------------------------------------
    # Get user question
    # --------------------------------------------------------

    question = input(
        "\nEnter your research question: "
    )


    # --------------------------------------------------------
    # Retrieve improved evidence
    # --------------------------------------------------------

    print(
        "\nRetrieving relevant evidence..."
    )


    retrieved_chunks = retrieve_relevant_chunks(
        question
    )


    # --------------------------------------------------------
    # Check retrieval result
    # --------------------------------------------------------

    if not retrieved_chunks:

        print(
            "\nNo sufficiently relevant evidence was found."
        )

        exit()


    print(
        f"\nRetrieved {len(retrieved_chunks)} evidence chunks."
    )


    # --------------------------------------------------------
    # Generate answer using Qwen
    # --------------------------------------------------------

    print(
        "\nGenerating answer using Qwen3.5:2B..."
    )


    answer = ask_qwen(
        question,
        retrieved_chunks
    )


    # --------------------------------------------------------
    # Display answer
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "             AI RESEARCH ANSWER"
    )

    print(
        "=========================================="
    )

    print(
        f"\n{answer}"
    )


    # --------------------------------------------------------
    # Display sources
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "                  SOURCES"
    )

    print(
        "=========================================="
    )


    displayed_sources = set()


    for chunk in retrieved_chunks:

        metadata = chunk.get(
            "metadata",
            {}
        )

        title = metadata.get(
            "title",
            "Unknown"
        )

        author = metadata.get(
            "author",
            "Unknown"
        )

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )


        # ----------------------------------------------------
        # Prevent duplicate paper + page combinations
        # ----------------------------------------------------

        source_key = (
            source,
            page
        )


        if source_key in displayed_sources:

            continue


        displayed_sources.add(
            source_key
        )


        print(
            f"\n- {title}"
        )

        print(
            f"  Author: {author}"
        )

        print(
            f"  Source: {source}"
        )

        print(
            f"  Page: {page}"
        )

