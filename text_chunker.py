from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_processor import extract_text_from_pdf


def split_text_into_chunks(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    all_chunks = []

    # Process every page separately
    for page in pages:

        page_text = page["text"]
        page_metadata = page["metadata"]

        # Skip empty pages
        if not page_text.strip():
            continue

        # Split page text
        page_chunks = splitter.split_text(
            page_text
        )

        # Attach page metadata to every chunk
        for chunk in page_chunks:

            chunk_metadata = {
                "title": page_metadata["title"],
                "author": page_metadata["author"],
                "source": page_metadata["source"],
                "page": page_metadata["page"]
            }

            all_chunks.append({
                "text": chunk,
                "metadata": chunk_metadata
            })

    return all_chunks


# -----------------------------
# Test
# -----------------------------

if __name__ == "__main__":

    pdf_path = "papers/paper2.pdf"

    pages = extract_text_from_pdf(
        pdf_path
    )

    chunks = split_text_into_chunks(
        pages
    )

    print(
        f"Total pages: {len(pages)}"
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    for i, chunk in enumerate(
        chunks[:5]
    ):

        print(
            f"\n--- Chunk {i + 1} ---"
        )

        print(
            "\nText:"
        )

        print(
            chunk["text"]
        )

        print(
            "\nMetadata:"
        )

        print(
            chunk["metadata"]
        )