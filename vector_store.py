import os

import chromadb
from sentence_transformers import SentenceTransformer

from pdf_processor import extract_text_from_pdf
from text_chunker import split_text_into_chunks


# ============================================================
# Configuration
# ============================================================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "research_papers"
PAPERS_FOLDER = "./papers"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# Load embedding model
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# Create ChromaDB client
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# ============================================================
# Get or create collection
# ============================================================

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# Check whether PDF is already indexed
# ============================================================

def is_pdf_already_indexed(pdf_name):

    results = collection.get(
        where={
            "source": pdf_name
        },
        limit=1
    )

    return len(results["ids"]) > 0


# ============================================================
# Add one PDF to ChromaDB
# ============================================================

def add_pdf_to_vector_db(pdf_path):

    file_name = os.path.basename(
        pdf_path
    )

    print(
        f"\n========================================"
    )

    print(
        f"Processing: {file_name}"
    )

    print(
        f"========================================"
    )


    # --------------------------------------------------------
    # Check whether PDF already exists
    # --------------------------------------------------------

    if is_pdf_already_indexed(file_name):

        print(
            f"Skipping: {file_name}"
        )

        print(
            "Reason: PDF is already indexed."
        )

        return False


    # --------------------------------------------------------
    # Extract text from PDF
    # --------------------------------------------------------

    pages = extract_text_from_pdf(
        pdf_path
    )

    print(
        f"Pages extracted: {len(pages)}"
    )


    if not pages:

        print(
            "No text found in PDF."
        )

        return False


    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = split_text_into_chunks(
        pages
    )

    print(
        f"Chunks created: {len(chunks)}"
    )


    if not chunks:

        print(
            "No chunks were created."
        )

        return False


    # --------------------------------------------------------
    # Extract chunk text
    # --------------------------------------------------------

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]


    # --------------------------------------------------------
    # Extract metadata
    # --------------------------------------------------------

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]


    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print(
        "Creating embeddings..."
    )

    embeddings = embedding_model.encode(
        chunk_texts,
        show_progress_bar=True
    ).tolist()


    # --------------------------------------------------------
    # Create deterministic IDs
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # We DO NOT use uuid.uuid4().
    #
    # The same PDF + same chunk index
    # will always receive the same ID.
    #
    # Example:
    #
    # test_paper.pdf_0
    # test_paper.pdf_1
    # test_paper.pdf_2
    #
    # This prevents duplicate insertion.
    # --------------------------------------------------------

    ids = []

    for index, chunk in enumerate(chunks):

        page = chunk["metadata"].get(
            "page",
            "unknown"
        )

        chunk_id = (
            f"{file_name}_"
            f"page_{page}_"
            f"chunk_{index}"
        )

        ids.append(
            chunk_id
        )


    # --------------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------------

    print(
        "Storing chunks in ChromaDB..."
    )

    collection.add(
        ids=ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=metadatas
    )


    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print(
        f"PDF successfully indexed: "
        f"{file_name}"
    )

    print(
        f"Chunks added: {len(chunks)}"
    )

    return True


# ============================================================
# Add all PDFs
# ============================================================

def add_all_pdfs():

    if not os.path.exists(
        PAPERS_FOLDER
    ):

        print(
            f"Folder not found: "
            f"{PAPERS_FOLDER}"
        )

        return


    pdf_files = [
        file
        for file in os.listdir(
            PAPERS_FOLDER
        )
        if file.lower().endswith(".pdf")
    ]


    if not pdf_files:

        print(
            "No PDF files found in papers folder."
        )

        return


    print(
        f"\nFound {len(pdf_files)} PDF file(s)."
    )


    new_pdfs = 0


    for pdf_file in pdf_files:

        pdf_path = os.path.join(
            PAPERS_FOLDER,
            pdf_file
        )

        added = add_pdf_to_vector_db(
            pdf_path
        )

        if added:

            new_pdfs += 1


    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "INGESTION COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"New PDFs indexed: {new_pdfs}"
    )

    print(
        f"Total chunks in ChromaDB: "
        f"{collection.count()}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    add_all_pdfs()