import chromadb
from vector_store import add_pdf_to_vector_db


# -----------------------------
# Connect to ChromaDB
# -----------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="research_papers"
)


# -----------------------------
# Remove old paper2.pdf chunks
# -----------------------------

print("\nRemoving old paper2.pdf data...")

results = collection.get(
    where={
        "source": "paper2.pdf"
    }
)

old_ids = results["ids"]

print(
    f"Found {len(old_ids)} existing chunks."
)


if old_ids:

    collection.delete(
        ids=old_ids
    )

    print(
        "Old paper2.pdf data removed."
    )

else:

    print(
        "No existing paper2.pdf data found."
    )


# -----------------------------
# Re-index paper2.pdf
# -----------------------------

print(
    "\nRe-indexing paper2.pdf..."
)

add_pdf_to_vector_db(
    "papers/paper2.pdf"
)


# -----------------------------
# Final count
# -----------------------------

print(
    f"\nTotal documents in ChromaDB: "
    f"{collection.count()}"
)