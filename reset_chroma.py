import chromadb

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

try:
    chroma_client.delete_collection(
        name="research_papers"
    )

    print("research_papers collection deleted successfully.")

except Exception as e:

    print("Could not delete collection:")
    print(e)