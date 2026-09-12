import chromadb
from collections import Counter, defaultdict


# ============================================================
# Configuration
# ============================================================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "research_papers"


# ============================================================
# Connect to ChromaDB
# ============================================================

print("Connecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


# ============================================================
# Basic statistics
# ============================================================

print("\n========================================")
print("CHROMADB STATISTICS")
print("========================================")

total_chunks = collection.count()

print(f"Total chunks: {total_chunks}")


# ============================================================
# Get all records
# ============================================================

result = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)

documents = result["documents"]
metadatas = result["metadatas"]
ids = result["ids"]


print(f"Records retrieved: {len(documents)}")


# ============================================================
# 1. Chunks per paper
# ============================================================

print("\n========================================")
print("CHUNKS PER PAPER")
print("========================================")

paper_counts = Counter()

for metadata in metadatas:

    source = metadata.get(
        "source",
        "Unknown"
    )

    paper_counts[source] += 1


for paper, count in paper_counts.items():

    print(
        f"{paper}: {count} chunks"
    )


# ============================================================
# 2. Page distribution
# ============================================================

print("\n========================================")
print("PAGE DISTRIBUTION")
print("========================================")

page_counts = defaultdict(Counter)

for metadata in metadatas:

    source = metadata.get(
        "source",
        "Unknown"
    )

    page = str(
        metadata.get(
            "page",
            "Unknown"
        )
    )

    page_counts[source][page] += 1


for source in sorted(page_counts):

    print(f"\n--- {source} ---")

    for page, count in sorted(
        page_counts[source].items(),
        key=lambda x: int(x[0])
        if x[0].isdigit()
        else 999999
    ):

        print(
            f"Page {page}: {count} chunks"
        )


# ============================================================
# 3. Check duplicate document text
# ============================================================

print("\n========================================")
print("DUPLICATE DOCUMENT CHECK")
print("========================================")

document_groups = defaultdict(list)

for index, document in enumerate(documents):

    if document is None:
        continue

    document_groups[document].append(index)


duplicate_groups = {
    document: indexes
    for document, indexes in document_groups.items()
    if len(indexes) > 1
}


print(
    f"Unique document texts: "
    f"{len(document_groups)}"
)

print(
    f"Duplicate document groups: "
    f"{len(duplicate_groups)}"
)


extra_duplicate_records = sum(
    len(indexes) - 1
    for indexes in duplicate_groups.values()
)


print(
    f"Extra duplicate records: "
    f"{extra_duplicate_records}"
)


# ============================================================
# 4. Show duplicate examples
# ============================================================

if duplicate_groups:

    print("\n========================================")
    print("DUPLICATE EXAMPLES")
    print("========================================")

    shown = 0

    for document, indexes in duplicate_groups.items():

        print(
            f"\n--- Duplicate Group {shown + 1} ---"
        )

        print(
            f"Occurrences: {len(indexes)}"
        )

        print(
            "\nText:"
        )

        print(
            document[:500]
        )

        print(
            "\nStored records:"
        )

        for index in indexes:

            print(
                f"\nID: {ids[index]}"
            )

            print(
                f"Metadata: {metadatas[index]}"
            )

        shown += 1

        # Don't flood the terminal.
        # Show maximum 10 duplicate groups.
        if shown >= 10:
            break

else:

    print(
        "\nNo exact duplicate document texts found."
    )


# ============================================================
# 5. Check duplicate IDs
# ============================================================

print("\n========================================")
print("DUPLICATE ID CHECK")
print("========================================")

id_counts = Counter(ids)

duplicate_ids = {
    record_id: count
    for record_id, count in id_counts.items()
    if count > 1
}


print(
    f"Unique IDs: {len(id_counts)}"
)

print(
    f"Duplicate IDs: {len(duplicate_ids)}"
)


if duplicate_ids:

    print(
        "\nDuplicate IDs found:"
    )

    for record_id, count in list(
        duplicate_ids.items()
    )[:20]:

        print(
            f"{record_id}: {count} occurrences"
        )

else:

    print(
        "No duplicate IDs found."
    )


# ============================================================
# 6. Specifically inspect test_paper.pdf page 3
# ============================================================

print("\n========================================")
print("test_paper.pdf - PAGE 3")
print("========================================")

page3_records = []

for index, metadata in enumerate(metadatas):

    source = metadata.get(
        "source",
        ""
    )

    page = str(
        metadata.get(
            "page",
            ""
        )
    )

    if (
        source == "test_paper.pdf"
        and page == "3"
    ):

        page3_records.append(index)


print(
    f"Chunks on page 3: "
    f"{len(page3_records)}"
)


for number, index in enumerate(
    page3_records,
    start=1
):

    print(
        f"\n--- Chunk {number} ---"
    )

    print(
        f"ID: {ids[index]}"
    )

    print(
        f"Metadata: {metadatas[index]}"
    )

    print(
        "\nText:"
    )

    print(
        documents[index][:700]
    )


# ============================================================
# 7. Specifically inspect paper2.pdf
# ============================================================

print("\n========================================")
print("paper2.pdf CHECK")
print("========================================")

paper2_records = []

for index, metadata in enumerate(metadatas):

    source = metadata.get(
        "source",
        ""
    )

    if source == "paper2.pdf":

        paper2_records.append(index)


print(
    f"paper2.pdf chunks: "
    f"{len(paper2_records)}"
)


if paper2_records:

    paper2_pages = Counter()

    for index in paper2_records:

        page = str(
            metadatas[index].get(
                "page",
                "Unknown"
            )
        )

        paper2_pages[page] += 1


    print(
        "\npaper2.pdf page distribution:"
    )

    for page, count in sorted(
        paper2_pages.items(),
        key=lambda x: int(x[0])
        if x[0].isdigit()
        else 999999
    ):

        print(
            f"Page {page}: {count} chunks"
        )


# ============================================================
# 8. Show first few records
# ============================================================

print("\n========================================")
print("FIRST 3 RECORDS")
print("========================================")


for i in range(
    min(3, len(documents))
):

    print(
        "\n=============================="
    )

    print(
        f"ID: {ids[i]}"
    )

    print(
        "\nDOCUMENT:"
    )

    print(
        documents[i][:300]
    )

    print(
        "\nMETADATA:"
    )

    print(
        metadatas[i]
    )


# ============================================================
# Done
# ============================================================

print("\n========================================")
print("DIAGNOSTIC COMPLETE")
print("========================================")

print(
    "\nDo NOT delete or modify chroma_db yet."
)

print(
    "Send me the complete output."
)