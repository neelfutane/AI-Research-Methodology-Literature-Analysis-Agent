import os
import sys
import requests
import streamlit as st
import chromadb

from agent import ask_agent, MODEL, OLLAMA_URL
from vector_store import add_pdf_to_vector_db, is_pdf_already_indexed

# ============================================================
# Page Configuration
# ============================================================

st.set_page_state = getattr(st, "set_page_state", None)
st.set_page_config(
    page_title="AI Research Methodology & Literature Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

PAPERS_DIR = "papers"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "research_papers"


# ============================================================
# Helper Functions
# ============================================================

def check_ollama_status():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            has_qwen = any(MODEL in m for m in models)
            return True, has_qwen, models
    except Exception:
        pass
    return False, False, []


def get_indexed_papers_overview():
    """
    Returns list of metadata dicts for currently indexed papers in ChromaDB.
    """
    papers_meta = []
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        records = collection.get(include=["metadatas"])
        if not records or not records["metadatas"]:
            return papers_meta

        # Group by source
        grouped = {}
        for meta in records["metadatas"]:
            if not meta:
                continue
            src = meta.get("source", "Unknown")
            if src not in grouped:
                grouped[src] = {
                    "source": src,
                    "title": meta.get("title", "Unknown"),
                    "author": meta.get("author", "Unknown"),
                    "pages": set(),
                    "chunks": 0
                }
            grouped[src]["chunks"] += 1
            page = meta.get("page")
            if page is not None:
                grouped[src]["pages"].add(page)

        for src, item in sorted(grouped.items()):
            papers_meta.append({
                "source": item["source"],
                "title": item["title"],
                "author": item["author"],
                "pages": len(item["pages"]),
                "chunks": item["chunks"]
            })
    except Exception:
        pass
    return papers_meta


# ============================================================
# Sidebar: System Status & Research Paper Management
# ============================================================

with st.sidebar:
    st.header("⚙️ System Status")
    ollama_ok, qwen_ok, available_models = check_ollama_status()

    col_status1, col_status2 = st.columns(2)
    with col_status1:
        if ollama_ok:
            st.success("Ollama: Online")
        else:
            st.error("Ollama: Offline")

    with col_status2:
        if qwen_ok:
            st.success(f"Model: {MODEL}")
        else:
            st.warning(f"Model: Missing {MODEL}")

    if not ollama_ok:
        st.info("Start Ollama service: `ollama serve` (or run Ollama app)")

    st.markdown("---")
    st.header("📄 Paper Management")

    # Upload PDF files
    uploaded_files = st.file_uploader(
        "Upload Research Papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple research papers. Files will be saved to papers/ and indexed into ChromaDB."
    )

    if uploaded_files:
        os.makedirs(PAPERS_DIR, exist_ok=True)
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            # Validate extension
            if not file_name.lower().endswith(".pdf"):
                st.error(f"'{file_name}' is not a PDF file.")
                continue

            # Validate non-empty
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) == 0:
                st.error(f"'{file_name}' is empty (0 bytes).")
                continue

            target_path = os.path.join(PAPERS_DIR, file_name)

            # Check if file exists on disk
            already_on_disk = os.path.exists(target_path)
            already_indexed = is_pdf_already_indexed(file_name)

            if already_indexed:
                st.info(f"ℹ️ '{file_name}' is already indexed in ChromaDB. Skipped re-ingestion.")
                continue

            # Confirmation if file exists on disk but not in index
            if already_on_disk:
                overwrite_key = f"overwrite_{file_name}"
                confirm_overwrite = st.checkbox(
                    f"File '{file_name}' exists on disk. Overwrite and index?",
                    key=overwrite_key
                )
                if not confirm_overwrite:
                    st.warning(f"Skipped '{file_name}' (waiting for overwrite confirmation).")
                    continue

            # Save to papers/
            try:
                with open(target_path, "wb") as f:
                    f.write(file_bytes)
            except Exception as e:
                st.error(f"Failed to save '{file_name}': {e}")
                continue

            # Index via existing vector_store pipeline
            with st.spinner(f"Indexing '{file_name}' into ChromaDB..."):
                try:
                    success = add_pdf_to_vector_db(target_path)
                    if success:
                        st.success(f"✅ Successfully indexed '{file_name}'!")
                    else:
                        st.warning(f"⚠️ Indexing returned False for '{file_name}'.")
                except Exception as e:
                    st.error(f"Error indexing '{file_name}': {e}")

    # Display currently indexed papers
    st.markdown("---")
    st.subheader("📚 Indexed Papers")
    papers_overview = get_indexed_papers_overview()

    if papers_overview:
        for p in papers_overview:
            with st.expander(f"📖 {p['source']}", expanded=False):
                st.markdown(f"**Title:** {p['title']}")
                st.markdown(f"**Author:** {p['author']}")
                st.markdown(f"**Pages:** {p['pages']}")
                st.markdown(f"**Indexed Chunks:** {p['chunks']}")
    else:
        st.write("No papers indexed yet.")


# ============================================================
# Main Area: Research Question & Analysis Interface
# ============================================================

st.title("📚 AI Research Methodology & Literature Agent")
st.caption(
    "Ground questions in research papers with automated tool routing, local LLM reasoning, and verified page citations."
)

# Example questions dropdown to facilitate testing
st.markdown("##### Quick Question Templates")
templates = [
    "-- Select an example query --",
    "What is supervised machine learning?",
    "Summarize test_paper.pdf, covering its main subject, key algorithms, applications, and conclusions.",
    "Show me page 3 of test_paper.pdf",
    "Compare CART and Naive Bayes in test_paper.pdf",
    "Compare the research goals, methodologies, and findings between test_paper.pdf and paper2.pdf",
    "What does test_paper.pdf say about quantum computing algorithms?"
]
selected_template = st.selectbox("Choose a sample query to populate the input:", templates, label_visibility="collapsed")

# Initialize session state for input
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

if selected_template != "-- Select an example query --":
    st.session_state.query_input = selected_template

question_text = st.text_area(
    "Enter your research question:",
    value=st.session_state.query_input,
    height=110,
    placeholder="e.g. Compare the methodologies of test_paper.pdf and paper2.pdf, or ask about a specific algorithm/page..."
)

col_btn, _ = st.columns([1, 5])
with col_btn:
    submit_clicked = st.button("Submit Question", type="primary", use_container_width=True)

if submit_clicked:
    if not question_text.strip():
        st.warning("Please enter a research question.")
    elif not ollama_ok or not qwen_ok:
        st.error(f"Cannot process question: Ollama service or '{MODEL}' model is unavailable. Please verify Ollama is running.")
    else:
        with st.spinner("Analyzing research papers with AI Agent..."):
            try:
                result = ask_agent(question_text.strip())

                # Display Results
                st.markdown("### 📝 Answer")
                st.write(result["answer"])

                st.markdown("---")
                # Source evidence & execution trace
                with st.expander("🔍 Inspection & Verification Details (Routing, Tool & Citations)", expanded=False):
                    col_meta1, col_meta2 = st.columns(2)
                    with col_meta1:
                        st.markdown(f"**Selected MCP Tool:** `{result['tool_name']}`")
                        st.markdown(f"**Arguments:** `{result['arguments']}`")
                    with col_meta2:
                        st.markdown(f"**LLM Model:** `{MODEL}`")
                        st.markdown(f"**Qwen Tool Decision:** `{result['qwen_decision'].get('tool', 'N/A')}`")

                    st.markdown("#### Retrieved Evidence & Citations")
                    if result["evidence"].strip():
                        st.text_area("Tool Output / Retrieved Evidence", value=result["evidence"], height=250, disabled=True)
                    else:
                        st.info("No evidence retrieved.")

            except Exception as e:
                st.error(f"An error occurred while analyzing the papers. Please check system logs. ({type(e).__name__})")
