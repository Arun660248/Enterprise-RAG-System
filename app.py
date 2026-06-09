import streamlit as st
import requests

TESTING_LOCALLY = True
if TESTING_LOCALLY:
    BASE_API_URL = "http://127.0.0.1:8000"
else:
    BASE_API_URL = "http://172.31.36.136:8000"

ASK_ENDPOINT = f"{BASE_API_URL}/ask"
UPLOAD_ENDPOINT = f"{BASE_API_URL}/upload"

st.set_page_config(
    page_title="Intelligent Document Assistant",
    page_icon="🤖",
    layout="centered"
)

# --- Sidebar Configuration & File Ingestion ---
with st.sidebar:
    st.title("🗂️ Document Control")
    st.caption("Enterprise RAG System v1.1")
    st.divider()

    # Drag-and-drop file uploader
    uploaded_file = st.file_uploader(
        "Ingest a new target PDF:",
        type=["pdf"],
        help="Uploading a new file will automatically hot-swap the vector memory store."
    )

    if uploaded_file is not None:
        # Prevent continuous loop by running upload only if state changes
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            with st.spinner("Chunking and vectorizing document..."):
                try:
                    # Prepare the file binary stream for the multipart/form-data request
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    upload_response = requests.post(UPLOAD_ENDPOINT, files=files)
                    upload_response.raise_for_status()

                    st.success(f"Successfully loaded: {uploaded_file.name}")
                    st.session_state.last_uploaded = uploaded_file.name
                except requests.exceptions.RequestException:
                    st.error("Failed to connect to the ingestion API. Check backend logs.")

    st.divider()

    # System disclaimer and operational notes
    st.markdown("""
    ### 🛡️ System Guidelines
    * **Scope:** Answers are grounded in the active document's context.
    * **Verification:** Always verify page numbers provided in citations.
    * **Notice:** AI systems can occasionally misinterpret text or generalize incorrectly. Please double-check critical compliance facts.
    """)
    st.divider()
    st.caption("Build by Arun Jyoti Chakraborty")

# --- Interface Main Framework ---
st.title("Enterprise RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome landing state (empty chat history context)
if not st.session_state.messages:
    st.markdown("""
    ### Document Intelligence Console
    Upload a document in the left panel to begin analysis. The system will slice the data into semantic chunks and build local vector relationships for pinpoint retrieval.
    """)

    st.markdown("#### Quick Analysis Triggers:")
    grid_col_1, grid_col_2 = st.columns(2)

    with grid_col_1:
        if st.button("📊 Generate Document Summary"):
            st.session_state.queued_query = "Provide a comprehensive high-level summary of this document, highlighting its core themes."
            st.rerun()

    with grid_col_2:
        if st.button("📌 Extract Core Takeaways"):
            st.session_state.queued_query = "List the top 5 most critical takeaways or actionable points found within this text."
            st.rerun()

# Handle pipeline queues from selection buttons
active_query = None
if "queued_query" in st.session_state and st.session_state.queued_query:
    active_query = st.session_state.queued_query
    st.session_state.queued_query = None

# Render historic logs
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capture live text field inputs
user_text_input = st.chat_input("Query the ingested assets...")
if user_text_input:
    active_query = user_text_input

# --- Live Execution Runtime ---
if active_query:
    # Render user side immediately
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # Process backend communication loop
    with st.chat_message("assistant"):
        with st.spinner("Scanning semantic vectors..."):
            try:
                payload = {"question": active_query}
                backend_reply = requests.post(ASK_ENDPOINT, json=payload)
                backend_reply.raise_for_status()

                parsed_json = backend_reply.json()
                text_response = parsed_json["answer"]
                citations = parsed_json.get("sources", [])

                # Build clean citation tracking markdown block
                formatted_citations = "\n\n**Verified Citations:**\n"
                if citations:
                    # Filter out duplicate source page tracking
                    seen_citations = set()
                    for item in citations:
                        doc_identity = item.get("file_name", "Context Asset")
                        page_id = item.get("page_number", "N/A")

                        unique_key = f"{doc_identity}-{page_id}"
                        if unique_key not in seen_citations:
                            seen_citations.add(unique_key)
                            formatted_citations += f"- 📄 `{doc_identity}` — Page {page_id}\n"
                else:
                    formatted_citations += "_Synthesized from broad contextual data._\n"

                complete_payload = text_response + formatted_citations
                st.markdown(complete_payload)
                st.session_state.messages.append({"role": "assistant", "content": complete_payload})

            except requests.exceptions.RequestException:
                st.error(
                    f"Endpoint Timeout: Unable to query port router. Ensure service at `{ASK_ENDPOINT}` is active.")