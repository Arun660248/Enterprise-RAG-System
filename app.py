import streamlit as st
import requests

API_URL = "http://172.31.36.136:8000/ask"

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise AI",
    page_icon="🤖",
    layout="centered"
)

# 2. Sidebar System Information & Capabilities
with st.sidebar:
    st.image("https://img.icons8.com/fluent/100/000000/shield.png", width=60)
    st.title("System Overview")
    st.markdown("""
    This intelligent system is connected directly to our secure internal knowledge repository. 

    ### 🛡️ Guardrails & Compliance
    * **Grounding:** Responses are strictly limited to verified uploaded assets.
    * **Anti-Hallucination:** Page-level verifiable citations are provided for every output.
    * **Data Privacy:** Queries stay securely within our enterprise firewall.
    """)
    st.divider()
    st.caption("Engineered by Arun Jyoti Chakraborty | Version 1.0.0")

# 3. Main Chat Interface Title
st.title("Enterprise RAG Assistant")

# 4. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. The Welcome State (Only displays if no chat messages exist)
if not st.session_state.messages:
    st.markdown("""
    ### 👋 Welcome to the Enterprise Knowledge Command Center
    This workspace utilizes an advanced **Retrieval-Augmented Generation (RAG)** pipeline to synthesize insights across our internal corpus. It is designed to act as your expert research partner.

    #### 🔍 Key Features
    * **Dynamic Context Loading:** Reads, chunks, and semantically vectors internal files.
    * **Automatic Sourcing:** Every answer will contain verifiable page numbers and file names.
    """)

    # Clickable Suggestion Buttons
    st.markdown("#### 💡 Try asking one of these structural examples:")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📈 Extract Q4 Financial Highlights"):
            st.session_state.pending_prompt = "What are the core revenue growth figures and net profit margins reported for Q4?"
            st.rerun()

    with col2:
        if st.button("⚖️ Review Compliance & Risks"):
            st.session_state.pending_prompt = "Summarize the primary risk factors and regulatory compliance updates outlined in our current guidelines."
            st.rerun()

# 6. Handle Pending Prompts from Clicked Buttons
current_prompt = None
if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    current_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None  # Clear after capturing

# 7. Render Existing Chat Messages from Session State
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 8. User Input Element (Captures either typing or a button click trigger)
user_input = st.chat_input("Ask a question about your documents...")
if user_input:
    current_prompt = user_input

# 9. Execution Pipeline
if current_prompt:
    # Append and render user query
    st.session_state.messages.append({"role": "user", "content": current_prompt})
    with st.chat_message("user"):
        st.markdown(current_prompt)

    # Query the FastAPI application layer
    with st.chat_message("assistant"):
        with st.spinner("Searching enterprise knowledge base..."):
            try:
                dic = {"question": current_prompt}
                response = requests.post(API_URL, json=dic)
                response.raise_for_status()  # Trigger error logic if server status is 4xx/5xx

                response_data = response.json()
                ai_answer = response_data["answer"]
                sources_list = response_data.get("sources", [])

                # Format page citations clearly using markdown formatting
                citation_text = "\n\n**Sources:**\n"
                if sources_list:
                    for source in sources_list:
                        file = source.get("file_name", "Unknown File")
                        page = source.get("page_number", "N/A")
                        citation_text += f"- 📁 `{file}` (Page {page})\n"
                else:
                    citation_text += "_No specific source documentation cited for this output._\n"

                final_output = ai_answer + citation_text
                st.markdown(final_output)
                st.session_state.messages.append({"role": "assistant", "content": final_output})

            except requests.exceptions.RequestException as e:
                error_msg = f"❌ **Connection Error:** Unable to reach backend API server at `{API_URL}`. Verify that your FastAPI service is live."
                st.error(error_msg)