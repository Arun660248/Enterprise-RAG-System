from typing import final

import  streamlit as st
import  requests
API_URL = "http://127.0.0.1:8000/ask"
st.set_page_config(page_title="Enterprise Ai",page_icon="🤖")
st.title("Enterprise RAG Assistant")
if "messages" not in st.session_state:
    st.session_state.messages=[]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Ask a question about your documents..."):
    # When the user hits enter, save their message to the memory vault
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching enterprise knowledge base..."):
            dic={"question":prompt}
            response=requests.post(API_URL,json=dic)
            response_data=response.json()
            ai_answer=response_data["answer"]
            sources_list=response_data["sources"]
            citation_text="\n\n**Sources:**\n"
            for source in sources_list:
                file = source["file_name"]
                page = source["page_number"]
                citation_text += f"- {file} (Page {page})\n"
            final_output=ai_answer+citation_text
            st.markdown(final_output)
            st.session_state.messages.append({"role": "assistant", "content": final_output})