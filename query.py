import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
os.environ["GOOGLE_GENAI_API_VERSION"] = "v1"
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"))
vector_store=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True )
retriever=vector_store.as_retriever(search_kwargs={"k": 3})
llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"))
prompt=ChatPromptTemplate.from_template( "You are an assistant for answering questions. "
 "Use the following pieces of retrieved context to answer the question."
" If the answer is not in the context, say 'I do not know'."
 " \n\nContext: {context} \n\nQuestion: {input}")
document_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,document_chain)
question = "What is Arun's major and CPI?"
response = rag_chain.invoke({"input": question})

print("\n--- AI ANSWER ---")
print(response["answer"])
print("\n--- SOURCES ---")
for i in response["context"]:
    print(i.metadata.get("source"))
    print(i.metadata.get("page"))