import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
class QueryRequest(BaseModel):
    question:str
class SourceItem(BaseModel):
    file_name:str
    page_number:int
class QueryResponse(BaseModel):
    answer:str
    sources:list[SourceItem]
app=FastAPI()
os.environ["GOOGLE_GENAI_API_VERSION"] = "v1"
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"))
vector_store=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True )
retriever=vector_store.as_retriever(search_kwargs={"k": 3})
llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"))
prompt=ChatPromptTemplate.from_template( "Use the following pieces of context to answer the question. "
"If the answer is not in the context, state 'This information is not in the provided document, but based on my general knowledge...' and then answer the question."
 " \n\nContext: {context} \n\nQuestion: {input}")
document_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,document_chain)
@app.get("/")
def health_check():
    return {"status": "Active", "message": "Enterprise RAG API is running."}
@app.post("/ask",response_model=QueryResponse)
def ask_question(request: QueryRequest):
    response = rag_chain.invoke({"input": request.question})
    extracted_sources = []

    for i in response["context"]:
        extracted_sources.append(SourceItem(file_name=i.metadata.get("source"),page_number=i.metadata.get("page")))
    return QueryResponse(answer=response["answer"], sources=extracted_sources)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)