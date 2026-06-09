import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
os.environ["GOOGLE_GENAI_API_VERSION"] = "v1"


# --- Models & Pydantic Definitions ---
class QueryRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    file_name: str
    page_number: int

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]

app = FastAPI()
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"))
prompt = ChatPromptTemplate.from_template(
    "Use the following pieces of context to answer the question.Output requirements(e.g.,conversational,medium length,bullet points(if needed)) "
    "If the answer is not in the context, state 'This information is not in the provided document, but based on my general knowledge...' and then answer the question."
    " \n\nContext: {context} \n\nQuestion: {input}")

document_chain = create_stuff_documents_chain(llm, prompt)
try:
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    rag_chain = create_retrieval_chain(retriever, document_chain)
    print("Existing FAISS index loaded successfully.")
except Exception as e:
    print("No FAISS index found. Awaiting document upload.")
    vector_store, retriever, rag_chain = None, None, None

@app.get("/")
def health_check():
    return {"status": "Active", "message": "Enterprise RAG API is running."}
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Receives a PDF, processes it, and updates the AI's active memory."""
    global vector_store, retriever, rag_chain
    file_path = f"./{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
    chunks = text_splitter.split_documents(pages)

    # 3. Create fresh vector database and save it
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")

    # 4. Rebuild the active chains
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    rag_chain = create_retrieval_chain(retriever, document_chain)

    # 5. Clean up the temporary PDF file to save space
    os.remove(file_path)

    return {"status": "Success", "message": f"Document '{file.filename}' processed and active."}
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not rag_chain:
        return QueryResponse(answer="System has no documents loaded. Please upload a PDF first.", sources=[])

    response = rag_chain.invoke({"input": request.question})
    extracted_sources = []

    for i in response["context"]:
        extracted_sources.append(SourceItem(
            file_name=i.metadata.get("source", "Unknown"),
            page_number=i.metadata.get("page", 0)
        ))
    return QueryResponse(answer=response["answer"], sources=extracted_sources)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)