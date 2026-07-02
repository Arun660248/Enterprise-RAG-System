import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
file_path = "sample.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
chunks = text_splitter.split_documents(pages)
print(len(chunks))

# Force v1 API via environment variable
os.environ["GOOGLE_GENAI_API_VERSION"] = "v1"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vector_store = FAISS.from_documents(chunks, embeddings)
save_path = "../backend/faiss_index" if os.path.exists("../backend") else "faiss_index"
vector_store.save_local(save_path)
print(f"Done! FAISS index saved to {save_path}.")