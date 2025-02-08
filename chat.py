from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set. Please check your .env file.")

url = "https://brainlox.com/courses/category/technical"

# Load data using LangChain URL loader
loader = WebBaseLoader(url)
docs = loader.load()

# Convert to text
text = " ".join([doc.page_content for doc in docs])

# Split text into chunks for embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(text)

# Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)

# Store in FAISS vector store
vector_store = FAISS.from_texts(chunks, embeddings)

# Save vector store locally
vector_store.save_local("faiss_index")

print("✅ Data extracted, embeddings created, and stored in FAISS!")
