from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings

data = PyPDFLoader("documentloaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 3000,chunk_overlap = 200)
chunks = splitter.split_documents(docs)

embedding_model = CohereEmbeddings(
    model="embed-english-v3.0"
)

vectorstore = Chroma.from_documents(
    embedding = embedding_model,
    documents = chunks,
    persist_directory="chroma_db"
)

