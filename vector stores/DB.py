from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.documents import Document

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

llm = ChatMistralAI(model="mistral-small-2506")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory= "chroma-db"
)


result = vectorstore.similarity_search("what is used for data analysis",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vectorstore.as_retriever()

multi_query_retriver = MultiQueryRetriever.from_llm(
    retriver=retriver,
    llm = llm
)

docs = retriver.invoke("what is deep learning")
for d in docs:
    print(d.page_content)

print("\n===== Similarity Search Results =====\n")

mmr_retriver = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)

mmr_docs = mmr_retriver.invoke("what is deep learning")
for doc in mmr_docs:
    print(doc.page_content)
