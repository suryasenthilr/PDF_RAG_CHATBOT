import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):

        with st.spinner("Processing document..."):

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            embeddings = CohereEmbeddings(model="embed-english-v3.0")

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

        st.success("Vector database created!")


if os.path.exists("chroma_db"):

    embeddings = CohereEmbeddings(model="embed-english-v3.0")

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retreiver = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5
        }
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            MessagesPlaceholder(variable_name="chat_history"),

            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "lc_history" not in st.session_state:
        st.session_state.lc_history = []

    if "input_key" not in st.session_state:
        st.session_state.input_key = 0

    st.divider()
    st.subheader("Ask Questions From the Book")

    # Display full chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.text_input("Enter your question",
                          key=f"q_{st.session_state.input_key}")

    if query:

        # Show user message immediately
        with st.chat_message("user"):
            st.write(query)

        docs = retreiver.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke({
            "context": context,
            "question": query,
            "chat_history": st.session_state.lc_history
        })

        response = llm.invoke(final_prompt)

        # Show AI message
        with st.chat_message("assistant"):
            st.write(response.content)

        # Save to histories
        st.session_state.chat_history.append({"role": "user",      "content": query})
        st.session_state.chat_history.append({"role": "assistant",  "content": response.content})
        st.session_state.lc_history.append(HumanMessage(content=query))
        st.session_state.lc_history.append(AIMessage(content=response.content))

        st.session_state.input_key += 1
        st.rerun()