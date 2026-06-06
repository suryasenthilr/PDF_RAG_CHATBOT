from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_cohere import CohereEmbeddings
from langchain_core.messages import HumanMessage, AIMessage


embedding_model = CohereEmbeddings(
    model="embed-english-v3.0"
)
vectorstore = Chroma(
    embedding_function=  embedding_model,
    persist_directory="chroma_db"
)

retreiver = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

chat_history = []

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

while True:
    query = input("You: ")
    if query =="0":
        break
    docs = retreiver.invoke(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context":context,
        "question":query,
        "chat_history":chat_history
    })

    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")

    chat_history.append(HumanMessage(content = query))
    chat_history.append(AIMessage(content = response.content))