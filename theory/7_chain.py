from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

ads = [
    Document(page_content="BMW 320d, дизель, пробег 140000 км, цена 14000 евро"),
    Document(page_content="BMW 320i, бензин, пробег 90000 км, цена 12000 евро"),
    Document(page_content="BMW 320xi, полный привод, пробег 110000 км, цена 13000 евро"),
]

store = InMemoryVectorStore.from_documents(
    ads,
    OpenAIEmbeddings(model="text-embedding-3-small"),
)

retriever = store.as_retriever(search_kwargs={"k": 2})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system", "Отвечай только по контексту. Если ответа в контексте нет, напиши: в объявлениях этого нет."),
    ("human", "Контекст:\n{context}\n\nВопрос: {question}"),
])

chain = (
    {"context": retriever | format_docs,
    "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

print(chain.invoke("Какой расход у BMW 320d?"))

# poetry run python 7_chain.py 