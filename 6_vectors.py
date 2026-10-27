from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

ads = [
    Document(
        page_content="BMW 320d, дизель, пробег 140000 км, цена 14000 евро",
        metadata={"model": "BMW 320d", "fuel": "diesel"},
    ),
    Document(
        page_content="BMW 320i, бензин, пробег 90000 км, цена 12000 евро",
        metadata={"model": "BMW 320i", "fuel": "petrol"},
    ),
    Document(
        page_content="BMW 320xi, полный привод, пробег 110000 км, цена 13000 евро",
        metadata={"model": "BMW 320xi", "fuel": "petrol"},
    ),
]

store = InMemoryVectorStore.from_documents(
    ads,
    OpenAIEmbeddings(model="text-embedding-3-small"),
)

found = store.similarity_search_with_score("дизель", k=2)

for doc, score in found:
    print(doc.page_content)
    print("Similarity:", score)