# poetry run python theory/14_files.py

from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
BASE_DIR = Path(__file__).resolve().parent

class CarAnswer(BaseModel):
    verdict: Literal["good", "uncertain", "bad"]
    estimated_price: int | None
    reason: str = Field(description="Почему такой вердикт")


CAR_FILES = {
    "bmw_320d.txt": {
        "model": "BMW 320d",
        "year": 2015,
        "fuel": "diesel",
    },
    "bmw_320i.txt": {
        "model": "BMW 320i",
        "year": 2016,
        "fuel": "petrol",
    },
    "bmw_320xi.txt": {
        "model": "BMW 320xi",
        "year": 2017,
        "fuel": "petrol",
    },
}

ads = []

for filename, metadata in CAR_FILES.items():
    path = BASE_DIR / filename

    text = path.read_text(encoding="utf-8")

    doc = Document(
        page_content=text,
        metadata={
            **metadata,
            "source": filename,
        },
    )

    ads.append(doc)

print("Загружено документов:", len(ads))

splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
chunks = splitter.split_documents(ads)
print("chunks:", len(chunks))

store = InMemoryVectorStore.from_documents(
    chunks,
    OpenAIEmbeddings(model="text-embedding-3-small"),
)
retriever = store.as_retriever(search_kwargs={"k": 2})


@tool
def search_catalog(query: str) -> str:
    """Ищет в объявлениях топливо, пробег, расход и состояние."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


@tool
def get_car_price(car: str) -> str:
    """Возвращает точную цену машины в евро."""
    data = {"BMW 320d": "14000", "BMW 320i": "12000", "BMW 320xi": "13000"}
    return data.get(car, "Car not found")


@tool
def get_car_year(car: str) -> str:
    """Возвращает год выпуска машины."""
    data = {"BMW 320d": "2015", "BMW 320i": "2016", "BMW 320xi": "2017"}
    return data.get(car, "Car not found")


memory = InMemorySaver(
    serde=JsonPlusSerializer(allowed_msgpack_modules=[("__main__", "CarAnswer")])
)

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[search_catalog, get_car_price, get_car_year],
    system_prompt="""
        Ты автоэксперт.
        Точную цену бери только через get_car_price.
        Год бери только через get_car_year.
        Топливо, пробег, расход и состояние бери только через search_catalog.
        Если инструмент вернул "Car not found" или в каталоге нет факта, не выдумывай значение.
        Отвечай кратко.
    """,
    response_format=CarAnswer,
    checkpointer=memory,
)

dialog = {"configurable": {"thread_id": "dialog-1"}}

agent.invoke(
    {"messages": [{"role": "user", "content": "Сколько стоит BMW 320d?"}]},
    dialog,
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user",
            "content": "Какой у неё расход и пробег?"}
            ]
    },
    dialog,
)

for message in result["messages"]:
    print(type(message).__name__)
    print("content:", message.content)
    print("tool_calls:", getattr(message, "tool_calls", None))
    print("tool_call_id:", getattr(message, "tool_call_id", None))
    print("---")

answer = result["structured_response"]
print(answer.verdict, answer.estimated_price, answer.reason)
