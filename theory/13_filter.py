# poetry run python theory/13_filter.py

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


class CarAnswer(BaseModel):
    verdict: Literal["good", "uncertain", "bad"]
    estimated_price: int | None
    reason: str = Field(description="Почему такой вердикт")


ads = [
    Document(
        page_content=(
            "BMW 320d 2015 года, дизель, пробег 140000 км, цена 14000 евро. "
            "Мотор 2.0 хорошо тянет по трассе и расходует около 5 литров. "
            "Коробка автомат переключается плавно, подвеска мягкая. "
            "Салон без прокуров, сервисная история есть, последний раз меняли масло и колодки. "
            "Минус: зимой долго греется, а турбина на таком пробеге уже требует проверки."
        ),
        metadata={"model": "BMW 320d", "year": 2015, "fuel": "diesel"},
    ),
    Document(
        page_content=(
            "BMW 320i 2016 года, бензин, пробег 90000 км, цена 12000 евро. "
            "Мотор 2.0 бодрый в городе и расходует около 8 литров. "
            "Привод задний, коробка автомат. Кузов без окрасов, два владельца. "
            "Минус: расход выше дизеля, на трассе шумнее 320d."
        ),
        metadata={"model": "BMW 320i", "year": 2016, "fuel": "petrol"},
    ),
    Document(
        page_content=(
            "BMW 320xi 2017 года, бензин, полный привод, пробег 110000 км, цена 13000 евро. "
            "Зимой устойчивее заднего привода. Расход около 9 литров. "
            "Коробка автомат, климат двухзонный. Минус: сложнее и дороже обслуживание полного привода."
        ),
        metadata={"model": "BMW 320xi", "year": 2017, "fuel": "petrol"},
    ),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
chunks = splitter.split_documents(ads)
print("chunks:", len(chunks))

store = InMemoryVectorStore.from_documents(
    chunks,
    OpenAIEmbeddings(model="text-embedding-3-small"),
)

retriever = store.as_retriever(search_kwargs={"k": 2})


@tool
def search_catalog(
    query: str,
    fuel: Literal["diesel", "petrol"] | None = None,
) -> str:
    """
    Ищет в каталоге автомобилей информацию о пробеге,
    расходе топлива и состоянии.

    Args:
        query: Поисковый запрос.
        fuel: Необязательный фильтр по типу топлива.
              diesel — дизель, petrol — бензин.
    """

    if fuel is not None:
        docs = store.similarity_search(
            query=query,
            k=3,
            filter=lambda doc: doc.metadata.get("fuel") == fuel,
        )
    else:
        docs = store.similarity_search(
            query=query,
            k=3,
        )

    if not docs:
        return "Ничего не найдено"

    return "\n\n".join(
        f"Model: {doc.metadata['model']}\n"
        f"Fuel: {doc.metadata['fuel']}\n"
        f"Text: {doc.page_content}"
        for doc in docs
    )


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
        Для поиска информации об автомобилях используй search_catalog.
        Если пользователь указывает тип топлива,
        передавай его через аргумент fuel.
        Не выдумывай значения.
    """,
    response_format=CarAnswer,
    checkpointer=memory,
)


questions = [
    "Какой расход у дизельных BMW?",
    "Какой расход у бензиновых BMW?",
    "Сравни расход всех BMW.",
]

for i, question in enumerate(questions):
    print(f"\n=== ВОПРОС {i + 1} ===")
    print(question)

    config = {
        "configurable": {"thread_id": f"metadata-{i}"}
    }

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            for msg in update.get("messages", []):
                for call in getattr(msg, "tool_calls", []):
                    print("TOOL:", call["name"])
                    print("ARGS:", call["args"])

                if msg.type == "tool":
                    print("RESULT:", msg.content[:500])

            if "structured_response" in update:
                print("ANSWER:", update["structured_response"])
