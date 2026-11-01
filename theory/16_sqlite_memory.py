# poetry run python theory/16_sqlite_memory.py
# poetry run python theory/16_sqlite_memory.py first
# poetry run python theory/16_sqlite_memory.py second

from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
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
import sys

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class CarAnswer(BaseModel):
    verdict: Literal["good", "uncertain", "bad"]
    estimated_price: int | None = Field(
        default=None,
        description="Цена в евро из get_car_price. Если цену ещё не запрашивали — null.",
    )
    reason: str = Field(
        description=(
            "Ответ именно на последний вопрос пользователя. "
            "Пиши заново, не копируй reason из предыдущего JSON."
        )
    )
    fuel_consumption: str | None = Field(
        default=None,
        description="Расход из search_catalog, например '5 л/100 км'. Если ещё не искали — null.",
    )
    mileage: int | None = Field(
        default=None,
        description="Пробег в километрах из search_catalog. Если ещё не искали — null.",
    )


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


DB_PATH = Path(__file__).resolve().parent / "dialogs.sqlite"

serde = JsonPlusSerializer(allowed_msgpack_modules=[("__main__", "CarAnswer")])

# Выбираем этап эксперимента через аргумент командной строки
if len(sys.argv) != 2 or sys.argv[1] not in {"first", "second"}:
    raise SystemExit(
        "Использование: python theory/16_sqlite.py first|second"
    )

mode = sys.argv[1]

with SqliteSaver.from_conn_string(DB_PATH) as memory:
    # Указываем сериализатор для пользовательского CarAnswer
    memory.serde = serde

    agent = create_agent(
        model=ChatOpenAI(model="gpt-4o-mini"),
        tools=[search_catalog, get_car_price, get_car_year],
        system_prompt="""
            Ты автоэксперт.

            Точную цену бери только через get_car_price.
            Год бери только через get_car_year.
            Расход, пробег, топливо и состояние
            бери только через search_catalog.

            Если информация отсутствует, не выдумывай.
            Отвечай кратко.

            Каждый ответ — полный объект CarAnswer.
            Уже известные поля сохраняй.
            reason пиши заново под последний вопрос:
            в нём должны быть факты, которые спросили сейчас.
            Не копируй reason из предыдущего JSON.
        """,
        response_format=CarAnswer,
        checkpointer=memory,
    )

    # Один и тот же thread_id для обоих запусков!
    dialog = {
        "configurable": {
            "thread_id": "sqlite-dialog-2"
        }
    }

    if mode == "first":
        question = "Сколько стоит BMW 320d?"
    else:
        question = "А какой у неё расход и пробег?"

    print("\nUSER:", question)

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        },
        config=dialog,
    )

    answer = result["structured_response"]

    print("\nANSWER:")
    print(answer.model_dump())

    print("\nHISTORY:")
    for message in result["messages"]:
        print(
            type(message).__name__,
            str(message.content),
        )

    print("\nDATABASE:", DB_PATH)
    print("EXISTS:", DB_PATH.exists())
