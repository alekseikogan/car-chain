# poetry run python theory/12_trim.py

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

from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain.agents.middleware import before_model
from langchain.agents import AgentState
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

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


from langchain_core.messages import AIMessage, ToolMessage


def validate_tool_calls(messages):
    pending = set()

    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                pending.add(call["id"])

        elif isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                raise ValueError(
                    f"ToolMessage without AIMessage: "
                    f"{message.tool_call_id}"
                )
            pending.remove(message.tool_call_id)

    if pending:
        raise ValueError(
            f"Unanswered tool calls: {pending}"
        )

    print("История корректна")


from langchain_core.messages import HumanMessage


def split_into_turns(messages):
    turns = []

    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append([])

        if not turns:
            raise ValueError("History must start with HumanMessage")

        turns[-1].append(message)

    return turns


def trim_complete_turns(messages, max_tokens=300):
    turns = split_into_turns(messages)

    kept = []

    for turn in reversed(turns):
        candidate = turn + kept

        if count_tokens_approximately(candidate) > max_tokens:
            break

        kept = candidate

    return kept


@before_model
def limit_history(state: AgentState, runtime: Runtime):
    messages = state["messages"]

    # Оставляем не более ~1200 токенов истории.
    trimmed = trim_complete_turns(
        messages,
        max_tokens=1200,
    )

    # Если последний ход не помещается в бюджет,
    # не удаляем его: обрабатываем отдельно.
    if not trimmed:
        print("[TRIM] Последний ход превышает бюджет")
        return None

    if len(trimmed) == len(messages):
        return None

    validate_tool_calls(trimmed)

    print(f"[TRIM] Было: {len(messages)}")
    print(f"[TRIM] Стало: {len(trimmed)}")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *trimmed,
        ]
    }


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
    middleware=[limit_history],  # Добавили
)

dialog = {"configurable": {"thread_id": "trim-test-1"}}

questions = [
    "Сколько стоит BMW 320d?",
    "Какой у неё расход?",
    "Какой у неё пробег?",
    "Какие у неё недостатки?",
    "Какого она года?",
]

for question in questions:
    print(f"\nUSER: {question}")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=dialog,
    )

    answer = result["structured_response"]

    print("ANSWER:", answer.reason)

    state = agent.get_state(dialog)

    print("Messages:", len(state.values["messages"]))
    print(
        "Approx tokens:",
        count_tokens_approximately(state.values["messages"]),
    )

    validate_tool_calls(state.values["messages"])