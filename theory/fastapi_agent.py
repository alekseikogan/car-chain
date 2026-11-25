# Агент и SqliteSaver создаются один раз при старте сервера.
# Каждый POST /ask только подкладывает вопрос в уже открытый диалог.
#
# Из корня проекта:
# poetry run uvicorn theory.fastapi_agent:app --host 127.0.0.1 --port 8000
# Форма запроса: http://127.0.0.1:8000/docs

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class CarAnswer(BaseModel):
    verdict: Literal["good", "uncertain", "bad"]
    estimated_price: int | None = Field(
        default=None,
        description="Цена автомобиля в евро. Если цену ещё не запрашивали — null.",
    )
    fuel_consumption: str | None = Field(
        default=None,
        description="Расход из search_catalog, например '5 л/100 км'. Если ещё не искали — null.",
    )
    reason: str = Field(
        description=(
            "Ответ именно на последний вопрос. "
            "Пиши заново, не копируй reason из предыдущего JSON."
        )
    )


class AskRequest(BaseModel):
    thread_id: str
    question: str


@tool
def get_car_price(car: str) -> str:
    """Возвращает точную цену автомобиля в евро."""
    prices = {
        "BMW 320d": "14000",
        "BMW 320i": "12000",
        "BMW 320xi": "13000",
    }
    return prices.get(car, "Car not found")


@tool
def search_catalog(car: str) -> str:
    """Возвращает расход топлива автомобиля."""
    catalog = {
        "BMW 320d": "5 л/100 км",
        "BMW 320i": "8 л/100 км",
        "BMW 320xi": "9 л/100 км",
    }
    return catalog.get(car, "Car not found")


# Отдельный файл, чтобы не смешать диалоги с theory/16_sqlite_memory.py.
DB_PATH = Path(__file__).resolve().parent / "fastapi_dialogs.sqlite"

# Чекпоинт хранит CarAnswer. Без этого SqliteSaver не сможет его прочитать обратно.
serde = JsonPlusSerializer(
    allowed_msgpack_modules=[(CarAnswer.__module__, CarAnswer.__name__)]
)


def build_agent(memory: SqliteSaver):
    return create_agent(
        model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=[get_car_price, search_catalog],
        response_format=CarAnswer,
        checkpointer=memory,
        system_prompt=(
            "Ты консультант по автомобилям. "
            "Точную цену получай только через get_car_price. "
            "Расход получай только через search_catalog. "
            "Не выдумывай информацию. Если данных нет, ставь null. "
            "Местоимения вроде «она» разрешай по истории этого диалога. "
            "Если из истории неясно, о какой машине речь, не вызывай инструменты "
            "и оставь цену и расход пустыми. "
            "Уже известные поля сохраняй. "
            "reason пиши заново под последний вопрос."
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # with держит соединение открытым, пока сервер работает.
    # Выход из with — только при остановке, не после запроса.
    with SqliteSaver.from_conn_string(str(DB_PATH)) as memory:
        memory.serde = serde
        app.state.agent = build_agent(memory)
        yield


app = FastAPI(lifespan=lifespan)


@app.post("/ask", response_model=CarAnswer)
def ask(body: AskRequest) -> CarAnswer:
    # Обычный def, не async: SqliteSaver умеет только синхронные вызовы.
    # FastAPI сам выполнит этот обработчик в пуле потоков.
    result = app.state.agent.invoke(
        {"messages": [{"role": "user", "content": body.question}]},
        config={"configurable": {"thread_id": body.thread_id}},
    )
    return result["structured_response"]
