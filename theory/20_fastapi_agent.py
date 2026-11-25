# Агент и SqliteSaver создаются один раз при старте сервера.
# Каждый POST /ask только подкладывает вопрос в уже открытый диалог.
#
# Из корня проекта:
# poetry run uvicorn theory.20_fastapi_agent:app --host 127.0.0.1 --port 8000
# Форма запроса: http://127.0.0.1:8000/docs
# Поток шагов: curl -N -G http://127.0.0.1:8000/ask/stream --data-urlencode "thread_id=demo" --data-urlencode "question=Сколько стоит BMW 320d?"

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field
from langgraph.types import interrupt, Command

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

@tool
def ask_approval(car: str, price: int) -> str:
    """Запрашивает подтверждение покупки автомобиля."""
    approved = interrupt({
        "question": f"Подтверди покупку автомобиля {car} за {price} евро?",
        "price": price,
    })
    if approved:
        return f"Пользователь подтвердил покупку автомобиля {car} за {price} евро."
    return "Пользователь отказался от покупки."


# Отдельный файл, чтобы не смешать диалоги с theory/16_sqlite_memory.py.
DB_PATH = Path(__file__).resolve().parent / "20_fastapi_dialogs.sqlite"

# Чекпоинт хранит CarAnswer. Без этого SqliteSaver не сможет его прочитать обратно.
serde = JsonPlusSerializer(
    allowed_msgpack_modules=[(CarAnswer.__module__, CarAnswer.__name__)]
)


def build_agent(memory: SqliteSaver):
    return create_agent(
        model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=[get_car_price, search_catalog, ask_approval],
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
            "reason пиши заново под последний вопрос. "
            "Если пользователь хочет купить машину, сначала узнай цену через get_car_price. "
            "Когда цена известна, перед итоговым ответом вызови ask_approval "
            "и дождись его результата. "
            "Покупку подтверждай только по ответу ask_approval. "
            "В каждом CarAnswer обязательно заполни verdict: good, uncertain или bad."
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


@app.post("/ask")
def ask(body: AskRequest):
    # Обычный def, не async: SqliteSaver умеет только синхронные вызовы.
    # FastAPI сам выполнит этот обработчик в пуле потоков.
    result = app.state.agent.invoke(
        {"messages": [{"role": "user", "content": body.question}]},
        config={"configurable": {"thread_id": body.thread_id}},
    )

    paused = result.get("__interrupt__")
    if paused:
        return {"status": "need_user", **paused[0].value}
    return {
        "status": "done",
        "answer": result["structured_response"].model_dump(),
}


def iter_agent_events(thread_id: str, question: str):
    # Тот же цикл, что в 11_stream.py: каждое событие уходит клиенту сразу.
    try:
        yield from _iter_agent_events(thread_id, question)
    except Exception as exc:
        yield f"\n[ERROR] {exc}\n"


def _iter_agent_events(thread_id: str, question: str):
    for part in app.state.agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=["updates", "messages"],
        version="v2",
    ):
        if part["type"] == "messages":
            token, _metadata = part["data"]
            # Аргументы инструмента и его ответ приходят отдельно, через updates.
            if getattr(token, "tool_call_chunks", None):
                continue
            if getattr(token, "type", None) == "tool":
                continue
            if token.text:
                yield token.text
            continue

        for node_name, update in part["data"].items():
            if node_name == "__interrupt__":
                for item in update:
                    payload = getattr(item, "value", item)
                    yield f"[PAUSED] {json.dumps(payload, ensure_ascii=False)}\n"
                continue

            if not isinstance(update, dict):
                continue

            for message in update.get("messages", []):
                for call in getattr(message, "tool_calls", None) or []:
                    yield (
                        f"[MODEL] Вызов: {call['name']} "
                        f"Аргументы: {call['args']}\n"
                    )
                if getattr(message, "type", None) == "tool":
                    yield f"[TOOL] {message.name}: {message.content}\n"

            answer = update.get("structured_response")
            if answer is not None:
                payload = answer.model_dump() if hasattr(answer, "model_dump") else answer
                yield f"\n[FINAL] {json.dumps(payload, ensure_ascii=False)}\n"


@app.get("/ask/stream")
def ask_stream(thread_id: str, question: str):
    return StreamingResponse(
        iter_agent_events(thread_id, question),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/resume")
def resume(thread_id: str, resume: bool):
    result = app.state.agent.invoke(
        Command(resume=resume),
        config={"configurable": {"thread_id": thread_id}},
    )
    return {"status": "success", "response": result["structured_response"]}
