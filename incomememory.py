# poetry run python incomememory.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from typing import Literal
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

load_dotenv()

class CarAnswer(BaseModel):
    verdict: Literal["good", "uncertain", "bad"]
    estimated_price: int | None
    reason: str = Field(description="Why you gave this verdict")

@tool
def get_car_price(car_model: str) -> str:
    """Возвращает стоимость авто."""
    return "18000 EUR"

@tool
def get_car_year(car: str) -> int:
    """Get the year of a car"""
    data = {
        "BMW 320d": 2015,
        "BMW 320i": 2016,
        "BMW 320xi": 2017,
    }
    return data.get(car, "Car not found")

model = ChatOpenAI(model="gpt-4o-mini")

# 2. Создаем объект памяти
memory = InMemorySaver(
    serde=JsonPlusSerializer(allowed_msgpack_modules=[("__main__", "CarAnswer")])
)

# 3. Передаем memory в параметр checkpointer
agent = create_agent(
    model=model,
    tools=[get_car_price, get_car_year],
    system_prompt="""
        Ты автоэксперт.
        Если пользователь спрашивает цену, используй get_car_price.
        Если спрашивает год, используй get_car_year.
        Если инструмент вернул "Car not found", не выдумывай значение.
        Отвечай кратко.
        """,
    response_format=CarAnswer,
    checkpointer=memory,
)

# -------------------------------------------------------------
# Использование: создаем конфигурацию с thread_id (ID диалога)
# -------------------------------------------------------------
dialog = {"configurable": {"thread_id": "dialog-1"}}

# ШАГ 1: Пользователь называет машину
agent.invoke(
    {"messages": [{"role": "user", "content": "Сколько стоит BMW 320d?"}]},
    config,
)

# ШАГ 2: Задаем вопрос БЕЗ указания марки машины
# Агент ПОМНИТ из предыдущего сообщения, что речь идет о BMW 320d!
res2 = agent.invoke(
    {"messages": [("human", "Какая у неё рыночная цена?")]}, 
    config=dialog
)

print(res2["messages"][-1].content)
# Выведет: "Рыночная цена BMW 320d составляет 18 000 EUR."