# poetry run python agent.py
from langchain_core.messages import ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent


load_dotenv()


class CarAnswer(BaseModel):
    verdict: Literal["good", "uncertain", "bad"]
    estimated_price: int | None
    reason: str = Field(description="Why you gave this verdict")


@tool
def get_car_price(car: str) -> int:
    """Get the price of a car in euros"""
    data = {
        "BMW 320d": 14000,
        "BMW 320i": 12000,
        "BMW 320xi": 13000,
    }
    return data.get(car, "Car not found")

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

agent = create_agent(
    model=model,
    tools=[get_car_price, get_car_year],
    system_prompt = """
        Ты автоэксперт.
        Если пользователь спрашивает цену конкретной машины,
        обязательно используй инструмент get_car_price.
        Если инструмент вернул "Car not found",
        скажи, что цена не найдена. Не придумывай цену.
        Отвечай кратко.
        """,
    response_format=CarAnswer,
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "BMW 320d: сколько стоит и какого года?"}
    ]
})

for message in result["messages"]:
    print(type(message).__name__)
    print("content:", message.content)
    print("tool_calls:", getattr(message, "tool_calls", None))
    print("---")

answer = result["structured_response"]
print(answer.verdict, answer.estimated_price, answer.reason)