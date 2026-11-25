from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


# ==========================================
# 1. НАСТРОЙКИ
# ==========================================

load_dotenv(
    Path(__file__).resolve().parents[1] / ".env"
)


# ==========================================
# 2. STRUCTURED OUTPUT
# ==========================================

class CarAnswer(BaseModel):
    verdict: Literal["good", "uncertain", "bad"]

    estimated_price: int | None = Field(
        default=None,
        description="Цена автомобиля в евро"
    )

    fuel_consumption: str | None = Field(
        default=None,
        description="Расход топлива"
    )

    reason: str


# ==========================================
# 3. ИНСТРУМЕНТЫ
# ==========================================

@tool
def get_car_price(car: str) -> str:
    """Возвращает точную цену автомобиля в евро."""

    prices = {
        "BMW 320d": 14000,
        "BMW 320i": 12000,
        "BMW 320xi": 13000,
    }

    return str(prices.get(car, "Car not found"))


@tool
def search_catalog(car: str) -> str:
    """Возвращает расход топлива автомобиля."""

    catalog = {
        "BMW 320d": "Дизель, расход 5 л/100 км",
        "BMW 320i": "Бензин, расход 8 л/100 км",
        "BMW 320xi": "Бензин, расход 9 л/100 км",
    }

    return catalog.get(car, "Car not found")


# ==========================================
# 4. АГЕНТ
# ==========================================

agent = create_agent(
    model=ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    ),
    tools=[
        get_car_price,
        search_catalog,
    ],
    response_format=CarAnswer,
    system_prompt=(
        "Ты консультант по автомобилям. "
        "Точную цену получай только через get_car_price. "
        "Расход получай только через search_catalog. "
        "Не выдумывай информацию. "
        "Если данные отсутствуют, используй null."
    ),
)


# ==========================================
# 5. ЗАПУСК
# ==========================================

question = "Сколько стоит BMW 320d и какой у неё расход?"

print("\nUSER:", question)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ]
    },
    config={
        "run_name": "BMW price and consumption",
        "tags": ["lesson-3", "bmw"],
    },
)

print("\nANSWER:")
print(result["structured_response"].model_dump())

# poetry run python theory/19_langsmith.py