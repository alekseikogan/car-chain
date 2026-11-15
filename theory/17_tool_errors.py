from pathlib import Path

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# ==========================================
# 1. ИНСТРУМЕНТ
# ==========================================

@tool
def get_car_price(car: str) -> str:
    """Возвращает цену автомобиля в евро."""

    prices = {
        "BMW 320d": 14000,
        "BMW 320i": 12000,
        "BMW 320xi": 13000,
    }

    if car not in prices:
        raise ValueError(f"Car not found: {car}")

    return str(prices[car])

# ==========================================
# 2. MIDDLEWARE
# ==========================================

@wrap_tool_call
def handle_tool_error(request, handler):

    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]

    print(f"\n[TOOL] Название: {tool_name}")
    print(f"[TOOL] Аргументы: {tool_args}")

    try:
        result = handler(request)

        print(f'[TOOL] Результат: {result}')
        return result

    except Exception as e:

        print(f'[TOOL] Ошибка: {e}')

        return ToolMessage(
            content=f"Ошибка: {e}",
            tool_call_id=request.tool_call["id"],
        )

# ==========================================
# 3. АГЕНТ
# ==========================================

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[get_car_price],
    middleware=[handle_tool_error],
    system_prompt=(
        "Ты консультант по автомобилям. "
        "Для получения цены всегда вызывай get_car_price. "
        "Если инструмент вернул Car not found, "
        "скажи, что автомобиля нет в каталоге."
    ),
)

# ==========================================
# 4. ЗАПУСК
# ==========================================

question = "Сколько стоит Audi A4?"

print(f"\nUSER: {question}")

result = agent.invoke({
    "messages": [
        {"role": "user", "content": question}
    ]
})

print("\nANSWER:")
print(result["messages"][-1].content)