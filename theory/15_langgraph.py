# poetry run python theory/15_langgraph.py

from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# === 1. ИНСТРУМЕНТЫ ===

@tool
def search_catalog(query: str) -> str:
    """Ищет в каталоге расход, пробег, топливо и состояние автомобилей."""
    data = {
        "BMW 320d": "Дизель, 140000 км, расход 5 л/100 км. Требует проверки турбины.",
        "BMW 320i": "Бензин, 90000 км, расход 8 л/100 км.",
        "BMW 320xi": "Бензин, 110000 км, расход 9 л/100 км.",
    }
    return "\n".join(f"{car}: {info}" for car, info in data.items())


@tool
def get_car_price(car: str) -> str:
    """Возвращает точную цену автомобиля в евро."""
    data = {
        "BMW 320d": "14000",
        "BMW 320i": "12000",
        "BMW 320xi": "13000",
    }
    return data.get(car, "Car not found")


@tool
def get_car_year(car: str) -> str:
    """Возвращает год выпуска автомобиля."""
    data = {
        "BMW 320d": "2015",
        "BMW 320i": "2016",
        "BMW 320xi": "2017",
    }
    return data.get(car, "Car not found")


# === 2. МОДЕЛЬ ===

tools = [search_catalog, get_car_price, get_car_year]

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


# === 3. УЗЕЛ МОДЕЛИ ===

def call_model(state: MessagesState):
    print("\n[MODEL] Вызываем LLM")

    response = llm_with_tools.invoke([
        SystemMessage(content="""
            Ты автоэксперт.
            Точную цену бери только через get_car_price.
            Год бери только через get_car_year.
            Расход и пробег бери только через search_catalog.
            Не выдумывай значения.
            Отвечай кратко.
        """),
        *state["messages"],
    ])

    print("[MODEL] tool_calls:", response.tool_calls)

    return {"messages": [response]}


# === 4. СОБИРАЕМ ГРАФ ===

builder = StateGraph(MessagesState)

builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "model")

builder.add_conditional_edges(
    "model",
    tools_condition,
)

builder.add_edge("tools", "model")

graph = builder.compile()


# === 5. ЗАПУСК ===
content = input("Введите запрос: ")

result = graph.invoke({
    "messages": [
        HumanMessage(content=content),
    ]
})


# === 6. ПРОСМОТР ИСТОРИИ ===

print("\n=== ВСЕ СООБЩЕНИЯ ===")

for message in result["messages"]:
    print("\nTYPE:", type(message).__name__)
    print("CONTENT:", message.content)
    print("TOOL CALLS:", getattr(message, "tool_calls", None))
    print("TOOL CALL ID:", getattr(message, "tool_call_id", None))

print("\n=== ФИНАЛЬНЫЙ ОТВЕТ ===")
print(result["messages"][-1].content)
