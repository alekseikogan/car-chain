from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


# ==========================================
# 1. STATE
# ==========================================

class CarState(TypedDict):
    car: str
    price: int
    approved: bool | None
    status: str


# ==========================================
# 2. УЗЛЫ
# ==========================================

def get_price(state: CarState):
    print("\n[NODE] get_price")

    prices = {
        "BMW 320d": 14000,
        "BMW 320i": 12000,
    }

    return {
        "price": prices[state["car"]],
        "status": "price_found",
    }


def should_ask_approval(state: CarState):
    print("\n[NODE] should_ask_approval")
    if state["price"] < 13000:
        return "no"

    return "yes"


def ask_approval(state: CarState):
    print("\n[NODE] ask_approval")

    approved = interrupt({
        "question": f"Подтвердить покупку {state['car']}?",
        "price": state["price"],
    })

    print("[RESUME] Получен ответ:", approved)

    return {
        "approved": approved,
    }


def buy_car(state: CarState):
    print("\n[NODE] buy_car")

    print(f"Покупаем {state['car']} за {state['price']} евро")

    return {
        "status": "purchased",
    }


def cancel_purchase(state: CarState):
    print("\n[NODE] cancel_purchase")

    return {
        "status": "cancelled",
    }


# ==========================================
# 3. УСЛОВИЕ
# ==========================================


def check_approval(state: CarState):
    if state["approved"]:
        return "buy"

    return "cancel"


builder = StateGraph(CarState)

builder.add_node("get_price", get_price)
builder.add_node("ask_approval", ask_approval)
builder.add_node("buy_car", buy_car)
builder.add_node("cancel_purchase", cancel_purchase)

builder.add_edge(START, "get_price")

builder.add_conditional_edges(
    "get_price",
    should_ask_approval,
    path_map={
        "yes": "ask_approval",
        "no": "buy_car",
    },
)
builder.add_edge("buy_car", END)
builder.add_edge("cancel_purchase", END)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

config = {
    "configurable": {
        "thread_id": "car-purchase-1"
    }
}

initial_state = {
    "car": "BMW 320i",
    "price": 0,
    "approved": None,
    "status": "new",
}

print("\n=== FIRST INVOKE ===")

result = graph.invoke(
    initial_state,
    config=config,
)

print("\nRESULT:")
print(result)


# ==========================================
# 7. ПОДТВЕРЖДЕНИЕ
# ==========================================
if result.get("__interrupt__"):
    print("\n=== SECOND INVOKE ===")
    answer = input("\nПодтвердить покупку? (yes/no): ")
    approved = answer.lower() == "yes"
    result = graph.invoke(Command(resume=approved), config=config)

    print("\nFINAL RESULT:")
    print(result)