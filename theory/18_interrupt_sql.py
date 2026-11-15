from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pathlib import Path

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


def auto_approve(state: CarState):
    print("\n[NODE] auto_approve")

    return {
        "approved": True,
        "status": "auto_approved",
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

DB_PATH = Path(__file__).resolve().parent / "18_interrupt.sqlite"

builder = StateGraph(CarState)

builder.add_node("get_price", get_price)
builder.add_node("ask_approval", ask_approval)
builder.add_node("buy_car", buy_car)
builder.add_node("cancel_purchase", cancel_purchase)
builder.add_node("auto_approve", auto_approve)

builder.add_edge(START, "get_price")

builder.add_conditional_edges(
    "get_price",
    should_ask_approval,
    path_map={
        "yes": "ask_approval",
        "no": "auto_approve",
    },
)

builder.add_conditional_edges(
    "ask_approval",
    check_approval,
    path_map={
        "buy": "buy_car",
        "cancel": "cancel_purchase",
    },
)

builder.add_edge("auto_approve", "buy_car")
builder.add_edge("buy_car", END)
builder.add_edge("cancel_purchase", END)

mode = input("Введите режим (first/second): ")
if mode not in ["first", "second"]:
    raise ValueError("Неверный режим")

initial_state = {
    "car": "BMW 320i",
    "price": 0,
    "approved": None,
    "status": "new",
}

with SqliteSaver.from_conn_string(str(DB_PATH)) as memory:
    graph = builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "car-purchase-1"}}
    
    if mode == "first":
        print("\n=== FIRST INVOKE ===")
        result = graph.invoke(initial_state, config)
        print("\nRESULT:")
        print(result)
        if result.get("__interrupt__"):
            print("\n[PAUSED] Граф ожидает подтверждения")
    else:
        print("\n=== SECOND INVOKE ===")
        state = graph.get_state(config)

        if "approved" not in state.next:
            raise RuntimeError(
                "Нет ожидающего подтверждения. "
                "Сначала запустите режим first."
            )
        
        answer = input("\nПодтвердить покупку? (yes/no): ")
        approved = answer.lower() == "yes"
        result = graph.invoke(Command(resume=approved), config)
        print("\nRESULT:")
        print(result)

# poetry run python 18_interrupt_sql.py