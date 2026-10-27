# poetry run python test.py
from langchain_core.messages import ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal


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


def get_car_info(car: str) -> str:
    return f"Information about {car}"

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты автоэксперт. Давай короткие советы. если инструмент не нашёл цену, не выдумывай её — скажи, что цена неизвестна"),
    ("human", "Данные авто: {car_info}\nВопрос пользователя: {question}")
])

model = ChatOpenAI(model="gpt-4o-mini")

model_with_tools = model.bind_tools([get_car_price, get_car_year])

structured_model = model_with_tools.with_structured_output(CarAnswer)


chain = RunnablePassthrough.assign(
    car_info=RunnableLambda(lambda data: get_car_info(data["car"]))
) | prompt | structured_model

# Модель НЕ пишет ответ. Переменная response содержит специальный объект AIMessage,
# у которого в поле response.tool_calls лежит команда: вызови get_car_price.
response = model_with_tools.invoke(
    "How much is a BMW 320d?"
)

# Формирование истории диалога
messages = [
    ("human", "How much is a BMW 320d?"), # 1. Вопрос пользователя
    response,                              # 2. Ответ модели: "Вызови инструмент!"
]

print("response.tool_calls: ", response.tool_calls)

# for tool_call in response.tool_calls:
#     if tool_call["name"] == "get_car_price":
#         result = get_car_price.invoke(tool_call["args"])

#         # Упаковываем результат функции в специальное сообщение ToolMessage
#         messages.append(
#             ToolMessage(
#                 content=str(result),
#                 tool_call_id=tool_call["id"], # Обязательно связываем результат с конкретным вызовом
#             )
#         )
#     elif tool_call["name"] == "get_car_year":
#         print("Вызывали бы get_car_year")

# final_response = model.invoke(messages)

# print(final_response.content)

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

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
# После invoke() структурированный результат находится в result["structured_response"].

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "How much is a BMW 320d?"}
    ]
})

answer = result["structured_response"]

print(answer)
print(answer.verdict)
print(answer.estimated_price)
print(answer.reason)