# poetry run python test.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()


class CarAnswer(BaseModel):
    verdict: Literal["good", "uncertain", "bad"]
    estimated_price: int
    reason: str = Field(description="Why you gave this verdict")


def get_car_info(car: str) -> str:
    return f"Information about {car}"

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты автоэксперт. Давай короткие советы."),
    ("human", "Данные авто: {car_info}\nВопрос пользователя: {question}")
])

model = ChatOpenAI(model="gpt-4o-mini")

structured_model = model.with_structured_output(CarAnswer)

chain = RunnablePassthrough.assign(
    car_info=RunnableLambda(lambda data: get_car_info(data["car"]))
) | prompt | structured_model

result = chain.invoke({
    "car": "BMW 320d",
    "question": "Is 140000 km a lot for a car?"
})


print(result)
