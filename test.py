# poetry run python test.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os
from dotenv import load_dotenv
load_dotenv()

def get_car_info(car: str) -> str:
    return f"Information about {car}"

car_info_runnable = RunnableLambda(get_car_info)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты автоэксперт. Давай короткие советы."),
    ("human", "Данные авто: {car_info}\nВопрос пользователя: {question}")
])

parser = StrOutputParser()

model = ChatOpenAI(model="gpt-4o-mini")

chain = RunnablePassthrough.assign(
    car_info=RunnableLambda(lambda data: get_car_info(data["car"]))
) | prompt | model | parser

result = chain.invoke({
    "car": "BMW 320d",
    "question": "Is 140000 km a lot for a car?"
})


print(result)