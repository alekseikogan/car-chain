from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

messages = [
    HumanMessage(content="Сколько стоит BMW 320d?"),
    AIMessage(
        content="",
        tool_calls=[{
            "name": "get_car_price",
            "args": {"car": "BMW 320d"},
            "id": "call_1",
            "type": "tool_call",
        }],
    ),
    ToolMessage(content="99999", tool_call_id="wrong"),
]

response = ChatOpenAI(model="gpt-4o-mini").invoke(messages)
print(response.content)

# poetry run python 8_messages.py