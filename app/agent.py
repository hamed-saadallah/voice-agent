from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

SYSTEM_PROMPT = (
    "You are a spoken voice agent for Northwind Goods. "
    "Keep every reply under 40 words. Use short sentences. "
    "Call tools for hours or orders. Never read JSON, IDs, "
    "or tool names out loud."
)

HOURS = {
    "monday": "9am to 6pm",
    "tuesday": "9am to 6pm",
    "wednesday": "9am to 6pm",
    "thursday": "9am to 6pm",
    "friday": "9am to 6pm",
    "saturday": "10am to 2pm",
    "sunday": "closed",
}

ORDERS = {
    "ORD-1001": "Shipped yesterday. Delivery is expected on Friday.",
    "ORD-2044": "Still in packing. It should leave the warehouse tomorrow.",
}


@tool
def get_store_hours(day: str) -> str:
    """Return opening hours for one weekday, such as saturday."""
    key = day.strip().lower()
    return HOURS.get(key, "I only know Monday through Sunday.")


@tool
def get_order_status(order_id: str) -> str:
    """Return shipping status for an order id such as ORD-1001."""
    return ORDERS.get(order_id.strip().upper(), "No order found with that id.")


tools = [get_store_hours, get_order_status]

model = ChatMistralAI(
    model="open-mistral-nemo",
    temperature=0,
    max_retries=2,
)
model_with_tools = model.bind_tools(tools)


def call_model(state: MessagesState) -> dict:
    response = model_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    )
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", "__end__": END},
)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=InMemorySaver())


def last_text(result: dict) -> str:
    content = result["messages"][-1].content
    if isinstance(content, str):
        return content.strip()
    return str(content)


def run_turn(transcript: str, thread_id: str) -> str:
    result = graph.invoke(
        {"messages": [HumanMessage(content=transcript)]},
        {"configurable": {"thread_id": thread_id}},
    )
    return last_text(result)