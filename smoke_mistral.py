from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = ChatMistralAI(
    model="open-mistral-nemo",
    temperature=0,
    max_retries=2,
)
response = model.invoke([
    SystemMessage(content="Reply in one short sentence."),
    HumanMessage(content="Confirm you are ready to run a voice agent."),
])
print(response.content)