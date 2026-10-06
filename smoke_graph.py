from dotenv import load_dotenv

load_dotenv()

from app.agent import run_turn

print(run_turn("What time do you close on Saturday?", "smoke"))
print(run_turn("Where is order ORD-1001?", "smoke"))