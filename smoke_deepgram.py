from dotenv import load_dotenv

load_dotenv()

from app.speech import synthesize, transcribe_url

print(transcribe_url("https://dpgr.am/bueller.wav"))
audio = synthesize("The voice agent is ready.")
with open("smoke_speak.mp3", "wb") as handle:
    handle.write(audio)
print(f"Wrote smoke_speak.mp3 ({len(audio)} bytes)")