import asyncio
import base64
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

load_dotenv()

from app.agent import run_turn
from app.speech import synthesize, transcribe

app = FastAPI(title="Voice agent")
INDEX = Path(__file__).resolve().parent.parent / "static" / "index.html"


@app.get("/")
async def index():
    return FileResponse(INDEX)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/voice")
async def voice(
    audio: UploadFile = File(...),
    session_id: str = Form("demo"),
):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    try:
        transcript = await asyncio.to_thread(transcribe, audio_bytes)
        reply = await asyncio.to_thread(run_turn, transcript, session_id)
        mp3 = await asyncio.to_thread(synthesize, reply)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "transcript": transcript,
        "reply": reply,
        "audio_base64": base64.b64encode(mp3).decode("ascii"),
        "session_id": session_id,
    }