import os
from functools import lru_cache

from deepgram import DeepgramClient
from deepgram.core.api_error import ApiError

STT_MODEL = "nova-3"
TTS_MODEL = "aura-2-thalia-en"


@lru_cache(maxsize=1)
def get_deepgram() -> DeepgramClient:
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is missing")
    return DeepgramClient(api_key=api_key)


def transcribe(audio_bytes: bytes) -> str:
    try:
        response = get_deepgram().listen.v1.media.transcribe_file(
            request=audio_bytes,
            model=STT_MODEL,
            language="en",
            smart_format=True,
        )
        text = response.results.channels[0].alternatives[0].transcript or ""
    except (ApiError, AttributeError, IndexError, TypeError) as exc:
        raise ValueError("Deepgram returned no transcript") from exc
    text = text.strip()
    if not text:
        raise ValueError("Deepgram returned an empty transcript")
    return text


def transcribe_url(url: str) -> str:
    response = get_deepgram().listen.v1.media.transcribe_url(
        url=url,
        model=STT_MODEL,
        language="en",
        smart_format=True,
    )
    text = response.results.channels[0].alternatives[0].transcript or ""
    return text.strip()


def synthesize(text: str) -> bytes:
    if not text.strip():
        raise ValueError("Cannot synthesize empty text")
    response = get_deepgram().speak.v1.audio.generate(
        text=text,
        model=TTS_MODEL,
    )
    audio = bytearray()
    for chunk in response:
        audio.extend(chunk)
    return bytes(audio)