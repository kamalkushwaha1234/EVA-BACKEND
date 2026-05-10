import os
import uuid
import json
import subprocess
from typing import Optional
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header
from fastapi.staticfiles import StaticFiles
import edge_tts
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from jose import jwt, JWTError
from pydantic import BaseModel

# =====================================================
# App & Base Config
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="STT + LLM + TTS API")
app.mount("/audio", StaticFiles(directory=UPLOAD_DIR), name="audio")

# =====================================================
# Single-user conversation history (file-backed)
# =====================================================
HISTORY_FILE = os.path.join(BASE_DIR, "conversation_history.json")
MAX_HISTORY = 20  # keep last 20 messages to avoid token overflow


def load_history() -> list[dict]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history: list[dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


conversation_history: list[dict] = load_history()

# =====================================================
# Whisper (STT) Config
# =====================================================
WHISPER_BIN = os.path.join(BASE_DIR, "build/bin/whisper-cli")
WHISPER_MODEL = os.path.join(BASE_DIR, "models/ggml-tiny.bin")

# =====================================================
# Azure OpenAI (GPT-4.1) Config
# =====================================================
AZURE_ENDPOINT = "https://models.github.ai/inference"
AZURE_MODEL = "openai/gpt-4.1"
AZURE_TOKEN = os.environ["GITHUB_TOKEN"]

azure_client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_TOKEN),
)

# =====================================================
# MongoDB Config
# =====================================================
MONGODB_URI = os.environ["MONGODB_URI"]
JWT_SECRET = os.environ.get("JWT_SECRET", "change-this-to-a-long-random-secret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client["eva_db"]
users_col = db["users"]
blacklist_col = db["token_blacklist"]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# =====================================================
# Auth Models
# =====================================================
class SignUpRequest(BaseModel):
    email: str
    name: str
    device_id: str
    password: str


class SignInRequest(BaseModel):
    email: str
    password: str


# =====================================================
# Auth Helpers
# =====================================================
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    return jwt.encode({"sub": user_id, "exp": expire}, JWT_SECRET, algorithm=JWT_ALGORITHM)


# =====================================================
# Startup: create DB indexes
# =====================================================
@app.on_event("startup")
async def startup():
    await users_col.create_index("email", unique=True)
    # blacklist tokens auto-expire after JWT lifetime
    await blacklist_col.create_index(
        "created_at", expireAfterSeconds=JWT_EXPIRE_HOURS * 3600
    )


# =====================================================
# Auth Endpoints
# =====================================================
@app.post("/auth/signup")
async def signup(data: SignUpRequest):
    existing = await users_col.find_one({"email": data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = {
        "email": data.email,
        "name": data.name,
        "device_id": data.device_id,
        "password": hash_password(data.password),
        "created_at": datetime.now(timezone.utc),
    }
    result = await users_col.insert_one(user)
    token = create_token(str(result.inserted_id))
    return {"message": "Account created", "token": token, "name": data.name}


@app.post("/auth/signin")
async def signin(data: SignInRequest):
    user = await users_col.find_one({"email": data.email})
    if not user or not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(str(user["_id"]))
    return {"message": "Signed in", "token": token, "name": user["name"]}


@app.post("/auth/signout")
async def signout(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split(" ", 1)[1]
    try:
        jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    await blacklist_col.insert_one({"token": token, "created_at": datetime.now(timezone.utc)})
    return {"message": "Signed out successfully"}


# =====================================================
# TTS Config
# =====================================================
VOICE_MAP = {
    "en": "en-US-AriaNeural",
    "hi": "hi-IN-MadhurNeural"
}


# =====================================================
# STT Endpoint
# =====================================================
@app.post("/stt")
async def speech_to_text(
    file: UploadFile = File(...),
    lang: str = "en"
):
    audio_id = uuid.uuid4().hex
    audio_path = os.path.join(UPLOAD_DIR, f"{audio_id}.wav")
    output_txt = os.path.join(UPLOAD_DIR, f"{audio_id}.txt")

    try:
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        cmd = [
            WHISPER_BIN,
            "-m", WHISPER_MODEL,
            "-f", audio_path,
            "-l", lang,
            "-t", "1",
            "--no-timestamps",
            "--output-txt",
            "--output-file", os.path.join(UPLOAD_DIR, audio_id)
        ]
        subprocess.run(cmd, check=True)

        with open(output_txt, "r", encoding="utf-8") as f:
            text = f.read().strip()

        return {"language": lang, "text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in (audio_path, output_txt):
            if os.path.exists(path):
                os.remove(path)


# =====================================================
# ASK Endpoint — with global conversation context
# =====================================================
@app.post("/ask")
async def ask_llm(
    question: str,
    system_prompt: Optional[str] = (
        "You are an AI tutor for children aged 2 to 17. "
        "Always explain things in simple, easy-to-understand language. "
        "Use fun examples to make concepts clear. "
        "Be friendly, kind, and encouraging at all times. "
        "Keep your answers short and to the point. "
        "Never answer any sexual, violent, or inappropriate questions. "
        "If a question is not suitable for children or is outside the age range of 2-17, "
        "politely refuse and redirect the child to ask a parent or teacher."
    ),
    temperature: float = 0.7
):
    global conversation_history

    # Add user question to history
    conversation_history.append({"role": "user", "content": question})

    # Trim oldest messages if history is too long
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]

    save_history(conversation_history)

    # Build message list for the LLM
    messages = [SystemMessage(system_prompt)]
    for entry in conversation_history:
        if entry["role"] == "user":
            messages.append(UserMessage(entry["content"]))
        elif entry["role"] == "assistant":
            messages.append(AssistantMessage(entry["content"]))

    try:
        response = azure_client.complete(
            model=AZURE_MODEL,
            temperature=temperature,
            top_p=1.0,
            messages=messages,
        )

        answer = response.choices[0].message.content

        # Save assistant reply into history
        conversation_history.append({"role": "assistant", "content": answer})
        save_history(conversation_history)

        return {
            "question": question,
            "answer": answer
        }

    except Exception as e:
        # Roll back user message if LLM call failed
        conversation_history.pop()
        save_history(conversation_history)
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# Clear history endpoint (optional utility)
# =====================================================
@app.post("/reset")
def reset_conversation():
    global conversation_history
    conversation_history = []
    save_history(conversation_history)
    return {"message": "Conversation history cleared"}


# =====================================================
# TTS Endpoint
# =====================================================
@app.post("/tts")
async def text_to_speech(
    request: Request,
    text: str,
    lang: str = "en"
):
    voice = VOICE_MAP.get(lang, VOICE_MAP["en"])
    audio_id = uuid.uuid4().hex
    audio_path = os.path.join(UPLOAD_DIR, f"{audio_id}.mp3")

    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(audio_path)
        return {
            "audio_url": str(request.url_for("audio", path=f"{audio_id}.mp3")),
            "filename": f"{audio_id}.mp3",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
