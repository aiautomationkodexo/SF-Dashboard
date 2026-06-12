import asyncio
import os
import uuid
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from livekit import api

load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("outbound-api")


# ── Request / Response models ──────────────────────────────────────────────────

class CallRequest(BaseModel):
    phone_number: str          # E.164 format, e.g. "+447311129945"
    agent_name: str = "test-agen"   # which LiveKit agent to dispatch

class CallResponse(BaseModel):
    success: bool
    room: str | None = None
    message: str


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Stories Flooring — Outbound Call API",
    description="POST /call to trigger Elliot to ring a customer.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ─────────────────────────────────────────────────────────────────────

async def initiate_call(phone_number: str, agent_name: str) -> str:
    """Creates a SIP participant and dispatches the agent. Returns the room name."""
    url        = os.getenv("LIVEKIT_URL")
    api_key    = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    trunk_id   = os.getenv("LIVEKIT_TRUNK_ID")

    if not all([url, api_key, api_secret, trunk_id]):
        raise RuntimeError("Missing LiveKit credentials in .env.local")

    api_url  = url.replace("wss://", "https://").replace("ws://", "http://")
    lkapi    = api.LiveKitAPI(api_url, api_key, api_secret)
    room_name = f"outbound-{uuid.uuid4().hex[:8]}"

    try:
        # 1. Create the room FIRST so it exists before we dispatch anything
        from livekit.protocol import room as room_proto
        await lkapi.room.create_room(
            room_proto.CreateRoomRequest(name=room_name)
        )
        logger.info(f"Room created: {room_name}")

        # 2. Dial the customer
        await lkapi.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=trunk_id,
                sip_call_to=phone_number,
                participant_identity=f"phone_{phone_number.strip('+')}",
                participant_name="Customer",
                room_name=room_name,
            )
        )
        logger.info(f"SIP call initiated to {phone_number} in room {room_name}")

        # 3. Dispatch the agent into the room (room is guaranteed to exist now)
        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=agent_name,
                room=room_name,
            )
        )
        logger.info(f"Agent '{agent_name}' dispatched to room {room_name}")

    finally:
        await lkapi.aclose()

    return room_name


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/call", response_model=CallResponse)
async def trigger_call(body: CallRequest):
    """
    Trigger an outbound call to the given phone number.

    Body:
        phone_number  – E.164 format, e.g. "+447311129945"
        agent_name    – (optional) LiveKit agent to dispatch, default "test-agen"

    Example:
        curl -X POST http://localhost:8000/call \\
             -H "Content-Type: application/json" \\
             -d '{"phone_number": "+447311129945"}'
    """
    phone = body.phone_number.strip()

    if not phone.startswith("+"):
        raise HTTPException(
            status_code=400,
            detail="phone_number must be in E.164 format, e.g. +447311129945",
        )

    try:
        room = await initiate_call(phone, body.agent_name)
        return CallResponse(success=True, room=room, message=f"Call initiated to {phone}")
    except Exception as e:
        logger.error(f"Failed to initiate call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
