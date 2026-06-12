import asyncio
import os
import uuid
from dotenv import load_dotenv
from livekit import api

load_dotenv(".env.local")

async def make_outbound_call(to_number: str):
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    trunk_id = os.getenv("LIVEKIT_TRUNK_ID")

    if not all([url, api_key, api_secret, trunk_id]):
        print("Error: Missing LiveKit credentials in .env.local")
        return

    api_url = url.replace("wss://", "https://").replace("ws://", "http://")
    lkapi = api.LiveKitAPI(api_url, api_key, api_secret)

    room_name = f"outbound-{uuid.uuid4().hex[:8]}"

    print(f"Initiating call to {to_number}...")
    print(f"Room: {room_name}")

    try:
        # Step 1: Dial the phone number via SIP (this makes the phone ring)
        await lkapi.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=trunk_id,
                sip_call_to=to_number,
                participant_identity=f"phone_{to_number.strip('+')}",
                participant_name="Customer",
                room_name=room_name,
            )
        )
        print(f"✅ Phone call initiated to {to_number}")

        # Step 2: Dispatch Elliot (test-agen) to the same room
        # This tells LiveKit which agent to use for this specific call
        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="test-agen",
                room=room_name,
            )
        )
        print(f"✅ Elliot (test-agen) dispatched to room: {room_name}")

    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        await lkapi.aclose()


if __name__ == "__main__":
    TARGET_NUMBER = "+447311129945"
    asyncio.run(make_outbound_call(TARGET_NUMBER))
