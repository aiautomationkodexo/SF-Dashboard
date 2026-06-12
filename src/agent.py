import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    inference,
)
from livekit.plugins import silero, openai, deepgram, cartesia

load_dotenv(".env.local")
logger = logging.getLogger("outbound-agent")

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="Say hello and talk like a freind. Ask the user if he wants to hear a story. If he says yes, tell him a story. If he says no, say goodbye and end the call.",
        )


@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=cartesia.TTS(),
        vad=ctx.proc.userdata["vad"],
    )

    await session.start(agent=Assistant(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    from livekit.agents import cli
    cli.run_app(server)
