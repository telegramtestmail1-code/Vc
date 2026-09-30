import os
import asyncio
import logging
from typing import Optional

import numpy as np
from telethon import TelegramClient, events
from pytgcalls import PyTgCalls, filters
from pytgcalls.types import Device, Direction, ExternalMedia, MediaStream, RecordStream, StreamFrames
from pytgcalls.types.raw import AudioParameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

API_ID = int(os.environ.get("TG_API_ID", "31157048"))
API_HASH = os.environ.get("TG_API_HASH", "ed0fea589fd64fe5985a373cb4ad9d84")
SESSION = os.environ.get("TG_SESSION", "combined_bot")
DEFAULT_GAIN_DB = float(os.environ.get("GAIN_DB", "12"))

SAMPLE_RATE = 48000
CHANNELS = 2
AUDIO = AudioParameters(bitrate=SAMPLE_RATE, channels=CHANNELS)

if not API_ID or not API_HASH:
    raise RuntimeError("Set TG_API_ID and TG_API_HASH environment variables.")

# IMPORTANT:
# These are created inside main(), after the WispByte event loop exists.
client: Optional[TelegramClient] = None
calls: Optional[PyTgCalls] = None

source_chat: Optional[int] = None
target_chat: Optional[int] = None
gain_db = DEFAULT_GAIN_DB
running = False
start_lock: Optional[asyncio.Lock] = None


def apply_gain(pcm: bytes, db: float) -> bytes:
    """
    Apply gain to signed 16-bit PCM with soft-knee limiting.

    Instead of hard-clipping (which creates harsh square-wave distortion),
    this uses a tanh-based soft limiter. Quiet parts get full linear gain,
    loud parts gently compress instead of clipping. This makes high boost
    values (up to +50 dB) sound loud but still clean.
    """
    if not pcm:
        return pcm

    # Normalize to float in [-1.0, 1.0].
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    # Apply linear gain.
    gain = 10.0 ** (db / 20.0)
    samples *= gain

    # Soft limiter: tanh squashes peaks smoothly toward ±1.
    # tanh(x) ≈ x for small x, and → ±1 for large x.
    samples = np.tanh(samples)

    # Back to int16 range.
    samples = np.clip(samples * 32767.0, -32768, 32767)

    return samples.astype(np.int16).tobytes()


async def open_raw_call(chat_id: int):
    if calls is None:
        raise RuntimeError("PyTgCalls is not initialized.")

    # Output: inject processed PCM into this voice chat.
    await calls.play(
        chat_id,
        MediaStream(ExternalMedia.AUDIO, AUDIO),
    )

    # Input: receive incoming VC audio frames.
    await calls.record(
        chat_id,
        RecordStream(True, AUDIO),
    )


async def start_bridge():
    global running

    if source_chat is None or target_chat is None:
        raise RuntimeError("Set both source and target first.")

    if calls is None:
        raise RuntimeError("PyTgCalls is not initialized.")

    if start_lock is None:
        raise RuntimeError("Start lock is not initialized.")

    async with start_lock:
        if running:
            return

        # Join source first, then target.
        await open_raw_call(source_chat)
        await open_raw_call(target_chat)

        running = True


async def stop_bridge():
    global running

    running = False

    if calls is None:
        return

    for chat_id in (source_chat, target_chat):
        if chat_id is not None:
            try:
                await calls.leave_call(chat_id)
            except Exception:
                pass


async def get_command_chat_id(event):
    if client is None:
        return None

    if event.is_reply:
        reply = await event.get_reply_message()
        if reply:
            return reply.chat_id

    parts = event.raw_text.split(maxsplit=1)
    if len(parts) != 2:
        return None

    value = parts[1].strip()

    try:
        return int(value)
    except ValueError:
        entity = await client.get_entity(value)
        return entity.id


def register_handlers():
    """Register Telethon and PyTgCalls handlers on the same active loop."""

    if client is None or calls is None:
        raise RuntimeError("Clients are not initialized.")

    @calls.on_update(
        filters.stream_frame(Direction.INCOMING, Device.MICROPHONE)
    )
    async def on_audio(_: PyTgCalls, update: StreamFrames):
        global source_chat, target_chat, running

        if not running or target_chat is None:
            return

        if update.chat_id != source_chat:
            return

        if not update.frames:
            return

        logging.info(
            "Incoming VC audio: %d frame(s) from %s",
            len(update.frames),
            update.chat_id,
        )

        arrays = [
            np.frombuffer(frame.frame, dtype=np.int16).astype(np.int32)
            for frame in update.frames
            if frame.frame
        ]

        if not arrays:
            return

        # Mix all incoming speaker frames.
        max_len = max(len(arr) for arr in arrays)
        mixed = np.zeros(max_len, dtype=np.int32)

        for arr in arrays:
            mixed[:len(arr)] += arr

        mixed //= len(arrays)
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)

        try:
            await calls.send_frame(
                target_chat,
                Device.MICROPHONE,
                apply_gain(mixed.tobytes(), gain_db),
            )
        except Exception as exc:
            logging.warning("send_frame failed: %s", exc)

    @client.on(events.NewMessage(pattern=r"^\.setsource(?:\s+.+)?$"))
    async def set_source(event):
        global source_chat

        chat_id = await get_command_chat_id(event)

        if chat_id is None:
            return await event.reply(
                "Usage: .setsource <chat_id/@username> "
                "or reply to a message."
            )

        source_chat = int(chat_id)
        await event.reply(f"✅ Source VC set: {source_chat}")

    @client.on(events.NewMessage(pattern=r"^\.settarget(?:\s+.+)?$"))
    async def set_target(event):
        global target_chat

        chat_id = await get_command_chat_id(event)

        if chat_id is None:
            return await event.reply(
                "Usage: .settarget <chat_id/@username> "
                "or reply to a message."
            )

        target_chat = int(chat_id)
        await event.reply(f"✅ Target VC set: {target_chat}")

    @client.on(events.NewMessage(pattern=r"^\.boost(?:\s+.+)?$"))
    async def set_boost(event):
        global gain_db

        parts = event.raw_text.split(maxsplit=1)

        if len(parts) != 2:
            return await event.reply(
                f"🔊 Current boost: +{gain_db:g} dB\n"
                "Usage: .boost 12"
            )

        try:
            value = float(parts[1])
        except ValueError:
            return await event.reply("Usage: .boost 0-50")

        if not 0 <= value <= 50:
            return await event.reply("⚠️ Boost range: 0 to 50 dB")

        gain_db = value

        await event.reply(
            f"🔊 Live boost set to +{gain_db:g} dB"
        )

    @client.on(events.NewMessage(pattern=r"^\.vcstart$"))
    async def vc_start(event):
        try:
            await start_bridge()

            await event.reply(
                "🟢 LIVE VC BOOST STARTED\n\n"
                f"Source: {source_chat}\n"
                f"Target: {target_chat}\n"
                f"Boost: +{gain_db:g} dB"
            )

        except Exception as exc:
            logging.exception("Start failed")
            await event.reply(f"❌ Start failed:\n{exc}")

    @client.on(events.NewMessage(pattern=r"^\.vcstop$"))
    async def vc_stop(event):
        await stop_bridge()
        await event.reply("🔴 Live VC boost stopped.")

    @client.on(events.NewMessage(pattern=r"^\.vcstatus$"))
    async def vc_status(event):
        await event.reply(
            "🎙️ VC BOOSTER STATUS\n\n"
            f"Source: {source_chat or 'Not set'}\n"
            f"Target: {target_chat or 'Not set'}\n"
            f"Boost: +{gain_db:g} dB\n"
            f"Status: {'🟢 LIVE' if running else '🔴 OFF'}"
        )

    @client.on(events.NewMessage(pattern=r"^\.vchelp$"))
    async def vc_help(event):
        await event.reply(
            "🎙️ LIVE VC BOOSTER\n\n"
            ".setsource <chat_id/@username> - source VC\n"
            ".settarget <chat_id/@username> - target VC\n"
            ".boost <0-50> - live gain\n"
            ".vcstart - start bridge\n"
            ".vcstop - stop bridge\n"
            ".vcstatus - status\n"
            ".vchelp - help\n\n"
            "You may also reply to a group message with "
            ".setsource or .settarget."
        )


async def main():
    global client, calls, start_lock

    # This coroutine is already running on WispByte's one event loop.
    # Create BOTH Telegram clients here so they bind to the same loop.
    start_lock = asyncio.Lock()

    client = TelegramClient(SESSION, API_ID, API_HASH)
    calls = PyTgCalls(client)

    register_handlers()

    await client.start()

    # PyTgCalls 2.3.3 exposes start() as a coroutine.
    await calls.start()

    me = await client.get_me()

    logging.info(
        "Logged in as %s",
        getattr(me, "username", None) or me.id,
    )
    logging.info("VC booster ready.")

    # Keep this exact loop alive for both Telethon and PyTgCalls.
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
