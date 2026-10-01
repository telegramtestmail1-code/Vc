import os
import asyncio
import logging
from typing import Optional, Set

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

# Optional: pre-approve users via env var, e.g. "12345,67890"
EXTRA_APPROVED = os.environ.get("APPROVED_USERS", "")

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

# ---------------- Approval system ----------------
owner_id: Optional[int] = None
approved_users: Set[int] = set()
approval_lock: Optional[asyncio.Lock] = None


def _load_extra_approved() -> Set[int]:
    ids: Set[int] = set()
    if not EXTRA_APPROVED.strip():
        return ids
    for part in EXTRA_APPROVED.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            logging.warning("Ignoring invalid APPROVED_USERS entry: %s", part)
    return ids


def is_authorized(user_id: Optional[int]) -> bool:
    if user_id is None:
        return False
    if owner_id is not None and user_id == owner_id:
        return True
    return user_id in approved_users


async def resolve_user_id(value: str) -> Optional[int]:
    """Resolve a username / id string to a numeric Telegram user ID."""
    value = value.strip()
    if not value:
        return None

    # Strip leading @ if present.
    if value.startswith("@"):
        value = value[1:]

    # Try as raw integer ID first.
    try:
        return int(value)
    except ValueError:
        pass

    # Resolve as username / entity.
    if client is None:
        return None

    try:
        entity = await client.get_entity(value)
    except Exception as exc:
        logging.warning("Could not resolve user %s: %s", value, exc)
        return None

    # Only accept real users (not chats/channels).
    uid = getattr(entity, "id", None)
    if uid is None:
        return None
    return int(uid)


# ---------------- Audio processing ----------------
def apply_gain(pcm: bytes, db: float) -> bytes:
    """
    Apply gain to signed 16-bit PCM with soft-knee limiting.

    Uses a tanh-based soft limiter so high boosts (up to +50 dB) sound
    loud but clean, instead of harshly clipped.
    """
    if not pcm:
        return pcm

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    gain = 10.0 ** (db / 20.0)
    samples *= gain

    # Soft limiter.
    samples = np.tanh(samples)

    samples = np.clip(samples * 32767.0, -32768, 32767)
    return samples.astype(np.int16).tobytes()


async def open_raw_call(chat_id: int):
    if calls is None:
        raise RuntimeError("PyTgCalls is not initialized.")

    await calls.play(
        chat_id,
        MediaStream(ExternalMedia.AUDIO, AUDIO),
    )
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

    # ---------------- Audio bridge ----------------
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

    # ---------------- Approval: owner-only commands ----------------
    @client.on(events.NewMessage(pattern=r"^\.approve(?:\s+.+)?$"))
    async def approve_cmd(event):
        sender_id = event.sender_id

        # Owner-only command. Silent for everyone else.
        if owner_id is None or sender_id != owner_id:
            return

        parts = event.raw_text.split(maxsplit=1)
        if len(parts) != 2:
            return await event.reply(
                "Usage: .approve <user_id/@username> "
                "or reply to a user's message."
            )

        # Reply-based approve.
        if event.is_reply:
            reply = await event.get_reply_message()
            if reply and reply.sender_id:
                target_id = int(reply.sender_id)
            else:
                target_id = await resolve_user_id(parts[1])
        else:
            target_id = await resolve_user_id(parts[1])

        if target_id is None:
            return await event.reply("❌ Could not resolve that user.")

        if target_id == owner_id:
            return await event.reply("ℹ️ That's you (the owner).")

        async with approval_lock:
            approved_users.add(target_id)

        await event.reply(
            f"✅ Approved user: `{target_id}`\n"
            f"Total approved: {len(approved_users)}"
        )

    @client.on(events.NewMessage(pattern=r"^\.disapprove(?:\s+.+)?$"))
    async def disapprove_cmd(event):
        sender_id = event.sender_id
        if owner_id is None or sender_id != owner_id:
            return

        parts = event.raw_text.split(maxsplit=1)
        if len(parts) != 2:
            return await event.reply(
                "Usage: .disapprove <user_id/@username> "
                "or reply to a user's message."
            )

        if event.is_reply:
            reply = await event.get_reply_message()
            if reply and reply.sender_id:
                target_id = int(reply.sender_id)
            else:
                target_id = await resolve_user_id(parts[1])
        else:
            target_id = await resolve_user_id(parts[1])

        if target_id is None:
            return await event.reply("❌ Could not resolve that user.")

        async with approval_lock:
            approved_users.discard(target_id)

        await event.reply(
            f"🚫 Disapproved user: `{target_id}`\n"
            f"Total approved: {len(approved_users)}"
        )

    @client.on(events.NewMessage(pattern=r"^\.approvedlist$"))
    async def approved_list(event):
        sender_id = event.sender_id
        if owner_id is None or sender_id != owner_id:
            return

        if not approved_users:
            return await event.reply("📋 No approved users yet.")

        lines = "\n".join(f"• `{uid}`" for uid in sorted(approved_users))
        await event.reply(f"📋 Approved users ({len(approved_users)}):\n{lines}")

    # ---------------- Bridge control commands ----------------
    @client.on(events.NewMessage(pattern=r"^\.setsource(?:\s+.+)?$"))
    async def set_source(event):
        global source_chat

        if not is_authorized(event.sender_id):
            return  # silent

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

        if not is_authorized(event.sender_id):
            return  # silent

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

        if not is_authorized(event.sender_id):
            return  # silent

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
        await event.reply(f"🔊 Live boost set to +{gain_db:g} dB")

    @client.on(events.NewMessage(pattern=r"^\.vcstart$"))
    async def vc_start(event):
        if not is_authorized(event.sender_id):
            return  # silent

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
        if not is_authorized(event.sender_id):
            return  # silent

        await stop_bridge()
        await event.reply("🔴 Live VC boost stopped.")

    @client.on(events.NewMessage(pattern=r"^\.vcstatus$"))
    async def vc_status(event):
        if not is_authorized(event.sender_id):
            return  # silent

        await event.reply(
            "🎙️ VC BOOSTER STATUS\n\n"
            f"Source: {source_chat or 'Not set'}\n"
            f"Target: {target_chat or 'Not set'}\n"
            f"Boost: +{gain_db:g} dB\n"
            f"Status: {'🟢 LIVE' if running else '🔴 OFF'}"
        )

    @client.on(events.NewMessage(pattern=r"^\.vchelp$"))
    async def vc_help(event):
        if not is_authorized(event.sender_id):
            return  # silent

        text = (
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

        if owner_id is not None and event.sender_id == owner_id:
            text += (
                "\n\n👑 OWNER COMMANDS\n"
                ".approve <id/@username> - allow a user\n"
                ".disapprove <id/@username> - revoke access\n"
                ".approvedlist - list approved users"
            )

        await event.reply(text)


async def main():
    global client, calls, start_lock, owner_id, approval_lock

    # This coroutine is already running on WispByte's one event loop.
    # Create BOTH Telegram clients here so they bind to the same loop.
    start_lock = asyncio.Lock()
    approval_lock = asyncio.Lock()

    client = TelegramClient(SESSION, API_ID, API_HASH)
    calls = PyTgCalls(client)

    register_handlers()

    await client.start()

    # PyTgCalls 2.3.3 exposes start() as a coroutine.
    await calls.start()

    me = await client.get_me()
    owner_id = int(me.id)

    # Load pre-approved users from env (optional).
    approved_users.update(_load_extra_approved())

    logging.info(
        "Logged in as %s (owner id=%s)",
        getattr(me, "username", None) or me.id,
        owner_id,
    )
    if approved_users:
        logging.info("Pre-approved users: %s", sorted(approved_users))
    logging.info("VC booster ready.")

    # Keep this exact loop alive for both Telethon and PyTgCalls.
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
