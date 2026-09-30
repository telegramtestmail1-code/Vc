LIVE TELEGRAM VC AUDIO BOOSTER

Environment:
TG_API_ID
TG_API_HASH
TG_SESSION (optional, default vc_booster)
GAIN_DB (optional, default 12)

Commands:
.setsource <chat_id/@username>
.settarget <chat_id/@username>
.boost <0-30>
.vcstart
.vcstop
.vcstatus
.vchelp

The source VC is captured as raw PCM frames. Incoming speaker frames
are mixed, gain is applied, and the resulting audio is sent to the
target VC.

First run:
python vc_booster.py

Telethon may ask for your phone number, login code, and 2FA password.
Keep the generated .session file private.

This is a Linux-host oriented build. Test the PyTgCalls wheel on your
specific hosting platform before using it continuously.
