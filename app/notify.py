import json
import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
NOTIFIED_JSON = STATE_DIR / "notified.json"

COOLDOWN_HOURS = 6
AI_ALERT_THRESHOLD = 0.60
AI_ALERT_TARGETS = {"amber_3h", "red_3h", "tc3_6h"}

AI_ALERT_TITLES = {
    "amber_3h": "Amber Rainstorm possible within 3h",
    "red_3h": "Red Rainstorm possible within 3h",
    "tc3_6h": "Typhoon Signal No. 3+ possible within 6h",
}


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_notified():
    if not NOTIFIED_JSON.exists():
        return {}
    with open(NOTIFIED_JSON, encoding="utf-8-sig") as f:
        return json.load(f)


def save_notified(state):
    STATE_DIR.mkdir(exist_ok=True)
    with open(NOTIFIED_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _within_cooldown(state, key, now):
    last = state.get(key)
    if not last:
        return False
    last_ts = datetime.fromisoformat(last)
    return now - last_ts < timedelta(hours=COOLDOWN_HOURS)


def ai_alert_keys(probs, state, now):
    keys = []
    for target in AI_ALERT_TARGETS:
        p = probs.get(target, 0.0)
        if p >= AI_ALERT_THRESHOLD and not _within_cooldown(state, target, now):
            keys.append(target)
    return keys


def send_email(subject, body, dry_run=False):
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_APP_PASSWORD", "")
    to = os.environ.get("NOTIFY_TO", "")
    if not (user and password and to):
        print(f"[notify] skipped (missing SMTP env): {subject}")
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content(body)
    if dry_run:
        print(f"[notify] dry-run email: {subject}")
        return True
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
        print(f"[notify] sent: {subject}")
        return True
    except Exception as e:
        print(f"[notify] email failed (data still saved): {e}")
        return False

