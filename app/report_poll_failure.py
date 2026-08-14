"""Record a failed poll for the static dashboard and send a throttled maintainer alert."""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from health import failure_status, load_health, publish
from notify import send_email

FAILURE_COOLDOWN_HOURS = 6


def parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def should_notify(previous, now):
    last = parse_time(previous.get("failure_notified_at"))
    return last is None or now - last >= timedelta(hours=FAILURE_COOLDOWN_HOURS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", default="poll workflow failed")
    args = parser.parse_args()

    previous = load_health()
    status = failure_status(args.reason)
    now = datetime.now(timezone.utc)
    if should_notify(previous, now) and os.environ.get("DISABLE_EMAIL") != "1":
        sent = send_email(
            "[Testing] Poll failed",
            "Testing could not update its weather snapshot.\n\n"
            f"Reason: {args.reason}\n"
            f"Last successful snapshot: {status.get('last_success_at') or 'unknown'}\n\n"
            "The dashboard will mark data as stale until the next successful poll.",
        )
        if sent:
            status["failure_notified_at"] = now.isoformat(timespec="seconds")
        elif previous.get("failure_notified_at"):
            status["failure_notified_at"] = previous["failure_notified_at"]
    elif previous.get("failure_notified_at"):
        status["failure_notified_at"] = previous["failure_notified_at"]

    publish(status)
    print(f"[health] recorded failed poll: {status['summary']}")


if __name__ == "__main__":
    main()
