# ──────────────────────────────────────────────────────────────
# email_sender.py
# ──────────────────────────────────────────────────────────────
"""
A small, self‑contained library that sends a pre‑designed HTML e‑mail
for the “LCAS Winter Gathering – Santa Pose AR Face”.

Usage:

    from email_sender import send_santa_email

    send_santa_email("ODeSilva@lincoln.ac.uk")

The script expects a `secrets.txt` file in the same directory:

    # secrets.txt
    SMTP_HOST   = smtp.example.com
    SMTP_PORT   = 465
    SMTP_USER   = your_email@example.com
    SMTP_PASS   = your_password
    FROM_ADDR   = your_email@example.com    # optional – defaults to SMTP_USER
"""

from __future__ import annotations

import ssl
import smtplib
from pathlib import Path
from email.message import EmailMessage
from email.utils import make_msgid, formatdate
from typing import Dict, Optional

# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
SECRETS_FILE: Path = Path("secrets.txt")          # Change if the file lives elsewhere
INLINE_IMAGE_PATH: Path = Path("logo.png")        # Optional – leave missing for no image
DEFAULT_DOMAIN: str = "DOMAIN_HERE"             # Used for make_msgid() if you want a custom domain

# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def _read_secrets(path: Path) -> Dict[str, str]:
    """Read a simple key=value file (comments start with '#')."""
    secrets: Dict[str, str] = {}
    if not path.is_file():
        raise FileNotFoundError(f"Secrets file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        secrets[key.strip().upper()] = val.strip()
    return secrets


def _create_email(subject: str, to: str, from_addr: str, img_cid: Optional[str]) -> EmailMessage:
    """Build the EmailMessage object with HTML body and optional inline image."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to
    # ---- NEW: required "Date" header ---------------------------------
    msg["Date"] = formatdate(localtime=True)      # RFC‑5322 compliant timestamp
    # (Optional) add a Message‑ID for better tracking
    msg["Message-ID"] = make_msgid(domain=DEFAULT_DOMAIN)


    # --- HTML body ----------------------------------------------------------
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background-color: #f0f0f0; margin: 0; padding: 0; }}
            .container {{ max-width: 600px; margin: 30px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.1); padding: 20px; }}
            h1 {{ color: #d00; text-align: center; }}
            p {{ line-height: 1.6; color: #333; }}
            .cta {{ display: inline-block; margin-top: 20px; padding: 12px 24px; background: #d00; color: #fff; text-decoration: none; border-radius: 4px; font-weight: bold; }}
            .footer {{ margin-top: 30px; font-size: 12px; color: #777; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            {f'<img src="cid:{img_cid}" alt="LCAS Logo" style="display:block; margin:auto; width:120px;">' if img_cid else ''}
            <h1>Santa Pose AR Face</h1>
            <p>Dear LCAS community,</p>
            <p>
                We’re thrilled to announce the <strong>Santa Pose AR Face</strong> for the upcoming
                <em>Winter Gathering</em>. This interactive AR experience lets you
                pose as Santa Claus and capture festive memories in a whole new way!
            </p>
            <p>
                <strong>Date:</strong> December 15, 2025<br>
                <strong>Time:</strong> 5:00 PM – 8:00 PM<br>
                <strong>Location:</strong> Main Auditorium & Online Stream
            </p>
            <p>
                <a href="https://example.com/santa-ar" class="cta">Get the AR App Now</a>
            </p>
            <p>
                We look forward to seeing you there and sharing some holiday cheer.
            </p>
            <div class="footer">
                &copy; 2025 LCAS – All rights reserved<br>
                <a href="https://lcas.example.com">LCAS Website</a>
            </div>
        </div>
    </body>
    </html>
    """
    msg.add_alternative(html, subtype="html")

    # --- Inline image -------------------------------------------------------
    if img_cid and INLINE_IMAGE_PATH.is_file():
        with INLINE_IMAGE_PATH.open("rb") as f:
            img_data = f.read()
        maintype, subtype = "image", INLINE_IMAGE_PATH.suffix.lstrip(".")
        msg.get_payload()[0].add_related(
            img_data,
            maintype=maintype,
            subtype=subtype,
            cid=img_cid,
            filename=INLINE_IMAGE_PATH.name,
        )
    return msg


# --------------------------------------------------------------------------- #
# PUBLIC API
# --------------------------------------------------------------------------- #
def send_santa_email(recipient: str) -> None:
    """
    Send the “Santa Pose AR Face” e‑mail to *recipient*.

    Parameters
    ----------
    recipient : str
        The e‑mail address to send the message to.

    Raises
    ------
    FileNotFoundError
        If the secrets file cannot be located.
    smtplib.SMTPException
        Any SMTP‑related error (authentication, connection, etc.).
    """
    # Load secrets once – the module will cache the result
    secrets = _read_secrets(SECRETS_FILE)

    smtp_host = secrets["SMTP_HOST"]
    smtp_port = int(secrets.get("SMTP_PORT", 465))
    smtp_user = secrets["SMTP_USER"]
    smtp_pass = secrets["SMTP_PASS"]
    from_addr = secrets.get("FROM_ADDR", smtp_user)

    # Prepare the message
    img_cid = make_msgid(domain=DEFAULT_DOMAIN)[1:-1]  # strip < and >
    msg = _create_email(
        subject="LCAS Winter Gathering – Santa Pose AR Face",
        to=recipient,
        from_addr=from_addr,
        img_cid=img_cid,
    )

    # Send it via a secure TLS connection
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
            print(f"✅ Email sent to {recipient}")
    except smtplib.SMTPException as exc:
        raise RuntimeError(f"Failed to send e‑mail: {exc}") from exc


# --------------------------------------------------------------------------- #
# Demo / self‑test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Replace with your own test address
    test_recipient = "foo.bar@baz.com"
    try:
        send_santa_email(test_recipient)
    except Exception as e:
        print(f"❌ {e}")