#!/usr/bin/env python3
"""
Stale File Handle / Mount Point Recovery Monitor

Monitors a path (such as a GPFS, NFS, or Lustre network mount) that is currently
offline or throwing "Stale file handle". Periodically checks the path until the
filesystem comes back online, then sends an email notification to the user and exits.

Usage Examples:
    # Positional arguments: <path> <interval> <email>
    python scripts/misc/check_stale_mount.py /gpfs/mnt/data 60 user@example.com

    # Named arguments with time units (e.g. 30s, 5m, 1h):
    python scripts/misc/check_stale_mount.py --path /gpfs/mnt/data --interval 5m --email user@example.com

    # Test email sending before starting monitoring:
    python scripts/misc/check_stale_mount.py --test-email --email user@example.com

    # Probe path once and exit (exit code 0 if online, 1 if offline):
    python scripts/misc/check_stale_mount.py /gpfs/mnt/data --run-once
"""

# pylint: disable=too-many-arguments,too-many-branches,too-many-return-statements

import argparse
from datetime import datetime
import email.utils
from email.mime.text import MIMEText
import os
import re
import shutil
import smtplib
import socket
import subprocess
import sys
import time


def parse_interval(val: str) -> float:
    """Parses time intervals like '60', '30s', '5m', '1h', '0.5h' into seconds."""
    val = val.strip()
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)$", val)
    if not match:
        raise ValueError(
            f"Invalid interval format: '{val}'. Expected number of seconds (e.g. 60) or with unit (e.g. '30s', '5m', '1h')"
        )
    num = float(match.group(1))
    unit = match.group(2).lower()
    if unit in ("", "s", "sec", "second", "seconds"):
        return num
    if unit in ("m", "min", "minute", "minutes"):
        return num * 60.0
    if unit in ("h", "hr", "hour", "hours"):
        return num * 3600.0
    if unit in ("d", "day", "days"):
        return num * 86400.0
    raise ValueError(f"Unknown time unit '{unit}' in interval: '{val}'")


def format_duration(seconds: float) -> str:
    """Formats seconds into human-readable duration like '1h 23m 45s'."""
    sec_int = int(seconds)
    hours, remainder = divmod(sec_int, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def probe_path(path: str, timeout: float = 15.0) -> tuple[bool, str]:
    """
    Probes the path using isolated subprocesses with timeouts.
    This protects against hanging indefinitely on dead/unresponsive network mounts (GPFS, NFS, etc.).

    Returns:
        (is_online, status_description)
    """
    # 1. Run 'ls -ld <path>' to check stat and metadata access
    try:
        res_ls = subprocess.run(
            ["ls", "-ld", path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if res_ls.returncode != 0:
            err = res_ls.stderr.strip() or res_ls.stdout.strip() or f"exit code {res_ls.returncode}"
            # Extract relevant error message
            lines = [ln.strip() for ln in err.splitlines() if ln.strip()]
            return False, lines[-1] if lines else err
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout:.1f}s ('ls -ld' hung - mount unresponsive)"
    except Exception as e:
        return False, f"Subprocess error during 'ls -ld': {e}"

    # 2. Check reading directory / file contents
    py_check_code = (
        "import sys, os\n"
        "p = sys.argv[1]\n"
        "st = os.stat(p)\n"
        "if os.path.isdir(p):\n"
        "    with os.scandir(p) as it:\n"
        "        for _ in zip(range(5), it): pass\n"
        "elif os.path.isfile(p):\n"
        "    with open(p, 'rb') as f:\n"
        "        f.read(64)\n"
    )
    try:
        res_py = subprocess.run(
            [sys.executable, "-c", py_check_code, path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if res_py.returncode != 0:
            err = res_py.stderr.strip() or res_py.stdout.strip() or f"exit code {res_py.returncode}"
            lines = [ln.strip() for ln in err.splitlines() if ln.strip()]
            return False, lines[-1] if lines else err
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout:.1f}s (reading path hung - mount unresponsive)"
    except Exception as e:
        return False, f"Python check failed: {e}"

    return True, "Accessible and responding normally"


def send_email(
    to_email: str,
    subject: str,
    body: str,
    sender: str | None = None,
    smtp_server: str | None = None,
    smtp_port: int | None = None,
    smtp_user: str | None = None,
    smtp_password: str | None = None,
    smtp_tls: bool = False,
    smtp_ssl: bool = False,
    mail_cmd: str | None = None,
) -> tuple[bool, str]:
    """
    Sends an email using the best available mechanism:
    1. Specified mail command (e.g. mail, mailx, sendmail)
    2. Configured SMTP server
    3. Auto-detected local mail command (mailx, mail, /usr/sbin/sendmail)
    4. Localhost SMTP (localhost:25)

    Returns:
        (success, message)
    """
    hostname = socket.gethostname()
    if not sender:
        username = os.environ.get("USER", "root")
        sender = f"{username}@{hostname}"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    msg["Date"] = email.utils.formatdate(localtime=True)

    errors = []

    # Method 1: If mail_cmd is explicitly passed
    if mail_cmd:
        try:
            cmd_name = os.path.basename(mail_cmd)
            if cmd_name in ("mail", "mailx"):
                proc = subprocess.run(
                    [mail_cmd, "-s", subject, to_email],
                    input=body,
                    text=True,
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
            elif "sendmail" in cmd_name:
                proc = subprocess.run(
                    [mail_cmd, "-t", "-oi"],
                    input=msg.as_string(),
                    text=True,
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
            else:
                proc = subprocess.run(
                    [mail_cmd, to_email],
                    input=body,
                    text=True,
                    capture_output=True,
                    timeout=15,
                    check=False,
                )

            if proc.returncode == 0:
                return True, f"Sent via command '{mail_cmd}'"
            err = proc.stderr.strip() or f"exit code {proc.returncode}"
            return False, f"Command '{mail_cmd}' failed: {err}"
        except Exception as e:
            return False, f"Failed executing '{mail_cmd}': {e}"

    # Method 2: If SMTP server is explicitly passed
    if smtp_server:
        port = smtp_port or (465 if smtp_ssl else (587 if smtp_tls else 25))
        password = smtp_password or os.environ.get("SMTP_PASSWORD", "")
        try:
            if smtp_ssl:
                server = smtplib.SMTP_SSL(smtp_server, port, timeout=15)
            else:
                server = smtplib.SMTP(smtp_server, port, timeout=15)

            if smtp_tls and not smtp_ssl:
                server.starttls()

            if smtp_user:
                server.login(smtp_user, password)

            server.send_message(msg)
            server.quit()
            return True, f"Sent via SMTP server {smtp_server}:{port}"
        except Exception as e:
            return False, f"SMTP delivery to {smtp_server}:{port} failed: {e}"

    # Method 3: Auto-detect local system tools (mailx / mail)
    for tool in ("mailx", "mail"):
        tool_path = shutil.which(tool)
        if tool_path:
            try:
                proc = subprocess.run(
                    [tool_path, "-s", subject, to_email],
                    input=body,
                    text=True,
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
                if proc.returncode == 0:
                    return True, f"Sent via '{tool_path}'"
                errors.append(f"{tool} failed: {proc.stderr.strip()}")
            except Exception as e:
                errors.append(f"{tool} error: {e}")

    # Method 4: Auto-detect sendmail binary
    sendmail_paths = [
        shutil.which("sendmail"),
        "/usr/sbin/sendmail",
        "/usr/lib/sendmail",
    ]
    for s_path in sendmail_paths:
        if s_path and os.path.exists(s_path):
            try:
                proc = subprocess.run(
                    [s_path, "-t", "-oi"],
                    input=msg.as_string(),
                    text=True,
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
                if proc.returncode == 0:
                    return True, f"Sent via '{s_path}'"
                errors.append(f"{s_path} failed: {proc.stderr.strip()}")
            except Exception as e:
                errors.append(f"{s_path} error: {e}")
            break

    # Method 5: Attempt localhost SMTP (127.0.0.1:25)
    try:
        server = smtplib.SMTP("127.0.0.1", 25, timeout=10)
        server.send_message(msg)
        server.quit()
        return True, "Sent via local SMTP relay (127.0.0.1:25)"
    except Exception as e:
        errors.append(f"Localhost SMTP (127.0.0.1:25) failed: {e}")

    summary = "; ".join(errors) if errors else "No mail utility or SMTP server found."
    return False, summary


def generate_recovery_report(path: str, start_time: datetime, check_count: int) -> str:
    """Builds a clear, diagnostic recovery report email."""
    now = datetime.now()
    duration_str = format_duration((now - start_time).total_seconds())
    hostname = socket.gethostname()
    try:
        fqdn = socket.getfqdn()
    except Exception:
        fqdn = hostname

    # Verification sample
    ls_info = ""
    try:
        res = subprocess.run(["ls", "-ld", path], capture_output=True, text=True, timeout=10, check=False)
        ls_info = res.stdout.strip()
    except Exception as e:
        ls_info = f"<error retrieving ls -ld: {e}>"

    sample_items = ""
    if os.path.isdir(path):
        try:
            res = subprocess.run(["ls", "-lh", path], capture_output=True, text=True, timeout=10, check=False)
            lines = res.stdout.splitlines()[:15]
            if lines:
                sample_items = "\nSample directory listing:\n" + "\n".join(lines)
                if len(res.stdout.splitlines()) > 15:
                    sample_items += "\n... [truncated]"
        except Exception:
            pass

    report = f"""Hello,

This is an automated alert from {hostname}.

The monitored filesystem path is now ONLINE and accessible:
    Path: {path}

Recovery Details:
----------------------------------------------------------------------
Host:                {fqdn} ({hostname})
Recovery Time:       {now.strftime('%Y-%m-%d %H:%M:%S %Z')}
Monitoring Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}
Total Downtime/Wait: {duration_str}
Checks Performed:    {check_count}

Verification Status:
----------------------------------------------------------------------
$ ls -ld {path}
{ls_info}
{sample_items}

----------------------------------------------------------------------
Notification generated by scripts/misc/check_stale_mount.py
"""
    return report


def build_parser() -> argparse.ArgumentParser:
    """Constructs and returns the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Monitor a stale mount / path until it recovers, then email the user and exit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s /gpfs/mnt/data 60 user@bnl.gov
  %(prog)s --path /gpfs/mnt/data --interval 5m --email user@bnl.gov
  %(prog)s --test-email --email user@bnl.gov
  %(prog)s /gpfs/mnt/data --run-once
""",
    )

    # Positional arguments (can be supplied without flags)
    parser.add_argument(
        "pos_path",
        nargs="?",
        default=None,
        metavar="PATH",
        help="Target filesystem path to monitor (if not passed via --path)",
    )
    parser.add_argument(
        "pos_interval",
        nargs="?",
        default=None,
        metavar="INTERVAL",
        help="Polling interval, e.g. 60, 30s, 5m, 1h (default: 60s)",
    )
    parser.add_argument(
        "pos_email",
        nargs="?",
        default=None,
        metavar="EMAIL",
        help="Recipient email address to notify on recovery (if not passed via --email)",
    )

    # Flag equivalents
    parser.add_argument("-p", "--path", dest="flag_path", default=None, metavar="PATH", help="Target filesystem path to monitor")
    parser.add_argument("-i", "--interval", dest="flag_interval", default=None, metavar="INTERVAL", help="Polling interval (default: 60s)")
    parser.add_argument("-e", "--email", dest="flag_email", default=None, metavar="EMAIL", help="Recipient email address")
    parser.add_argument("-s", "--subject", default=None, help="Custom email subject")
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=15.0,
        help="Timeout in seconds for each path probe to avoid hanging on dead mounts (default: 15.0s)",
    )
    parser.add_argument(
        "--test-email",
        action="store_true",
        help="Send a test email immediately to verify email settings, then exit",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Probe path once and exit immediately (exit 0 if online, 1 if offline)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode: suppress periodic check log messages",
    )

    # Email configuration group
    mail_grp = parser.add_argument_group("Email / SMTP Configuration Options")
    mail_grp.add_argument(
        "--mail-cmd",
        default=None,
        help="Force specific local mail binary (e.g. 'mailx', 'mail', or '/usr/sbin/sendmail')",
    )
    mail_grp.add_argument(
        "--smtp-server",
        default=None,
        help="SMTP server hostname or IP (e.g. 'localhost', 'smtp.bnl.gov')",
    )
    mail_grp.add_argument(
        "--smtp-port",
        type=int,
        default=None,
        help="SMTP server port (defaults: 25 plain, 587 TLS, 465 SSL)",
    )
    mail_grp.add_argument("--smtp-user", default=None, help="SMTP username for authentication")
    mail_grp.add_argument(
        "--smtp-password",
        default=None,
        help="SMTP password (or set via SMTP_PASSWORD environment variable)",
    )
    mail_grp.add_argument("--smtp-tls", action="store_true", help="Use STARTTLS for SMTP")
    mail_grp.add_argument("--smtp-ssl", action="store_true", help="Use SSL/TLS for SMTP")
    mail_grp.add_argument("--sender", default=None, help="Sender email address (default: <user>@<hostname>)")

    return parser


def main() -> int:
    """Main CLI entry point for monitoring stale mount points."""
    parser = build_parser()
    args = parser.parse_args()

    # Resolve arguments (flags override positional)
    path = args.flag_path or args.pos_path
    raw_interval = args.flag_interval or args.pos_interval or "60s"
    email_addr = args.flag_email or args.pos_email

    # Email parameters dict for convenience
    mail_kwargs = {
        "sender": args.sender,
        "smtp_server": args.smtp_server,
        "smtp_port": args.smtp_port,
        "smtp_user": args.smtp_user,
        "smtp_password": args.smtp_password,
        "smtp_tls": args.smtp_tls,
        "smtp_ssl": args.smtp_ssl,
        "mail_cmd": args.mail_cmd,
    }

    # Handle test-email mode
    if args.test_email:
        if not email_addr:
            parser.error("Recipient email is required for --test-email. Specify via argument or --email.")
        print(f"Sending test email to {email_addr}...")
        test_subject = f"[Test] Filesystem Monitor Email Test from {socket.gethostname()}"
        test_body = (
            f"Hello,\n\n"
            f"This is a test notification from check_stale_mount.py on host {socket.gethostname()}.\n"
            f"If you received this message, your email configuration is working properly!\n\n"
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        ok, msg = send_email(to_email=email_addr, subject=test_subject, body=test_body, **mail_kwargs)
        if ok:
            print(f"[SUCCESS] Test email sent successfully: {msg}")
            return 0
        print(f"[FAILED] Could not send test email: {msg}", file=sys.stderr)
        return 1

    # For monitoring or run-once, path is required
    if not path:
        parser.error("Path is required. Specify as first positional argument or via --path.")

    # Parse interval
    try:
        interval = parse_interval(raw_interval)
        if interval <= 0:
            raise ValueError("Interval must be positive")
    except ValueError as e:
        parser.error(str(e))

    # For monitoring loop, email is required
    if not args.run_once and not email_addr:
        parser.error("Recipient email is required. Specify as third positional argument or via --email.")

    # Handle run-once mode
    if args.run_once:
        print(f"Probing path: {path} (timeout: {args.timeout}s)...")
        online, status_desc = probe_path(path, timeout=args.timeout)
        if online:
            print(f"[ONLINE] {status_desc}")
            return 0
        print(f"[OFFLINE] {status_desc}")
        return 1

    # Standard monitoring mode
    start_time = datetime.now()
    hostname = socket.gethostname()
    print("=" * 70)
    print("STALE MOUNT RECOVERY MONITOR")
    print("=" * 70)
    print(f"  Target Path:   {path}")
    print(f"  Interval:      {interval}s ({raw_interval})")
    print(f"  Notification:  {email_addr}")
    print(f"  Probe Timeout: {args.timeout}s")
    print(f"  Host:          {hostname}")
    print(f"  Started:       {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Initial check
    initial_online, initial_status = probe_path(path, timeout=args.timeout)
    if initial_online:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Target path is ALREADY ONLINE: {initial_status}")
        subject = args.subject or f"[Alert] Filesystem Online: {path} on {hostname}"
        body = generate_recovery_report(path, start_time, 1)
        print(f"Sending notification email to {email_addr}...")
        ok, msg = send_email(to_email=email_addr, subject=subject, body=body, **mail_kwargs)
        if ok:
            print(f"[SUCCESS] Email delivered: {msg}")
        else:
            print(f"[WARNING] Email delivery failed: {msg}", file=sys.stderr)
        return 0

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initial state: OFFLINE ({initial_status})")
    print("Waiting for filesystem to come back online (Press Ctrl+C to stop)...")

    check_count = 0
    try:
        while True:
            time.sleep(interval)
            check_count += 1
            now = datetime.now()
            online, status_desc = probe_path(path, timeout=args.timeout)

            if online:
                # Issue went away!
                print("\n" + "=" * 70)
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS: Filesystem is back ONLINE!")
                print(f"Path '{path}' is now accessible: {status_desc}")
                print("=" * 70)

                subject = args.subject or f"[Resolved] Filesystem back online: {path} on {hostname}"
                body = generate_recovery_report(path, start_time, check_count)

                print(f"Sending notification email to {email_addr}...")
                ok, msg = send_email(to_email=email_addr, subject=subject, body=body, **mail_kwargs)
                if ok:
                    print(f"[SUCCESS] Email delivered successfully: {msg}")
                else:
                    print(f"[WARNING] Failed to deliver email: {msg}", file=sys.stderr)
                    print("=" * 70)
                    print("Report content was:")
                    print(body)
                    print("=" * 70)

                print("Task complete. Exiting.")
                return 0

            # Still offline
            if not args.quiet:
                elapsed = format_duration((now - start_time).total_seconds())
                timestamp = now.strftime("%H:%M:%S")
                print(f"[{timestamp}] Check #{check_count} (Elapsed: {elapsed}): Still offline ({status_desc})")

    except KeyboardInterrupt:
        total_time = format_duration((datetime.now() - start_time).total_seconds())
        print(f"\nMonitoring stopped by user after {check_count} checks ({total_time}). Exiting.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
