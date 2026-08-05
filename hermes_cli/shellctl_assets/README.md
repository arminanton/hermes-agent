# Hermes shellctl — SSH-layer media & file bridge

Move **images, PDFs, audio (TTS + mic), and any file** between the Hermes host
(the box you SSH into, running the TUI) and **your local machine** — over your
**existing SSH connection**, with no extra tunnel, no ControlMaster requirement,
and no dependency on iTerm2 / a specific terminal / a specific OS.

Works on macOS, Linux, WSL, and any host that runs `ssh` (PuTTY included).
The client is a **single zero-dependency Python 3 file** (stdlib only) — installs
on a locked-down corporate Mac with no admin/sudo and no package manager.

## Why this exists

The Hermes TUI runs on the *remote* host, so its "clipboard", "microphone", and
filesystem are the **remote host's**, not yours. shellctl bridges that gap: a tiny
HTTP listener on your machine, reachable by the remote host through a reverse SSH
forward, so the agent can pull/push bytes to/from *your* machine.

```
Your machine (Mac/Linux/WSL)        existing SSH (any transport)      Hermes host
┌────────────────────────┐                                          ┌──────────────┐
│ hermes-shellctl daemon │  ◄── RemoteForward 127.0.0.1:8765 ──────►│ TUI /get /say │
│  • serves local files  │      (reverse: host reaches your box)     │ hermes-shell- │
│  • mic / speakers / clip│                                          │   bridge      │
└────────────────────────┘                                          └──────────────┘
```

## Install (run on the HERMES host)

```sh
hermes install shellctl --ssh-host <the-host-alias-you-ssh-to>
```

This prints a token + the exact 4 steps. Summary:

1. **Save the client on your machine:**
   ```sh
   ssh <host> 'hermes install shellctl --print-client' > ~/.hermes-shellctl
   chmod +x ~/.hermes-shellctl
   ```
2. **Add ONE block to `~/.ssh/config` on your machine** (covers plain `ssh` and
   tmux-wrapping helpers like `sshp`; **no ControlMaster needed**):
   ```
   Host <host>
       RemoteForward 127.0.0.1:8765 127.0.0.1:8765
   ```
3. **Run the daemon on your machine** (leave it in a tab, or add to Login Items):
   ```sh
   HERMES_SHELLCTL_TOKEN=<token> python3 ~/.hermes-shellctl daemon --port 8765
   ```
4. **SSH in normally.** Verify from the host:
   ```sh
   <bridge> ping     # → {"ok": true, "caps": {...}}
   ```

## Use (in the TUI)

| Command | What it does |
|---|---|
| `/get <local-path>` | Pull a file FROM your machine; auto-attaches it to the turn (image/pdf/any) |
| `/grab` | Pull your machine's **clipboard image** and attach it |
| `/send <host-path>` | Push a Hermes-side file TO your machine and open it locally |
| `/say <text>` | Gateway TTS → **play on your speakers** |
| `/listen [secs]` | **Record your mic** → STT → transcript becomes the next turn |

(`/paste` is unchanged — it's the existing clipboard-attach. shellctl's clipboard
pull is `/grab` to avoid collision.)

## Optional local helpers (better fidelity, not required)

- **Clipboard images:** macOS `pngpaste` (`brew install pngpaste`) or Linux `xclip`.
  Without them, macOS falls back to AppleScript.
- **Mic recording:** macOS `sox` (`brew install sox`) or `ffmpeg`; Linux `arecord`
  (alsa-utils) or `ffmpeg`.
- **Audio playback:** macOS `afplay` (built in); Linux `paplay`/`aplay`/`ffplay`/`mpv`.

`hermes-shellctl daemon` reports which capabilities are available at startup and
via `/ping`.

## Security

- The listener binds **127.0.0.1 only** and is reached solely through your SSH
  reverse-forward — nothing on the network can reach it.
- Every request is gated by a **shared token** (`HERMES_SHELLCTL_TOKEN`),
  generated per-install and stored `0600`.
- 64 MB per-transfer cap.

## Files

- Client (your machine): `~/.hermes-shellctl` (single file, stdlib only)
- Host orchestrator: `<HERMES_HOME>/shellctl/hermes-shellbridge`
- Token + config: `<HERMES_HOME>/shellctl/bridge-token`, `bridge.env` (both `0600`)
- Canonical source (ships with hermes_cli): `hermes_cli/shellctl_assets/`

## Notes

- The gateway slash-command handler lives in `tui_gateway/server.py`
  (`_run_shellbridge_command` + the `slash.exec` interception). It runs in the
  main gateway process, so a **gateway restart** is needed for the `/get` etc.
  commands to become active after install.
- The bridge reuses the dashboard's existing `/api/audio/speak` (TTS) and
  `/api/audio/transcribe` (STT) endpoints, and lands pulled files in the gateway
  images dir so the normal attach pipeline handles them.
