"""
claude_cli.py — drop-in replacement for anthropic.Anthropic
Uses `claude -p` subprocess (Pro subscription), no API key required.
"""
import subprocess


class _Content:
    def __init__(self, text: str):
        self.text = text


class _Response:
    def __init__(self, text: str):
        self.content = [_Content(text)]


class _Messages:
    def create(self, *, model="claude-sonnet-4-6", max_tokens=1000,
               messages=None, system=None, **kwargs):
        prompt = ""
        if messages:
            for m in messages:
                if m.get("role") == "user":
                    c = m.get("content", "")
                    if isinstance(c, str):
                        prompt = c
                    elif isinstance(c, list):
                        prompt = " ".join(
                            p.get("text", "") for p in c
                            if isinstance(p, dict) and p.get("type") == "text"
                        )

        full_prompt = f"{system}\n\nHuman: {prompt}" if system else prompt

        cmd = ["claude", "-p", "-", "--model", model]
        result = subprocess.run(cmd, input=full_prompt, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(f"claude CLI error: {result.stderr.strip()}")

        return _Response(result.stdout.strip())


class Anthropic:
    """Drop-in replacement for anthropic.Anthropic using the claude CLI (Pro subscription)."""

    def __init__(self, api_key=None, **kwargs):
        self.messages = _Messages()
