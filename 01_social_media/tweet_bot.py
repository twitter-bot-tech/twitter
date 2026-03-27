#!/usr/bin/env python3
"""
Prediction Markets Daily Tweet Bot
Generates and posts daily tweets (with optional comparison chart) about prediction markets.
Supports weekly polls and weekly analytics reports.
"""
from __future__ import annotations

import os
import re
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add parent directory to path so claude_cli can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates
from dotenv import load_dotenv
from claude_cli import Anthropic
import requests
import tweepy

# Load environment variables from parent .env (primary) then script dir .env
script_dir = Path(__file__).parent
load_dotenv(script_dir.parent / ".env")
load_dotenv(script_dir / ".env", override=False)

# Configure logging
log_file = script_dir / "logs" / "tweet.log"
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger(__name__)

TWEET_MAX_CHARS = 270
TWEET_IDS_FILE = script_dir / "tweet_ids.json"
MIN_COOLDOWN_MINUTES = 30  # minimum gap between same-type tweets

MOONX_CTA = "\n\nTrack it live → bydfi.com/en/moonx/markets/trending"
# CTA is appended after generation; content prompts use reduced char budget
TWEET_CONTENT_MAX = TWEET_MAX_CHARS - len(MOONX_CTA)

SYSTEM_PROMPT = """You are the sharpest prediction markets analyst on Crypto Twitter.
You've studied every major platform — Polymarket, Kalshi, Manifold, PredictIt — and you have strong opinions.
You write tweets that make people stop scrolling and say "wait, is that actually true?"

Your voice is modeled on the best CT accounts:
- OPINIONATED like a founder: take a clear, defensible stance. Never "on one hand / on the other hand."
- DATA-FIRST: lead with the number that changes the frame, then explain why it matters
- CONTRARIAN by default: if the market consensus looks wrong, say so loudly
- ABSURD COMPARISONS to make numbers real: "0.4% — less likely than getting struck by lightning. Twice."
- NEVER promotional — your tweets inform and provoke, they never advertise

HOOK LAW — the first 10 words decide everything:
- Lead with a SPECIFIC NUMBER, a COUNTER-INTUITIVE conclusion, or a STRONG TAKE that demands a reaction
- The hook must make the reader feel: surprise, the urge to argue, or "I never thought about it that way"
- Bad: "Prediction markets are getting more popular..."
- Good: "Polymarket's market cap is now 6x PumpFun's. The market for information is bigger than the market for memes."
- Bad: "Interesting price movement on Polymarket today..."
- Good: "The market just priced in a 0.4% chance. That's less likely than you getting struck by lightning. Twice."
- Bad: "Check out what's happening on Polymarket..."
- Good: "Most prediction market traders don't lose because they predict wrong. They lose because platforms take 5-10% per trade."
- The reader must feel something after the first line."""

TEXT_ONLY_PROMPT = """Write a single tweet about prediction markets from a first-person sharing perspective.

Rotate between these 6 high-performing angles (pick the one that hits hardest today):
1. THE MARKET IS WRONG: Pick a current prediction market price and argue it's miscalibrated
   Example opener: "Polymarket gives this event 15%. I think they're wrong. Here's why..."
2. ABSURD COMPARISON: Use an unexpected comparison to make odds feel real
   Example opener: "0.4% chance. That's less likely than you getting struck by lightning. Twice."
3. COUNTER-INTUITIVE: Start with a conclusion that surprises people
   Example opener: "Most prediction market traders don't lose because they predict wrong. They lose because..."
4. SCALE COMPARISON: Contrast prediction markets to something people know (makes the space feel huge or tiny)
   Example opener: "Polymarket's market cap is now 6x PumpFun's. The market for information > market for memes."
5. RANKING / LIST: Structured comparison that people want to save
   Example opener: "Prediction market platforms ranked by what actually matters:\nLiquidity: Polymarket\nFees: Manifold\nVariety: Kalshi"
6. HOT TAKE + CHANGE MY MIND: Clear, provocative opinion that invites argument
   Example opener: "Prediction markets will replace election polling by 2028. Change my mind."

Style rules:
- First line MUST be a hook — counter-intuitive, absurd comparison, or hot take
- NEVER start with "I've been thinking" or "Interesting observation" — too weak
- Maximum {max_chars} characters (strictly enforced)
- English only
- End with EITHER 1-2 hashtags OR a 1-sentence question that invites replies (not both every time — alternate)
  Question endings: "What's your take?" / "Am I wrong?" / "Which side are you on?" / "Change my mind."
- Use line breaks to separate ideas. Format:
  Line 1: Strong hook
  (blank line)
  Line 2-3: Supporting detail, the "why"
  (blank line)
  Last line: hashtags OR question
- Output only the tweet text""".format(max_chars=TWEET_MAX_CHARS)

COMPARISON_PROMPT = """Generate a prediction market platform comparison tweet WITH chart data.
Write the tweet from a first-person researcher/participant perspective.

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "tweet": "Tweet text here (max {max_chars} chars, first-person sharing voice, use \\n\\n between paragraphs for blank lines, 1-2 hashtags on last line)",
  "chart_title": "Short chart title",
  "metric": "The metric being compared (e.g. Monthly Volume, Active Markets, User Base)",
  "platforms": ["Polymarket", "Kalshi", "Manifold", "PredictIt"],
  "values": [100, 45, 20, 10],
  "unit": "$ Millions"
}}

Tweet style examples:
- "I ran the numbers on prediction market volumes — the gap between platforms is wider than most think..."
- "Been studying how Kalshi vs Polymarket handle liquidity. The difference is striking."

The values should be realistic relative numbers.""".format(
    max_chars=TWEET_MAX_CHARS
)

POLL_PROMPT = """Generate a Twitter poll question about crypto or prediction markets.

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "tweet": "Poll intro text here (max 200 chars, engaging question framing, first-person voice)",
  "options": ["Option A", "Option B", "Option C"],
  "duration_minutes": 1440
}}

Rules:
- 2 to 4 options (Twitter limit)
- Each option max 25 characters
- duration_minutes must be between 5 and 10080 (7 days); use 1440 for 24 hours
- Topics: which platform will dominate, best prediction market use case, crypto market direction, etc.
- Make the poll genuinely interesting and discussion-provoking

Example topics:
- "Which prediction market will have the most volume by end of 2025?"
- "What's the next big event prediction markets will go crazy for?"
- "Will crypto prediction markets outgrow sports betting within 2 years?"
"""

REPORT_PROMPT = """Analyze this week's Twitter engagement data from a prediction markets account and write a brief, insightful summary.

Tweet data (JSON):
{tweet_data}

Write a short Markdown analysis (3-5 bullet points) covering:
- Which tweet performed best and why it likely resonated
- Any patterns in engagement (time, topic, format)
- One actionable recommendation for next week's content

Keep it concise and practical. Use bullet points."""


POLYMARKET_API_BASE = "https://gamma-api.polymarket.com/markets"

CLOSING_SOON_PROMPT = """You are a prediction markets analyst. Based on the following Polymarket markets that close within 48 hours, write a single engaging tweet.

Markets data:
{markets}

Tweet rules:
- First line MUST be a counter-intuitive or data-driven hook about what the odds reveal
- Do NOT start with "These markets close soon" — that's a headline, not a hook
- Strong openers: "The market is pricing X at only Y% — most people think it's much higher..."
  or "Only 48h left on this bet. The odds tell a very different story than the headlines..."
- Highlight the most interesting market and its current odds
- Max {max_chars} characters (strictly enforced — a CTA line will be appended after)
- English only
- Use line breaks to separate ideas. Format:
  Line 1: Hook — what the odds reveal that's surprising or counter-intuitive
  (blank line)
  Line 2-3: Specific odds, volume, and why it matters
  (blank line)
  Last line: 1-2 hashtags (e.g. #Polymarket #PredictionMarkets)
- Output only the tweet text (no CTA — it will be added automatically)"""

TRENDING_PROMPT = """You are a prediction markets analyst. Based on today's highest-volume Polymarket markets, write a single engaging tweet.

Markets data:
{markets}
{kol_context}

Tweet rules:
- First line MUST be a strong hook — lead with the surprising odds number, an absurd comparison, or a contrarian take
- Do NOT start with "Traders are piling into" — generic. Lead with WHAT the market is actually saying
- Strong openers:
  "$21M just flooded into this market. Here's what that actually means for..."
  "The market gives [event] only X% — less likely than [absurd comparison]."
  "I think [market] is mispriced. Here's my read..."
  "[Odds]% chance. Meanwhile, headlines are saying the opposite."
- Max {max_chars} characters (strictly enforced)
- English only
- Use line breaks to separate ideas. Format:
  Line 1: Hook — the surprising odds/signal with a specific number
  (blank line)
  Line 2-3: Context — why it matters, what smart money is actually saying
  (blank line)
  Last line: End with a question ("What's your take?" / "Am I reading this wrong?" / "Which side are you on?") OR 1-2 hashtags — alternate between these, do NOT always use hashtags
- IMPORTANT: Before the tweet text, output exactly one line in this format (do NOT include it in the tweet):
  MARKET_INDEX: N
  where N is the 0-based index (from the markets list above) of the ONE market your tweet focuses on.
- Output only: the MARKET_INDEX line, then the tweet text (no promotional links)"""

SMART_MONEY_PROMPT = """You are a prediction markets analyst. Based on Polymarket markets with significant price moves today, write a single engaging tweet signaling where smart money is moving.

Markets data:
{markets}
{kol_context}

Tweet rules:
- First line MUST lead with the raw price move number — that IS the hook
- Do NOT start with "Smart money just moved" — show the DATA first, interpret second
- Strong openers:
  "23% → 61% in 24h. Someone knows something."
  "Odds just collapsed from 71% to 19% overnight. Here's what changed..."
  "I've been watching this market all week. This move is either a mistake or insider knowledge."
- After the hook: explain which market, why the move is significant, what it might signal
- Max {max_chars} characters (strictly enforced)
- English only
- Use line breaks to separate ideas. Format:
  Line 1: Hook — the raw price move with exact numbers
  (blank line)
  Line 2-3: Market context — what's happening and why it matters
  (blank line)
  Last line: End with a sharp question ("What's your read?" / "Is this a signal or noise?" / "Am I wrong here?") OR 1-2 hashtags — alternate between these
- Output only the tweet text (no promotional links)"""


THREAD_PROMPT = """Generate a "MoonX Market Brief" weekly thread — 7 connected tweets about the biggest prediction market moves.

Markets data (top movers):
{markets}

Today's date: {date}
{kol_context}

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "tweets": [
    "tweet1 text here",
    "tweet2 text here",
    "tweet3 text here",
    "tweet4 text here",
    "tweet5 text here",
    "tweet6 text here",
    "tweet7 text here"
  ]
}}

Rules for each tweet:
- Tweet 1 (HOOK): Must start with "🧵 MoonX Market Brief" then a blank line, then one punchy line summarizing the week's biggest signal. End with "Thread 👇"
- Tweet 2: Biggest mover — exact odds change (e.g. "23% → 61%"), what market, why it matters. End with a question.
- Tweet 3: Second biggest story — lead with a counter-intuitive angle or absurd comparison.
- Tweet 4: Cross-platform insight — compare same event across Polymarket/Kalshi if relevant, or contrast two markets that are sending conflicting signals.
- Tweet 5: "The market I think is WRONG right now" — pick one market and argue it's mispriced. Be specific.
- Tweet 6: Key takeaway — what does this week's data tell us about the next 7 days? One clear prediction.
- Tweet 7 (CLOSE): "Follow @MoonXBYDFi for daily market signals.\\n\\nTrack it live → bydfi.com/en/moonx/markets/trending"
- Each tweet max 270 characters
- Only tweet 7 contains a link/CTA
- All other tweets end with a question or strong take — no hashtags except tweet 7
- English only, data-driven throughout"""


def _ensure_line_breaks(text: str) -> str:
    """确保推文段落之间有空行（\\n\\n），Twitter 才会渲染出视觉间隔。"""
    # 先压缩超过两个的连续换行
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    # 已有空行，格式正确
    if '\n\n' in text:
        return text

    # 只有单个换行（无空行）→ 全部升级为空行
    if '\n' in text:
        text = re.sub(r'\n(?!\n)', '\n\n', text)
        return text.strip()

    # 完全没有换行 → 在句子边界和 hashtag 前插入空行
    text = re.sub(r'\s+(#\w)', r'\n\n\1', text)
    text = re.sub(r'([.!?])\s+([A-Z"\'$])', r'\1\n\n\2', text)
    return text.strip()


def _truncate_tweet(text: str, max_chars: int = TWEET_MAX_CHARS) -> str:
    """截断推文，优先在换行处截断保留完整的行。"""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit('\n', 1)[0]
    if len(truncated) > 10:
        return truncated.strip()
    return text[:max_chars].rsplit(' ', 1)[0].strip()


KOL_HANDLES = ["yuexiaoyu", "0xpengyu", "haze0x"]


def fetch_kol_context() -> str:
    """Fetch recent tweets from reference KOL accounts for content inspiration.
    Returns formatted string or empty string on failure."""
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        return ""
    try:
        client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=False)
        sections = []
        for handle in KOL_HANDLES:
            try:
                user_resp = client.get_user(username=handle)
                if not user_resp.data:
                    continue
                tweets_resp = client.get_users_tweets(
                    id=user_resp.data.id,
                    max_results=5,
                    exclude=["retweets", "replies"],
                )
                if not tweets_resp.data:
                    continue
                lines = [f"@{handle}:"]
                for t in tweets_resp.data[:3]:
                    text = t.text[:180] + "…" if len(t.text) > 180 else t.text
                    lines.append(f"  • {text}")
                sections.append("\n".join(lines))
            except Exception:
                continue
        return "\n\n".join(sections) if sections else ""
    except Exception as e:
        logger.debug("KOL context fetch failed: %s", e)
        return ""


def call_gemini(prompt: str) -> str:
    client = Anthropic()
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _fmt_kol_context(raw: str) -> str:
    """Format KOL tweets into a prompt section, or return empty string."""
    if not raw.strip():
        return ""
    return f"\nWhat top CT voices are posting today (use as inspiration — build on, contrast, or extend their angle):\n{raw}\n"


def get_twitter_client() -> tweepy.Client:
    """Build and return an authenticated tweepy.Client."""
    api_key = os.getenv("TWITTER_API_KEY")
    api_secret = os.getenv("TWITTER_API_SECRET")
    access_token = os.getenv("TWITTER_ACCESS_TOKEN")
    access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

    for name, val in [
        ("TWITTER_API_KEY", api_key),
        ("TWITTER_API_SECRET", api_secret),
        ("TWITTER_ACCESS_TOKEN", access_token),
        ("TWITTER_ACCESS_TOKEN_SECRET", access_token_secret),
    ]:
        if not val:
            raise ValueError(f"{name} not set in environment")

    return tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )


def track_tweet_id(tweet_id: str, tweet_text: str, tweet_type: str = "generic") -> None:
    """Append a tweet record to tweet_ids.json."""
    records = load_tweet_ids()
    records.append({
        "id": str(tweet_id),
        "type": tweet_type,
        "text": tweet_text,
        "posted_at": datetime.now(timezone.utc).isoformat(),
    })
    TWEET_IDS_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    logger.info("Tracked tweet ID: %s (type: %s)", tweet_id, tweet_type)


def load_tweet_ids() -> list[dict]:
    """Load tweet records from tweet_ids.json."""
    if not TWEET_IDS_FILE.exists():
        return []
    try:
        return json.loads(TWEET_IDS_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load tweet_ids.json: %s", e)
        return []


def _within_cooldown(tweet_type: str, minutes: int = MIN_COOLDOWN_MINUTES) -> bool:
    """Return True if a tweet of the same type was posted within `minutes` ago."""
    records = load_tweet_ids()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    for r in reversed(records):
        if r.get("type") != tweet_type:
            continue
        try:
            posted_at = datetime.fromisoformat(r["posted_at"])
            if posted_at >= cutoff:
                logger.warning(
                    "Cooldown active: last %s tweet posted %s (< %d min ago). Skipping.",
                    tweet_type, r["posted_at"], minutes,
                )
                return True
        except (ValueError, KeyError):
            continue
    return False


def fetch_polymarket_markets(params: dict) -> list[dict]:
    """Fetch markets from Polymarket Gamma API."""
    try:
        resp = requests.get(POLYMARKET_API_BASE, params=params, timeout=(5, 15))
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("Failed to fetch Polymarket data: %s", e)
        raise


def _fetch_price_history(market: dict, days: int = 14) -> list[dict]:
    """Fetch YES price history from Polymarket CLOB API (max 14 days, hourly).
    Returns list of {t: unix_ts, p: 0-1 price} or [] on failure.
    """
    clob_ids_raw = market.get("clobTokenIds", "[]") or "[]"
    if isinstance(clob_ids_raw, str):
        try:
            clob_ids = json.loads(clob_ids_raw)
        except Exception:
            clob_ids = []
    else:
        clob_ids = list(clob_ids_raw)

    token_id = clob_ids[0] if clob_ids else ""
    if not token_id:
        return []

    import time as _time
    now = int(_time.time())
    start = now - days * 24 * 3600
    try:
        resp = requests.get(
            "https://clob.polymarket.com/prices-history",
            params={"market": token_id, "startTs": start, "endTs": now, "fidelity": 60},
            timeout=12,
        )
        if resp.status_code == 200:
            pts = resp.json().get("history", [])
            logger.info("Fetched %d price history points for market", len(pts))
            return pts
    except Exception as exc:
        logger.warning("Price history fetch failed: %s", exc)
    return []


# ── Polymarket-style card ─────────────────────────────────────────────────────

def _wrap_question(question: str, max_chars: int = 28) -> list:
    """Word-wrap question into lines of max_chars, max 3 lines."""
    words = question.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if len(test) <= max_chars:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:3]


def _build_chart_svg(history_pts: list, yes_pct: float) -> str:
    """Build an inline SVG price-history line chart (600×160)."""
    W, H = 600, 160
    PAD_L, PAD_R, PAD_T, PAD_B = 4, 52, 12, 24

    if not history_pts or len(history_pts) < 2:
        return ""

    ys_raw = [float(p["p"]) * 100 for p in history_pts]
    y_min, y_max = min(ys_raw), max(ys_raw)
    if y_max - y_min < 2:
        return ""

    pad = max((y_max - y_min) * 0.12, 2)
    y_lo = max(0, y_min - pad)
    y_hi = min(100, y_max + pad)

    def sx(i):
        return PAD_L + (i / (len(ys_raw) - 1)) * (W - PAD_L - PAD_R)

    def sy(v):
        return PAD_T + (1 - (v - y_lo) / (y_hi - y_lo)) * (H - PAD_T - PAD_B)

    pts = [(sx(i), sy(v)) for i, v in enumerate(ys_raw)]
    line_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    fill_d = line_d + f" L {pts[-1][0]:.1f},{H - PAD_B} L {pts[0][0]:.1f},{H - PAD_B} Z"

    # Y-axis grid lines and labels
    grid_lines = []
    import math
    rng = y_hi - y_lo
    step = 5 if rng <= 30 else 10
    start = math.ceil(y_lo / step) * step
    tick_vals = []
    v = start
    while v <= y_hi + 0.01:
        tick_vals.append(v)
        v += step

    for tv in tick_vals:
        gy = sy(tv)
        grid_lines.append(
            f'<line x1="{PAD_L}" y1="{gy:.1f}" x2="{W - PAD_R}" y2="{gy:.1f}" '
            f'stroke="#E5E7EB" stroke-width="1" stroke-dasharray="3,3"/>'
        )
        grid_lines.append(
            f'<text x="{W - PAD_R + 6}" y="{gy:.1f}" dy="4" '
            f'fill="#9CA3AF" font-size="11" font-family="system-ui">{tv:.0f}%</text>'
        )

    ex, ey = pts[-1]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 {W} {H}" preserveAspectRatio="none">
  <defs>
    <linearGradient id="fillGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#3B82F6" stop-opacity="0.18"/>
      <stop offset="100%" stop-color="#3B82F6" stop-opacity="0"/>
    </linearGradient>
  </defs>
  {''.join(grid_lines)}
  <path d="{fill_d}" fill="url(#fillGrad)"/>
  <path d="{line_d}" fill="none" stroke="#3B82F6" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="#3B82F6"/>
</svg>"""
    return svg


def _build_card_html(market: dict, yes_pct: float, vol_str: str,
                     change_pct: float, thumb_b64: str, history_pts: list) -> str:
    """Render market card as HTML (1200×628). Dark OLED + MoonX brand."""
    no_pct = 100 - yes_pct
    yes_payout = round(100 / yes_pct * 100) if yes_pct > 0 else 999
    no_pct_val = no_pct
    no_payout  = round(100 / no_pct * 100) if no_pct > 0 else 999
    question   = market.get("question", "")

    # Change badge
    if change_pct >= 0.5:
        chg_html = f'<span class="badge badge-up">▲ {change_pct:.0f}%</span>'
    elif change_pct <= -0.5:
        chg_html = f'<span class="badge badge-dn">▼ {abs(change_pct):.0f}%</span>'
    else:
        chg_html = ""

    # Thumbnail
    if thumb_b64:
        thumb_html = f'<img src="data:image/jpeg;base64,{thumb_b64}" class="thumb"/>'
    else:
        thumb_html = '<div class="thumb thumb-empty"><span>?</span></div>'

    # Stat pill color: blue if >50%, red if <50%
    chance_cls = "stat-yes" if yes_pct >= 50 else "stat-no"

    chart_svg   = _build_chart_svg(history_pts, yes_pct)
    if chart_svg:
        chart_block = f'<div class="chart-wrap">{chart_svg}</div>'
    else:
        # Fallback: large YES/NO stats block when no price history available
        yes_w = max(yes_pct, 2) if yes_pct > 0 else 2
        no_w  = max(no_pct, 2) if no_pct > 0 else 2
        chart_block = f"""<div class="odds-bar-wrap">
  <div class="odds-big-row">
    <div class="odds-big-stat yes-stat">
      <div class="odds-big-pct yes-color">{yes_pct:.1f}%</div>
      <div class="odds-big-label">YES</div>
      <div class="odds-big-payout">$100 → ${yes_payout}</div>
    </div>
    <div class="odds-divider"></div>
    <div class="odds-big-stat no-stat">
      <div class="odds-big-pct no-color">{no_pct:.1f}%</div>
      <div class="odds-big-label">NO</div>
      <div class="odds-big-payout">$100 → ${no_payout}</div>
    </div>
  </div>
  <div class="odds-bar">
    <div class="odds-fill odds-fill-yes" style="flex:{yes_w}"></div>
    <div class="odds-fill odds-fill-no"  style="flex:{no_w}"></div>
  </div>
</div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;500;600;700;800&family=Orbitron:wght@600;700;800;900&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}

  body {{
    width:1200px; height:628px; overflow:hidden;
    background:#080C14;
    font-family: "Exo 2", "Segoe UI", Arial, sans-serif;
  }}

  .card {{
    width:1200px; height:628px;
    background:#080C14;
    display:grid;
    grid-template-columns: 320px 1fr;
    position:relative;
    overflow:hidden;
  }}

  /* ── decorative top border ── */
  .card::before {{
    content:'';
    position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, #FF6B00 0%, #FF9A3C 50%, transparent 100%);
    z-index:10;
  }}

  /* ══ LEFT PANEL ══ */
  .left {{
    background: linear-gradient(170deg, #111827 0%, #0D1520 100%);
    border-right: 1px solid rgba(255,107,0,0.15);
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    padding:36px 28px; gap:20px;
    position:relative; overflow:hidden;
  }}
  /* orange glow behind thumbnail */
  .left::after {{
    content:''; position:absolute;
    width:280px; height:280px; border-radius:50%;
    background: radial-gradient(circle, rgba(255,107,0,0.12) 0%, transparent 65%);
    top:50%; left:50%; transform:translate(-50%,-50%);
    pointer-events:none;
  }}

  .thumb {{
    width:128px; height:128px; border-radius:14px;
    object-fit:cover;
    border:1.5px solid rgba(255,107,0,0.30);
    box-shadow: 0 0 24px rgba(255,107,0,0.20), 0 8px 32px rgba(0,0,0,0.6);
    position:relative; z-index:1;
  }}
  .thumb-empty {{
    width:128px; height:128px; border-radius:14px;
    background:rgba(255,107,0,0.06);
    border:1.5px solid rgba(255,107,0,0.20);
    display:flex; align-items:center; justify-content:center;
    position:relative; z-index:1;
  }}
  .thumb-empty span {{ font-size:44px; color:rgba(255,107,0,0.25); font-weight:700; }}

  /* Big probability number */
  .stat-block {{
    text-align:center; position:relative; z-index:1;
  }}
  .stat-pct {{
    font-family:"Orbitron", monospace;
    font-size:68px; font-weight:900; line-height:1;
    letter-spacing:-1px;
  }}
  .stat-yes {{
    color:#10B981;
    text-shadow: 0 0 20px rgba(16,185,129,0.45);
  }}
  .stat-no {{
    color:#F87171;
    text-shadow: 0 0 20px rgba(248,113,113,0.45);
  }}
  .stat-label {{
    font-size:11px; font-weight:600;
    color:rgba(255,255,255,0.35);
    margin-top:6px; letter-spacing:2.5px; text-transform:uppercase;
  }}

  /* Change badge */
  .badge {{
    display:inline-flex; align-items:center; gap:4px;
    padding:4px 12px; border-radius:20px;
    font-size:13px; font-weight:700;
    position:relative; z-index:1;
    font-family:"Orbitron", monospace;
  }}
  .badge-up {{
    background:rgba(16,185,129,0.12);
    color:#10B981;
    border:1px solid rgba(16,185,129,0.25);
  }}
  .badge-dn {{
    background:rgba(248,113,113,0.12);
    color:#F87171;
    border:1px solid rgba(248,113,113,0.25);
  }}

  /* Polymarket source */
  .source {{
    font-size:11px; color:rgba(255,255,255,0.20);
    letter-spacing:0.5px; position:relative; z-index:1;
  }}
  .source span {{ color:rgba(255,107,0,0.50); }}

  /* ══ RIGHT PANEL ══ */
  .right {{
    display:flex; flex-direction:column;
    padding:28px 40px 22px;
    gap:0;
    background: #080C14;
  }}

  /* Top bar: tag + volume */
  .meta {{
    display:flex; align-items:center; justify-content:space-between;
    margin-bottom:12px;
  }}
  .meta-tag {{
    font-size:11px; font-weight:700; color:#FF6B00;
    background:rgba(255,107,0,0.10);
    border:1px solid rgba(255,107,0,0.25);
    padding:3px 10px; border-radius:20px;
    letter-spacing:1px; text-transform:uppercase;
  }}
  .meta-vol {{
    font-size:12px; color:rgba(255,255,255,0.30);
    font-family:"Orbitron", monospace; font-weight:500;
  }}

  /* Question text */
  .question {{
    font-size:21px; font-weight:700;
    color:#F1F5F9;
    line-height:1.45; margin-bottom:16px;
    flex-shrink:0;
  }}

  /* Divider */
  .sep {{
    height:1px;
    background:linear-gradient(90deg, rgba(255,107,0,0.20) 0%, rgba(255,255,255,0.04) 100%);
    margin-bottom:14px;
  }}

  /* Chart */
  .chart-wrap {{ height:198px; margin-bottom:14px; overflow:hidden; }}

  /* Odds bar fallback */
  .odds-bar-wrap {{
    flex:1; margin-bottom:14px;
    display:flex; flex-direction:column;
    justify-content:space-between;
    gap:14px;
  }}
  .odds-big-row {{
    flex:1; display:flex; align-items:stretch;
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:12px; overflow:hidden;
  }}
  .odds-big-stat {{
    flex:1; display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:5px;
    padding:16px 0;
  }}
  .odds-divider {{ width:1px; background:rgba(255,255,255,0.06); }}
  .odds-big-pct {{
    font-family:"Orbitron", monospace;
    font-size:48px; font-weight:900; line-height:1; letter-spacing:-1px;
  }}
  .yes-color {{ color:#10B981; text-shadow:0 0 16px rgba(16,185,129,0.35); }}
  .no-color  {{ color:#F87171; text-shadow:0 0 16px rgba(248,113,113,0.35); }}
  .odds-big-label {{
    font-size:11px; font-weight:700;
    color:rgba(255,255,255,0.30);
    letter-spacing:2px; text-transform:uppercase;
  }}
  .odds-big-payout {{
    font-size:11px; color:rgba(255,255,255,0.20);
  }}
  .odds-bar {{
    display:flex; height:8px; border-radius:4px; overflow:hidden;
    flex-shrink:0;
  }}
  .odds-fill-yes {{ background:linear-gradient(90deg,#059669,#10B981); }}
  .odds-fill-no  {{ background:linear-gradient(90deg,#DC2626,#EF4444); }}

  /* YES / NO buttons */
  .buttons {{ display:flex; gap:12px; margin-bottom:6px; }}
  .btn {{
    flex:1; height:48px; border-radius:10px; border:none;
    font-size:15px; font-weight:700; color:#fff;
    font-family:"Exo 2", sans-serif;
    display:flex; align-items:center; justify-content:space-between;
    padding:0 18px; cursor:default;
  }}
  .btn-yes {{
    background:linear-gradient(135deg,#065F46,#059669);
    border:1px solid rgba(16,185,129,0.30);
    box-shadow:0 0 16px rgba(16,185,129,0.15);
  }}
  .btn-no {{
    background:linear-gradient(135deg,#7F1D1D,#DC2626);
    border:1px solid rgba(248,113,113,0.25);
    box-shadow:0 0 16px rgba(248,113,113,0.12);
  }}
  .btn-label {{ font-weight:800; letter-spacing:0.5px; }}
  .btn-price {{
    font-family:"Orbitron", monospace;
    font-weight:600; font-size:14px;
    opacity:0.85;
  }}

  /* Payout hints */
  .sub-row {{ display:flex; gap:12px; margin-bottom:6px; }}
  .sub {{ flex:1; text-align:center; font-size:11px; color:rgba(255,255,255,0.18); }}

  /* Footer: watermark */
  .footer {{
    display:flex; align-items:center; justify-content:flex-end; gap:6px;
  }}
  .moonx-dot {{
    width:8px; height:8px; border-radius:50%;
    background:#FF6B00;
    box-shadow:0 0 8px rgba(255,107,0,0.8);
  }}
  .watermark {{
    font-family:"Orbitron", monospace;
    font-size:12px; font-weight:700;
    color:rgba(255,107,0,0.55);
    letter-spacing:0.5px;
  }}
</style>
</head>
<body>
<div class="card">

  <!-- LEFT -->
  <div class="left">
    {thumb_html}
    <div class="stat-block">
      <div class="stat-pct {chance_cls}">{yes_pct:.0f}%</div>
      <div class="stat-label">probability</div>
    </div>
    {chg_html}
    <div class="source"><span>Polymarket</span> &nbsp;·&nbsp; {vol_str} vol</div>
  </div>

  <!-- RIGHT -->
  <div class="right">
    <div class="meta">
      <span class="meta-tag">Prediction Market</span>
      <span class="meta-vol">{vol_str} 24H VOL</span>
    </div>
    <div class="question">{question}</div>
    <div class="sep"></div>
    {chart_block}
    <div class="buttons">
      <button class="btn btn-yes">
        <span class="btn-label">YES</span>
        <span class="btn-price">{round(yes_pct)}¢</span>
      </button>
      <button class="btn btn-no">
        <span class="btn-label">NO</span>
        <span class="btn-price">{round(no_pct)}¢</span>
      </button>
    </div>
    <div class="sub-row">
      <div class="sub">$100 → ${yes_payout}</div>
      <div class="sub">$100 → ${no_payout}</div>
    </div>
    <div class="footer">
      <div class="moonx-dot"></div>
      <div class="watermark">@moonx_bydfi</div>
    </div>
  </div>

</div>
</body>
</html>"""


def create_market_chart(market: dict) -> str | None:
    """Generate market card: HTML → Playwright screenshot → PNG."""
    import base64
    from playwright.sync_api import sync_playwright

    # ── Parse data ───────────────────────────────────────────────────────────────
    raw_prices = market.get("outcomePrices", "[]")
    if isinstance(raw_prices, str):
        try:
            prices = json.loads(raw_prices)
        except Exception:
            prices = []
    else:
        prices = raw_prices
    try:
        yes_pct = round(float(prices[0]) * 100, 1) if prices else 50.0
    except (ValueError, TypeError):
        yes_pct = 50.0

    volume = market.get("volume24hr", 0) or 0
    if volume >= 1_000_000:
        vol_str = f"${volume / 1e6:.1f}M"
    elif volume >= 1_000:
        vol_str = f"${volume / 1e3:.0f}K"
    else:
        vol_str = f"${volume:.0f}"

    price_change = float(market.get("oneDayPriceChange", 0) or 0)
    change_pct = round(price_change * 100, 1)

    image_url = market.get("image", "")
    history_pts = _fetch_price_history(market)

    # ── Download thumbnail → base64 ──────────────────────────────────────────────
    thumb_b64 = ""
    if image_url:
        try:
            r = requests.get(image_url, timeout=10)
            r.raise_for_status()
            thumb_b64 = base64.b64encode(r.content).decode()
        except Exception as e:
            logger.warning("Thumbnail download failed: %s", e)

    # ── Build HTML ───────────────────────────────────────────────────────────────
    html = _build_card_html(market, yes_pct, vol_str, change_pct,
                            thumb_b64, history_pts)
    html_path = script_dir / "market_card.html"
    html_path.write_text(html, encoding="utf-8")

    # ── Screenshot via Playwright ────────────────────────────────────────────────
    png_path = str(script_dir / "market_chart.png")
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1200, "height": 628})
            page.goto(f"file://{html_path.resolve()}")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=png_path, clip={"x": 0, "y": 0, "width": 1200, "height": 628})
            browser.close()
    except Exception as e:
        logger.error("Playwright screenshot failed: %s", e)
        return None

    logger.info("Market card saved → %s", png_path)
    return png_path


def _format_market_for_prompt(market: dict) -> str:
    """Format a single market dict into a human-readable string for prompts."""
    question = market.get("question", "Unknown")

    # Parse outcomePrices — may be a JSON string or already a list
    raw_prices = market.get("outcomePrices", "[]")
    if isinstance(raw_prices, str):
        try:
            prices = json.loads(raw_prices)
        except (json.JSONDecodeError, ValueError):
            prices = []
    else:
        prices = raw_prices

    if len(prices) >= 2:
        try:
            yes_pct = round(float(prices[0]) * 100, 1)
            no_pct = round(float(prices[1]) * 100, 1)
            odds_str = f"YES {yes_pct}% / NO {no_pct}%"
        except (ValueError, TypeError):
            odds_str = "N/A"
    else:
        odds_str = "N/A"

    volume = market.get("volume24hr", 0) or 0
    price_change = market.get("oneDayPriceChange", 0) or 0
    end_date = market.get("endDate", "Unknown")
    if end_date and end_date != "Unknown":
        end_date = end_date[:10]  # keep YYYY-MM-DD only

    return (
        f"- {question}\n"
        f"  Odds: {odds_str} | 24h Volume: ${volume:,.0f} | "
        f"Price Change: {price_change:+.1%} | Closes: {end_date}"
    )


def generate_tweet() -> tuple[str, str | None]:
    """Generate tweet text only. Returns (tweet_text, None)."""
    text = call_gemini(TEXT_ONLY_PROMPT)
    text = _ensure_line_breaks(text)
    text = _truncate_tweet(text)
    return text, None


def generate_comparison_tweet() -> tuple[str, str]:
    """Generate tweet + comparison chart. Returns (tweet_text, image_path)."""
    raw = call_gemini(COMPARISON_PROMPT)

    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)
    tweet_text = _ensure_line_breaks(data["tweet"])
    tweet_text = _truncate_tweet(tweet_text)

    image_path = create_comparison_chart(
        title=data["chart_title"],
        metric=data["metric"],
        platforms=data["platforms"],
        values=data["values"],
        unit=data.get("unit", ""),
    )
    return tweet_text, image_path


def create_comparison_chart(title, metric, platforms, values, unit) -> str:
    """Horizontal bar chart comparing platforms. Random color theme."""
    import random
    THEMES = [
        {"BG": "#0B0E1A", "CARD_BG": "#111827", "BORDER": "#1E2D45",
         "TEXT": "#F0F4FF", "GRAY": "#6B7A99"},
        {"BG": "#0E0E0E", "CARD_BG": "#181818", "BORDER": "#2A2A2A",
         "TEXT": "#F5F5F5", "GRAY": "#888888"},
        {"BG": "#0C0718", "CARD_BG": "#150E2A", "BORDER": "#2A1F45",
         "TEXT": "#EEE8FF", "GRAY": "#7A6A9A"},
    ]
    t = random.choice(THEMES)
    BG, CARD_BG, TEXT, GRAY, BORDER = t["BG"], t["CARD_BG"], t["TEXT"], t["GRAY"], t["BORDER"]
    BAR_COLORS = ["#FF6B00", "#00E5A0", "#7B6EF6", "#FF4D6D"]

    # Sort descending
    pairs = sorted(zip(values, platforms), reverse=True)
    vals_s = [v for v, _ in pairs]
    plats_s = [p for _, p in pairs]
    max_val = max(vals_s)

    fig, ax = plt.subplots(figsize=(12, 6.28))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD_BG)

    y_pos = list(range(len(plats_s)))
    bars = ax.barh(y_pos, vals_s,
                   color=[BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(vals_s))],
                   height=0.52, zorder=3)

    for bar, val in zip(bars, vals_s):
        ax.text(bar.get_width() + max_val * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val} {unit}", va="center", ha="left",
                color=TEXT, fontsize=12, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plats_s, fontsize=13, color=TEXT)
    ax.set_xlim(0, max_val * 1.28)
    ax.set_title(title, color=TEXT, fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel(f"{metric} ({unit})", color=GRAY, fontsize=11)
    ax.tick_params(colors=TEXT, labelsize=10, length=0)
    ax.xaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.6, zorder=0, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    fig.text(0.98, 0.02, "MoonX · bydfi.com", ha="right",
             color=GRAY, fontsize=9, style="italic")
    plt.tight_layout()
    image_path = str(script_dir / "chart.png")
    plt.savefig(image_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    logger.info("Comparison chart saved: %s", image_path)
    return image_path


def post_tweet(tweet_text: str, image_path: str | None = None, tweet_type: str = "generic") -> str:
    """Post a tweet via Twitter API. Attaches image if provided."""
    api_key = os.getenv("TWITTER_API_KEY")
    api_secret = os.getenv("TWITTER_API_SECRET")
    access_token = os.getenv("TWITTER_ACCESS_TOKEN")
    access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

    media_ids = None
    if image_path:
        # Media upload requires v1.1 API
        auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
        api_v1 = tweepy.API(auth)
        media = api_v1.media_upload(filename=image_path)
        media_ids = [media.media_id]
        logger.info("Image uploaded, media_id: %s", media.media_id)

    client = get_twitter_client()
    kwargs = {"text": tweet_text}
    if media_ids:
        kwargs["media_ids"] = media_ids

    response = client.create_tweet(**kwargs)
    tweet_id = response.data["id"]
    track_tweet_id(tweet_id, tweet_text, tweet_type=tweet_type)
    return tweet_id


def generate_poll() -> tuple[str, list[str], int]:
    """Call Gemini to generate a poll. Returns (tweet_text, options, duration_minutes)."""
    raw = call_gemini(POLL_PROMPT)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)
    tweet_text = data["tweet"]
    options = data["options"]
    duration_minutes = int(data.get("duration_minutes", 1440))

    # Enforce Twitter poll constraints
    options = [o[:25] for o in options[:4]]  # max 4 options, each max 25 chars
    duration_minutes = max(5, min(duration_minutes, 10080))  # 5 min to 7 days

    return tweet_text, options, duration_minutes


def post_poll() -> str:
    """Generate and post a Twitter poll. Returns tweet ID."""
    logger.info("Generating poll with Gemini...")
    tweet_text, options, duration_minutes = generate_poll()
    logger.info("Poll question: %s | Options: %s | Duration: %d min",
                tweet_text, options, duration_minutes)

    client = get_twitter_client()
    response = client.create_tweet(
        text=tweet_text,
        poll_options=options,
        poll_duration_minutes=duration_minutes,
    )
    tweet_id = response.data["id"]
    track_tweet_id(tweet_id, f"[POLL] {tweet_text}")
    logger.info("SUCCESS — Poll posted (ID: %s)", tweet_id)
    return tweet_id


def generate_report() -> str:
    """Fetch metrics for last 7 days of tweets, generate Markdown report, save to file."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    all_records = load_tweet_ids()

    recent_records = [
        r for r in all_records
        if datetime.fromisoformat(r["posted_at"]) >= cutoff
    ]

    if not recent_records:
        logger.warning("No tweets found in the last 7 days; skipping report.")
        return ""

    logger.info("Fetching metrics for %d tweets...", len(recent_records))
    ids = [r["id"] for r in recent_records]

    client = get_twitter_client()
    response = client.get_tweets(
        ids=ids,
        tweet_fields=["public_metrics", "created_at"],
        user_auth=True,
    )

    tweet_data = []
    if response.data:
        metrics_map = {str(t.id): t for t in response.data}
        for record in recent_records:
            t = metrics_map.get(record["id"])
            if t:
                m = t.public_metrics or {}
                tweet_data.append({
                    "id": record["id"],
                    "text": record["text"],
                    "posted_at": record["posted_at"],
                    "likes": m.get("like_count", 0),
                    "retweets": m.get("retweet_count", 0),
                    "replies": m.get("reply_count", 0),
                    "impressions": m.get("impression_count", 0),
                })

    # Build Markdown table
    table_rows = []
    for row in tweet_data:
        short_text = (row["text"][:60] + "…") if len(row["text"]) > 60 else row["text"]
        table_rows.append(
            f"| {row['posted_at'][:10]} | {short_text} | "
            f"{row['likes']} | {row['retweets']} | {row['replies']} | {row['impressions']} |"
        )

    table = (
        "| Date | Tweet | Likes | Retweets | Replies | Impressions |\n"
        "|------|-------|-------|----------|---------|-------------|\n"
        + "\n".join(table_rows)
    )

    # Gemini analysis
    gemini_analysis = call_gemini(
        REPORT_PROMPT.format(tweet_data=json.dumps(tweet_data, indent=2))
    )

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_content = f"""# Weekly Tweet Report — {today}

## Engagement Data

{table}

## Gemini Analysis

{gemini_analysis}
"""

    reports_dir = script_dir / "logs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"report_{today}.md"
    report_path.write_text(report_content, encoding="utf-8")
    logger.info("Report saved: %s", report_path)
    return str(report_path)


def post_closing_soon() -> str:
    """Fetch markets closing within 48h with >$100k volume and post a tweet."""
    if _within_cooldown("closing_soon"):
        sys.exit(0)
    logger.info("Fetching closing-soon markets from Polymarket...")
    try:
        markets = fetch_polymarket_markets({
            "closed": "false",
            "order": "endDate",
            "ascending": "true",
            "limit": 20,
        })
    except Exception:
        logger.warning("Polymarket unavailable, falling back to generic tweet.")
        tweet_text, _ = generate_tweet()
        tweet_id = post_tweet(tweet_text)
        logger.info("SUCCESS — Fallback tweet posted (ID: %s)", tweet_id)
        return tweet_id

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=48)

    filtered = []
    for m in markets:
        end_raw = m.get("endDate", "")
        if not end_raw:
            continue
        # Normalize timezone suffix: "Z" → "+00:00"
        if end_raw.endswith("Z"):
            end_raw = end_raw[:-1] + "+00:00"
        try:
            end_dt = datetime.fromisoformat(end_raw)
        except ValueError:
            continue
        volume = m.get("volume24hr", 0) or 0
        if end_dt <= cutoff and volume > 1_000:
            filtered.append(m)

    if not filtered:
        logger.info("No closing-soon markets meeting criteria; exiting.")
        sys.exit(0)

    # Take top 3 by volume24hr
    top3 = sorted(filtered, key=lambda m: m.get("volume24hr", 0), reverse=True)[:3]
    markets_text = "\n".join(_format_market_for_prompt(m) for m in top3)

    prompt = CLOSING_SOON_PROMPT.format(markets=markets_text, max_chars=TWEET_CONTENT_MAX)
    tweet_text = _ensure_line_breaks(call_gemini(prompt))
    tweet_text = _truncate_tweet(tweet_text, max_chars=TWEET_CONTENT_MAX) + MOONX_CTA

    image_path = create_market_chart(top3[0])
    logger.info("Posting closing-soon tweet (%d chars): %s", len(tweet_text), tweet_text)
    tweet_id = post_tweet(tweet_text, image_path, tweet_type="closing_soon")
    logger.info("SUCCESS — Closing-soon tweet posted (ID: %s)", tweet_id)
    return tweet_id


def post_trending() -> str:
    """Fetch today's highest-volume markets and post a tweet."""
    if _within_cooldown("trending"):
        sys.exit(0)
    logger.info("Fetching trending markets from Polymarket...")
    try:
        markets = fetch_polymarket_markets({
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
            "limit": 10,
        })
    except Exception:
        logger.warning("Polymarket unavailable, falling back to generic tweet.")
        tweet_text, _ = generate_tweet()
        tweet_id = post_tweet(tweet_text)
        logger.info("SUCCESS — Fallback tweet posted (ID: %s)", tweet_id)
        return tweet_id

    if not markets:
        logger.info("No trending markets returned; exiting.")
        sys.exit(0)

    top5 = markets[:5]
    markets_text = "\n".join(_format_market_for_prompt(m) for m in top5)

    kol_ctx = _fmt_kol_context(fetch_kol_context()).replace("{", "{{").replace("}", "}}")
    prompt = TRENDING_PROMPT.format(markets=markets_text, max_chars=TWEET_MAX_CHARS, kol_context=kol_ctx)
    raw_response = call_gemini(prompt)

    # Parse MARKET_INDEX line to match image to the market Gemini actually wrote about
    chosen_market = top5[0]
    lines = raw_response.strip().splitlines()
    if lines and lines[0].startswith("MARKET_INDEX:"):
        try:
            idx = int(lines[0].split(":", 1)[1].strip())
            if 0 <= idx < len(top5):
                chosen_market = top5[idx]
        except (ValueError, IndexError):
            pass
        raw_response = "\n".join(lines[1:]).lstrip("\n")

    tweet_text = _ensure_line_breaks(raw_response)
    tweet_text = _truncate_tweet(tweet_text, max_chars=TWEET_MAX_CHARS)

    image_path = create_market_chart(chosen_market)
    logger.info("Posting trending tweet (%d chars): %s", len(tweet_text), tweet_text)
    tweet_id = post_tweet(tweet_text, image_path, tweet_type="trending")
    logger.info("SUCCESS — Trending tweet posted (ID: %s)", tweet_id)
    return tweet_id


def post_smart_money() -> str:
    """Fetch high-volume markets with >5% price change and post a tweet."""
    if _within_cooldown("smart_money"):
        sys.exit(0)
    logger.info("Fetching smart-money markets from Polymarket...")
    try:
        markets = fetch_polymarket_markets({
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
            "limit": 50,
        })
    except Exception:
        logger.warning("Polymarket unavailable, falling back to generic tweet.")
        tweet_text, _ = generate_tweet()
        tweet_id = post_tweet(tweet_text)
        logger.info("SUCCESS — Fallback tweet posted (ID: %s)", tweet_id)
        return tweet_id

    filtered = [
        m for m in markets
        if abs(m.get("oneDayPriceChange", 0) or 0) > 0.05
    ]

    if not filtered:
        logger.info("No smart-money markets with >5%% price change; exiting.")
        sys.exit(0)

    # Top 3 by absolute price change
    top3 = sorted(filtered, key=lambda m: abs(m.get("oneDayPriceChange", 0) or 0), reverse=True)[:3]
    markets_text = "\n".join(_format_market_for_prompt(m) for m in top3)

    kol_ctx = _fmt_kol_context(fetch_kol_context()).replace("{", "{{").replace("}", "}}")
    prompt = SMART_MONEY_PROMPT.format(markets=markets_text, max_chars=TWEET_MAX_CHARS, kol_context=kol_ctx)
    tweet_text = _ensure_line_breaks(call_gemini(prompt))
    tweet_text = _truncate_tweet(tweet_text, max_chars=TWEET_MAX_CHARS)

    image_path = create_market_chart(top3[0])
    logger.info("Posting smart-money tweet (%d chars): %s", len(tweet_text), tweet_text)
    tweet_id = post_tweet(tweet_text, image_path, tweet_type="smart_money")
    logger.info("SUCCESS — Smart-money tweet posted (ID: %s)", tweet_id)
    return tweet_id


def post_thread() -> str:
    """Post a weekly 'MoonX Market Brief' thread (7 tweets). Only runs once per 5 days."""
    if _within_cooldown("thread", minutes=7200):  # 5 days = 7200 min
        sys.exit(0)
    logger.info("Fetching markets for weekly thread...")
    try:
        markets = fetch_polymarket_markets({
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
            "limit": 20,
        })
    except Exception:
        logger.warning("Polymarket unavailable, skipping thread.")
        sys.exit(0)

    from datetime import date as _date
    import time as _time

    today = _date.today().strftime("%b %d")
    markets_text = "\n".join(_format_market_for_prompt(m) for m in markets[:10])

    kol_ctx = _fmt_kol_context(fetch_kol_context()).replace("{", "{{").replace("}", "}}")
    prompt = THREAD_PROMPT.format(markets=markets_text, date=today, kol_context=kol_ctx)
    raw = call_gemini(prompt)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
        tweets = [t for t in data.get("tweets", []) if t.strip()]
    except Exception as e:
        logger.error("Failed to parse thread JSON: %s | raw: %s", e, raw[:300])
        sys.exit(1)

    if not tweets:
        logger.info("No tweets generated for thread; exiting.")
        sys.exit(0)

    client = get_twitter_client()

    # Post first tweet
    first_resp = client.create_tweet(text=_truncate_tweet(tweets[0]))
    first_id = first_resp.data["id"]
    track_tweet_id(first_id, tweets[0], tweet_type="thread")
    logger.info("Thread tweet 1 posted (ID: %s)", first_id)

    # Post subsequent tweets as replies
    reply_to = first_id
    for i, tweet_text in enumerate(tweets[1:], start=2):
        _time.sleep(2)
        resp = client.create_tweet(
            text=_truncate_tweet(tweet_text),
            in_reply_to_tweet_id=reply_to,
        )
        reply_to = resp.data["id"]
        track_tweet_id(reply_to, tweet_text, tweet_type="thread")
        logger.info("Thread tweet %d posted (ID: %s)", i, reply_to)

    logger.info("SUCCESS — Thread posted (%d tweets, first ID: %s)", len(tweets), first_id)
    return first_id


def main():
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--with-image", action="store_true",
                       help="Generate comparison chart and attach to tweet")
    group.add_argument("--poll", action="store_true",
                       help="Generate and post a weekly crypto/prediction markets poll")
    group.add_argument("--report", action="store_true",
                       help="Generate weekly engagement report from tweet metrics")
    group.add_argument("--closing-soon", action="store_true",
                       help="Post about Polymarket markets closing within 48h with >$100k volume")
    group.add_argument("--trending", action="store_true",
                       help="Post about today's highest-volume Polymarket markets")
    group.add_argument("--smart-money", action="store_true",
                       help="Post about Polymarket markets with >5%% price change (smart money signal)")
    group.add_argument("--thread", action="store_true",
                       help="Post weekly MoonX Market Brief thread (7 tweets, runs once per 5 days)")
    args = parser.parse_args()

    logger.info("=== Tweet Bot Starting ===")

    if args.poll:
        try:
            post_poll()
        except Exception as e:
            logger.error("Failed to post poll: %s", e)
            sys.exit(1)

    elif args.report:
        try:
            report_path = generate_report()
            if report_path:
                logger.info("Report generated: %s", report_path)
        except Exception as e:
            logger.error("Failed to generate report: %s", e)
            sys.exit(1)

    elif args.closing_soon:
        try:
            post_closing_soon()
        except SystemExit:
            raise
        except Exception as e:
            logger.error("Failed to post closing-soon tweet: %s", e)
            sys.exit(1)

    elif args.trending:
        try:
            post_trending()
        except SystemExit:
            raise
        except Exception as e:
            logger.error("Failed to post trending tweet: %s", e)
            sys.exit(1)

    elif args.smart_money:
        try:
            post_smart_money()
        except SystemExit:
            raise
        except Exception as e:
            logger.error("Failed to post smart-money tweet: %s", e)
            sys.exit(1)

    elif args.thread:
        try:
            post_thread()
        except SystemExit:
            raise
        except Exception as e:
            logger.error("Failed to post thread: %s", e)
            sys.exit(1)

    else:
        try:
            if args.with_image:
                logger.info("Generating comparison tweet + chart with Gemini...")
                tweet_text, image_path = generate_comparison_tweet()
            else:
                logger.info("Generating tweet with Gemini...")
                tweet_text, image_path = generate_tweet()
            logger.info("Generated tweet (%d chars): %s", len(tweet_text), tweet_text)
        except Exception as e:
            logger.error("Failed to generate tweet: %s", e)
            sys.exit(1)

        try:
            logger.info("Posting tweet to Twitter...")
            tweet_id = post_tweet(tweet_text, image_path)
            logger.info("SUCCESS — Tweet posted (ID: %s)", tweet_id)
        except Exception as e:
            logger.error("Failed to post tweet: %s", e)
            sys.exit(1)

    logger.info("=== Tweet Bot Done ===")


if __name__ == "__main__":
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.pid_lock import acquire_lock; acquire_lock()
    main()
