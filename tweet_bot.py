#!/usr/bin/env python3
"""
Prediction Markets Daily Tweet Bot
Generates and posts daily tweets (with optional comparison chart) about prediction markets.
Supports weekly polls and weekly analytics reports.
"""
from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google import genai
import tweepy

# Load environment variables from .env file in script directory
script_dir = Path(__file__).parent
load_dotenv(script_dir / ".env")

# Configure logging
log_file = script_dir / "logs" / "tweet.log"
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

TWEET_MAX_CHARS = 270
TWEET_IDS_FILE = script_dir / "tweet_ids.json"

SYSTEM_PROMPT = """You are a builder and active participant in the prediction markets space.
You write tweets from a first-person sharing perspective — like someone who is deeply in the trenches,
researching platforms, spotting patterns, and sharing genuine insights with fellow enthusiasts.

Your voice is:
- Personal and direct ("I've been thinking...", "IMO...", "What I find interesting is...")
- Analytical but accessible — you use analogies to explain complex mechanics
- Willing to take a clear stance, including contrarian views
- Curious and open-minded, treating competitors as peers worth studying
- Never generic or promotional — always grounded in a real observation"""

TEXT_ONLY_PROMPT = """Write a single tweet about prediction markets from a first-person sharing perspective.

Rotate between these angles:
- A personal observation or pattern you've noticed across platforms
- A contrarian or counterintuitive take on prediction market trends
- Comparing prediction markets to another space (meme coins, DeFi, sports betting, etc.)
- A genuine question or insight about the future direction of the space

Style rules:
- Write as if you're sharing a real thought, not broadcasting a headline
- Start with "I", "Been thinking", "Hot take:", "IMO", "One thing I keep coming back to", etc.
- Maximum {max_chars} characters (strictly enforced)
- English only
- 1-2 hashtags max, placed naturally at the end
- Output only the tweet text""".format(max_chars=TWEET_MAX_CHARS)

COMPARISON_PROMPT = """Generate a prediction market platform comparison tweet WITH chart data.
Write the tweet from a first-person researcher/participant perspective.

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "tweet": "Tweet text here (max {max_chars} chars, first-person sharing voice, 1-2 hashtags at end)",
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


def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in environment")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="models/gemini-2.5-flash-lite",
        contents=SYSTEM_PROMPT + "\n\n" + prompt,
    )
    return response.text.strip()


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


def track_tweet_id(tweet_id: str, tweet_text: str) -> None:
    """Append a tweet record to tweet_ids.json."""
    records = load_tweet_ids()
    records.append({
        "id": str(tweet_id),
        "text": tweet_text,
        "posted_at": datetime.now(timezone.utc).isoformat(),
    })
    TWEET_IDS_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    logger.info("Tracked tweet ID: %s", tweet_id)


def load_tweet_ids() -> list[dict]:
    """Load tweet records from tweet_ids.json."""
    if not TWEET_IDS_FILE.exists():
        return []
    try:
        return json.loads(TWEET_IDS_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load tweet_ids.json: %s", e)
        return []


def generate_tweet() -> tuple[str, str | None]:
    """Generate tweet text only. Returns (tweet_text, None)."""
    text = call_gemini(TEXT_ONLY_PROMPT)
    if len(text) > TWEET_MAX_CHARS:
        text = text[:TWEET_MAX_CHARS].rsplit(" ", 1)[0]
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
    tweet_text = data["tweet"]
    if len(tweet_text) > TWEET_MAX_CHARS:
        tweet_text = tweet_text[:TWEET_MAX_CHARS].rsplit(" ", 1)[0]

    image_path = create_comparison_chart(
        title=data["chart_title"],
        metric=data["metric"],
        platforms=data["platforms"],
        values=data["values"],
        unit=data.get("unit", ""),
    )
    return tweet_text, image_path


def create_comparison_chart(title, metric, platforms, values, unit) -> str:
    """Render a bar chart and save to PNG. Returns file path."""
    COLORS = ["#6C63FF", "#FF6584", "#43C6AC", "#FFB347"]
    BG = "#0f0f1a"
    TEXT = "#e0e0e0"
    GRID = "#2a2a3d"

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.bar(platforms, values, color=COLORS[: len(platforms)], width=0.5, zorder=3)

    # Value labels on bars
    max_val = max(values)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.02,
            f"{val} {unit}",
            ha="center", va="bottom",
            color=TEXT, fontsize=11, fontweight="bold",
        )

    ax.set_title(title, color=TEXT, fontsize=15, fontweight="bold", pad=16)
    ax.set_ylabel(f"{metric} ({unit})", color=TEXT, fontsize=11)
    ax.tick_params(colors=TEXT, labelsize=11)
    ax.yaxis.grid(True, color=GRID, linestyle="--", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.set_ylim(0, max_val * 1.2)

    # Watermark
    fig.text(0.98, 0.02, "#PredictionMarkets", ha="right", color="#555577",
             fontsize=9, style="italic")

    plt.tight_layout()
    image_path = str(script_dir / "chart.png")
    plt.savefig(image_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    logger.info("Chart saved: %s", image_path)
    return image_path


def post_tweet(tweet_text: str, image_path: str | None = None) -> str:
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
    track_tweet_id(tweet_id, tweet_text)
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
    options = options[:4]  # max 4 options
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
    main()
