#!/usr/bin/env python3
"""
Prediction Markets Daily Tweet Bot
Generates and posts a daily tweet (with optional comparison chart) about prediction markets.
"""
from __future__ import annotations

import os
import sys
import json
import logging
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

    for name, val in [
        ("TWITTER_API_KEY", api_key),
        ("TWITTER_API_SECRET", api_secret),
        ("TWITTER_ACCESS_TOKEN", access_token),
        ("TWITTER_ACCESS_TOKEN_SECRET", access_token_secret),
    ]:
        if not val:
            raise ValueError(f"{name} not set in environment")

    media_ids = None
    if image_path:
        # Media upload requires v1.1 API
        auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
        api_v1 = tweepy.API(auth)
        media = api_v1.media_upload(filename=image_path)
        media_ids = [media.media_id]
        logger.info("Image uploaded, media_id: %s", media.media_id)

    client = tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )
    kwargs = {"text": tweet_text}
    if media_ids:
        kwargs["media_ids"] = media_ids

    response = client.create_tweet(**kwargs)
    return response.data["id"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-image", action="store_true",
                        help="Generate comparison chart and attach to tweet")
    args = parser.parse_args()

    logger.info("=== Tweet Bot Starting ===")

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
