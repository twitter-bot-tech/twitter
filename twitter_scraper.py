"""
twitter_scraper.py — 直接调 Twitter 内部 GraphQL/REST API（无官方 API 费用）
使用浏览器 session cookies，模拟网页请求。

.env 需配置：
  TWITTER_SCRAPER_AUTH_TOKEN=<from browser cookie>
  TWITTER_SCRAPER_CT0=<from browser cookie>
"""
from __future__ import annotations
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).parent / ".env", override=True)

logger = logging.getLogger(__name__)

# Twitter 内部 Bearer Token（公开固定值，网页版 JS 里的）
_BEARER = "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"


def _session() -> requests.Session:
    auth_token = os.getenv("TWITTER_SCRAPER_AUTH_TOKEN", "")
    ct0 = os.getenv("TWITTER_SCRAPER_CT0", "")
    if not auth_token or not ct0:
        raise RuntimeError("TWITTER_SCRAPER_AUTH_TOKEN / CT0 未配置")

    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {_BEARER}",
        "x-csrf-token": ct0,
        "x-twitter-auth-type": "OAuth2Session",
        "x-twitter-active-user": "yes",
        "x-twitter-client-language": "en",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Referer": "https://x.com/",
        "Origin": "https://x.com",
    })
    s.cookies.set("auth_token", auth_token, domain=".x.com")
    s.cookies.set("ct0", ct0, domain=".x.com")
    return s


_FEATURES_USER = json.dumps({
    "hidden_profile_subscriptions_enabled": True,
    "rweb_tipjar_consumption_enabled": False,
    "responsive_web_graphql_exclude_directive_enabled": True,
    "verified_phone_label_enabled": False,
    "subscriptions_verification_info_is_identity_verified_enabled": True,
    "subscriptions_verification_info_verified_since_enabled": True,
    "highlights_tweets_tab_ui_enabled": True,
    "responsive_web_twitter_article_notes_tab_enabled": True,
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
    "responsive_web_graphql_timeline_navigation_enabled": True,
}, separators=(",", ":"))


def get_user(username: str) -> dict | None:
    """
    按 username 获取用户信息，返回 dict 或 None。
    字段：id, username, name, followers_count, description, verified, url
    """
    username = username.lstrip("@")
    variables = json.dumps({"screen_name": username}, separators=(",", ":"))
    try:
        s = _session()
        resp = s.get(
            "https://x.com/i/api/graphql/qW5u-DAuXpMEG0zA1F7UGQ/UserByScreenName",
            params={"variables": variables, "features": _FEATURES_USER},
            timeout=15,
        )
        if resp.status_code == 429:
            logger.warning("Twitter 限速 (429)，等待 60 秒...")
            time.sleep(60)
            return get_user(username)
        if resp.status_code != 200:
            logger.warning(f"get_user {username}: HTTP {resp.status_code}")
            return None
        data = resp.json()
        user = data.get("data", {}).get("user", {}).get("result", {})
        if not user or user.get("__typename") == "UserUnavailable":
            return None
        legacy = user.get("legacy", {})
        return {
            "id": user.get("rest_id", ""),
            "username": legacy.get("screen_name", ""),
            "name": legacy.get("name", ""),
            "followers_count": legacy.get("followers_count", 0),
            "description": legacy.get("description", ""),
            "verified": legacy.get("verified", False) or legacy.get("is_blue_verified", False),
            "url": legacy.get("url", ""),
            "entities": legacy.get("entities", {}),
        }
    except Exception as e:
        logger.error(f"get_user {username}: {e}")
        return None


def get_user_tweets(user_id: str, limit: int = 20, days: int = 0) -> list:
    """
    获取用户最近推文列表。
    返回 list of dict：id, text, created_at, likes, retweets, replies, impressions
    """
    variables = json.dumps({
        "userId": str(user_id),
        "count": min(limit, 100),
        "includePromotedContent": False,
        "withQuickPromoteEligibilityTweetFields": False,
        "withVoice": False,
        "withV2Timeline": True,
    }, separators=(",", ":"))
    features = json.dumps({
        "rweb_tipjar_consumption_enabled": False,
        "responsive_web_graphql_exclude_directive_enabled": True,
        "verified_phone_label_enabled": False,
        "creator_subscriptions_tweet_preview_api_enabled": True,
        "responsive_web_graphql_timeline_navigation_enabled": True,
        "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
        "tweetypie_unmention_optimization_enabled": True,
        "responsive_web_edit_tweet_api_enabled": True,
        "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
        "view_counts_everywhere_api_enabled": True,
        "longform_notetweets_consumption_enabled": True,
        "responsive_web_twitter_article_tweet_consumption_enabled": False,
        "tweet_awards_web_tipping_enabled": False,
        "freedom_of_speech_not_reach_fetch_enabled": True,
        "standardized_nudges_misinfo": True,
        "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
        "rweb_video_timestamps_enabled": True,
        "longform_notetweets_rich_text_read_enabled": True,
        "longform_notetweets_inline_media_enabled": True,
        "responsive_web_enhance_cards_enabled": False,
    }, separators=(",", ":"))
    try:
        s = _session()
        resp = s.get(
            "https://x.com/i/api/graphql/V7H0Ap3_Hh2FyS75OCDO3Q/UserTweets",
            params={"variables": variables, "features": features},
            timeout=15,
        )
        if resp.status_code == 429:
            logger.warning("Twitter 限速 (429)，等待 60 秒...")
            time.sleep(60)
            return get_user_tweets(user_id, limit, days)
        if resp.status_code != 200:
            logger.warning(f"get_user_tweets {user_id}: HTTP {resp.status_code}")
            return []

        cutoff = None
        if days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        tweets = []
        timeline = (resp.json().get("data", {}).get("user", {})
                    .get("result", {}).get("timeline_v2", {})
                    .get("timeline", {}).get("instructions", []))
        for instruction in timeline:
            for entry in instruction.get("entries", []):
                content = entry.get("content", {})
                item_content = content.get("itemContent", {})
                tweet_result = item_content.get("tweet_results", {}).get("result", {})
                if not tweet_result:
                    continue
                legacy = tweet_result.get("legacy", {})
                if not legacy or legacy.get("retweeted_status_id_str"):
                    continue  # 跳过转推
                created_str = legacy.get("created_at", "")
                try:
                    created_at = datetime.strptime(created_str, "%a %b %d %H:%M:%S +0000 %Y").replace(tzinfo=timezone.utc)
                except Exception:
                    created_at = None
                if cutoff and created_at and created_at < cutoff:
                    continue
                metrics = {
                    "likes": legacy.get("favorite_count", 0),
                    "retweets": legacy.get("retweet_count", 0),
                    "replies": legacy.get("reply_count", 0),
                    "impressions": (tweet_result.get("views", {}).get("count") or 0),
                }
                tweets.append({
                    "id": legacy.get("id_str", ""),
                    "text": legacy.get("full_text", ""),
                    "created_at": created_at,
                    **metrics,
                })
                if len(tweets) >= limit:
                    break
        return tweets
    except Exception as e:
        logger.error(f"get_user_tweets {user_id}: {e}")
        return []


def search_tweets(query: str, limit: int = 50) -> list:
    """
    搜索推文，返回 list of dict（含 user 子 dict）。
    """
    variables = json.dumps({
        "rawQuery": query,
        "count": min(limit, 100),
        "querySource": "typed_query",
        "product": "Latest",
    }, separators=(",", ":"))
    features = json.dumps({
        "rweb_tipjar_consumption_enabled": False,
        "responsive_web_graphql_exclude_directive_enabled": True,
        "verified_phone_label_enabled": False,
        "creator_subscriptions_tweet_preview_api_enabled": True,
        "responsive_web_graphql_timeline_navigation_enabled": True,
        "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
        "tweetypie_unmention_optimization_enabled": True,
        "responsive_web_edit_tweet_api_enabled": True,
        "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
        "view_counts_everywhere_api_enabled": True,
        "longform_notetweets_consumption_enabled": True,
        "responsive_web_twitter_article_tweet_consumption_enabled": False,
        "tweet_awards_web_tipping_enabled": False,
        "freedom_of_speech_not_reach_fetch_enabled": True,
        "standardized_nudges_misinfo": True,
        "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
        "rweb_video_timestamps_enabled": True,
        "longform_notetweets_rich_text_read_enabled": True,
        "longform_notetweets_inline_media_enabled": True,
        "responsive_web_enhance_cards_enabled": False,
    }, separators=(",", ":"))
    try:
        s = _session()
        resp = s.get(
            "https://x.com/i/api/graphql/gkjsKepM6gl_HmFWoWKfgg/SearchTimeline",
            params={"variables": variables, "features": features},
            timeout=15,
        )
        if resp.status_code == 429:
            logger.warning("Twitter 限速 (429)，等待 60 秒...")
            time.sleep(60)
            return search_tweets(query, limit)
        if resp.status_code != 200:
            logger.warning(f"search_tweets '{query}': HTTP {resp.status_code}")
            return []

        tweets = []
        instructions = (resp.json().get("data", {}).get("search_by_raw_query", {})
                        .get("search_timeline", {}).get("timeline", {})
                        .get("instructions", []))
        for instruction in instructions:
            for entry in instruction.get("entries", []):
                content = entry.get("content", {})
                item_content = content.get("itemContent", {})
                tweet_result = item_content.get("tweet_results", {}).get("result", {})
                if not tweet_result:
                    continue
                legacy = tweet_result.get("legacy", {})
                if not legacy:
                    continue
                user_result = tweet_result.get("core", {}).get("user_results", {}).get("result", {})
                user_legacy = user_result.get("legacy", {})
                user = {
                    "id": user_result.get("rest_id", ""),
                    "username": user_legacy.get("screen_name", ""),
                    "name": user_legacy.get("name", ""),
                    "followers_count": user_legacy.get("followers_count", 0),
                    "description": user_legacy.get("description", ""),
                    "verified": user_legacy.get("verified", False) or user_legacy.get("is_blue_verified", False),
                    "entities": user_legacy.get("entities", {}),
                }
                tweets.append({
                    "id": legacy.get("id_str", ""),
                    "text": legacy.get("full_text", ""),
                    "likes": legacy.get("favorite_count", 0),
                    "retweets": legacy.get("retweet_count", 0),
                    "replies": legacy.get("reply_count", 0),
                    "user": user,
                })
                if len(tweets) >= limit:
                    break
        return tweets
    except Exception as e:
        logger.error(f"search_tweets '{query}': {e}")
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("测试 twitter_scraper 连接...")
    user = get_user("Polymarket")
    if user:
        print(f"✅ @{user['username']} | {user['followers_count']:,} 粉丝 | {user['description'][:60]}")
    else:
        print("❌ 获取用户失败，请检查 .env 中的 TWITTER_SCRAPER_* 配置")
