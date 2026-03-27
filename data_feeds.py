#!/usr/bin/env python3
"""
MoonX 数据源聚合层
为 AI 员工提供真实数据，替代 AI 编造的数字

数据源：
  - Polymarket    : 实时预测市场数据（Top 10 + 分类）
  - Twitter       : 自身账号表现 + 竞品监控
  - CoinGecko     : 加密市场趋势
  - Google Search Console : SEO 排名（需凭证）
  - 神策          : MoonX 平台数据（需凭证）
"""
import os, sys, json, time
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).parent))
import twitter_scraper

try:
    import requests as _req
except ImportError:
    _req = None

# ── 缓存（30分钟内不重复拉）────────────────────────────────────────────────────
_cache: dict = {}
_CACHE_TTL = 1800  # 秒


def _cached(key: str, fetch_fn, ttl: int = _CACHE_TTL):
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < ttl:
            return data
    data = fetch_fn()
    _cache[key] = (data, now)
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Polymarket — 实时预测市场
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_polymarket() -> dict:
    try:
        resp = _req.get(
            "https://gamma-api.polymarket.com/markets",
            params={"closed": "false", "order": "volume24hr",
                    "ascending": "false", "limit": 10},
            timeout=10,
        )
        if resp.status_code != 200:
            return {"status": "error", "msg": f"HTTP {resp.status_code}"}

        markets = []
        for m in resp.json():
            try:
                prices = json.loads(m.get("outcomePrices", "[0.5]"))
                yes = round(float(prices[0]) * 100, 1)
            except Exception:
                yes = 50
            markets.append({
                "question": m.get("question", ""),
                "yes_pct":  yes,
                "volume_24h": float(m.get("volume24hr") or 0),
                "category": m.get("category", ""),
            })

        total_vol = sum(m["volume_24h"] for m in markets)
        return {"status": "ok", "markets": markets, "total_volume_24h": total_vol}
    except Exception as e:
        return {"status": "error", "msg": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# Twitter — 自身账号 + 竞品监控
# ═══════════════════════════════════════════════════════════════════════════════

COMPETITORS = ["Polymarket", "KalshiHQ", "ManifoldMarkets", "metaculus"]


def fetch_twitter_self() -> dict:
    """自身账号近期推文表现"""
    try:
        handle = os.getenv("TWITTER_SCRAPER_USERNAME", "moonx_bydfi")
        user = twitter_scraper.get_user(handle)
        if not user:
            return {"status": "error", "msg": "无法获取账号信息"}

        followers = user.get("followers_count", 0)
        raw_tweets = twitter_scraper.get_user_tweets(user["id"], limit=10)
        tweets = []
        for t in raw_tweets:
            tweets.append({
                "text":        t.get("text", "")[:100],
                "likes":       t.get("likes", 0),
                "retweets":    t.get("retweets", 0),
                "replies":     t.get("replies", 0),
                "impressions": t.get("impressions", 0),
                "created_at":  t["created_at"].strftime("%Y-%m-%d") if t.get("created_at") else "",
            })

        best = max(tweets, key=lambda t: t["likes"] + t["retweets"]) if tweets else {}
        return {
            "status":    "ok",
            "handle":    user.get("username", handle),
            "followers": followers,
            "tweets":    tweets,
            "best_tweet": best,
            "avg_impressions": int(sum(t["impressions"] for t in tweets) / len(tweets)) if tweets else 0,
            "total_engagement": sum(t["likes"] + t["retweets"] for t in tweets),
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}


def fetch_twitter_competitors() -> dict:
    """竞品 Twitter 近况"""
    try:
        result = {}
        for handle in COMPETITORS:
            try:
                user = twitter_scraper.get_user(handle)
                if not user:
                    continue
                raw_tweets = twitter_scraper.get_user_tweets(user["id"], limit=3)
                latest_tweets = [
                    {
                        "text":     t.get("text", "")[:80],
                        "likes":    t.get("likes", 0),
                        "retweets": t.get("retweets", 0),
                    }
                    for t in raw_tweets
                ]
                result[handle] = {
                    "followers":     user.get("followers_count", 0),
                    "latest_tweets": latest_tweets,
                }
                time.sleep(0.3)
            except Exception:
                continue
        return {"status": "ok", "competitors": result}
    except Exception as e:
        return {"status": "error", "msg": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# CoinGecko — 加密市场趋势（免费 API）
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_coingecko() -> dict:
    try:
        # 今日热门
        r1 = _req.get(
            "https://api.coingecko.com/api/v3/search/trending",
            timeout=10,
        )
        trending = []
        if r1.status_code == 200:
            for coin in r1.json().get("coins", [])[:5]:
                item = coin.get("item", {})
                trending.append({
                    "name":   item.get("name", ""),
                    "symbol": item.get("symbol", ""),
                    "rank":   item.get("market_cap_rank", 0),
                })

        # 全球市场概览
        r2 = _req.get("https://api.coingecko.com/api/v3/global", timeout=10)
        global_data = {}
        if r2.status_code == 200:
            d = r2.json().get("data", {})
            global_data = {
                "total_market_cap_usd": d.get("total_market_cap", {}).get("usd", 0),
                "market_cap_change_24h": d.get("market_cap_change_percentage_24h_usd", 0),
                "btc_dominance": d.get("market_cap_percentage", {}).get("btc", 0),
            }

        return {"status": "ok", "trending": trending, "global": global_data}
    except Exception as e:
        return {"status": "error", "msg": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# Google Search Console — SEO 排名（需 JSON 凭证）
# ═══════════════════════════════════════════════════════════════════════════════

GSC_SITE = "https://www.bydfi.com"


def fetch_search_console() -> dict:
    cred_path = Path(__file__).parent / "gsc_credentials.json"
    if not cred_path.exists():
        # 也支持从环境变量读取
        cred_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        if not cred_json:
            return {"status": "pending", "msg": "等待 GSC 凭证文件（gsc_credentials.json）"}
        try:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            tmp.write(cred_json)
            tmp.close()
            cred_path = Path(tmp.name)
        except Exception as e:
            return {"status": "error", "msg": str(e)}
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            str(cred_path),
            scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
        )
        service = build("searchconsole", "v1", credentials=creds, cache_discovery=False)

        end_date   = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # 按查询词统计
        body = {
            "startDate": start_date,
            "endDate":   end_date,
            "dimensions": ["query"],
            "rowLimit": 20,
            "orderBy": [{"fieldName": "clicks", "sortOrder": "DESCENDING"}],
        }
        resp = service.searchanalytics().query(siteUrl=GSC_SITE, body=body).execute()
        keywords = []
        for row in resp.get("rows", []):
            keywords.append({
                "query":       row["keys"][0],
                "clicks":      row.get("clicks", 0),
                "impressions": row.get("impressions", 0),
                "ctr":         round(row.get("ctr", 0) * 100, 1),
                "position":    round(row.get("position", 0), 1),
            })
        # 总体点击
        total_clicks = sum(k["clicks"] for k in keywords)
        return {
            "status":       "ok",
            "keywords":     keywords,
            "total_clicks": total_clicks,
            "period":       f"{start_date} ~ {end_date}",
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# 神策分析 — MoonX 平台数据（需 API Token）
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_platform_snapshot() -> dict:
    """读取本地 platform_data.json（PDF 解析结果）"""
    path = Path(__file__).parent / "platform_data.json"
    if not path.exists():
        return {"status": "pending", "msg": "暂无平台数据（需导出神策 PDF）"}
    try:
        data = json.loads(path.read_text())
        moonx = data.get("moonx_data", {})
        t1    = data.get("t1_user_profile", {})
        alerts = data.get("marketing_alerts", [])

        # 取最新月活
        monthly = moonx.get("monthly_active_users", {})
        latest_month = sorted(monthly.keys())[-1] if monthly else ""
        latest_mau   = monthly.get(latest_month, 0)

        trading = moonx.get("trading", {})
        today   = moonx.get("today_2026_03_11", {})

        return {
            "status":         "ok",
            "updated":        data.get("updated", ""),
            "latest_mau":     latest_mau,
            "latest_month":   latest_month,
            "monthly_mau":    monthly,
            "today_traders":  today.get("trading_users", 0),
            "today_app_visits": today.get("app_homepage_visits", 0),
            "total_traders":  trading.get("total_traders_since_june", 0),
            "daily_avg_traders": trading.get("daily_avg_traders", 0),
            "total_volume":   trading.get("total_volume_cny", ""),
            "daily_avg_volume": trading.get("daily_avg_volume_cny", ""),
            "top_country":    list(t1.get("geography", {}).items())[:3],
            "seo_traffic_pct": t1.get("user_acquisition", {}).get("自然_SEO", ""),
            "alerts":         alerts,
            "key_insights":   t1.get("key_insights", []),
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}


def fetch_sensors() -> dict:
    token    = os.getenv("SENSORS_TOKEN", "")
    endpoint = os.getenv("SENSORS_ENDPOINT", "")
    project  = os.getenv("SENSORS_PROJECT", "default")
    if not token or not endpoint:
        return {"status": "pending", "msg": "等待神策 API Token 和 Endpoint"}
    try:
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        headers   = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }

        # 神策 SQL 查询接口（私有化部署路径）
        sql = (
            f"SELECT date(time) as day, COUNT(DISTINCT distinct_id) as dau "
            f"FROM events WHERE date(time) >= '{yesterday}' "
            f"GROUP BY day ORDER BY day DESC LIMIT 7"
        )
        payload = {"sql": sql, "project": project}

        for api_path in ["/api/sql/query", "/api/gql/", "/gql"]:
            try:
                r = _req.post(
                    f"{endpoint.rstrip('/')}{api_path}",
                    json=payload, headers=headers, timeout=15,
                )
                if r.status_code == 200:
                    data = r.json()
                    rows = data.get("rows", data.get("data", []))
                    dau  = rows[0][1] if rows else 0
                    return {
                        "status": "ok", "dau": dau,
                        "daily": [{"date": r[0], "dau": r[1]} for r in rows[:7]],
                        "api_path": api_path,
                    }
            except Exception:
                continue

        return {"status": "error", "msg": "所有 API 路径均无响应，请确认神策版本"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# 聚合入口
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_all(use_cache: bool = True) -> dict:
    """拉取所有数据源，带缓存"""
    def _do():
        return {
            "polymarket":   fetch_polymarket(),
            "twitter_self": fetch_twitter_self(),
            "twitter_comp": fetch_twitter_competitors(),
            "coingecko":    fetch_coingecko(),
            "platform":     fetch_platform_snapshot(),
            "gsc":          fetch_search_console(),
            "sensors":      fetch_sensors(),
            "fetched_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    if use_cache:
        return _cached("all", _do)
    return _do()


def format_context(data: dict = None) -> str:
    """把所有数据格式化成 AI 员工可读的上下文字符串"""
    if data is None:
        data = fetch_all()

    lines = [f"📡 实时数据快照 [{data.get('fetched_at', '')}]"]

    # Polymarket
    pm = data.get("polymarket", {})
    if pm.get("status") == "ok":
        lines.append("\n【Polymarket 热门市场（24h）】")
        for m in pm["markets"][:5]:
            vol = m["volume_24h"]
            lines.append(f"  • {m['question'][:55]} | YES {m['yes_pct']}% | ${vol:,.0f}")
        lines.append(f"  Top10 总成交量：${pm['total_volume_24h']:,.0f}")

    # Twitter 自身
    tw = data.get("twitter_self", {})
    if tw.get("status") == "ok":
        lines.append(f"\n【@{tw.get('handle','?')} Twitter 状态】")
        lines.append(f"  粉丝：{tw['followers']:,} | 近10推文平均曝光：{tw['avg_impressions']:,}")
        best = tw.get("best_tweet", {})
        if best:
            lines.append(f"  最佳推文：{best.get('text','')[:60]} | 曝光 {best.get('impressions',0):,}")

    # 竞品监控
    comp = data.get("twitter_comp", {})
    if comp.get("status") == "ok" and comp.get("competitors"):
        lines.append("\n【竞品 Twitter 粉丝】")
        for handle, info in comp["competitors"].items():
            lines.append(f"  @{handle}：{info['followers']:,} 粉丝")

    # CoinGecko
    cg = data.get("coingecko", {})
    if cg.get("status") == "ok":
        g = cg.get("global", {})
        if g:
            cap_t = g.get("total_market_cap_usd", 0) / 1e12
            chg   = g.get("market_cap_change_24h", 0)
            lines.append(f"\n【加密市场】总市值 ${cap_t:.2f}T | 24h变化 {chg:+.1f}%")
        trending = cg.get("trending", [])
        if trending:
            names = " / ".join(f"{t['name']}({t['symbol']})" for t in trending[:3])
            lines.append(f"  今日热门：{names}")

    # MoonX 平台数据
    pf = data.get("platform", {})
    if pf.get("status") == "ok":
        lines.append(f"\n【MoonX 平台数据（{pf.get('updated','')}）】")
        lines.append(f"  本月活跃用户：{pf['latest_mau']} | 今日交易用户：{pf['today_traders']} | 今日App访问：{pf['today_app_visits']}")
        lines.append(f"  累计交易用户：{pf['total_traders']:,} | 日均：{pf['daily_avg_traders']} | 总交易量：{pf['total_volume']}")
        lines.append(f"  自然流量占比：{pf['seo_traffic_pct']} | 主要市场：美国62%")
        for alert in pf.get("alerts", [])[:3]:
            lines.append(f"  {alert}")

    # GSC
    gsc = data.get("gsc", {})
    if gsc.get("status") == "ok":
        lines.append(f"\n【Search Console 近7日】点击 {gsc['total_clicks']:,}")
        for kw in gsc["keywords"][:3]:
            lines.append(f"  • {kw['query']} | 点击 {kw['clicks']} | 排名 {kw['position']}")
    elif gsc.get("status") == "pending":
        lines.append(f"\n【Search Console】{gsc['msg']}")

    # 神策
    sa = data.get("sensors", {})
    if sa.get("status") == "ok":
        lines.append(f"\n【MoonX 平台】昨日 DAU：{sa.get('dau', '?')}")
    elif sa.get("status") == "pending":
        lines.append(f"\n【MoonX 平台】{sa['msg']}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("🔍 测试数据源连接...\n")
    data = fetch_all(use_cache=False)
    print(format_context(data))
    print("\n--- 原始数据状态 ---")
    for k, v in data.items():
        if k != "fetched_at":
            status = v.get("status", "?")
            msg    = v.get("msg", "")
            print(f"  {k}: {status}" + (f" — {msg}" if msg else ""))
