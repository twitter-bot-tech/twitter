# /insights — 每周推文表现分析与优化建议

你是 MoonX Twitter 账号 @moonx_bydfi 的运营分析师。

## 任务

分析过去 7 天的推文表现，输出结构化洞察 + 下周优化建议。

## 执行步骤

### Step 1：拉取本周推文数据

运行以下 Python 代码，从 tweet_ids.json 和 tweet.log 提取本周发出的推文列表：

```python
import json, re
from pathlib import Path
from datetime import datetime, timedelta, timezone

cutoff = datetime.now(timezone.utc) - timedelta(days=7)

# 从 tweet_ids.json 拿文案
tweets = json.loads(Path('/Users/coco/agent-twitter/01_social_media/tweet_ids.json').read_text())
recent = [t for t in tweets if datetime.fromisoformat(t['posted_at']) >= cutoff]
print(f"本周发推数: {len(recent)}")
for t in recent:
    ts = datetime.fromisoformat(t['posted_at']).strftime('%m-%d %H:%M')
    print(f"[{ts}] {t['id']} | {t['text'][:80]}")
```

### Step 2：通过 Twitter API 拉取每条推文的互动数据

```python
import os
from pathlib import Path
from dotenv import load_dotenv
import tweepy

load_dotenv(Path('/Users/coco/agent-twitter/.env'))

client = tweepy.Client(
    bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
    consumer_key=os.getenv('TWITTER_API_KEY'),
    consumer_secret=os.getenv('TWITTER_API_SECRET'),
    access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
    access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
)

# 拉取本周推文 ID 的互动指标
tweet_ids = [t['id'] for t in recent]
results = []
for tid in tweet_ids:
    try:
        resp = client.get_tweet(
            tid,
            tweet_fields=['public_metrics', 'created_at', 'text'],
        )
        m = resp.data.public_metrics
        results.append({
            'id': tid,
            'text': resp.data.text[:100],
            'created_at': resp.data.created_at,
            'likes': m['like_count'],
            'retweets': m['retweet_count'],
            'replies': m['reply_count'],
            'impressions': m['impression_count'],
            'engagement_rate': round((m['like_count'] + m['retweet_count'] + m['reply_count']) / max(m['impression_count'], 1) * 100, 2),
        })
    except Exception as e:
        print(f"跳过 {tid}: {e}")

# 按 engagement_rate 排序
results.sort(key=lambda x: x['engagement_rate'], reverse=True)
for r in results:
    print(f"ER:{r['engagement_rate']}% | 👍{r['likes']} RT{r['retweets']} 💬{r['replies']} 👁{r['impressions']} | {r['text'][:60]}")
```

### Step 3：分析 + 输出报告

根据以上数据，按以下结构输出本周 Insights 报告：

---

## 📊 MoonX Twitter 本周 Insights（{本周日期范围}）

### 概览
- 本周发推：N 条
- 平均互动率：X%
- 总曝光：Xk

### 🏆 本周 Top 3 推文
列出互动率最高的 3 条，注明：话题类型、发布时间、为什么表现好

### ❌ 表现最差的推文
列出互动率最低的 1-2 条，分析原因

### 📈 规律发现
从以下维度找规律（有数据支撑才写，没有就跳过）：
- **话题类型**：政治类 vs 体育类 vs 加密类，哪类 ER 更高？
- **发布时间**：哪个时间段互动更好？
- **文案风格**：问句结尾 vs hashtag 结尾 vs 无结尾，哪类更好？
- **有图 vs 无图**：配图推文的曝光 vs 纯文字

### 🎯 下周优化建议（3 条，具体可执行）
每条建议格式：
> **[改什么]** 原来做法 → 建议做法。**理由**：数据支撑一句话。

---

## 注意事项
- 数据不够（< 5 条推文）时，降低置信度，建议"仅供参考"
- Twitter API 拉不到数据时，基于 tweet.log 的发推记录做定性分析
- 不要猜测，只说数据能支撑的结论

### Step 4：自动迭代 CLAUDE.md

报告输出后，对比上次 insights 的摩擦点：
1. 读取 `CLAUDE.md` 现有规则
2. 根据本周新发现，判断是否需要增删规则（例如：某类话题 ER 持续低 → 加入"减少此类话题"规则）
3. 有变更时直接修改 CLAUDE.md，并在对话中说明改了什么、为什么
4. 没有新发现时跳过，不做无意义修改

### Step 5：更新 memory 时间戳

执行完后，更新 memory 文件 `recurring_tasks.md` 中 `insights` 的 `last_run` 为今天日期。
