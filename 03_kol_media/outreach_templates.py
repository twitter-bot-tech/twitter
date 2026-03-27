#!/usr/bin/env python3
"""
外联话术模板 — GMGN 预测市场
包含：KOL DM 模板、媒体投稿邮件模板、跟进邮件模板
"""

# ══════════════════════════════════════════
# KOL DM 模板（Twitter/X 私信）
# ══════════════════════════════════════════

KOL_DM_CRYPTO = """
Hey {name} 👋

Big fan of your work on prediction markets — your take on {recent_topic} was spot on.

I'm reaching out because we're building something you might find interesting: a platform that tracks smart money moves across prediction markets (think GMGN, but for Polymarket/Kalshi).

We're onboarding a small group of early KOL partners before our public launch. Here's what we're offering:
• Exclusive early access to our smart money dashboard
• Revenue share on every user you refer
• Custom analytics reports tailored to your content

Would love to hop on a quick call or share more details if you're curious.

Cheers,
{sender_name}
{platform_name}
"""

KOL_DM_STOCKS = """
Hey {name},

Love your coverage of {recent_topic} — really sharp analysis.

We're launching a prediction market aggregator that tracks where smart money is flowing on events like Fed decisions, earnings calls, and macro moves. Think of it as Bloomberg Terminal meets Polymarket.

We're looking for a few sharp finance voices to partner with for our launch:
• Early access to our smart money tracking dashboard
• Revenue share on referrals
• Co-branded market reports you can share with your audience

Interested in learning more? Happy to send over details or jump on a quick call.

Best,
{sender_name}
{platform_name}
"""

# ══════════════════════════════════════════
# 媒体投稿邮件模板
# ══════════════════════════════════════════

MEDIA_EMAIL_CRYPTO = """
Subject: Exclusive: New Platform Brings Smart Money Tracking to Prediction Markets

Hi {editor_name},

I'm reaching out from {platform_name}, a new prediction market aggregator that's bringing the "smart money" tracking experience — familiar to crypto traders via tools like GMGN — to platforms like Polymarket and Kalshi.

Why this matters for your readers:
• Prediction markets now process $800M+ in daily volume with 100K+ daily active traders
• Until now, there's been no easy way to track where sophisticated money is moving
• We're the first platform to aggregate smart money signals across multiple prediction markets

We'd love to offer {media_name} an exclusive first look before our public launch, including:
• Interview with our founding team
• Early access to our smart money dashboard
• Proprietary data on prediction market flows

Would you be open to a briefing this week? Happy to work around your schedule.

Best regards,
{sender_name}
{sender_title}
{platform_name}
{sender_email}
"""

MEDIA_EMAIL_FINANCE = """
Subject: New Fintech Platform Lets Retail Investors Track Smart Money in Prediction Markets

Hi {editor_name},

I wanted to introduce {platform_name} — a new platform that aggregates prediction market data and shows retail investors where institutional and sophisticated money is flowing on events like Fed rate decisions, earnings surprises, and election outcomes.

Key data points that might interest your readers:
• Global prediction markets: $800M+ daily volume, growing 40% YoY
• Our platform aggregates signals from Polymarket, Kalshi, and 5+ other markets
• Early users are seeing 23% better returns by following smart money signals

We'd love to offer {media_name} an exclusive story opportunity, including data access and founder interviews.

Would you have 20 minutes for a briefing this week?

Best,
{sender_name}
{sender_title}
{platform_name}
{sender_email}
"""

# ══════════════════════════════════════════
# 跟进邮件（7天后无回复）
# ══════════════════════════════════════════

FOLLOW_UP_EMAIL = """
Subject: Re: {original_subject}

Hi {name},

Just wanted to bump this up in case it got buried.

We're finalizing our launch partner list this week and wanted to make sure you had a chance to consider it before we close the early access window.

Happy to make it super easy — even a 15-min call or a quick email exchange works.

Thanks for your time either way!

{sender_name}
{platform_name}
"""

# ══════════════════════════════════════════
# 签约合作条款要点（谈判话术）
# ══════════════════════════════════════════

PARTNERSHIP_TERMS = """
合作条款要点（KOL 谈判用）

【我们提供】
1. 独家早期访问权限（抢在公众前使用产品）
2. 专属推荐码 + 佣金分成（每成功注册用户的 X%）
3. 定制化聪明钱数据报告（可直接用于内容创作）
4. 联合品牌活动机会

【我们期望】
1. 每月至少 2 条提及我们产品的原创内容
2. 使用专属推荐码（用于追踪转化）
3. 内容需标注 #Ad 或 #Sponsored（合规要求）

【谈判底线】
- 头部KOL（>50万粉）：可提供固定费用 + 分成
- 中腰部KOL（1-10万粉）：纯分成模式，无固定费用
- 媒体：提供独家数据 + 采访机会，不支付稿费
"""

# ══════════════════════════════════════════
# BD 议价话术（Telegram / TG 谈判场景）
# ══════════════════════════════════════════

# 标准返佣方案说明（首次介绍时用）
BD_REVSHARE_INTRO_EN = """
Here's what we can offer:

- ${upfront} upfront per video
- Plus rev-share on top:
  - Futures referrals → 65% to you (industry standard is 20–30%)
  - Meme coin referrals → 50%
  - Prediction market referrals → 40%

To explain how the rev-share works: every time a viewer signs up through your link and trades, you earn a cut of their trading fees — ongoing, not one-time. The 65% on futures is one of the highest rates in the space.

This is our founding partner rate. We offer more because we want quality partners, not volume.
"""

BD_REVSHARE_INTRO_ZH = """
我们的合作方案如下：

- 每个视频 ${upfront} 预付
- 额外叠加分成返佣：
  - 合约引流 → 65%（行业普遍是 20-30%）
  - Meme 币引流 → 50%
  - 预测市场引流 → 40%

返佣逻辑：你的受众通过你的链接注册后，每次交易你都能拿到手续费分成，持续收入，不是一次性的。65% 合约返佣是市场上最高档之一。

这是创始合伙人专属费率，我们宁可少合作，要质量不要数量。
"""

# 对方要求报价时，先反问让对方开价
BD_ASK_RATE_EN = "Thanks! What's your typical rate per video? Let's start there."
BD_ASK_RATE_ZH = "好的，你们单个视频的报价是多少？先了解一下。"

# 对方报价后，要求先看数据再谈预算
BD_ASK_STATS_EN = """
Before we confirm budget, could you share your channel stats with screenshots — subscribers, average views per video, and audience country breakdown? Want to make sure it's a good fit for MoonX's target users.
"""
BD_ASK_STATS_ZH = """
确认预算前，能发一下你的频道数据截图吗——订阅数、平均播放量、受众国家分布（有截图最好）？确认和 MoonX 目标受众的匹配度后再推进。
"""

# 对方问目标市场时
BD_TARGET_MARKET_EN = "Our primary target markets are the US, Europe, and Southeast Asia. Does your audience align with these regions?"
BD_TARGET_MARKET_ZH = "我们的主要目标市场是美国、欧洲和东南亚。你的受众主要来自这些地区吗？"

# 让对方先报价（报价前用）
BD_ASK_RATE_AFTER_STATS_EN = "Thanks for sharing! What's your typical rate per video? Let's start there."
BD_ASK_RATE_AFTER_STATS_ZH = "感谢分享！你们单个视频的报价是多少？先了解一下。"

# 对方要求预付但价格偏高 → 提议先纯分成
BD_COUNTER_REVSHARE_ONLY_EN = """
That's a bit above our current budget for new partners. Would you consider starting with rev-share only for the first video? If the results are good, we're happy to discuss paid packages after.
"""
BD_COUNTER_REVSHARE_ONLY_ZH = """
这个价格对新合作来说略超我们预算。能否考虑第一个视频先纯分成合作？效果好的话，后续我们很愿意谈付费套餐。
"""

# 提出 $50 现金 + $50 期货 bonus 方案
BD_OFFER_50_PLUS_BONUS_EN = """
I hear you on the production costs. Let me help you out — I'll apply for $50 cash + $50 futures bonus credit for you. The bonus credit can be used directly for futures trading on MoonX, so it's real value you can put to work. That brings the total to $100, plus the 65% rev-share on top. Let me know if that works!
"""
BD_OFFER_50_PLUS_BONUS_ZH = """
理解你们的制作成本压力。我来帮你申请 $50 现金 + $50 合约交易 bonus，bonus 可以直接在 MoonX 上用于合约交易，是真实可用的价值。这样总计 $100，再加上 65% 返佣。你看这样可以吗？
"""

# 受众地区不匹配时，用来压低预付
BD_AUDIENCE_MISMATCH_EN = """
Thanks for sharing the stats! Looking at your audience data, most of your viewers are from {regions} — these markets aren't our primary target right now, so our budget allocation for this region is limited. The ${upfront} upfront is the most we can offer given that. The rev-share is where the real upside is if your audience engages. Let me know if you'd like to move forward on that basis.
"""
BD_AUDIENCE_MISMATCH_ZH = """
感谢分享数据！看了你的受众分布，主要集中在 {regions}——这些地区目前不是我们的核心目标市场，公司在这个方向的预算比较有限。${upfront} 预付已经是我们能给的上限了。返佣才是真正的长期收益空间，如果你的受众活跃度高，收入会持续增长。你看这样能接受吗？
"""

# 订阅高但播放量低，守住预算上限
BD_LOW_VIEWS_HOLD_EN = """
I appreciate your offer, but based on your current views-to-subscriber ratio, this is the maximum I'm able to apply for internally. The $100 cash + $50 futures bonus + 65% rev-share is our best package at this tier. Happy to revisit the upfront after we see how the first video performs.
"""
BD_LOW_VIEWS_HOLD_ZH = """
理解你的立场，但基于你目前的播放量和订阅数比例，这已经是我内部能申请到的上限了。$100 现金 + $50 期货 bonus + 65% 返佣是这个量级合作伙伴的最优方案。第一个视频跑完数据后，我们可以再聊涨预付的事。
"""

# 对方继续还价，表示尝试申请但不保证
BD_TRY_APPLY_EN = """
I hear you. Let me try to push for $150 cash + $50 futures bonus internally — no guarantees, but I'll see what I can do. Will get back to you shortly.
"""
BD_TRY_APPLY_ZH = """
理解你的立场。我去内部帮你争取一下 $150 现金 + $50 期货 bonus，不保证能批下来，但我会尽力。稍后回复你。
"""

# 市场行情不好，申请不下来高预算
BD_BUDGET_CUT_EN = """
Honestly, ${amount} is something I can't get approved internally — the market has been rough and everyone is cutting budgets right now. The $100 cash + $50 futures bonus is genuinely the ceiling I'm working with. I know it's not ideal, but the 65% rev-share is where the real long-term value is. Let me know if you want to give it a shot.
"""
BD_BUDGET_CUT_ZH = """
说实话，${amount} 我内部申请不下来——现在行情不好，大家都在缩减预算。$100 现金 + $50 期货 bonus 真的是我能拿到的上限。我知道不是你期望的数字，但 65% 返佣才是长期收益的关键。你看要不要试一下？
"""

# 差距太大，礼貌收场留后路
BD_CLOSE_GRACEFULLY_EN = "Totally understand — no hard feelings at all! Let's stay in touch and revisit when our budget opens up. Looking forward to collaborating in the future!"
BD_CLOSE_GRACEFULLY_ZH = "完全理解，没关系！保持联系，等我们预算恢复后再聊。期待以后有合作机会！"

# 截图需包含频道名称（存档用）
BD_ASK_CHANNEL_SCREENSHOT_EN = "Could you take a screenshot that includes your channel name? I need it for our internal records. Thanks!"
BD_ASK_CHANNEL_SCREENSHOT_ZH = "能截一张包含你频道名称的图吗？我需要提交内部存档用，谢谢！"

# 预测市场上线时间节点（告知对方等待）
BD_LAUNCH_DATE_EN = """
Great! Our prediction market feature launches on April 7. We're preparing the campaign and content brief before then. I'll send you all the details — referral link, talking points, and assets — closer to the launch date. Stay tuned!
"""
BD_LAUNCH_DATE_ZH = """
太好了！我们的预测市场功能将于 4 月 7 日上线。上线前我们会准备好活动方案和内容素材，到时会把推广链接、内容要点和素材一并发给你，请稍等！
"""

# ══════════════════════════════════════════════════════════════
# 媒体询价模板（Sponsored Content / Brand Placement）
# 场景：向媒体询问赞助内容刊载报价和受众数据
# ══════════════════════════════════════════════════════════════

# ── 套一：加密媒体（Decrypt / CoinDesk / BeInCrypto 等）──────────────────────
# 切入点：预测市场 + 链上智能资金，读者本身是币圈用户

MEDIA_INQUIRY_CRYPTO_EN = """\
Subject: Sponsored Content Inquiry — MoonX x {media_name}

Hi {contact_name},

I'm Kelly, Head of Marketing at BYDFi. We recently launched MoonX \
(https://www.bydfi.com/en/moonx/markets/trending) — a real-time smart money \
terminal that aggregates whale flows across Polymarket, Kalshi, and on-chain \
prediction markets into one feed.

We're planning a sponsored content campaign targeting crypto traders in the \
US, Europe, and Southeast Asia, and {media_name} is at the top of our list \
given your readership in this space.

A few things we're looking to understand:

1. Do you offer sponsored articles / native content placements?
2. What are your typical rates (CPM, flat fee, or per placement)?
3. Could you share a media kit or audience breakdown (monthly unique visitors, \
geo split, reader demographics)?
4. What's the lead time for booking a slot?

We're moving quickly — ideally looking to kick off in the next 2–3 weeks. \
Happy to jump on a quick call if that's easier.

Thanks,
Kelly
Head of Marketing | BYDFi MoonX
{sender_email} | TG: @BDkelly\
"""

MEDIA_INQUIRY_CRYPTO_ZH = """\
主题：赞助内容合作询价 — MoonX x {media_name}

你好 {contact_name}，

我是 Kelly，BYDFi 营销负责人。我们近期上线了 MoonX \
（https://www.bydfi.com/en/moonx/markets/trending）——一个聚合 Polymarket、\
Kalshi 及链上预测市场鲸鱼资金流向的实时智能资金终端。

我们计划针对欧美和东南亚加密交易用户做一轮赞助内容投放，{media_name} \
是我们的首选媒体之一。

想了解以下几点：

1. 你们是否提供赞助文章 / 原生内容刊载？
2. 通常的报价方式是什么（CPM、按篇固定费用，还是其他）？
3. 能否提供媒体包或受众数据（月独立访客量、地区分布、读者画像）？
4. 预定档期的提前周期是多少？

我们时间比较紧，希望 2-3 周内能启动。如果电话沟通更方便，随时可以安排。

谢谢，
Kelly
营销负责人 | BYDFi MoonX
{sender_email} | TG: @BDkelly\
"""

# ── 套二：财经媒体（Bloomberg / Forbes / Axios / FT 等）─────────────────────
# 切入点：美股+大选预测市场，读者是传统金融受众

MEDIA_INQUIRY_FINANCE_EN = """\
Subject: Sponsored Content Inquiry — Prediction Markets Platform Targeting \
Your Finance Audience

Hi {contact_name},

I'm Kelly, Head of Marketing at BYDFi. We run MoonX \
(https://www.bydfi.com/en/moonx/markets/trending) — a platform that helps \
retail investors track where sophisticated money is flowing across prediction \
markets, including US stock events, Fed decisions, and election outcomes on \
Polymarket and Kalshi.

We're looking to run a sponsored content campaign targeted at your finance \
and investment-focused readership in the US and Europe.

Could you help us with the following:

1. Do you offer sponsored / branded content placements (native articles, \
newsletter sponsorships, display)?
2. What are your rates — CPM, flat-fee per placement, or package pricing?
3. Do you have a media kit with audience data (monthly reach, geo split, \
income/age demographics)?
4. What's your typical booking lead time?

We're targeting a Q2 campaign launch and are actively comparing placements \
across several outlets. Happy to discuss further over a quick call.

Best,
Kelly
Head of Marketing | BYDFi MoonX
{sender_email} | TG: @BDkelly\
"""

MEDIA_INQUIRY_FINANCE_ZH = """\
主题：赞助内容合作询价 — 预测市场平台欲投放财经受众

你好 {contact_name}，

我是 Kelly，BYDFi 营销负责人。我们运营 MoonX \
（https://www.bydfi.com/en/moonx/markets/trending）——帮助散户投资者追踪 \
Polymarket、Kalshi 等预测市场上机构和智能资金的流向，覆盖美股事件、美联储 \
决议、选举结果等热点场景。

我们计划在欧美财经类媒体上投放一轮赞助内容，面向你们的投资者读者群体，\
{media_name} 是我们重点考量的媒体之一。

希望了解以下信息：

1. 是否提供赞助 / 品牌内容刊载（原生文章、Newsletter 赞助、展示广告）？
2. 报价方式是什么——CPM、按篇固定费用，还是套餐定价？
3. 有没有含受众数据的媒体包（月触达量、地区分布、收入/年龄画像）？
4. 预定档期通常需要提前多久？

我们计划 Q2 启动投放，目前在多家媒体间做横向比较。有需要可以约一个快速通话。

谢谢，
Kelly
营销负责人 | BYDFi MoonX
{sender_email} | TG: @BDkelly\
"""

# ── 跟进邮件（5天无回复后使用）────────────────────────────────────────────────

MEDIA_INQUIRY_FOLLOWUP_EN = """\
Subject: Re: Sponsored Content Inquiry — MoonX x {media_name}

Hi {contact_name},

Just following up on my note from {days_ago} days ago — wanted to make sure \
it didn't get buried.

We're finalizing our Q2 media plan this week and {media_name} is still on our \
shortlist. Even a rough ballpark on rates would help us move forward.

Happy to keep it brief — a media kit or a one-liner on pricing works too.

Thanks,
Kelly | BYDFi MoonX | {sender_email}\
"""

MEDIA_INQUIRY_FOLLOWUP_ZH = """\
主题：回复：赞助内容合作询价 — MoonX x {media_name}

你好 {contact_name}，

跟进一下 {days_ago} 天前发的那封邮件，怕被淹没了。

我们这周在敲定 Q2 媒体投放计划，{media_name} 仍在我们的候选名单里。\
哪怕是一个大概的报价范围也能帮助我们推进决策。

不需要太复杂，发一份媒体包或者直接告诉我价格区间就可以。

谢谢，
Kelly | BYDFi MoonX | {sender_email}\
"""


if __name__ == "__main__":
    # 示例：生成一份 KOL DM
    print("=" * 50)
    print("KOL DM 示例（加密方向）")
    print("=" * 50)
    print(KOL_DM_CRYPTO.format(
        name="Alex",
        recent_topic="Polymarket's US election volume",
        sender_name="Your Name",
        platform_name="GMGN Prediction Markets"
    ))
    print("\n" + "=" * 50)
    print("媒体邮件示例（加密媒体）")
    print("=" * 50)
    print(MEDIA_EMAIL_CRYPTO.format(
        editor_name="Editor",
        platform_name="GMGN Prediction Markets",
        media_name="Decrypt",
        sender_name="Your Name",
        sender_title="Head of Marketing",
        sender_email="marketing@yourplatform.com"
    ))
