# Current Projects — MoonX 营销团队

## 进行中

### 1. KOL 媒体外联（持续）
- **目标**：每周接触 ≥5 个 Web3 KOL，月签约 ≥2 个
- **工具**：`03_kol_media/web3_kol_scraper.py`（每日自动收集，6 渠道）
- **数据**：Google Sheets `GOOGLE_SHEET_ID_KOL`（每日自动同步）
- **渠道**：Twitter、YouTube、Telegram、CoinTelegraph、CoinDesk、Lunarcrush
- **当前进展**：自动化已上线，每日收集 ~100 KOL，Telegram 和 CoinTelegraph 产出最多

### 2. Twitter 内容运营（持续）
- **目标**：@moonx_bydfi 每日 2-3 条高质量推文
- **工具**：`scheduler.py`（launchd 调度），`01_social_media/` 脚本
- **内容类型**：市场行情、产品更新、KOL 互动、热点跟进

### 3. SEO 产品需求（进行中 → 待开发评审）
- **PRD**：`outbox/seo/moonx-seo-prd/prd/prd_v1.0.html`（v1.1，原型已确认，待开发评审）
- **原型**：`outbox/seo/moonx-seo-prd/prototype/prototype_v1.0.html`（已嵌入 PRD）
- **流程图**：`outbox/seo/moonx-seo-prd/flowcharts/flowcharts_v1.0.html`（v1.1，4 张 Mermaid 图）
- **接口文档**：`outbox/seo/moonx-seo-prd/annex/MoonX_SEO_接口需求文档_v3_source.html`（V4.0，17 章，含文章接口 + CMS API + Predict 按钮接口）
- **待确认**：Meme 学院 URL 路径（/learn/meme vs /en/moonx/guide/buy）、DEX 报价接口、批量预测市场查询方案
- **下一步**：提交开发评审

### 4. 增长实验（规划中）
- **目录**：`04_growth/`
- **目标**：用户获客漏斗优化，新用户转化率提升

## 基础设施
- **Lark 机器人**：每日自动推送各业务线报告（KOL/SEO/Social/Growth/Strategy）
- **服务器**：Fly.io Singapore `moonx-lark-server`（Flask）
- **调度**：macOS launchd + APScheduler

## 关键链接
- Google Sheet KOL 名单：`https://docs.google.com/spreadsheets/d/1fC3p_LG3psiBwuFupLOuJCPMWIYlj0UXZ5Ow2W9yWQo`
- Lark KOL 群：`LARK_KOL` webhook 已配置
