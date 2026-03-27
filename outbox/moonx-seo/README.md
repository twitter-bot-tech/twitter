# MoonX SEO Platform — Next.js Prototype

A full-stack Next.js 14 prototype implementing all 7 SEO page types + CMS backend for MoonX (prediction market aggregator).

## Quick Start

```bash
cd outbox/moonx-seo
npm install
npm run db:migrate       # creates SQLite dev.db + runs migrations
npm run db:seed          # seeds 3 articles + 3 templates + 2 banners + 3 tokens
npm run dev              # start dev server at http://localhost:3000
```

## Verify SSR (Google can index)

```bash
# Check SSR — should return page content, not empty HTML
curl -s http://localhost:3000/en/moonx/learn/prediction/polymarket-vs-kalshi | grep -i "polymarket vs kalshi"

# Check JSON-LD schema injection
curl -s http://localhost:3000/en/moonx/learn/prediction/polymarket-vs-kalshi | grep "application/ld+json"

# Check sitemap
curl http://localhost:3000/sitemap.xml

# Check noindex on dead token
curl -s http://localhost:3000/en/moonx/solana/token/DEAD123 | grep "noindex"
```

## Pages Implemented

| Route | Type | REQ |
|-------|------|-----|
| `/en/moonx/learn/prediction` | Article list | REQ-01 |
| `/en/moonx/learn/prediction/[slug]` | Article detail | REQ-02 |
| `/en/moonx/learn/meme` | Meme list | REQ-03 |
| `/en/moonx/learn/meme/[slug]` | Meme detail | REQ-04 |
| `/en/moonx/guide/prediction/[event-slug]` | Event guide (programmatic) | REQ-05 |
| `/en/moonx/markets/stocks/[ticker]` | Stock prediction (programmatic) | REQ-06 |
| `/en/moonx/solana/token/[contract]` | Meme token page (programmatic) | REQ-07 |
| `/en/moonx/markets/trending` | Trending list | REQ-08 |

## Admin CMS

Access at `http://localhost:3000/admin`

**Workflow:** `draft → in_review → reviewing → published | rejected`

API endpoints:
- `POST /api/articles` — create article
- `PUT /api/articles/:id` — update article
- `PATCH /api/articles/:id/status` — status transitions `{action: "submit"|"approve"|"reject"|"publish"}`

## Meme Token 7 Tools

1. **OddsWidget** — Polymarket odds, CSR 60s refresh
2. **SecurityScore** — RugCheck API risk score
3. **TokenLifecycle** — Age counter + lifecycle stage
4. **HolderChart** — Recharts pie chart of holder distribution
5. **ProfitCalculator** — Pure frontend P&L calculator
6. **SentimentPill** — Community sentiment score
7. **SmartMoneyFeed** — Tracked wallet activity, CSR 2min refresh

## Content Freshness Strategy

- Hot tokens (vol > $1M): `revalidate = 60s`
- Regular tokens: `revalidate = 300s`
- Dead tokens (vol < $500 + age > 48h): `noindex = true` + grey banner
- Articles: `revalidate = 3600s`

## Production Upgrade Path

1. **Database**: Change `schema.prisma` datasource to `postgresql` + update `DATABASE_URL`
2. **Cache**: Replace `node-cache` in `lib/cache.ts` with `ioredis` — same interface
3. **Token data**: Add real Helius API key to `.env`
4. **Sitemap**: Auto-populates from DB — no manual work needed
5. **Deploy**: `npm run build` (includes `prisma generate + migrate deploy`)
