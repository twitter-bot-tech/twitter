export const CATEGORIES = ['All', 'Bitcoin', 'Ethereum', 'Solana', 'Altcoin', 'DeFi', 'Crypto Headlines']

export const CATEGORY_SLUGS: Record<string, string> = {
  'Bitcoin': 'bitcoin',
  'Ethereum': 'ethereum',
  'Solana': 'solana',
  'Altcoin': 'altcoin',
  'DeFi': 'defi',
  'Crypto Headlines': 'crypto-headlines',
}

export const SLUG_TO_CATEGORY: Record<string, string> = Object.fromEntries(
  Object.entries(CATEGORY_SLUGS).map(([k, v]) => [v, k])
)

export function readingTime(content: string): number {
  const words = content.trim().split(/\s+/).length
  return Math.max(1, Math.ceil(words / 200))
}

export interface FAQ {
  q: string
  a: string
}

export interface NewsItem {
  id: number
  slug: string
  title: string
  excerpt: string
  content: string
  author: string
  source: string
  category: string
  date: string
  color: string
  tag: string
  keywords: string[]
  faqs: FAQ[]
}

export const NEWS_ITEMS: NewsItem[] = [
  {
    id: 1,
    slug: 'bitcoin-ath-institutional-inflows-2b',
    title: 'Bitcoin Hits New ATH as Institutional Inflows Surpass $2B in a Single Week',
    excerpt: 'Spot Bitcoin ETFs recorded their largest single-week inflow since launch, pushing BTC above previous highs as Wall Street demand accelerates.',
    content: `Spot Bitcoin ETFs recorded their largest single-week inflow since launch last week, with combined net inflows exceeding $2 billion across all U.S.-listed products. BlackRock's IBIT alone accounted for $1.1 billion of that total, bringing its assets under management to a new high.

The surge in institutional demand coincides with Bitcoin breaking above its previous all-time high, a milestone that market participants attribute to a combination of ETF-driven demand and easing macroeconomic uncertainty.

**What's driving the rally?**

Analysts point to several converging factors. First, the SEC's approval of spot Bitcoin ETFs earlier this year unlocked a wave of pent-up institutional demand that had previously been locked out of the market. Wealth management platforms and registered investment advisors are now able to offer BTC exposure to clients without requiring custody solutions.

Second, the Bitcoin halving earlier this year reduced daily new supply by 50%, from roughly 900 BTC per day to 450 BTC. With demand outstripping newly minted supply by a wide margin, upward price pressure has been sustained.

**Prediction market outlook**

On Polymarket, the probability of Bitcoin reaching $150,000 by June 2026 currently sits at 62%, up from 41% a month ago. Kalshi's equivalent contract is pricing in a 58% chance.

MoonX users can track live odds across all prediction platforms in real time on the [Trending Markets](/en/moonx/markets/trending) page.

**What to watch**

Key levels to monitor include the $100K psychological support zone on any pullbacks. Options market data shows heavy call open interest at $120K and $150K strikes for the June expiry, suggesting institutional players are positioned for further upside.`,
    author: 'CoinDesk',
    source: 'CoinDesk',
    category: 'Bitcoin',
    date: '2026-03-26T05:30:00Z',
    color: 'from-orange-500 to-yellow-500',
    tag: 'BTC',
    keywords: ['Bitcoin ATH', 'Bitcoin ETF inflows', 'BlackRock IBIT', 'BTC price 2026', 'institutional Bitcoin'],
    faqs: [
      { q: 'Why is Bitcoin hitting a new all-time high?', a: 'Spot Bitcoin ETFs are driving record institutional inflows — over $2B in a single week — while the 2024 halving cut new supply by 50%. Demand is far outpacing new supply.' },
      { q: 'What is IBIT and why does it matter?', a: 'IBIT is BlackRock\'s iShares Bitcoin Trust, the world\'s largest spot Bitcoin ETF. It now holds over $25B in BTC, making BlackRock one of the largest Bitcoin holders on earth.' },
      { q: 'What does the prediction market say about Bitcoin\'s price?', a: 'Polymarket currently gives a 62% probability that BTC will reach $150,000 by June 2026. You can track live odds on MoonX.' },
    ],
  },
  {
    id: 2,
    slug: 'ethereum-pectra-upgrade-testnet',
    title: "Ethereum's Pectra Upgrade Goes Live on Testnet — What's Changing",
    excerpt: 'The Pectra hard fork introduces account abstraction and validator improvements. Developers say mainnet deployment is on track for Q2.',
    content: `The Ethereum Pectra upgrade successfully deployed on the Holesky testnet this week, marking a major milestone ahead of its anticipated mainnet launch in Q2 2026. Core developers confirmed no critical bugs were detected during the initial deployment phase.

**Key changes in Pectra**

Pectra combines two previously separate upgrade tracks — Prague (execution layer) and Electra (consensus layer) — into a single coordinated hard fork. The most significant changes include:

**EIP-7702: Account abstraction for EOAs.** This proposal allows externally owned accounts (regular user wallets) to temporarily take on smart contract behavior during a transaction. In practice, this means wallets can sponsor gas fees for users, batch multiple transactions into one, and support social recovery without migrating to a new address.

**EIP-7251: Increased validator max balance.** Validators can now consolidate up to 2,048 ETH in a single validator, reducing the total number of validators required to secure the network and improving efficiency.

**EIP-6110: Faster validator onboarding.** New validators can now be activated on-chain in roughly 13 minutes versus the current multi-day queue, significantly improving the staking experience.

**Timeline**

Developers have targeted a mainnet activation in late Q2 2026, pending successful testnet validation. A shadow fork against mainnet state is scheduled for next month.

For those tracking Ethereum's competitive positioning against Solana and other L1s, the account abstraction improvements in Pectra could narrow the user experience gap significantly.`,
    author: 'The Block',
    source: 'The Block',
    category: 'Ethereum',
    date: '2026-03-26T04:15:00Z',
    color: 'from-blue-500 to-indigo-500',
    tag: 'ETH',
    keywords: ['Ethereum Pectra upgrade', 'EIP-7702 account abstraction', 'Ethereum hard fork 2026', 'ETH testnet', 'Ethereum validators'],
    faqs: [
      { q: 'What is the Ethereum Pectra upgrade?', a: 'Pectra is Ethereum\'s next major hard fork, combining the Prague (execution layer) and Electra (consensus layer) upgrades. It introduces account abstraction for regular wallets and allows validators to consolidate up to 2,048 ETH.' },
      { q: 'When will Pectra go live on mainnet?', a: 'Developers are targeting a mainnet activation in late Q2 2026, pending successful testnet validation on Holesky.' },
      { q: 'What is EIP-7702 account abstraction?', a: 'EIP-7702 lets regular Ethereum wallets temporarily behave like smart contracts during a transaction, enabling gas sponsorship, transaction batching, and social recovery without migrating to a new address.' },
    ],
  },
  {
    id: 3,
    slug: 'solana-dex-volume-50b-record',
    title: 'Solana DEX Volume Breaks $50B Monthly Record, Outpacing Ethereum L2s',
    excerpt: "Jupiter and Raydium drove Solana's decentralized exchange volumes to an all-time high, with meme token trading accounting for 38% of activity.",
    content: `Solana's decentralized exchange ecosystem set a new monthly volume record in March 2026, surpassing $50 billion in aggregate trading activity. The milestone was driven largely by Jupiter, which alone processed over $28 billion, and Raydium, which contributed $14 billion.

For context, the combined DEX volume of Arbitrum, Base, and Optimism — Ethereum's three largest Layer 2 networks — totaled $42 billion over the same period.

**Meme tokens lead activity**

Meme token trading accounted for 38% of Solana's DEX volume, according to data from DeFiLlama. The launch of Pump.fun's new bonding curve mechanism in February triggered a wave of new token launches, with several meme coins reaching market caps above $100 million within hours of launch.

MoonX's [Solana token pages](/en/moonx/solana/token/) track smart money wallet activity, security scores, and holder concentration for newly launched tokens in real time.

**Competitive dynamics**

The volume gap between Solana and Ethereum L2s reflects both technical differences and cultural shifts. Solana's sub-second finality and sub-cent transaction fees make it the preferred venue for high-frequency retail trading. Ethereum L2s continue to dominate in DeFi lending, derivatives, and institutional use cases.

**What this means for SOL price**

Fee revenue flowing to SOL stakers increases as network activity rises. Analysts at Messari estimate SOL's annualized protocol revenue has reached $800 million, a metric that supports higher fundamental valuations. Prediction markets currently give a 71% probability that SOL will trade above $300 by end of Q2.`,
    author: 'DeFiLlama',
    source: 'DeFiLlama',
    category: 'Solana',
    date: '2026-03-26T03:00:00Z',
    color: 'from-purple-500 to-pink-500',
    tag: 'SOL',
    keywords: ['Solana DEX volume', 'Jupiter DEX', 'Raydium', 'Solana vs Ethereum L2', 'SOL meme tokens 2026'],
    faqs: [
      { q: 'How much DEX volume does Solana do?', a: 'Solana broke its monthly DEX volume record in March 2026, surpassing $50 billion — more than the combined volume of Arbitrum, Base, and Optimism.' },
      { q: 'Which DEXs drive Solana volume?', a: 'Jupiter is the largest aggregator with $28B monthly volume, followed by Raydium at $14B. Meme token launches on Pump.fun contribute significantly to activity.' },
      { q: 'Why is Solana popular for meme tokens?', a: 'Sub-second finality and sub-cent transaction fees make Solana ideal for high-frequency retail trading. MoonX tracks smart money activity on Solana meme tokens in real time.' },
    ],
  },
  {
    id: 4,
    slug: 'xrp-fees-spike-ripple-cto-explains',
    title: "XRP Fees Suddenly Spike: Ripple CTO Explains What's Happening",
    excerpt: "Transaction costs on the XRP Ledger surged 40x following a wave of NFT minting activity. Ripple's CTO confirmed a temporary network congestion issue.",
    content: `Transaction fees on the XRP Ledger spiked dramatically this week, reaching 40x normal levels at peak congestion. The sudden surge caught many users off guard, with some transactions failing due to insufficient fee amounts set by wallets that hadn't updated their fee estimation logic.

Ripple's CTO David Schwartz addressed the situation in a series of posts on X, explaining that a coordinated wave of NFT minting activity had overwhelmed the network's fee market temporarily.

**What caused the spike?**

The XRP Ledger's fee escalation mechanism is designed to automatically increase transaction costs when the network is congested, prioritizing higher-fee transactions. A single actor minting a large collection of NFTs in rapid succession triggered this mechanism, causing fees to spike from the baseline 0.00001 XRP to roughly 0.0004 XRP per transaction.

While still cheap in absolute terms compared to Ethereum or Solana during congestion events, the relative spike was enough to cause failed transactions for users whose wallets had hard-coded fee limits.

**Resolution**

The congestion cleared within approximately 90 minutes as the minting activity concluded. Schwartz noted that the network behaved exactly as designed, and that the spike reflected the fee market working correctly rather than a bug.

The incident has renewed discussion about whether the XRP Ledger's fee mechanism needs adjustment to handle NFT and tokenization use cases that generate bursty transaction patterns.`,
    author: 'CoinPedia',
    source: 'CoinPedia',
    category: 'Altcoin',
    date: '2026-03-26T02:45:00Z',
    color: 'from-cyan-500 to-blue-500',
    tag: 'XRP',
    keywords: ['XRP fees spike', 'XRP Ledger congestion', 'Ripple CTO', 'XRP NFT minting', 'XRP transaction costs'],
    faqs: [
      { q: 'Why did XRP fees spike?', a: 'A coordinated wave of NFT minting on the XRP Ledger triggered the network\'s fee escalation mechanism, pushing transaction costs 40x above normal for roughly 90 minutes.' },
      { q: 'Are XRP fees still high?', a: 'No. The congestion cleared within 90 minutes once the minting activity concluded. Normal fees on the XRP Ledger are a fraction of a cent.' },
      { q: 'What did Ripple\'s CTO say about the fee spike?', a: 'David Schwartz confirmed the network behaved as designed — the fee escalation mechanism worked correctly and there was no bug. The spike reflected the fee market functioning under unusual load.' },
    ],
  },
  {
    id: 5,
    slug: 'uniswap-v4-10b-tvl-60-days',
    title: 'Uniswap v4 Surpasses $10B in Total Value Locked Within 60 Days',
    excerpt: "The new hook-based architecture attracted liquidity providers with custom fee tiers, catapulting Uniswap v4 to the top of the DeFi TVL rankings.",
    content: `Uniswap v4 has amassed over $10 billion in total value locked just 60 days after its mainnet launch, making it one of the fastest-growing DeFi protocols in history. The milestone was reached as liquidity migrated from v3 pools and new capital entered attracted by the novel hook architecture.

**What makes v4 different?**

The defining feature of Uniswap v4 is its hook system, which allows developers to attach custom logic to liquidity pools. Hooks can fire before or after swaps, liquidity additions, and removals, enabling use cases that were impossible in previous versions.

Popular hook implementations already live on mainnet include:

- **Dynamic fee hooks** that adjust trading fees based on volatility, reducing impermanent loss for LPs during turbulent markets
- **TWAP oracle hooks** that provide on-chain price feeds directly from pool data
- **Limit order hooks** that allow traders to set target prices for execution without relying on off-chain infrastructure
- **MEV protection hooks** that detect and block sandwich attacks at the pool level

**Liquidity provider returns**

Early data suggests that LPs using dynamic fee hooks are earning 15-25% higher returns compared to equivalent v3 positions, primarily by capturing more fee revenue during high-volatility periods while remaining competitive during stable markets.

The success of v4 has reinvigorated interest in the UNI governance token, which has rallied 85% since the launch date.`,
    author: 'DeFi Pulse',
    source: 'DeFi Pulse',
    category: 'DeFi',
    date: '2026-03-26T01:30:00Z',
    color: 'from-pink-500 to-rose-500',
    tag: 'UNI',
    keywords: ['Uniswap v4 TVL', 'Uniswap hooks', 'DeFi TVL 2026', 'UNI token', 'AMM liquidity'],
    faqs: [
      { q: 'What is Uniswap v4?', a: 'Uniswap v4 is the latest version of the largest decentralized exchange. It introduces a hook system that lets developers attach custom logic to liquidity pools, enabling dynamic fees, limit orders, and MEV protection.' },
      { q: 'How much TVL does Uniswap v4 have?', a: 'Uniswap v4 surpassed $10 billion in total value locked within 60 days of its mainnet launch, making it one of the fastest-growing DeFi protocols ever.' },
      { q: 'Are Uniswap v4 LPs earning more than v3?', a: 'Early data shows LPs using dynamic fee hooks earn 15–25% higher returns than equivalent v3 positions, primarily by capturing more fees during volatile markets.' },
    ],
  },
  {
    id: 6,
    slug: 'sec-approves-solana-spot-etf',
    title: 'SEC Approves First Solana Spot ETF — Launch Expected Next Month',
    excerpt: 'Regulators greenlit three competing Solana ETF proposals simultaneously. Asset managers plan to list on major exchanges by late April.',
    content: `The U.S. Securities and Exchange Commission approved three competing spot Solana ETF proposals simultaneously on Wednesday, following a pattern similar to the multi-issuer approval of spot Bitcoin ETFs in early 2024. The products from VanEck, Fidelity, and 21Shares are expected to begin trading on major exchanges by late April 2026.

**Why Solana, why now?**

The SEC's decision reflects growing regulatory comfort with proof-of-stake assets following the agency's internal review of staking mechanics. Unlike Bitcoin's proof-of-work, Solana's consensus mechanism raised questions about whether staking rewards might constitute securities. The approved ETFs will not pass staking yields to shareholders initially, sidestepping this issue while the regulatory framework develops.

**Market impact**

SOL surged 18% in the hours following the announcement. Prediction markets had assigned a 74% probability to Solana ETF approval in Q1 2026 prior to the news, suggesting the market had largely priced in the event. The remaining upside reflects uncertainty around launch timing and initial inflow volumes.

Analysts at Galaxy Digital estimate first-year inflows could reach $3-5 billion, assuming the Solana ETFs capture a similar share of total crypto ETF AUM as SOL's market cap relative to BTC.

**What comes next**

Following Solana, market participants are watching closely for XRP and Litecoin ETF decisions, with multiple applications pending. The SEC has until June to respond to several outstanding filings.`,
    author: 'Bloomberg Crypto',
    source: 'Bloomberg',
    category: 'Crypto Headlines',
    date: '2026-03-25T23:00:00Z',
    color: 'from-green-500 to-teal-500',
    tag: 'ETF',
    keywords: ['Solana ETF approved', 'SOL spot ETF', 'SEC crypto ETF 2026', 'VanEck Solana ETF', 'Fidelity Solana ETF'],
    faqs: [
      { q: 'Is there a Solana spot ETF?', a: 'Yes. The SEC approved three competing Solana spot ETFs simultaneously in March 2026 from VanEck, Fidelity, and 21Shares. Trading is expected to begin in late April 2026.' },
      { q: 'Will the Solana ETF offer staking rewards?', a: 'No, initially. The approved products will not pass staking yields to shareholders while the SEC develops a clearer regulatory framework for proof-of-stake assets.' },
      { q: 'How much money could flow into Solana ETFs?', a: 'Galaxy Digital analysts estimate first-year inflows of $3–5 billion, assuming Solana ETFs capture a market share proportional to SOL\'s relative market cap.' },
    ],
  },
  {
    id: 7,
    slug: 'blackrock-adds-500m-bitcoin-ibit',
    title: 'BlackRock Adds $500M More Bitcoin to IBIT ETF in Single Day',
    excerpt: "BlackRock's iShares Bitcoin Trust saw its largest single-day inflow on record, bringing total assets under management past the $25B mark.",
    content: `BlackRock's iShares Bitcoin Trust (IBIT) recorded its largest single-day inflow on record Thursday, with $500 million flowing into the fund and pushing total assets under management beyond $25 billion for the first time.

The inflow coincides with Bitcoin's move to new all-time highs, as momentum investors and institutions that had been waiting for price confirmation of the bull market entered positions.

**Institutional adoption accelerating**

IBIT's AUM milestone is significant because it establishes the ETF as a major holder of Bitcoin globally. At current prices, BlackRock's IBIT holds approximately 220,000 BTC, representing roughly 1% of the total circulating supply.

Wealth managers report that client demand for Bitcoin exposure through regulated vehicles has accelerated meaningfully in recent months. Many RIAs that previously had internal restrictions on crypto allocations have updated their investment policies to permit ETF-based exposure.

**Fee revenue and competitive dynamics**

BlackRock has waived its sponsor fee for IBIT through the end of Q2 2026 for the first $5 billion in assets. As the fund has grown well beyond that threshold, the fee waiver is no longer relevant for most assets, and the fund now generates approximately $62.5 million in annualized fee revenue at current AUM levels.

Competing products from Fidelity, Invesco, and Bitwise have collectively gathered $12 billion in AUM, with BlackRock commanding a dominant market share of approximately 52% of total spot Bitcoin ETF assets.`,
    author: 'Reuters',
    source: 'Reuters',
    category: 'Bitcoin',
    date: '2026-03-25T20:00:00Z',
    color: 'from-orange-400 to-amber-500',
    tag: 'BTC',
    keywords: ['BlackRock IBIT inflow', 'Bitcoin ETF record', 'IBIT AUM $25B', 'BlackRock Bitcoin', 'spot Bitcoin ETF 2026'],
    faqs: [
      { q: 'How much Bitcoin does BlackRock hold?', a: 'BlackRock\'s IBIT ETF holds approximately 220,000 BTC — about 1% of all Bitcoin in circulation — with over $25 billion in assets under management.' },
      { q: 'What was BlackRock\'s record single-day Bitcoin ETF inflow?', a: '$500 million in a single day, recorded in March 2026, bringing IBIT past the $25B AUM milestone.' },
      { q: 'What fee does BlackRock charge for IBIT?', a: 'BlackRock charges a standard sponsor fee that generates approximately $62.5M in annualized revenue at current AUM levels. A temporary fee waiver for the first $5B in assets has since expired.' },
    ],
  },
  {
    id: 8,
    slug: 'avalanche-subnets-300-new-q1-2026',
    title: 'Avalanche Subnets Surge: Over 300 New Networks Launched in Q1 2026',
    excerpt: 'Gaming and enterprise sectors are driving subnet adoption, with Avalanche processing more daily transactions than it did in all of 2024.',
    content: `The Avalanche ecosystem saw explosive growth in Q1 2026, with over 300 new subnets launching during the quarter. The milestone marks a turning point for Avalanche's subnet strategy, which had faced skepticism about real-world adoption since its introduction.

**Gaming leads adoption**

Gaming applications represent the largest category of new subnet launches, accounting for roughly 40% of Q1 deployments. The appeal is straightforward: subnets allow game developers to run their own blockchain with custom gas tokens, transaction throughput, and validator requirements — without competing for blockspace with DeFi applications.

Notable gaming subnet launches in Q1 include the official blockchain for a major Korean gaming studio and a dedicated chain for a card game with over 2 million registered users.

**Enterprise use cases**

Enterprise and institutional subnets make up another 25% of deployments, with use cases ranging from tokenized real-world assets to private transaction networks for financial institutions. Avalanche's permissioned subnet model allows enterprises to control validator sets, addressing regulatory requirements around data sovereignty.

**Network metrics**

Daily transaction counts on Avalanche's primary network and subnets combined have exceeded levels seen in all of 2024, driven by the cumulative activity of the new subnet ecosystem. AVAX's price has responded positively, with the token up 120% year-to-date.

The subnet growth validates Avalanche's core thesis that the future of blockchain is a network of interoperable chains rather than a single monolithic ledger.`,
    author: 'CryptoSlate',
    source: 'CryptoSlate',
    category: 'Altcoin',
    date: '2026-03-25T18:30:00Z',
    color: 'from-red-500 to-orange-500',
    tag: 'AVAX',
    keywords: ['Avalanche subnets 2026', 'AVAX subnet growth', 'Avalanche gaming blockchain', 'AVAX enterprise blockchain', 'Avalanche Q1 2026'],
    faqs: [
      { q: 'What are Avalanche subnets?', a: 'Subnets are custom blockchains built on Avalanche that share its security model. They allow developers to run their own chain with custom gas tokens, throughput, and validator requirements.' },
      { q: 'How many Avalanche subnets launched in Q1 2026?', a: 'Over 300 new subnets launched in Q1 2026, with gaming and enterprise sectors accounting for the majority of deployments.' },
      { q: 'Why do game developers choose Avalanche subnets?', a: 'Subnets give game developers their own dedicated blockchain, eliminating competition for blockspace with DeFi applications and allowing custom token economics for in-game assets.' },
    ],
  },
  {
    id: 9,
    slug: 'aave-protocol-revenue-100m-q1-buyback',
    title: 'Aave Protocol Revenue Tops $100M in Q1 — Governance Vote Eyes Buyback',
    excerpt: "Aave's treasury hit a milestone as rising borrow demand across Ethereum and Arbitrum drove record fee generation. A token buyback proposal is live for vote.",
    content: `Aave, the largest decentralized lending protocol by total value locked, generated over $100 million in protocol revenue during Q1 2026 — a new quarterly record. The milestone has catalyzed a governance proposal to use a portion of treasury funds for AAVE token buybacks.

**What's driving revenue growth**

Borrow demand across Aave's markets has surged alongside crypto price appreciation, as traders take out loans against their crypto collateral to leverage long positions without selling their underlying assets. Aave's Ethereum and Arbitrum deployments account for the majority of revenue, though newer deployments on Base and Optimism are growing rapidly.

The protocol's GHO stablecoin, which Aave mints directly and earns full interest on, has also grown to $850 million in circulation, adding a new revenue stream that wasn't present a year ago.

**The buyback proposal**

A governance proposal submitted this week would allocate 20% of monthly protocol revenue to purchasing AAVE tokens on the open market and distributing them to stakers in the Safety Module. Proponents argue this creates a direct link between protocol success and token value that is currently missing.

Opponents raise concerns about the opportunity cost of using treasury funds for buybacks rather than funding ecosystem growth or building additional safety reserves.

Voting closes in seven days. Current on-chain sentiment shows approximately 67% of voting power in favor of the proposal.

**AAVE price reaction**

AAVE has risen 28% since the revenue milestone and buyback proposal were announced, reflecting market anticipation of the governance outcome.`,
    author: 'Bankless',
    source: 'Bankless',
    category: 'DeFi',
    date: '2026-03-25T16:00:00Z',
    color: 'from-violet-500 to-purple-500',
    tag: 'AAVE',
    keywords: ['Aave revenue Q1 2026', 'AAVE token buyback', 'GHO stablecoin', 'Aave governance', 'DeFi lending protocol'],
    faqs: [
      { q: 'How much revenue did Aave generate in Q1 2026?', a: 'Aave generated over $100 million in protocol revenue in Q1 2026, driven by rising borrow demand and GHO stablecoin growth across Ethereum and Arbitrum.' },
      { q: 'What is the Aave buyback proposal?', a: 'A governance proposal would allocate 20% of monthly protocol revenue to purchasing AAVE tokens on the open market and distributing them to Safety Module stakers, directly linking protocol success to token value.' },
      { q: 'What is GHO?', a: 'GHO is Aave\'s native stablecoin, which the protocol mints directly and earns full interest on. It has grown to $850 million in circulation as of Q1 2026.' },
    ],
  },
  {
    id: 10,
    slug: 'ethereum-l2-tvl-crosses-50b',
    title: 'Ethereum Layer 2 Total Value Locked Crosses $50B for First Time',
    excerpt: 'Arbitrum, Base, and Optimism together now hold more TVL than most standalone L1 blockchains, signaling mainstream adoption of rollup technology.',
    content: `The combined total value locked across Ethereum Layer 2 networks surpassed $50 billion for the first time this week, a figure that places the L2 ecosystem above all standalone Layer 1 blockchains except Ethereum itself and, by some measures, Solana.

Arbitrum leads with $22 billion in TVL, followed by Base at $16 billion and Optimism at $8 billion. The remaining $4 billion is distributed across smaller rollups including zkSync, Starknet, and Scroll.

**What's driving L2 growth**

Several dynamics are converging to accelerate TVL accumulation on Layer 2s:

**Lower costs post-EIP-4844.** The Dencun upgrade earlier this year introduced blob transactions that reduced L2 data costs by approximately 90%, making L2 transactions consistently cheap and predictable for users.

**Institutional DeFi migration.** Major DeFi protocols have deployed on L2s with incentive programs, attracting TVL from the Ethereum mainnet as users seek better yield-to-fee ratios.

**Base's consumer app success.** Coinbase's Base network has attracted a wave of consumer-facing applications, bringing new users into the ecosystem who may not have previously interacted with DeFi.

**Implications for ETH**

ETH's value accrual from L2 activity occurs through two mechanisms: blob fees paid by L2s to post data to Ethereum mainnet, and the burn of ETH through EIP-1559 on mainnet transactions triggered by L2 activity. As L2 usage grows, analysts expect Ethereum's annualized ETH burn rate to increase, supporting a supply reduction narrative for the asset.`,
    author: 'L2Beat',
    source: 'L2Beat',
    category: 'Ethereum',
    date: '2026-03-25T14:00:00Z',
    color: 'from-blue-400 to-sky-500',
    tag: 'L2',
    keywords: ['Ethereum L2 TVL $50B', 'Arbitrum TVL', 'Base network growth', 'Ethereum rollup 2026', 'Layer 2 DeFi'],
    faqs: [
      { q: 'Which Ethereum Layer 2 has the most TVL?', a: 'Arbitrum leads with $22 billion in TVL, followed by Base at $16 billion and Optimism at $8 billion, as of March 2026.' },
      { q: 'Why are Ethereum L2s growing so fast?', a: 'The Dencun upgrade cut L2 data costs by ~90%, making transactions consistently cheap. Combined with institutional DeFi migration and Base\'s consumer apps, TVL has accelerated sharply.' },
      { q: 'How does L2 growth benefit ETH holders?', a: 'L2s pay blob fees to Ethereum mainnet and trigger EIP-1559 burns. As L2 usage grows, Ethereum\'s annualized ETH burn rate increases, reducing circulating supply.' },
    ],
  },
]
