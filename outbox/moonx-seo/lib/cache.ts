import NodeCache from 'node-cache'

// Global singleton cache (simulates Redis for prototype)
// In production: swap to Redis via ioredis with same interface
const cache = new NodeCache({ stdTTL: 300, checkperiod: 60 })

export interface CacheOptions {
  ttl?: number  // seconds
}

export function cacheGet<T>(key: string): T | undefined {
  return cache.get<T>(key)
}

export function cacheSet<T>(key: string, value: T, options?: CacheOptions): void {
  const ttl = options?.ttl ?? 300
  cache.set(key, value, ttl)
}

export function cacheDel(key: string): void {
  cache.del(key)
}

export async function cachedFetch<T>(
  key: string,
  fetcher: () => Promise<T>,
  options?: CacheOptions
): Promise<T> {
  const cached = cacheGet<T>(key)
  if (cached !== undefined) return cached

  const data = await fetcher()
  cacheSet(key, data, options)
  return data
}

// Specific TTL presets
export const TTL = {
  HOT_TOKEN: 60,        // 1 min for volume > $1M
  TOKEN: 300,           // 5 min default
  MARKET_ODDS: 60,      // 1 min for odds
  TRENDING: 120,        // 2 min for trending list
  LONG: 3600,           // 1 hour for stable data
} as const
