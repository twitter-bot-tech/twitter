/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ['www.bydfi.com', 'polymarket.com', 'rugcheck.xyz'],
  },
  async rewrites() {
    return []
  },
}

module.exports = nextConfig
