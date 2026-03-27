import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { SiteNav } from '@/components/layout/SiteNav'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: {
    default: 'MoonX — Prediction Market Aggregator',
    template: '%s | MoonX',
  },
  description: 'MoonX aggregates Polymarket, Kalshi, and Manifold prediction markets in one place. Track smart money, find meme token opportunities, and trade with better insights.',
  metadataBase: new URL('https://www.bydfi.com'),
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://www.bydfi.com/en/moonx',
    siteName: 'MoonX by BYDFi',
    images: [{ url: '/og-default.png', width: 1200, height: 630 }],
  },
  twitter: {
    card: 'summary_large_image',
    site: '@MoonX_BYDFi',
  },
  robots: { index: true, follow: true },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <SiteNav />
        {children}
      </body>
    </html>
  )
}
