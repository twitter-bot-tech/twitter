-- CreateTable
CREATE TABLE "Article" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "category" TEXT NOT NULL,
    "slug" TEXT NOT NULL,
    "h1" TEXT NOT NULL,
    "seoTitle" TEXT NOT NULL,
    "metaDesc" TEXT NOT NULL,
    "coverImage" TEXT,
    "content" TEXT NOT NULL,
    "author" TEXT NOT NULL DEFAULT 'MoonX Team',
    "tags" TEXT NOT NULL DEFAULT '[]',
    "faqs" TEXT NOT NULL DEFAULT '[]',
    "ctaText" TEXT,
    "ctaUrl" TEXT,
    "status" TEXT NOT NULL DEFAULT 'draft',
    "scheduledAt" DATETIME,
    "publishedAt" DATETIME,
    "reviewNote" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL
);

-- CreateTable
CREATE TABLE "ProgrammaticTemplate" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "pageType" TEXT NOT NULL,
    "titleTpl" TEXT NOT NULL,
    "descTpl" TEXT NOT NULL,
    "contentTpl" TEXT NOT NULL,
    "faqTpl" TEXT NOT NULL DEFAULT '[]',
    "noindexDays" INTEGER NOT NULL DEFAULT 30,
    "volumeMin" INTEGER NOT NULL DEFAULT 100000,
    "updatedAt" DATETIME NOT NULL
);

-- CreateTable
CREATE TABLE "Banner" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "pages" TEXT NOT NULL,
    "imageUrl" TEXT NOT NULL,
    "linkUrl" TEXT NOT NULL,
    "altText" TEXT NOT NULL,
    "position" TEXT NOT NULL,
    "startsAt" DATETIME,
    "endsAt" DATETIME,
    "active" BOOLEAN NOT NULL DEFAULT true
);

-- CreateTable
CREATE TABLE "TokenSnapshot" (
    "contract" TEXT NOT NULL PRIMARY KEY,
    "symbol" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "price" REAL NOT NULL DEFAULT 0,
    "priceChange" REAL NOT NULL DEFAULT 0,
    "volume24h" REAL NOT NULL DEFAULT 0,
    "mcap" REAL NOT NULL DEFAULT 0,
    "holders" INTEGER NOT NULL DEFAULT 0,
    "lpUsd" REAL NOT NULL DEFAULT 0,
    "launchedAt" DATETIME NOT NULL,
    "polyOdds" TEXT,
    "riskScore" TEXT,
    "noindex" BOOLEAN NOT NULL DEFAULT false,
    "updatedAt" DATETIME NOT NULL
);

-- CreateIndex
CREATE UNIQUE INDEX "Article_slug_key" ON "Article"("slug");
