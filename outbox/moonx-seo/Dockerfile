FROM node:20-alpine

RUN apk add --no-cache libc6-compat openssl

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

# Generate Prisma client (no DB needed at build time)
# Skip migrate deploy — volume DB not available during docker build
ENV NEXT_TELEMETRY_DISABLED=1
ENV DATABASE_URL=file:./dev.db
RUN npx prisma generate && npx next build

# Pre-create the volume mount point
RUN mkdir -p /data

EXPOSE 3000
ENV PORT=3000
ENV NODE_ENV=production

# At container startup: migrate DB on volume, then start Next.js
CMD ["sh", "-c", "npx prisma migrate deploy && npm start"]
