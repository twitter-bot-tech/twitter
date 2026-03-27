#!/usr/bin/env python3
"""
Medium Publisher — 从 Chrome 提取 session cookie，注入 Playwright 自动发布
"""

import sys
import time
import json
import logging
from pathlib import Path
import browser_cookie3
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

BASE_DIR = Path(__file__).parent
LOG_DIR  = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "medium_publisher.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── 文章配置 ──
ARTICLE_TITLE = "How Prediction Market Smart Money Moves Before the Rest of the World Notices"
ARTICLE_TAGS  = ["Prediction Markets", "Smart Money", "Crypto", "Polymarket", "Trading"]
ARTICLE_PATH  = Path(__file__).parent.parent / "outbox" / "2026-03-05_SEO文章_prediction-market-smart-money.md"


def load_article_body() -> str:
    """跳过文件顶部的 SEO 元数据注释，从正文第一行（# 标题）开始提取"""
    text = ARTICLE_PATH.read_text(encoding="utf-8")
    lines = text.split("\n")
    body, in_body = [], False
    for line in lines:
        # 跳过 **Target Keyword:** 等元数据行，从第一个 # 标题开始
        if not in_body:
            if line.startswith("# ") and not line.startswith("# SEO Article"):
                in_body = True
        if in_body:
            body.append(line)
    return "\n".join(body).strip()


def get_medium_cookies() -> list:
    cookies = browser_cookie3.chrome(domain_name=".medium.com")
    result = []
    for c in cookies:
        result.append({
            "name":   c.name,
            "value":  c.value,
            "domain": ".medium.com",
            "path":   "/",
        })
    logger.info("  从 Chrome 提取了 %d 个 Medium cookies", len(result))
    return result


def publish_draft(page, draft_url: str) -> str:
    """打开已有草稿页，点击 Publish 发布"""
    logger.info(f"▶ 打开草稿: {draft_url}")
    page.goto(draft_url, wait_until="networkidle", timeout=45000)
    time.sleep(5)
    page.mouse.click(640, 300)
    time.sleep(3)

    # 点 Publish 按钮
    page.evaluate("""
        () => {
            const btns = Array.from(document.querySelectorAll('button'));
            const pub = btns.find(b => b.textContent.trim() === 'Publish');
            if (pub) pub.click();
        }
    """)
    time.sleep(4)

    # 点 Publish now
    for attempt in range(5):
        clicked = page.evaluate("""
            () => {
                const btns = Array.from(document.querySelectorAll('button, [role="button"]'));
                const btn = btns.find(b =>
                    b.textContent.includes('Publish now') ||
                    b.textContent.includes('Publish story')
                );
                if (btn) { btn.scrollIntoView(); btn.click(); return btn.textContent.trim(); }
                return null;
            }
        """)
        if clicked:
            logger.info(f"  点击: '{clicked}'")
            break
        time.sleep(2)

    time.sleep(6)
    url = page.url
    logger.info(f"  ✅ 发布完成: {url}")
    page.screenshot(path=str(LOG_DIR / "medium_published.png"))
    return url


def publish_article(page, dry_run: bool = False) -> str:
    logger.info("▶ 打开新建文章页...")
    try:
        page.goto("https://medium.com/new-story", wait_until="networkidle", timeout=45000)
    except PlaywrightTimeoutError:
        logger.warning("  networkidle超时，继续尝试")
    time.sleep(6)
    # 点击页面中心触发编辑器初始化
    page.mouse.click(640, 450)
    time.sleep(3)

    # ── 1. 标题 ──
    logger.info("  输入标题...")
    found = False
    for sel in [
        'div.section-inner h3',
        'h3[data-placeholder="Title"]',
        'h2[data-placeholder="Title"]',
        '[data-testid="editorTitle"]',
        'div[data-placeholder="Title"]',
    ]:
        try:
            page.wait_for_selector(sel, timeout=6000)
            page.click(sel)
            logger.info(f"  标题 selector 命中: {sel}")
            found = True
            break
        except Exception:
            continue

    if not found:
        # 最终兜底：用 JS 找第一个 contenteditable 并点击
        clicked = page.evaluate("""
            () => {
                const el = document.querySelector('[contenteditable="true"]');
                if (el) { el.click(); el.focus(); return true; }
                return false;
            }
        """)
        if clicked:
            logger.info("  fallback: JS点击 contenteditable")
        else:
            logger.error("  找不到编辑器，截图留存")
            page.screenshot(path=str(LOG_DIR / "editor_fail.png"))
            raise RuntimeError("编辑器未加载")

    page.keyboard.type(ARTICLE_TITLE, delay=20)
    time.sleep(0.5)
    page.keyboard.press("Enter")
    time.sleep(0.5)

    # ── 2. 粘贴正文 ──
    logger.info("  粘贴正文...")
    body = load_article_body()

    # 用 JS 写剪贴板
    page.evaluate(
        "(text) => navigator.clipboard.writeText(text).catch(() => { "
        "const el = document.createElement('textarea'); el.value = text; "
        "document.body.appendChild(el); el.select(); "
        "document.execCommand('copy'); document.body.removeChild(el); })",
        body
    )
    time.sleep(0.3)
    page.keyboard.press("Meta+v")
    time.sleep(4)
    logger.info("  正文已粘贴 (%d 字符)", len(body))

    if dry_run:
        logger.info("  [DRY RUN] 内容写入完成，不执行发布")
        # 滚回顶部截图
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)
        shot = LOG_DIR / "medium_draft_preview.png"
        page.screenshot(path=str(shot), full_page=False)
        logger.info("  截图(顶部): %s", shot)
        return "DRY_RUN"

    # ── 3. 点 Publish ──
    logger.info("  点击 Publish...")
    # JS兜底：找所有按钮文本匹配Publish（不含"Publish now"）
    page.evaluate("""
        () => {
            const btns = Array.from(document.querySelectorAll('button'));
            const pub = btns.find(b => b.textContent.trim() === 'Publish');
            if (pub) pub.click();
        }
    """)
    time.sleep(4)

    # ── 4. Tags（跳过，不影响发布）──
    logger.info("  跳过 Tags（不影响发布）")

    # ── 5. Publish now ──
    logger.info("  Publish now...")
    time.sleep(3)
    # 滚到底部让发布面板完整显示
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(2)

    published = False
    for attempt in range(5):
        # 截图记录当前状态
        page.screenshot(path=str(LOG_DIR / f"publish_step_{attempt}.png"))

        clicked = page.evaluate("""
            () => {
                const btns = Array.from(document.querySelectorAll('button, [role="button"]'));
                const keywords = ['Publish now', 'Publish story', 'Publish Now', 'Publish Story'];
                for (const kw of keywords) {
                    const btn = btns.find(b => b.textContent.trim().includes(kw));
                    if (btn) { btn.scrollIntoView(); btn.click(); return btn.textContent.trim(); }
                }
                // 找绿色/主色调按钮作为兜底
                const greenBtn = btns.find(b =>
                    b.textContent.trim().toLowerCase().includes('publish') &&
                    !b.textContent.trim().toLowerCase().includes('draft')
                );
                if (greenBtn) { greenBtn.scrollIntoView(); greenBtn.click(); return greenBtn.textContent.trim(); }
                return null;
            }
        """)
        if clicked:
            logger.info(f"  点击按钮: '{clicked}'（第{attempt+1}次）")
            published = True
            break
        time.sleep(2)

    if not published:
        logger.warning("  未找到 Publish now 按钮，截图留存")
        page.screenshot(path=str(LOG_DIR / "publish_now_fail.png"))

    time.sleep(6)
    url = page.url
    logger.info("  ✅ 发布完成: %s", url)

    # 截图
    shot = LOG_DIR / "medium_published.png"
    page.screenshot(path=str(shot))
    logger.info("  截图: %s", shot)
    return url


def main():
    dry_run = "--send" not in sys.argv
    if dry_run:
        logger.info("⚠️  测试模式（加 --send 真实发布）")

    logger.info("=" * 50)
    logger.info("📰 Medium 自动发布 — Chrome Cookie 注入模式")
    logger.info("=" * 50)

    cookies = get_medium_cookies()
    if not cookies:
        logger.error("  未找到 Medium cookies，请先用 Chrome 登录 medium.com")
        sys.exit(1)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,   # 方便调试，可改 True
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )

        # 注入 cookies
        context.add_cookies(cookies)
        logger.info("  Cookies 注入完成")

        page = context.new_page()

        # 验证登录
        page.goto("https://medium.com", wait_until="domcontentloaded", timeout=20000)
        time.sleep(3)
        if "Sign in" in page.title() or "Sign In" in page.content()[:500]:
            logger.error("  ✗ Cookie 注入后仍未登录，Session 可能已过期")
            browser.close()
            sys.exit(1)
        logger.info("  ✅ 登录验证通过")

        url = publish_article(page, dry_run=dry_run)

        if url and url != "DRY_RUN":
            record = {
                "title": ARTICLE_TITLE,
                "url": url,
                "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            rec_file = LOG_DIR / "published_articles.json"
            records = json.loads(rec_file.read_text()) if rec_file.exists() else []
            records.append(record)
            rec_file.write_text(json.dumps(records, indent=2, ensure_ascii=False))
            logger.info("  记录: %s", rec_file)
            print(f"\n✅ 文章已发布: {url}")

        browser.close()


if __name__ == "__main__":
    main()
