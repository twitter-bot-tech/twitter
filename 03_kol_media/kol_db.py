#!/usr/bin/env python3
"""
KOL CRM — SQLite 数据库模块
所有 KOL/媒体数据统一存储，替换原 Excel 方案
DB 位置：Fly.io → /data/kol_crm.db，本地 → 03_kol_media/kol_crm.db
"""
import sqlite3
import os
import re
import hashlib
from datetime import datetime, date, timedelta
from pathlib import Path
from contextlib import contextmanager

_ON_FLY = bool(os.getenv("FLY_APP_NAME"))
DB_DIR  = Path("/data") if _ON_FLY else Path(__file__).parent
DB_PATH = DB_DIR / "kol_crm.db"


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS kols (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT    NOT NULL DEFAULT 'youtube',
    channel_id   TEXT,
    channel_url  TEXT,
    name         TEXT    NOT NULL,
    platform     TEXT    DEFAULT 'YouTube',
    subscribers  INTEGER DEFAULT 0,
    tier         TEXT    DEFAULT 'C级',
    email        TEXT,
    description  TEXT,
    category     TEXT,
    country      TEXT,
    twitter      TEXT,
    collect_date TEXT,
    status       TEXT    DEFAULT '待发送',
    tg_handle    TEXT,
    tg_status    TEXT    DEFAULT 'not_invited',
    utm_code     TEXT,
    notes            TEXT,
    sequence_step    INTEGER DEFAULT 0,
    next_followup_at TEXT,
    last_contact_at  TEXT,
    score            INTEGER DEFAULT 0,
    created_at       TEXT,
    updated_at       TEXT,
    UNIQUE(channel_id)
);

CREATE TABLE IF NOT EXISTS contacts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kol_id     INTEGER NOT NULL,
    sent_at    TEXT,
    subject    TEXT,
    template   TEXT,
    status     TEXT    DEFAULT 'sent',
    FOREIGN KEY(kol_id) REFERENCES kols(id)
);

CREATE TABLE IF NOT EXISTS replies (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    kol_id             INTEGER NOT NULL,
    email_from         TEXT,
    subject            TEXT,
    body_snippet       TEXT,
    detected_at        TEXT,
    intent             TEXT    DEFAULT 'unknown',
    classified_at      TEXT,
    auto_response_sent INTEGER DEFAULT 0,
    auto_response_at   TEXT,
    FOREIGN KEY(kol_id) REFERENCES kols(id)
);

CREATE TABLE IF NOT EXISTS negotiations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kol_id       INTEGER NOT NULL,
    stage        TEXT    DEFAULT 'initial',
    price_usd    REAL,
    content_type TEXT,
    deliverables TEXT,
    notes        TEXT,
    created_at   TEXT,
    updated_at   TEXT,
    FOREIGN KEY(kol_id) REFERENCES kols(id)
);

CREATE TABLE IF NOT EXISTS contracts (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    kol_id             INTEGER NOT NULL,
    status             TEXT    DEFAULT 'draft',
    contract_date      TEXT,
    total_value_usd    REAL,
    revenue_share_rate REAL,
    deliverables       TEXT,
    signed_at          TEXT,
    notes              TEXT,
    FOREIGN KEY(kol_id) REFERENCES kols(id)
);

CREATE TABLE IF NOT EXISTS content (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    kol_id             INTEGER NOT NULL,
    contract_id        INTEGER,
    brief_sent_at      TEXT,
    draft_url          TEXT,
    draft_submitted_at TEXT,
    approved_at        TEXT,
    published_at       TEXT,
    published_url      TEXT,
    platform           TEXT,
    FOREIGN KEY(kol_id) REFERENCES kols(id)
);

CREATE TABLE IF NOT EXISTS performance (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kol_id        INTEGER NOT NULL,
    utm_code      TEXT,
    clicks        INTEGER DEFAULT 0,
    registrations INTEGER DEFAULT 0,
    trades        INTEGER DEFAULT 0,
    revenue_usd   REAL    DEFAULT 0.0,
    measured_at   TEXT,
    FOREIGN KEY(kol_id) REFERENCES kols(id)
);

CREATE TABLE IF NOT EXISTS payments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kol_id     INTEGER NOT NULL,
    amount_usd REAL,
    currency   TEXT    DEFAULT 'USD',
    due_date   TEXT,
    paid_at    TEXT,
    status     TEXT    DEFAULT 'pending',
    notes      TEXT,
    FOREIGN KEY(kol_id) REFERENCES kols(id)
);

CREATE TABLE IF NOT EXISTS media (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    type         TEXT    DEFAULT 'crypto',
    website      TEXT,
    email        TEXT,
    priority     TEXT    DEFAULT 'B',
    status       TEXT    DEFAULT '待联系',
    contact_name TEXT,
    notes        TEXT,
    collect_date TEXT,
    created_at   TEXT,
    updated_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_kols_status    ON kols(status);
CREATE INDEX IF NOT EXISTS idx_kols_email     ON kols(email);
CREATE INDEX IF NOT EXISTS idx_kols_collect   ON kols(collect_date);
CREATE INDEX IF NOT EXISTS idx_kols_channel   ON kols(channel_id);
CREATE INDEX IF NOT EXISTS idx_replies_kol    ON replies(kol_id);
CREATE INDEX IF NOT EXISTS idx_replies_intent ON replies(intent);
CREATE INDEX IF NOT EXISTS idx_contacts_kol   ON contacts(kol_id);
CREATE INDEX IF NOT EXISTS idx_media_status   ON media(status);

CREATE TABLE IF NOT EXISTS activities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kol_id     INTEGER NOT NULL,
    type       TEXT    NOT NULL,
    content    TEXT    DEFAULT '',
    operator   TEXT    DEFAULT 'auto',
    created_at TEXT,
    FOREIGN KEY(kol_id) REFERENCES kols(id)
);

CREATE INDEX IF NOT EXISTS idx_activities_kol  ON activities(kol_id);
CREATE INDEX IF NOT EXISTS idx_activities_time ON activities(created_at);

CREATE TABLE IF NOT EXISTS followups (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kol_id          INTEGER NOT NULL UNIQUE,
    step            INTEGER DEFAULT 1,
    scheduled_at    TEXT,
    last_sent_at    TEXT,
    status          TEXT    DEFAULT 'pending',
    FOREIGN KEY(kol_id) REFERENCES kols(id)
);
CREATE INDEX IF NOT EXISTS idx_followups_sched ON followups(scheduled_at);

CREATE TABLE IF NOT EXISTS pending_actions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kol_id      INTEGER NOT NULL,
    type        TEXT    NOT NULL,
    context     TEXT    DEFAULT '{}',
    status      TEXT    DEFAULT 'open',
    created_at  TEXT,
    resolved_at TEXT,
    resolved_by TEXT,
    FOREIGN KEY(kol_id) REFERENCES kols(id)
);
CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_actions(status);
CREATE INDEX IF NOT EXISTS idx_pending_kol    ON pending_actions(kol_id);
"""


# ALTER TABLE 迁移（列已存在时静默跳过）
_MIGRATIONS = [
    "ALTER TABLE kols ADD COLUMN sequence_step INTEGER DEFAULT 0",
    "ALTER TABLE kols ADD COLUMN next_followup_at TEXT",
    "ALTER TABLE kols ADD COLUMN last_contact_at TEXT",
    "ALTER TABLE kols ADD COLUMN score INTEGER DEFAULT 0",
]


def _now_bjt() -> str:
    """返回北京时间 ISO 字符串"""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")


def _today_bjt() -> str:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")


@contextmanager
def get_db(path: Path = None):
    """返回 SQLite 连接（context manager，自动 commit/close）"""
    conn = sqlite3.connect(str(path or DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: Path = None):
    """初始化数据库，创建所有表，并运行迁移"""
    with get_db(path) as conn:
        conn.executescript(_SCHEMA)
        for sql in _MIGRATIONS:
            try:
                conn.execute(sql)
            except Exception:
                pass  # 列已存在，忽略
    print(f"[kol_db] 数据库已初始化: {path or DB_PATH}")


def generate_utm_code(name: str) -> str:
    """为 KOL 生成唯一 UTM 标识（kol_name_xxxx）"""
    slug = re.sub(r'[^a-z0-9]', '', name.lower())[:12]
    suffix = hashlib.md5(name.encode()).hexdigest()[:4]
    return f"kol_{slug}_{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# KOL CRUD
# ─────────────────────────────────────────────────────────────────────────────

def upsert_kol(data: dict, path: Path = None) -> int:
    """
    插入或更新 KOL（以 channel_id 去重）。
    返回 kols.id。
    data 必须包含 name，channel_id 可为空（此时按 name+email 查重）。
    """
    now = _now_bjt()
    today = _today_bjt()
    channel_id = data.get("channel_id") or ""

    with get_db(path) as conn:
        if channel_id:
            row = conn.execute(
                "SELECT id, status FROM kols WHERE channel_id=?", (channel_id,)
            ).fetchone()
        else:
            row = None

        if row:
            # 已存在：仅更新非状态字段（不覆盖已有联系状态）
            conn.execute("""
                UPDATE kols SET
                    name=COALESCE(?,name), channel_url=COALESCE(?,channel_url),
                    subscribers=MAX(COALESCE(?,0), subscribers),
                    tier=COALESCE(?,tier), email=COALESCE(NULLIF(?,''),(SELECT email FROM kols WHERE channel_id=?)),
                    description=COALESCE(?,description), twitter=COALESCE(?,twitter),
                    country=COALESCE(?,country), updated_at=?
                WHERE channel_id=?
            """, (
                data.get("name"), data.get("channel_url"),
                data.get("subscribers", 0),
                data.get("tier"), data.get("email", ""), channel_id,
                data.get("description"), data.get("twitter"),
                data.get("country"), now,
                channel_id,
            ))
            return row["id"]
        else:
            utm = generate_utm_code(data.get("name", ""))
            conn.execute("""
                INSERT INTO kols
                    (source, channel_id, channel_url, name, platform, subscribers,
                     tier, email, description, category, country, twitter,
                     collect_date, status, utm_code, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                data.get("source", "youtube"),
                channel_id or None,
                data.get("channel_url", ""),
                data["name"],
                data.get("platform", "YouTube"),
                data.get("subscribers", 0),
                data.get("tier", "C级"),
                data.get("email", ""),
                data.get("description", ""),
                data.get("category", ""),
                data.get("country", ""),
                data.get("twitter", ""),
                data.get("collect_date", today),
                data.get("status", "待发送"),
                utm,
                now, now,
            ))
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def get_kols_to_send(daily_limit: int = 10, source: str = None,
                     path: Path = None) -> list[dict]:
    """返回待发送 KOL 列表（有邮箱、状态为待发送）"""
    query = """
        SELECT id, name, email, tier, description, subscribers, utm_code, source
        FROM kols
        WHERE status='待发送' AND email != '' AND email IS NOT NULL
    """
    params = []
    if source:
        query += " AND source=?"
        params.append(source)
    query += " ORDER BY CASE tier WHEN 'KALSHI' THEN 0 WHEN 'PM' THEN 1 WHEN 'C级' THEN 2 WHEN 'D级' THEN 3 WHEN 'B级' THEN 4 ELSE 5 END, subscribers DESC"
    if daily_limit:
        query += f" LIMIT {daily_limit}"

    with get_db(path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def mark_sent(kol_id: int, subject: str = "", template: str = "",
              path: Path = None):
    """外联发送成功后更新状态，并写入活动日志 + 安排7天后跟进"""
    now = _now_bjt()
    from datetime import date, timedelta
    followup_date = (date.today() + timedelta(days=7)).isoformat()
    with get_db(path) as conn:
        row = conn.execute("SELECT sequence_step FROM kols WHERE id=?", (kol_id,)).fetchone()
        step = (row["sequence_step"] or 0) + 1 if row else 1
        conn.execute(
            """UPDATE kols SET status='已发送', last_contact_at=?,
               next_followup_at=?, sequence_step=?, updated_at=? WHERE id=?""",
            (now, followup_date, step, now, kol_id)
        )
        conn.execute(
            "INSERT INTO contacts (kol_id, sent_at, subject, template, status) VALUES (?,?,?,?,'sent')",
            (kol_id, now, subject, template)
        )
        conn.execute(
            "INSERT INTO activities (kol_id, type, content, operator, created_at) VALUES (?,?,?,?,?)",
            (kol_id, "email_sent", f"第{step}封邮件 | 模板: {template} | 主题: {subject}", "auto", now)
        )
        # 写入 followups 表（upsert）
        conn.execute("""
            INSERT INTO followups (kol_id, step, scheduled_at, last_sent_at, status)
            VALUES (?,?,?,?,'pending')
            ON CONFLICT(kol_id) DO UPDATE SET
                step=excluded.step, scheduled_at=excluded.scheduled_at,
                last_sent_at=excluded.last_sent_at, status='pending'
        """, (kol_id, step, followup_date, now))


def get_kols_by_email(path: Path = None) -> dict[str, dict]:
    """返回 {email: {id, name, status}} 用于回复匹配"""
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT id, name, email, status FROM kols WHERE email != '' AND email IS NOT NULL AND status='已发送'"
        ).fetchall()
        return {r["email"].lower(): dict(r) for r in rows}


def record_reply(kol_id: int, email_from: str, subject: str = "",
                 body_snippet: str = "", path: Path = None) -> int:
    """记录收到的回复，返回 reply_id"""
    now = _now_bjt()
    with get_db(path) as conn:
        # 避免重复记录同一封邮件
        existing = conn.execute(
            "SELECT id FROM replies WHERE kol_id=? AND email_from=? AND subject=?",
            (kol_id, email_from, subject)
        ).fetchone()
        if existing:
            return existing["id"]

        conn.execute(
            """INSERT INTO replies (kol_id, email_from, subject, body_snippet, detected_at)
               VALUES (?,?,?,?,?)""",
            (kol_id, email_from, subject, body_snippet[:500], now)
        )
        reply_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            "UPDATE kols SET status='已回复', next_followup_at=NULL, updated_at=? WHERE id=?",
            (now, kol_id)
        )
        conn.execute(
            "INSERT INTO activities (kol_id, type, content, operator, created_at) VALUES (?,?,?,?,?)",
            (kol_id, "reply_received", f"主题: {subject} | {body_snippet[:100]}", "auto", now)
        )
        conn.execute(
            "UPDATE followups SET status='replied' WHERE kol_id=?", (kol_id,)
        )
        return reply_id


def get_unclassified_replies(path: Path = None) -> list[dict]:
    """返回 intent='unknown' 的回复记录，含 KOL 基础信息"""
    with get_db(path) as conn:
        rows = conn.execute("""
            SELECT r.id, r.kol_id, r.email_from, r.subject, r.body_snippet,
                   r.detected_at, k.name, k.email, k.tier
            FROM replies r
            JOIN kols k ON k.id = r.kol_id
            WHERE r.intent='unknown' AND r.auto_response_sent=0
            ORDER BY r.detected_at ASC
        """).fetchall()
        return [dict(r) for r in rows]


def set_reply_intent(reply_id: int, intent: str, path: Path = None):
    """更新回复意图分类"""
    with get_db(path) as conn:
        conn.execute(
            "UPDATE replies SET intent=?, classified_at=? WHERE id=?",
            (intent, _now_bjt(), reply_id)
        )
        # 同步更新 KOL 状态
        row = conn.execute("SELECT kol_id FROM replies WHERE id=?", (reply_id,)).fetchone()
        if row and intent == "rejected":
            conn.execute(
                "UPDATE kols SET status='已拒绝', updated_at=? WHERE id=?",
                (_now_bjt(), row["kol_id"])
            )
        elif row and intent == "interested":
            conn.execute(
                "UPDATE kols SET status='TG接触', updated_at=? WHERE id=?",
                (_now_bjt(), row["kol_id"])
            )


def mark_auto_response_sent(reply_id: int, path: Path = None):
    with get_db(path) as conn:
        conn.execute(
            "UPDATE replies SET auto_response_sent=1, auto_response_at=? WHERE id=?",
            (_now_bjt(), reply_id)
        )


# ─────────────────────────────────────────────────────────────────────────────
# 谈判 / 合同 / 付款
# ─────────────────────────────────────────────────────────────────────────────

def add_negotiation(kol_id: int, price_usd: float = None, content_type: str = "",
                    deliverables: str = "", notes: str = "", path: Path = None) -> int:
    now = _now_bjt()
    with get_db(path) as conn:
        conn.execute("""
            INSERT INTO negotiations (kol_id, price_usd, content_type, deliverables, notes, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?)
        """, (kol_id, price_usd, content_type, deliverables, notes, now, now))
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def add_contract(kol_id: int, total_value_usd: float = None,
                 revenue_share_rate: float = None, deliverables: str = "",
                 notes: str = "", path: Path = None) -> int:
    now = _now_bjt()
    with get_db(path) as conn:
        conn.execute("""
            INSERT INTO contracts (kol_id, status, total_value_usd, revenue_share_rate,
                                   deliverables, contract_date, notes)
            VALUES (?,?,?,?,?,?,?)
        """, (kol_id, "draft", total_value_usd, revenue_share_rate, deliverables, now[:10], notes))
        kol_id_inserted = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "UPDATE kols SET status='已签约', updated_at=? WHERE id=?",
            (now, kol_id)
        )
        return kol_id_inserted


def add_payment(kol_id: int, amount_usd: float, due_date: str,
                notes: str = "", path: Path = None) -> int:
    with get_db(path) as conn:
        conn.execute("""
            INSERT INTO payments (kol_id, amount_usd, due_date, status, notes)
            VALUES (?,?,?,'pending',?)
        """, (kol_id, amount_usd, due_date, notes))
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def mark_payment_paid(payment_id: int, path: Path = None):
    with get_db(path) as conn:
        conn.execute(
            "UPDATE payments SET status='paid', paid_at=? WHERE id=?",
            (_now_bjt(), payment_id)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Media CRUD
# ─────────────────────────────────────────────────────────────────────────────

def upsert_media(data: dict, path: Path = None) -> int:
    now = _now_bjt()
    today = _today_bjt()
    name = data.get("name", "").strip()
    if not name:
        raise ValueError("media name is required")

    with get_db(path) as conn:
        row = conn.execute("SELECT id FROM media WHERE name=?", (name,)).fetchone()
        if row:
            conn.execute("""
                UPDATE media SET
                    type=COALESCE(?,type), website=COALESCE(?,website),
                    email=COALESCE(NULLIF(?,''), (SELECT email FROM media WHERE name=?)),
                    priority=COALESCE(?,priority), contact_name=COALESCE(?,contact_name),
                    updated_at=?
                WHERE name=?
            """, (
                data.get("type"), data.get("website"),
                data.get("email", ""), name,
                data.get("priority"), data.get("contact_name"),
                now, name,
            ))
            return row["id"]
        else:
            conn.execute("""
                INSERT INTO media (name, type, website, email, priority, contact_name,
                                   notes, collect_date, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                name, data.get("type", "crypto"), data.get("website", ""),
                data.get("email", ""), data.get("priority", "B"),
                data.get("contact_name", ""), data.get("notes", ""),
                data.get("collect_date", today), now, now,
            ))
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def get_media_to_send(path: Path = None) -> list[dict]:
    with get_db(path) as conn:
        rows = conn.execute("""
            SELECT id, name, type, email, priority, contact_name
            FROM media
            WHERE status='待联系' AND email != '' AND email IS NOT NULL AND priority='A'
            ORDER BY name
        """).fetchall()
        return [dict(r) for r in rows]


def mark_media_inquired(media_id: int, path: Path = None):
    today = _today_bjt()
    with get_db(path) as conn:
        conn.execute(
            "UPDATE media SET status=?, updated_at=? WHERE id=?",
            (f"已询价 {today}", _now_bjt(), media_id)
        )


def mark_media_replied(media_id: int, path: Path = None):
    today = _today_bjt()
    with get_db(path) as conn:
        conn.execute(
            "UPDATE media SET status=?, updated_at=? WHERE id=?",
            (f"已回复 {today}", _now_bjt(), media_id)
        )


# ─────────────────────────────────────────────────────────────────────────────
# 统计（日报用）
# ─────────────────────────────────────────────────────────────────────────────

def get_kol_stats(path: Path = None) -> dict:
    """返回 KOL 维度统计，供日报使用"""
    today      = _today_bjt()
    yesterday  = (date.today() - timedelta(days=1)).isoformat()
    before_y   = (date.today() - timedelta(days=2)).isoformat()

    with get_db(path) as conn:
        def _count(sql, params=()):
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else 0

        total        = _count("SELECT COUNT(*) FROM kols")
        total_email  = _count("SELECT COUNT(*) FROM kols WHERE email!='' AND email IS NOT NULL")
        total_sent   = _count("SELECT COUNT(*) FROM kols WHERE status IN ('已发送','已回复','已签约','谈判中','已发布','已完成')")
        total_replied = _count("SELECT COUNT(*) FROM kols WHERE status IN ('已回复','已签约','谈判中','已发布','已完成','已拒绝')")
        total_signed = _count("SELECT COUNT(*) FROM kols WHERE status='已签约'")

        today_total  = _count("SELECT COUNT(*) FROM kols WHERE collect_date=?", (today,))
        today_email  = _count("SELECT COUNT(*) FROM kols WHERE collect_date=? AND email!=''", (today,))
        yest_total   = _count("SELECT COUNT(*) FROM kols WHERE collect_date=?", (yesterday,))
        yest_email   = _count("SELECT COUNT(*) FROM kols WHERE collect_date=? AND email!=''", (yesterday,))
        by_total     = _count("SELECT COUNT(*) FROM kols WHERE collect_date=?", (before_y,))
        by_email     = _count("SELECT COUNT(*) FROM kols WHERE collect_date=? AND email!=''", (before_y,))

        # 发送数从 contacts 表读（按 sent_at 日期匹配）
        sent_today   = _count("SELECT COUNT(*) FROM contacts WHERE sent_at LIKE ?", (f"{today}%",))
        sent_yest    = _count("SELECT COUNT(*) FROM contacts WHERE sent_at LIKE ?", (f"{yesterday}%",))
        sent_by      = _count("SELECT COUNT(*) FROM contacts WHERE sent_at LIKE ?", (f"{before_y}%",))

        # 回复数从 replies 表读
        replied_today = _count("SELECT COUNT(*) FROM replies WHERE detected_at LIKE ?", (f"{today}%",))
        replied_yest  = _count("SELECT COUNT(*) FROM replies WHERE detected_at LIKE ?", (f"{yesterday}%",))
        replied_by    = _count("SELECT COUNT(*) FROM replies WHERE detected_at LIKE ?", (f"{before_y}%",))

        # 意图分布
        intent_rows = conn.execute("""
            SELECT intent, COUNT(*) as cnt FROM replies
            WHERE intent != 'unknown' GROUP BY intent
        """).fetchall()
        intents = {r["intent"]: r["cnt"] for r in intent_rows}

    return {
        "kol": {
            "total":          total,         "today_total":    today_total,
            "yest_total":     yest_total,    "by_total":       by_total,
            "total_email":    total_email,   "today_email":    today_email,
            "yest_email":     yest_email,    "by_email":       by_email,
            "total_sent":     total_sent,    "sent_today":     sent_today,
            "sent_yest":      sent_yest,     "sent_by":        sent_by,
            "total_replied":  total_replied, "replied_today":  replied_today,
            "replied_yest":   replied_yest,  "replied_by":     replied_by,
            "total_signed":   total_signed,
            "intents":        intents,
        }
    }


def get_media_stats(path: Path = None) -> dict:
    today      = _today_bjt()
    yesterday  = (date.today() - timedelta(days=1)).isoformat()
    before_y   = (date.today() - timedelta(days=2)).isoformat()

    with get_db(path) as conn:
        def _count(sql, params=()):
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else 0

        total        = _count("SELECT COUNT(*) FROM media")
        total_email  = _count("SELECT COUNT(*) FROM media WHERE email!='' AND email IS NOT NULL")
        total_sent   = _count("SELECT COUNT(*) FROM media WHERE status LIKE '已询价%'")
        total_replied = _count("SELECT COUNT(*) FROM media WHERE status LIKE '已回复%'")

        today_total  = _count("SELECT COUNT(*) FROM media WHERE collect_date=?", (today,))
        today_email  = _count("SELECT COUNT(*) FROM media WHERE collect_date=? AND email!=''", (today,))
        yest_total   = _count("SELECT COUNT(*) FROM media WHERE collect_date=?", (yesterday,))
        yest_email   = _count("SELECT COUNT(*) FROM media WHERE collect_date=? AND email!=''", (yesterday,))
        by_total     = _count("SELECT COUNT(*) FROM media WHERE collect_date=?", (before_y,))
        by_email     = _count("SELECT COUNT(*) FROM media WHERE collect_date=? AND email!=''", (before_y,))

        # 询价/回复按 updated_at 日期
        sent_today   = _count("SELECT COUNT(*) FROM media WHERE status LIKE '已询价%' AND updated_at LIKE ?", (f"{today}%",))
        sent_yest    = _count("SELECT COUNT(*) FROM media WHERE status LIKE '已询价%' AND updated_at LIKE ?", (f"{yesterday}%",))
        sent_by      = _count("SELECT COUNT(*) FROM media WHERE status LIKE '已询价%' AND updated_at LIKE ?", (f"{before_y}%",))
        replied_today = _count("SELECT COUNT(*) FROM media WHERE status LIKE '已回复%' AND updated_at LIKE ?", (f"{today}%",))
        replied_yest  = _count("SELECT COUNT(*) FROM media WHERE status LIKE '已回复%' AND updated_at LIKE ?", (f"{yesterday}%",))
        replied_by    = _count("SELECT COUNT(*) FROM media WHERE status LIKE '已回复%' AND updated_at LIKE ?", (f"{before_y}%",))

    return {
        "media": {
            "total":         total,         "today_total":    today_total,
            "yest_total":    yest_total,    "by_total":       by_total,
            "total_email":   total_email,   "today_email":    today_email,
            "yest_email":    yest_email,    "by_email":       by_email,
            "total_sent":    total_sent,    "sent_today":     sent_today,
            "sent_yest":     sent_yest,     "sent_by":        sent_by,
            "total_replied": total_replied, "replied_today":  replied_today,
            "replied_yest":  replied_yest,  "replied_by":     replied_by,
        }
    }


def get_detail_kols(path: Path = None) -> list[dict]:
    """返回有邮箱的 KOL 明细（用于日报明细卡）"""
    with get_db(path) as conn:
        rows = conn.execute("""
            SELECT name, channel_url, tier, subscribers, email, status, utm_code
            FROM kols
            WHERE email != '' AND email IS NOT NULL
            ORDER BY
                CASE status
                    WHEN '已签约' THEN 0 WHEN '谈判中' THEN 1 WHEN '已回复' THEN 2
                    WHEN '已发送' THEN 3 ELSE 4
                END,
                subscribers DESC
            LIMIT 200
        """).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Activity Log
# ─────────────────────────────────────────────────────────────────────────────

def log_activity(kol_id: int, type: str, content: str = "",
                 operator: str = "auto", path: Path = None):
    """写入活动日志"""
    with get_db(path) as conn:
        conn.execute(
            "INSERT INTO activities (kol_id, type, content, operator, created_at) VALUES (?,?,?,?,?)",
            (kol_id, type, content, operator, _now_bjt())
        )


def add_kol_note(kol_id: int, content: str, operator: str = "kelly",
                 path: Path = None):
    """添加手动备注（同时写入 activities + kols.notes）"""
    now = _now_bjt()
    with get_db(path) as conn:
        conn.execute(
            "INSERT INTO activities (kol_id, type, content, operator, created_at) VALUES (?,?,?,?,?)",
            (kol_id, "note", content, operator, now)
        )
        conn.execute(
            "UPDATE kols SET notes=?, updated_at=? WHERE id=?",
            (content, now, kol_id)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up 序列
# ─────────────────────────────────────────────────────────────────────────────

def schedule_followup(kol_id: int, days: int = 7, path: Path = None):
    """安排 N 天后跟进"""
    from datetime import date, timedelta
    followup_date = (date.today() + timedelta(days=days)).isoformat()
    now = _now_bjt()
    with get_db(path) as conn:
        conn.execute(
            "UPDATE kols SET next_followup_at=?, updated_at=? WHERE id=?",
            (followup_date, now, kol_id)
        )
        conn.execute("""
            INSERT INTO followups (kol_id, scheduled_at, status)
            VALUES (?,?,'pending')
            ON CONFLICT(kol_id) DO UPDATE SET scheduled_at=excluded.scheduled_at, status='pending'
        """, (kol_id, followup_date))


def get_followup_queue(path: Path = None) -> list[dict]:
    """返回今天需要 follow-up 的 KOL 列表"""
    today = _today_bjt()
    with get_db(path) as conn:
        rows = conn.execute("""
            SELECT k.id, k.name, k.email, k.tier, k.sequence_step,
                   k.next_followup_at, k.status, f.step
            FROM kols k
            JOIN followups f ON f.kol_id = k.id
            WHERE f.scheduled_at <= ? AND f.status='pending'
              AND k.status='已发送' AND k.email != ''
            ORDER BY f.scheduled_at
        """, (today,)).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# 谈判 upsert（更新已有或新建）
# ─────────────────────────────────────────────────────────────────────────────

def upsert_negotiation(kol_id: int, data: dict, path: Path = None) -> int:
    """更新或新建谈判记录，并写入活动日志"""
    now = _now_bjt()
    with get_db(path) as conn:
        existing = conn.execute(
            "SELECT id FROM negotiations WHERE kol_id=? ORDER BY created_at DESC LIMIT 1",
            (kol_id,)
        ).fetchone()
        # 确保 payment_terms 列存在（migration）
        cols = {r[1] for r in conn.execute("PRAGMA table_info(negotiations)").fetchall()}
        if "payment_terms" not in cols:
            conn.execute("ALTER TABLE negotiations ADD COLUMN payment_terms TEXT DEFAULT ''")

        if existing:
            conn.execute("""
                UPDATE negotiations SET
                    stage         = COALESCE(?,stage),
                    price_usd     = COALESCE(?,price_usd),
                    content_type  = COALESCE(?,content_type),
                    deliverables  = COALESCE(?,deliverables),
                    payment_terms = COALESCE(NULLIF(?,''),payment_terms),
                    notes         = COALESCE(?,notes),
                    updated_at    = ?
                WHERE id=?
            """, (
                data.get("stage"), data.get("price_usd"),
                data.get("content_type"), data.get("deliverables"),
                data.get("payment_terms"), data.get("notes"), now, existing["id"]
            ))
            neg_id = existing["id"]
        else:
            conn.execute("""
                INSERT INTO negotiations
                    (kol_id, stage, price_usd, content_type, deliverables, payment_terms, notes, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                kol_id, data.get("stage", "initial"), data.get("price_usd"),
                data.get("content_type", ""), data.get("deliverables", ""),
                data.get("payment_terms", ""), data.get("notes", ""), now, now
            ))
            neg_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # 如果状态升级为谈判中
        if data.get("stage") in ("price_quoted", "terms_agreed"):
            conn.execute(
                "UPDATE kols SET status='谈判中', updated_at=? WHERE id=? AND status NOT IN ('已签约','已发布','已完成')",
                (now, kol_id)
            )

        content = f"阶段:{data.get('stage','')} 报价:${data.get('price_usd','')} 类型:{data.get('content_type','')} 备注:{data.get('notes','')[:50]}"
        conn.execute(
            "INSERT INTO activities (kol_id, type, content, operator, created_at) VALUES (?,?,?,?,?)",
            (kol_id, "negotiation", content, "kelly", now)
        )
        return neg_id


def change_kol_status(kol_id: int, new_status: str, operator: str = "kelly",
                      path: Path = None):
    """手动变更 KOL 状态并记录日志"""
    now = _now_bjt()
    with get_db(path) as conn:
        old = conn.execute("SELECT status FROM kols WHERE id=?", (kol_id,)).fetchone()
        old_status = old["status"] if old else "?"
        conn.execute(
            "UPDATE kols SET status=?, updated_at=? WHERE id=?",
            (new_status, now, kol_id)
        )
        conn.execute(
            "INSERT INTO activities (kol_id, type, content, operator, created_at) VALUES (?,?,?,?,?)",
            (kol_id, "status_change", f"{old_status} → {new_status}", operator, now)
        )


# ─────────────────────────────────────────────────────────────────────────────
# KOL 详情（详情页用）
# ─────────────────────────────────────────────────────────────────────────────

def get_kol_detail(kol_id: int, path: Path = None) -> dict | None:
    """获取 KOL 完整数据：基础信息 + 联系历史 + 回复 + 活动 + 谈判 + 合同 + 付款"""
    with get_db(path) as conn:
        kol = conn.execute("SELECT * FROM kols WHERE id=?", (kol_id,)).fetchone()
        if not kol:
            return None

        contacts = conn.execute(
            "SELECT * FROM contacts WHERE kol_id=? ORDER BY sent_at DESC",
            (kol_id,)
        ).fetchall()

        replies = conn.execute(
            "SELECT * FROM replies WHERE kol_id=? ORDER BY detected_at DESC",
            (kol_id,)
        ).fetchall()

        negotiation = conn.execute(
            "SELECT * FROM negotiations WHERE kol_id=? ORDER BY created_at DESC LIMIT 1",
            (kol_id,)
        ).fetchone()

        contract = conn.execute(
            "SELECT * FROM contracts WHERE kol_id=? ORDER BY contract_date DESC LIMIT 1",
            (kol_id,)
        ).fetchone()

        payments = conn.execute(
            "SELECT * FROM payments WHERE kol_id=? ORDER BY due_date",
            (kol_id,)
        ).fetchall()

        followup = conn.execute(
            "SELECT * FROM followups WHERE kol_id=?", (kol_id,)
        ).fetchone()

        # 合并时间线：activities + 未被 activities 覆盖的 contacts/replies
        activities = conn.execute(
            "SELECT * FROM activities WHERE kol_id=? ORDER BY created_at DESC LIMIT 200",
            (kol_id,)
        ).fetchall()

        return {
            "kol":         dict(kol),
            "contacts":    [dict(r) for r in contacts],
            "replies":     [dict(r) for r in replies],
            "negotiation": dict(negotiation) if negotiation else {},
            "contract":    dict(contract) if contract else {},
            "payments":    [dict(r) for r in payments],
            "followup":    dict(followup) if followup else {},
            "activities":  [dict(r) for r in activities],
        }


def find_kol_by_name(name: str, path: Path = None) -> list[dict]:
    """按姓名模糊搜索 KOL，返回列表（含 id/name/status/email/tg_handle）"""
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT id, name, status, email, tg_handle, tier, subscribers FROM kols "
            "WHERE LOWER(name) LIKE ? ORDER BY subscribers DESC LIMIT 10",
            (f"%{name.lower()}%",)
        ).fetchall()
        return [dict(r) for r in rows]


def update_kol_tg(kol_id: int, tg_handle: str, path: Path = None):
    """记录 TG handle，更新状态为 TG接触，并推送报价待确认事项"""
    now = _now_bjt()
    with get_db(path) as conn:
        conn.execute(
            "UPDATE kols SET tg_handle=?, tg_status='invited', status='TG接触', updated_at=? WHERE id=?",
            (tg_handle, now, kol_id)
        )
        conn.execute(
            "INSERT INTO activities (kol_id, type, content, operator, created_at) VALUES (?,?,?,?,?)",
            (kol_id, "tg_contact", f"TG handle: {tg_handle}", "kelly", now)
        )
    # TG 接触成功 → 推报价待确认（create_pending_action 内部自动去重）
    create_pending_action(kol_id, "quote_needed", {
        "tg_handle": tg_handle,
        "note": "KOL 已加 TG，待发送报价",
    }, path=path)


def record_content_published(kol_id: int, url: str, platform: str = "",
                             contract_id: int = None, path: Path = None) -> int:
    """记录内容发布 URL，更新 KOL 状态为已发布"""
    now = _now_bjt()
    with get_db(path) as conn:
        conn.execute("""
            INSERT INTO content (kol_id, contract_id, published_at, published_url, platform)
            VALUES (?,?,?,?,?)
        """, (kol_id, contract_id, now[:10], url, platform))
        content_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "UPDATE kols SET status='已发布', updated_at=? WHERE id=?",
            (now, kol_id)
        )
        conn.execute(
            "INSERT INTO activities (kol_id, type, content, operator, created_at) VALUES (?,?,?,?,?)",
            (kol_id, "content_published", f"URL: {url}", "kelly", now)
        )
        return content_id


# ─────────────────────────────────────────────────────────────────────────────
# 待处理事项（Pending Actions）
# ─────────────────────────────────────────────────────────────────────────────

def create_pending_action(kol_id: int, action_type: str,
                          context: dict = None, path: Path = None) -> int:
    """
    创建一条待 Kelly 处理的事项。
    action_type: 'quote_needed' | 'contract_review' | 'content_review'
    context: 附加信息，如 {'reply_snippet': '...'}
    自动去重：同一 kol_id + type 若已有 open 记录则不重复创建。
    """
    import json as _json
    ctx_str = _json.dumps(context or {}, ensure_ascii=False)
    now = _now_bjt()
    with get_db(path) as conn:
        existing = conn.execute(
            "SELECT id FROM pending_actions WHERE kol_id=? AND type=? AND status='open'",
            (kol_id, action_type)
        ).fetchone()
        if existing:
            return existing[0]
        conn.execute(
            "INSERT INTO pending_actions (kol_id, type, context, status, created_at) VALUES (?,?,?,?,?)",
            (kol_id, action_type, ctx_str, "open", now)
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def get_pending_actions(status: str = "open", path: Path = None) -> list[dict]:
    """返回待处理事项列表，附带 KOL 基本信息"""
    import json as _json
    with get_db(path) as conn:
        rows = conn.execute("""
            SELECT p.id, p.kol_id, p.type, p.context, p.created_at,
                   k.name, k.platform, k.tier, k.subscribers,
                   k.email, k.tg_handle, k.status as kol_status,
                   k.channel_url,
                   (SELECT body_snippet FROM replies
                    WHERE kol_id=k.id ORDER BY detected_at DESC LIMIT 1) as last_reply,
                   (SELECT intent FROM replies
                    WHERE kol_id=k.id ORDER BY detected_at DESC LIMIT 1) as last_intent
            FROM pending_actions p
            JOIN kols k ON k.id = p.kol_id
            WHERE p.status = ?
            ORDER BY p.created_at DESC
        """, (status,)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["context"] = _json.loads(d["context"] or "{}")
            except Exception:
                d["context"] = {}
            result.append(d)
        return result


def resolve_pending_action(action_id: int, resolved_by: str = "kelly",
                           path: Path = None):
    """将一条待处理事项标记为已解决"""
    now = _now_bjt()
    with get_db(path) as conn:
        conn.execute(
            "UPDATE pending_actions SET status='resolved', resolved_at=?, resolved_by=? WHERE id=?",
            (now, resolved_by, action_id)
        )


def get_pending_counts(path: Path = None) -> dict:
    """返回各类型待处理数量（用于总览徽章）"""
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT type, COUNT(*) as cnt FROM pending_actions WHERE status='open' GROUP BY type"
        ).fetchall()
        counts = {"quote_needed": 0, "contract_review": 0, "content_review": 0}
        for r in rows:
            counts[r["type"]] = r["cnt"]
        counts["total"] = sum(counts.values())
        return counts


if __name__ == "__main__":
    init_db()
    print(f"[OK] DB 初始化完成: {DB_PATH}")
