#!/usr/bin/env python3
"""
MoonX KOL CRM — 完整版
python3 crm_preview.py → 自动打开 http://localhost:9999
"""
import sqlite3, json, re, csv, io, threading, webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

DB_PATH = Path(__file__).parent / "kol_crm.db"
KOL_STATUSES = ['待发送', '已发送', '已回复', 'TG接触', '谈判中', '已签约', '审核中', '已发布', '已完成', '已拒绝', '冷却']


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ─── API: LIST ──────────────────────────────────────

def api_kols(params):
    status = params.get("status", [""])[0]
    search = params.get("q", [""])[0].lower()
    tier   = params.get("tier", [""])[0]
    page   = int(params.get("page", ["1"])[0])
    limit  = int(params.get("limit", ["50"])[0])

    with get_db() as conn:
        where, args = [], []
        if status:
            where.append("status=?"); args.append(status)
        if tier:
            where.append("tier=?"); args.append(tier)
        if search:
            where.append("(LOWER(name) LIKE ? OR LOWER(email) LIKE ? OR LOWER(description) LIKE ?)")
            args += [f"%{search}%"] * 3
        sql_where = ("WHERE " + " AND ".join(where)) if where else ""

        total = conn.execute(f"SELECT COUNT(*) FROM kols {sql_where}", args).fetchone()[0]
        rows  = conn.execute(
            f"SELECT id,name,platform,subscribers,tier,email,status,country,twitter,collect_date "
            f"FROM kols {sql_where} ORDER BY subscribers DESC LIMIT ? OFFSET ?",
            args + [limit, (page - 1) * limit]
        ).fetchall()

        stats = {}
        for r in conn.execute("SELECT status, COUNT(*) FROM kols GROUP BY status").fetchall():
            stats[r[0] or "未知"] = r[1]

        tiers = [r[0] for r in conn.execute(
            "SELECT DISTINCT tier FROM kols WHERE tier!='' ORDER BY tier"
        ).fetchall()]

    return {"rows": [dict(r) for r in rows], "total": total,
            "stats": stats, "tiers": tiers, "page": page, "limit": limit}


# ─── API: DETAIL ────────────────────────────────────

def api_kol_detail(kol_id):
    with get_db() as conn:
        kol = conn.execute("SELECT * FROM kols WHERE id=?", (kol_id,)).fetchone()
        if not kol:
            return None
        contacts = conn.execute(
            "SELECT id, sent_at, subject, template, status FROM contacts "
            "WHERE kol_id=? ORDER BY sent_at DESC LIMIT 30",
            (kol_id,)
        ).fetchall()
        replies = conn.execute(
            "SELECT id, email_from, subject, body_snippet, detected_at, intent "
            "FROM replies WHERE kol_id=? ORDER BY detected_at DESC LIMIT 30",
            (kol_id,)
        ).fetchall()
        negotiations = conn.execute(
            "SELECT id, stage, price_usd, content_type, deliverables, notes, created_at "
            "FROM negotiations WHERE kol_id=? ORDER BY created_at DESC",
            (kol_id,)
        ).fetchall()
    return {
        "kol": dict(kol),
        "contacts": [dict(r) for r in contacts],
        "replies": [dict(r) for r in replies],
        "negotiations": [dict(r) for r in negotiations],
    }


# ─── API: INBOX ─────────────────────────────────────

def api_inbox(params):
    intent_filter = params.get("intent", [""])[0]
    with get_db() as conn:
        cond = "WHERE r.detected_at IS NOT NULL"
        args = []
        if intent_filter:
            cond += " AND r.intent=?"
            args.append(intent_filter)
        rows = conn.execute(f"""
            SELECT r.id, r.kol_id, r.email_from, r.subject, r.body_snippet,
                   r.detected_at, r.intent,
                   k.name as kol_name, k.platform, k.tier,
                   k.subscribers, k.status as kol_status
            FROM replies r
            JOIN kols k ON r.kol_id = k.id
            {cond}
            ORDER BY r.detected_at DESC
            LIMIT 200
        """, args).fetchall()
        counts = {}
        for r in conn.execute(
            "SELECT intent, COUNT(*) FROM replies WHERE detected_at IS NOT NULL GROUP BY intent"
        ).fetchall():
            counts[r[0] or "unknown"] = r[1]
    return {"rows": [dict(r) for r in rows], "intent_counts": counts}


# ─── API: MEDIA ─────────────────────────────────────

def api_media(params):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,name,type,website,email,priority,status,notes FROM media ORDER BY priority,name"
        ).fetchall()
    return {"rows": [dict(r) for r in rows]}


# ─── API: EXPORT CSV ────────────────────────────────

def api_export_csv():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,name,platform,subscribers,tier,email,twitter,country,status,collect_date,notes "
            "FROM kols ORDER BY subscribers DESC"
        ).fetchall()
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows([dict(r) for r in rows])
    return buf.getvalue()


# ─── MUTATIONS ──────────────────────────────────────

def do_update_status(kol_id, data):
    with get_db() as conn:
        conn.execute("UPDATE kols SET status=?, updated_at=? WHERE id=?",
                     (data["status"], datetime.now().isoformat(), kol_id))
    return {}


def do_update_notes(kol_id, data):
    with get_db() as conn:
        conn.execute("UPDATE kols SET notes=?, updated_at=? WHERE id=?",
                     (data.get("notes", ""), datetime.now().isoformat(), kol_id))
    return {}


def do_add_negotiation(kol_id, data):
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO negotiations
              (kol_id, stage, price_usd, content_type, deliverables, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (kol_id,
              data.get("stage", "initial"),
              data.get("price_usd") or None,
              data.get("content_type", ""),
              data.get("deliverables", ""),
              data.get("notes", ""),
              now, now))
    return {}


# ─── HTML ───────────────────────────────────────────

def build_html():
    statuses_js = json.dumps(KOL_STATUSES, ensure_ascii=False)
    return r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>MoonX KOL CRM</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  :root { --brand: #FF6B00; }
  body { font-family: -apple-system, 'Inter', sans-serif; }

  /* Status badges */
  .badge-待发送 { background:#1e3a5f; color:#60a5fa; }
  .badge-已发送 { background:#1a3a2a; color:#4ade80; }
  .badge-已回复 { background:#3a2a1a; color:#fb923c; }
  .badge-谈判中 { background:#2a1a3a; color:#c084fc; }
  .badge-已签约 { background:#1a1a3a; color:#818cf8; }
  .badge-执行中 { background:#3a3a1a; color:#facc15; }
  .badge-完成   { background:#1a2a2a; color:#6ee7b7; }

  /* Tier badges */
  .tier-A { background:#7c3aed22; color:#a78bfa; }
  .tier-B { background:#0369a122; color:#38bdf8; }
  .tier-C { background:#15803d22; color:#4ade80; }
  .tier-D { background:#71717a22; color:#a1a1aa; }

  /* Intent badges */
  .intent-感兴趣   { background:#14532d; color:#4ade80; }
  .intent-待确认   { background:#713f12; color:#fcd34d; }
  .intent-拒绝     { background:#7f1d1d; color:#fca5a5; }
  .intent-positive { background:#14532d; color:#4ade80; }
  .intent-interested { background:#14532d; color:#4ade80; }
  .intent-negative { background:#7f1d1d; color:#fca5a5; }
  .intent-unknown  { background:#1f2937; color:#9ca3af; }

  /* Kanban */
  .kanban-col { min-width: 210px; flex-shrink: 0; min-height: 400px; }
  .kol-card { cursor: pointer; transition: transform 0.15s, box-shadow 0.15s; }
  .kol-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
  .drag-over { border: 2px dashed #FF6B00 !important; }

  /* Drawer */
  #drawer {
    position: fixed; right: 0; top: 0; height: 100vh; width: 500px;
    transform: translateX(100%); transition: transform 0.25s cubic-bezier(.4,0,.2,1);
    z-index: 50; border-left: 1px solid #1f2937;
    display: flex; flex-direction: column;
  }
  #drawer.open { transform: translateX(0); }
  #backdrop {
    position: fixed; inset: 0; background: rgba(0,0,0,0.55);
    z-index: 40; display: none;
  }
  #backdrop.show { display: block; }

  /* Tab */
  .tab-btn { border-bottom: 2px solid transparent; color: #9ca3af; transition: color 0.15s, border-color 0.15s; }
  .tab-btn.active { border-bottom-color: #FF6B00; color: white; }
  .tab-btn:hover:not(.active) { color: #e5e7eb; }

  /* Line clamp */
  .line-clamp-2 { display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
  .line-clamp-3 { display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }

  /* Scrollbar */
  ::-webkit-scrollbar { width:5px; height:5px; }
  ::-webkit-scrollbar-track { background:#0f172a; }
  ::-webkit-scrollbar-thumb { background:#374151; border-radius:3px; }
  ::-webkit-scrollbar-thumb:hover { background:#4b5563; }

  /* Toast */
  #toast {
    position:fixed; bottom:24px; left:50%; transform:translateX(-50%) translateY(80px);
    background:#1f2937; border:1px solid #374151; color:#d1fae5;
    padding:10px 20px; border-radius:8px; font-size:13px;
    transition:transform 0.3s; z-index:100;
  }
  #toast.show { transform:translateX(-50%) translateY(0); }
</style>
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen">

<!-- Backdrop -->
<div id="backdrop" onclick="closeDrawer()"></div>
<!-- Toast -->
<div id="toast"></div>

<!-- Top bar -->
<div class="border-b border-gray-800 bg-gray-900 px-6 py-3 flex items-center justify-between fixed top-0 left-0 right-0 z-30">
  <div class="flex items-center gap-3">
    <div class="w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-sm" style="background:#FF6B00">M</div>
    <span class="font-semibold text-white">MoonX KOL CRM</span>
    <span class="text-gray-600 text-sm">/</span>
    <span class="text-gray-400 text-sm">外联管理</span>
  </div>
  <input id="searchBox" type="text" placeholder="搜索 KOL 名称、邮箱..."
    class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 w-64 focus:outline-none focus:border-orange-500"
    oninput="debounceSearch(this.value)">
</div>

<!-- Layout -->
<div class="flex pt-14 h-screen overflow-hidden">

  <!-- Sidebar -->
  <aside class="w-52 bg-gray-900 border-r border-gray-800 flex-shrink-0 fixed left-0 top-14 bottom-0 overflow-y-auto">
    <div class="px-3 pt-4">

      <div class="text-xs text-gray-500 uppercase tracking-wider px-3 mb-2">工作区</div>
      <button onclick="showView('table')" id="nav-table"
        class="nav-item w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 bg-gray-800 text-white hover:bg-gray-700 mb-1">
        <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M3 14h18M10 6h11M10 18h11M3 6h.01M3 18h.01"/></svg>
        KOL 列表
      </button>
      <button onclick="showView('kanban')" id="nav-kanban"
        class="nav-item w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 text-gray-400 hover:bg-gray-800 mb-1">
        <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 0v10"/></svg>
        外联看板
      </button>
      <button onclick="showView('inbox')" id="nav-inbox"
        class="nav-item w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 text-gray-400 hover:bg-gray-800 mb-1">
        <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"/></svg>
        回复收件箱
      </button>
      <button onclick="showView('media')" id="nav-media"
        class="nav-item w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 text-gray-400 hover:bg-gray-800 mb-1">
        <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z"/></svg>
        媒体库
      </button>

      <div class="text-xs text-gray-500 uppercase tracking-wider px-3 mb-2 mt-5">Tier 筛选</div>
      <div id="tierFilters"></div>

      <div class="mt-5 px-3">
        <div class="text-xs text-gray-500 uppercase tracking-wider mb-2">状态统计</div>
        <div id="statusStats" class="space-y-0.5"></div>
      </div>
    </div>
  </aside>

  <!-- Main content -->
  <main class="flex-1 ml-52 overflow-auto">

    <!-- TABLE VIEW -->
    <div id="view-table">
      <div class="sticky top-0 z-10 bg-gray-950 px-4 py-3 border-b border-gray-800 flex items-center gap-3">
        <span id="totalCount" class="text-sm text-gray-400"></span>
        <div class="flex-1"></div>
        <button onclick="exportCSV()"
          class="px-3 py-1.5 text-xs text-gray-400 border border-gray-700 rounded-lg hover:border-gray-500 hover:text-gray-200 transition-colors">
          导出 CSV
        </button>
      </div>
      <table class="w-full text-sm">
        <thead class="sticky top-12 bg-gray-900 border-b border-gray-800 z-10">
          <tr class="text-left text-gray-500 text-xs uppercase tracking-wider">
            <th class="px-4 py-3 font-medium">名称</th>
            <th class="px-4 py-3 font-medium">订阅数</th>
            <th class="px-4 py-3 font-medium">Tier</th>
            <th class="px-4 py-3 font-medium">邮箱</th>
            <th class="px-4 py-3 font-medium">Twitter</th>
            <th class="px-4 py-3 font-medium">地区</th>
            <th class="px-4 py-3 font-medium">收录日期</th>
            <th class="px-4 py-3 font-medium">状态</th>
          </tr>
        </thead>
        <tbody id="kolTableBody" class="divide-y divide-gray-800/40"></tbody>
      </table>
      <div id="pagination" class="p-4 flex items-center justify-between border-t border-gray-800 text-sm"></div>
    </div>

    <!-- KANBAN VIEW -->
    <div id="view-kanban" class="hidden h-full overflow-x-auto p-4">
      <div id="kanbanBoard" class="flex gap-3 h-full" style="min-width: max-content;"></div>
    </div>

    <!-- INBOX VIEW -->
    <div id="view-inbox" class="hidden flex h-full">
      <div class="w-40 border-r border-gray-800 p-3 flex-shrink-0 overflow-y-auto">
        <div class="text-xs text-gray-500 uppercase tracking-wider mb-3 px-2">意向筛选</div>
        <div id="intentFilters" class="space-y-1"></div>
      </div>
      <div class="flex-1 overflow-y-auto">
        <div id="inboxList" class="p-4 space-y-2 max-w-3xl"></div>
      </div>
    </div>

    <!-- MEDIA VIEW -->
    <div id="view-media" class="hidden p-5">
      <div id="mediaGrid"></div>
    </div>

  </main>
</div>

<!-- KOL Detail Drawer -->
<div id="drawer" class="bg-gray-900">
  <!-- Header -->
  <div class="flex-shrink-0 border-b border-gray-800 px-5 pt-5 pb-0">
    <div class="flex items-start justify-between mb-3">
      <div class="flex-1 min-w-0">
        <h2 id="drawerName" class="text-base font-semibold text-white truncate"></h2>
        <div id="drawerMeta" class="text-xs text-gray-400 mt-0.5"></div>
      </div>
      <button onclick="closeDrawer()"
        class="text-gray-500 hover:text-white ml-3 mt-0.5 flex-shrink-0 p-1 rounded hover:bg-gray-800 transition-colors">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
      </button>
    </div>
    <div class="flex gap-0">
      <button onclick="switchTab('overview')" id="tab-overview" class="tab-btn active px-4 py-2.5 text-sm font-medium">概览</button>
      <button onclick="switchTab('contacts')" id="tab-contacts" class="tab-btn px-4 py-2.5 text-sm font-medium">联系记录</button>
      <button onclick="switchTab('replies')" id="tab-replies" class="tab-btn px-4 py-2.5 text-sm font-medium">回复</button>
      <button onclick="switchTab('negotiations')" id="tab-negotiations" class="tab-btn px-4 py-2.5 text-sm font-medium">谈判</button>
    </div>
  </div>
  <!-- Tab contents -->
  <div class="flex-1 overflow-y-auto">
    <div id="tab-content-overview" class="p-5"></div>
    <div id="tab-content-contacts" class="p-5 hidden"></div>
    <div id="tab-content-replies" class="p-5 hidden"></div>
    <div id="tab-content-negotiations" class="p-5 hidden"></div>
  </div>
</div>

<script>
const KOL_STATUSES = __STATUSES__;
let currentPage = 1, currentStatus = '', currentTier = '', currentSearch = '';
let currentInboxIntent = '';
let drawerKolId = null, drawerData = null, currentDrawerTab = 'overview';
let searchTimer = null, dragId = null, dragFrom = null;
let toastTimer = null;

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function fmtSubs(n) {
  if (!n) return '—';
  if (n >= 1000000) return (n/1000000).toFixed(1) + 'M';
  if (n >= 1000) return Math.round(n/1000) + 'K';
  return String(n);
}
function fmtDate(s) {
  if (!s) return '—';
  return String(s).substring(0, 10);
}
function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2000);
}

// ── API ─────────────────────────────────────────────

async function loadKols() {
  const p = new URLSearchParams();
  if (currentStatus) p.set('status', currentStatus);
  if (currentTier)   p.set('tier', currentTier);
  if (currentSearch) p.set('q', currentSearch);
  p.set('page', currentPage);
  const data = await fetch('/api/kols?' + p).then(r => r.json());
  renderTable(data);
  renderStats(data.stats);
  renderTierFilters(data.tiers);
}

async function loadKanban() {
  const data = await fetch('/api/kols?limit=500').then(r => r.json());
  renderKanban(data.rows);
}

async function loadInbox() {
  const p = new URLSearchParams();
  if (currentInboxIntent) p.set('intent', currentInboxIntent);
  const data = await fetch('/api/inbox?' + p).then(r => r.json());
  renderInbox(data);
}

async function loadMedia() {
  const data = await fetch('/api/media').then(r => r.json());
  renderMedia(data.rows);
}

async function openDrawer(id, tab) {
  drawerKolId = id;
  const data = await fetch('/api/kols/' + id).then(r => r.json());
  drawerData = data;
  document.getElementById('drawerName').textContent = data.kol.name || '';
  document.getElementById('drawerMeta').textContent =
    [data.kol.platform, data.kol.tier ? 'Tier ' + data.kol.tier : '', fmtSubs(data.kol.subscribers)]
      .filter(Boolean).join(' · ');
  switchTab(tab || 'overview');
  document.getElementById('drawer').classList.add('open');
  document.getElementById('backdrop').classList.add('show');
}

function closeDrawer() {
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('backdrop').classList.remove('show');
  drawerKolId = null; drawerData = null;
}

function switchTab(tab) {
  currentDrawerTab = tab;
  ['overview','contacts','replies','negotiations'].forEach(t => {
    document.getElementById('tab-' + t).classList.toggle('active', t === tab);
    document.getElementById('tab-content-' + t).classList.toggle('hidden', t !== tab);
  });
  if (drawerData) renderDrawerTab(tab);
}

// ── DRAWER RENDER ────────────────────────────────────

function renderDrawerTab(tab) {
  if (!drawerData) return;
  const { kol, contacts, replies, negotiations } = drawerData;

  if (tab === 'overview') {
    document.getElementById('tab-content-overview').innerHTML = `
      <div class="space-y-5">
        <div>
          <label class="text-xs text-gray-500 uppercase tracking-wider block mb-1.5">外联状态</label>
          <select onchange="updateStatus(${kol.id}, this.value)"
            class="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-orange-500">
            ${KOL_STATUSES.map(s => `<option value="${s}" ${s === kol.status ? 'selected' : ''}>${s}</option>`).join('')}
          </select>
        </div>

        <div class="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
          <div>
            <div class="text-xs text-gray-500 mb-1">邮箱</div>
            <div class="text-gray-200 text-xs break-all">${kol.email ? esc(kol.email) : '<span class="text-gray-600">未填写</span>'}</div>
          </div>
          <div>
            <div class="text-xs text-gray-500 mb-1">Twitter</div>
            <div>${kol.twitter ? `<a href="${esc(kol.twitter)}" target="_blank" class="text-blue-400 hover:underline text-xs">@链接</a>` : '<span class="text-gray-600 text-xs">—</span>'}</div>
          </div>
          <div>
            <div class="text-xs text-gray-500 mb-1">地区</div>
            <div class="text-gray-300 text-xs">${esc(kol.country) || '—'}</div>
          </div>
          <div>
            <div class="text-xs text-gray-500 mb-1">TG Handle</div>
            <div class="text-gray-300 text-xs">${esc(kol.tg_handle) || '—'}</div>
          </div>
          <div>
            <div class="text-xs text-gray-500 mb-1">收录日期</div>
            <div class="text-gray-300 text-xs">${fmtDate(kol.collect_date)}</div>
          </div>
          <div>
            <div class="text-xs text-gray-500 mb-1">频道</div>
            <div class="text-xs">${kol.channel_url ? `<a href="${esc(kol.channel_url)}" target="_blank" class="text-blue-400 hover:underline">打开</a>` : '<span class="text-gray-600">—</span>'}</div>
          </div>
        </div>

        <div class="pt-1 border-t border-gray-800">
          <label class="text-xs text-gray-500 uppercase tracking-wider block mb-2">备注</label>
          <textarea id="notesInput" rows="4"
            class="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 resize-none focus:outline-none focus:border-orange-500"
            placeholder="添加备注...">${esc(kol.notes || '')}</textarea>
          <button onclick="saveNotes(${kol.id})"
            class="mt-2 px-4 py-1.5 text-xs text-white rounded-lg hover:opacity-90 transition-opacity" style="background:#FF6B00">
            保存备注
          </button>
        </div>
      </div>`;
  }

  if (tab === 'contacts') {
    const el = document.getElementById('tab-content-contacts');
    if (!contacts.length) {
      el.innerHTML = '<p class="text-gray-500 text-sm">暂无联系记录</p>';
    } else {
      el.innerHTML = `<div class="text-xs text-gray-500 mb-3">共 ${contacts.length} 条联系记录</div>` +
        contacts.map(c => `
          <div class="bg-gray-800 rounded-lg p-3 mb-2.5 border border-gray-700/60">
            <div class="flex items-center justify-between mb-1.5">
              <span class="text-xs text-gray-400">${fmtDate(c.sent_at)}</span>
              <span class="text-xs px-2 py-0.5 rounded-full bg-green-900/40 text-green-300">${esc(c.status)}</span>
            </div>
            <div class="text-sm text-gray-100">${esc(c.subject) || '<span class="text-gray-500">(无主题)</span>'}</div>
            ${c.template ? `<div class="text-xs text-gray-500 mt-1">模板: ${esc(c.template)}</div>` : ''}
          </div>`).join('');
    }
  }

  if (tab === 'replies') {
    const el = document.getElementById('tab-content-replies');
    const intentLabel = { '感兴趣':'感兴趣','待确认':'待确认','拒绝':'拒绝','positive':'正面','negative':'拒绝','interested':'感兴趣','unknown':'未分类' };
    if (!replies.length) {
      el.innerHTML = '<p class="text-gray-500 text-sm">暂无回复记录</p>';
    } else {
      el.innerHTML = `<div class="text-xs text-gray-500 mb-3">共 ${replies.length} 条回复</div>` +
        replies.map(r => `
          <div class="bg-gray-800 rounded-lg p-3 mb-2.5 border border-gray-700/60">
            <div class="flex items-center justify-between mb-1.5">
              <span class="text-xs text-gray-400">${fmtDate(r.detected_at)}</span>
              ${r.intent ? `<span class="text-xs px-2 py-0.5 rounded-full intent-${r.intent}">${intentLabel[r.intent] || esc(r.intent)}</span>` : ''}
            </div>
            <div class="text-sm text-gray-100 mb-1">${esc(r.subject) || '<span class="text-gray-500">(无主题)</span>'}</div>
            ${r.body_snippet ? `<div class="text-xs text-gray-400 line-clamp-3 mt-1">${esc(r.body_snippet)}</div>` : ''}
          </div>`).join('');
    }
  }

  if (tab === 'negotiations') {
    const stageLabel = { initial:'初步接触', negotiating:'报价中', agreed:'口头确认' };
    let html = `
      <div class="flex items-center justify-between mb-4">
        <span class="text-sm font-medium text-white">谈判记录</span>
        <button onclick="toggleNegForm()"
          class="text-xs px-3 py-1.5 text-white rounded-lg hover:opacity-90" style="background:#FF6B00">+ 添加记录</button>
      </div>

      <div id="negForm" class="hidden bg-gray-800 rounded-xl p-4 mb-4 border border-gray-700">
        <div class="grid grid-cols-2 gap-3 mb-3">
          <div>
            <label class="text-xs text-gray-500 mb-1 block">谈判阶段</label>
            <select id="negStage" class="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-orange-500">
              <option value="initial">初步接触</option>
              <option value="negotiating">报价中</option>
              <option value="agreed">口头确认</option>
            </select>
          </div>
          <div>
            <label class="text-xs text-gray-500 mb-1 block">报价 (USD)</label>
            <input id="negPrice" type="number" placeholder="500"
              class="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-orange-500">
          </div>
          <div>
            <label class="text-xs text-gray-500 mb-1 block">内容形式</label>
            <input id="negType" type="text" placeholder="推文 + 视频"
              class="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-orange-500">
          </div>
          <div>
            <label class="text-xs text-gray-500 mb-1 block">交付物</label>
            <input id="negDelivery" type="text" placeholder="1条推文，1个视频"
              class="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-orange-500">
          </div>
        </div>
        <div class="mb-3">
          <label class="text-xs text-gray-500 mb-1 block">备注</label>
          <textarea id="negNotes" rows="2" placeholder="补充说明..."
            class="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm text-gray-200 resize-none focus:outline-none focus:border-orange-500"></textarea>
        </div>
        <div class="flex gap-2">
          <button onclick="submitNegotiation(${kol.id})"
            class="px-4 py-1.5 text-xs text-white rounded-lg hover:opacity-90" style="background:#FF6B00">提交</button>
          <button onclick="toggleNegForm()"
            class="px-4 py-1.5 text-xs text-gray-400 bg-gray-700 rounded-lg hover:bg-gray-600">取消</button>
        </div>
      </div>`;

    if (!negotiations.length) {
      html += '<p class="text-gray-500 text-sm">暂无谈判记录，点击上方「添加记录」开始</p>';
    } else {
      html += negotiations.map(n => `
        <div class="bg-gray-800 rounded-lg p-3.5 mb-2.5 border border-gray-700/60">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs px-2 py-0.5 rounded-full bg-purple-900/40 text-purple-300">
              ${stageLabel[n.stage] || esc(n.stage)}
            </span>
            <span class="text-xs text-gray-500">${fmtDate(n.created_at)}</span>
          </div>
          ${n.price_usd != null ? `<div class="text-lg font-bold text-orange-400 mb-2">$${Number(n.price_usd).toLocaleString()}</div>` : ''}
          <div class="space-y-1 text-xs text-gray-400">
            ${n.content_type ? `<div>形式: <span class="text-gray-200">${esc(n.content_type)}</span></div>` : ''}
            ${n.deliverables ? `<div>交付: <span class="text-gray-200">${esc(n.deliverables)}</span></div>` : ''}
            ${n.notes ? `<div class="pt-2 mt-2 border-t border-gray-700 text-gray-400">${esc(n.notes)}</div>` : ''}
          </div>
        </div>`).join('');
    }
    document.getElementById('tab-content-negotiations').innerHTML = html;
  }
}

// ── VIEWS RENDER ─────────────────────────────────────

function renderTable(data) {
  document.getElementById('totalCount').textContent = `共 ${data.total} 位 KOL`;
  document.getElementById('kolTableBody').innerHTML = data.rows.map(r => `
    <tr class="hover:bg-gray-800/50 transition-colors cursor-pointer" onclick="openDrawer(${r.id})">
      <td class="px-4 py-3">
        <div class="font-medium text-white text-sm">${esc(r.name)}</div>
        <div class="text-xs text-gray-500 mt-0.5">${esc(r.platform)}</div>
      </td>
      <td class="px-4 py-3 text-gray-300 font-mono text-sm">${fmtSubs(r.subscribers)}</td>
      <td class="px-4 py-3">
        <span class="tier-${(r.tier||'').charAt(0)} px-2 py-0.5 rounded text-xs font-medium">${esc(r.tier) || '—'}</span>
      </td>
      <td class="px-4 py-3 text-gray-400 text-xs">${r.email ? esc(r.email) : '<span class="text-gray-600">无邮箱</span>'}</td>
      <td class="px-4 py-3 text-xs">
        ${r.twitter ? `<a href="${esc(r.twitter)}" target="_blank" class="text-blue-400 hover:underline" onclick="event.stopPropagation()">@链接</a>` : '<span class="text-gray-600">—</span>'}
      </td>
      <td class="px-4 py-3 text-gray-400 text-xs">${esc(r.country) || '—'}</td>
      <td class="px-4 py-3 text-gray-500 text-xs">${fmtDate(r.collect_date)}</td>
      <td class="px-4 py-3">
        <span class="badge-${r.status} px-2 py-0.5 rounded-full text-xs font-medium">${esc(r.status)}</span>
      </td>
    </tr>`).join('');

  const pages = Math.ceil(data.total / data.limit) || 1;
  document.getElementById('pagination').innerHTML = `
    <span class="text-gray-500 text-sm">第 ${data.page} / ${pages} 页</span>
    <div class="flex gap-2">
      <button onclick="changePage(${data.page - 1})" ${data.page <= 1 ? 'disabled' : ''}
        class="px-3 py-1.5 text-xs border border-gray-700 rounded-lg disabled:opacity-30 hover:border-gray-500 transition-colors">上一页</button>
      <button onclick="changePage(${data.page + 1})" ${data.page >= pages ? 'disabled' : ''}
        class="px-3 py-1.5 text-xs border border-gray-700 rounded-lg disabled:opacity-30 hover:border-gray-500 transition-colors">下一页</button>
    </div>`;
}

function renderStats(stats) {
  const colors = {
    '待发送':'text-blue-400','已发送':'text-green-400','已回复':'text-orange-400',
    '谈判中':'text-purple-400','已签约':'text-indigo-400','执行中':'text-yellow-400','完成':'text-teal-400'
  };
  document.getElementById('statusStats').innerHTML = KOL_STATUSES.map(k => `
    <button onclick="filterStatus('${k}')"
      class="w-full flex items-center justify-between px-2 py-1 rounded text-xs hover:bg-gray-800 transition-colors ${currentStatus === k ? 'bg-gray-800' : ''}">
      <span class="${colors[k] || 'text-gray-400'}">${k}</span>
      <span class="text-gray-500">${stats[k] || 0}</span>
    </button>`).join('');
}

function renderTierFilters(tiers) {
  const el = document.getElementById('tierFilters');
  if (el.innerHTML) return;
  el.innerHTML = ['', ...tiers].map(t => `
    <button onclick="filterTier('${t}')"
      class="w-full text-left px-3 py-1.5 rounded-lg text-xs text-gray-400 hover:bg-gray-800 mb-0.5 transition-colors">
      ${t || '全部'}
    </button>`).join('');
}

function renderKanban(rows) {
  const colColors = {
    '待发送':'text-blue-400','已发送':'text-green-400','已回复':'text-orange-400',
    '谈判中':'text-purple-400','已签约':'text-indigo-400','执行中':'text-yellow-400','完成':'text-teal-400'
  };
  const countBg = {
    '待发送':'bg-blue-900/40 text-blue-300','已发送':'bg-green-900/40 text-green-300',
    '已回复':'bg-orange-900/40 text-orange-300','谈判中':'bg-purple-900/40 text-purple-300',
    '已签约':'bg-indigo-900/40 text-indigo-300','执行中':'bg-yellow-900/40 text-yellow-300',
    '完成':'bg-teal-900/40 text-teal-300'
  };
  const cols = {};
  KOL_STATUSES.forEach(s => cols[s] = []);
  rows.forEach(r => { if (cols[r.status] !== undefined) cols[r.status].push(r); });

  document.getElementById('kanbanBoard').innerHTML = KOL_STATUSES.map(status => `
    <div class="kanban-col bg-gray-900 rounded-xl p-3 border border-gray-800 flex flex-col"
      ondragover="event.preventDefault(); this.classList.add('drag-over')"
      ondragleave="this.classList.remove('drag-over')"
      ondrop="onDrop(event,'${status}')">
      <div class="flex items-center justify-between mb-3 flex-shrink-0">
        <span class="text-sm font-semibold ${colColors[status]}">${status}</span>
        <span class="text-xs ${countBg[status]} px-2 py-0.5 rounded-full">${cols[status].length}</span>
      </div>
      <div class="space-y-2 overflow-y-auto flex-1">
        ${cols[status].slice(0, 50).map(r => `
          <div class="kol-card bg-gray-800 rounded-lg p-3 border border-gray-700"
            draggable="true"
            ondragstart="onDragStart(event,${r.id},'${r.status}')"
            onclick="openDrawer(${r.id})">
            <div class="font-medium text-sm text-white mb-1.5 truncate">${esc(r.name)}</div>
            <div class="flex items-center justify-between">
              <span class="tier-${(r.tier||'').charAt(0)} px-1.5 py-0.5 rounded text-xs">${esc(r.tier) || '—'}</span>
              <span class="text-xs text-gray-500">${fmtSubs(r.subscribers)}</span>
            </div>
            ${r.email ? `<div class="text-xs text-gray-500 mt-1 truncate">${esc(r.email)}</div>` : ''}
          </div>`).join('')}
        ${cols[status].length > 50 ? `<div class="text-xs text-gray-500 text-center py-1">还有 ${cols[status].length - 50} 条</div>` : ''}
      </div>
    </div>`).join('');
}

function renderInbox(data) {
  const intentLabel = {
    '感兴趣':'感兴趣','待确认':'待确认','拒绝':'拒绝',
    'positive':'正面','negative':'拒绝','interested':'感兴趣','unknown':'未分类'
  };
  const total = data.rows.length;

  document.getElementById('intentFilters').innerHTML =
    `<button onclick="filterIntent('')"
      class="w-full text-left px-2 py-1.5 rounded text-xs hover:bg-gray-800 transition-colors mb-1 ${!currentInboxIntent ? 'bg-gray-800 text-white' : 'text-gray-400'}">
      全部 <span class="text-gray-500 ml-1">${total}</span>
    </button>` +
    Object.entries(data.intent_counts).map(([intent, count]) => `
      <button onclick="filterIntent('${intent}')"
        class="w-full text-left px-2 py-1.5 rounded text-xs hover:bg-gray-800 transition-colors mb-1 ${currentInboxIntent === intent ? 'bg-gray-800 text-white' : 'text-gray-400'}">
        <span class="intent-${intent} px-1.5 py-0.5 rounded text-xs">${intentLabel[intent] || intent}</span>
        <span class="text-gray-500 ml-1">${count}</span>
      </button>`).join('');

  if (!data.rows.length) {
    document.getElementById('inboxList').innerHTML =
      '<p class="text-gray-500 text-sm">暂无回复记录</p>';
    return;
  }
  document.getElementById('inboxList').innerHTML = data.rows.map(r => `
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-orange-500/40 transition-colors cursor-pointer"
      onclick="openDrawer(${r.kol_id}, 'replies')">
      <div class="flex items-start justify-between gap-3 mb-2">
        <div class="min-w-0">
          <span class="font-medium text-white text-sm">${esc(r.kol_name)}</span>
          <span class="text-gray-500 text-xs ml-2">${esc(r.platform)}</span>
          ${r.tier ? `<span class="tier-${(r.tier||'').charAt(0)} px-1.5 py-0.5 rounded text-xs ml-1">${esc(r.tier)}</span>` : ''}
        </div>
        <div class="flex items-center gap-2 flex-shrink-0">
          ${r.intent ? `<span class="text-xs px-2 py-0.5 rounded-full intent-${r.intent}">${intentLabel[r.intent] || esc(r.intent)}</span>` : ''}
          <span class="text-xs text-gray-500">${fmtDate(r.detected_at)}</span>
        </div>
      </div>
      <div class="text-sm text-gray-300 mb-1">${esc(r.subject) || '<span class="text-gray-500">(无主题)</span>'}</div>
      ${r.body_snippet ? `<div class="text-xs text-gray-400 line-clamp-2">${esc(r.body_snippet)}</div>` : ''}
    </div>`).join('');
}

function renderMedia(rows) {
  const pColors = {'A':'text-red-400 bg-red-900/30','B':'text-orange-400 bg-orange-900/30','C':'text-blue-400 bg-blue-900/30'};
  document.getElementById('mediaGrid').innerHTML =
    `<div class="grid grid-cols-4 gap-3">` +
    rows.map(r => `
      <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-orange-500/50 transition-colors">
        <div class="flex items-start justify-between mb-2">
          <div class="font-medium text-white text-sm flex-1 min-w-0 mr-2">${esc(r.name)}</div>
          <span class="${pColors[r.priority] || 'text-gray-500 bg-gray-800'} text-xs px-2 py-0.5 rounded font-medium flex-shrink-0">${esc(r.priority)}</span>
        </div>
        <div class="text-xs text-gray-500 mb-2">${esc(r.type)}</div>
        ${r.website ? `<a href="${esc(r.website)}" target="_blank" class="text-xs text-blue-400 hover:underline block mb-1 truncate">${esc(r.website)}</a>` : ''}
        ${r.email ? `<div class="text-xs text-gray-400 truncate">${esc(r.email)}</div>` : ''}
        <div class="mt-2">
          <span class="text-xs ${(r.status||'').includes('已') ? 'text-green-400' : 'text-gray-500'}">${esc(r.status) || '待联系'}</span>
        </div>
      </div>`).join('') +
    `</div>`;
}

// ── ACTIONS ──────────────────────────────────────────

async function updateStatus(kolId, status) {
  await fetch('/api/kols/' + kolId + '/status', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({status})
  });
  toast('状态已更新');
  if (drawerData) drawerData.kol.status = status;
  const kanban = document.getElementById('view-kanban');
  if (!kanban.classList.contains('hidden')) loadKanban();
  else loadKols();
}

async function saveNotes(kolId) {
  const notes = document.getElementById('notesInput').value;
  await fetch('/api/kols/' + kolId + '/notes', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({notes})
  });
  if (drawerData) drawerData.kol.notes = notes;
  toast('备注已保存');
}

function toggleNegForm() {
  document.getElementById('negForm').classList.toggle('hidden');
}

async function submitNegotiation(kolId) {
  const data = {
    stage:       document.getElementById('negStage').value,
    price_usd:   parseFloat(document.getElementById('negPrice').value) || null,
    content_type: document.getElementById('negType').value.trim(),
    deliverables: document.getElementById('negDelivery').value.trim(),
    notes:        document.getElementById('negNotes').value.trim(),
  };
  await fetch('/api/kols/' + kolId + '/negotiations', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(data)
  });
  // Reload drawer data
  drawerData = await fetch('/api/kols/' + kolId).then(r => r.json());
  renderDrawerTab('negotiations');
  toast('谈判记录已添加');
}

// ── DRAG & DROP ───────────────────────────────────────

function onDragStart(e, id, status) { dragId = id; dragFrom = status; }
async function onDrop(e, targetStatus) {
  e.currentTarget.classList.remove('drag-over');
  if (!dragId || dragFrom === targetStatus) return;
  await fetch('/api/kols/' + dragId + '/status', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({status: targetStatus})
  });
  dragId = null; dragFrom = null;
  loadKanban();
}

// ── FILTERS & NAV ─────────────────────────────────────

function filterStatus(s) { currentStatus = (currentStatus === s) ? '' : s; currentPage = 1; loadKols(); }
function filterTier(t)   { currentTier = t; currentPage = 1; loadKols(); }
function changePage(p)   { currentPage = p; loadKols(); }
function filterIntent(i) { currentInboxIntent = i; loadInbox(); }
function debounceSearch(v) {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { currentSearch = v; currentPage = 1; loadKols(); }, 300);
}
function exportCSV() { window.open('/api/kols/export', '_blank'); }

function showView(v) {
  ['table','kanban','inbox','media'].forEach(n => {
    document.getElementById('view-' + n).classList.toggle('hidden', n !== v);
    const nav = document.getElementById('nav-' + n);
    if (n === v) { nav.classList.add('bg-gray-800','text-white'); nav.classList.remove('text-gray-400'); }
    else         { nav.classList.remove('bg-gray-800','text-white'); nav.classList.add('text-gray-400'); }
  });
  if (v === 'kanban') loadKanban();
  if (v === 'inbox')  loadInbox();
  if (v === 'media')  loadMedia();
}

loadKols();
</script>
</body>
</html>""".replace('__STATUSES__', statuses_js)


# ─── HTTP HANDLER ────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        path   = parsed.path

        if path == "/":
            return self.send_html(build_html())

        if path == "/api/kols":
            return self.send_json(api_kols(params))

        if path == "/api/kols/export":
            csv_data = api_export_csv().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="kol_export.csv"')
            self.send_header("Content-Length", len(csv_data))
            self.end_headers()
            self.wfile.write(csv_data)
            return

        m = re.match(r"^/api/kols/(\d+)$", path)
        if m:
            detail = api_kol_detail(int(m.group(1)))
            return self.send_json(detail if detail else {}, 200 if detail else 404)

        if path == "/api/inbox":
            return self.send_json(api_inbox(params))

        if path == "/api/media":
            return self.send_json(api_media(params))

        self.send_response(404); self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        data = self.read_body()

        m = re.match(r"^/api/kols/(\d+)/status$", path)
        if m:
            return self.send_json(do_update_status(int(m.group(1)), data))

        m = re.match(r"^/api/kols/(\d+)/notes$", path)
        if m:
            return self.send_json(do_update_notes(int(m.group(1)), data))

        m = re.match(r"^/api/kols/(\d+)/negotiations$", path)
        if m:
            return self.send_json(do_add_negotiation(int(m.group(1)), data))

        self.send_response(404); self.end_headers()

    def log_message(self, *a):
        pass


# ─── MAIN ────────────────────────────────────────────

if __name__ == "__main__":
    port   = 9999
    server = HTTPServer(("localhost", port), Handler)
    url    = f"http://localhost:{port}"
    print(f"MoonX KOL CRM 已启动: {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已关闭")
