#!/bin/bash
# KOL 收集每日状态检查（Headless 模式）
# 用法：bash scripts/daily_kol_check.sh
# 也可配置 launchd 定时运行

cd /Users/coco/agent-twitter

OUTPUT_FILE="/tmp/kol-status-$(date +%Y%m%d).txt"

claude --print \
  --allowedTools "Bash,Read,Grep,Glob" \
  "检查今日 KOL 收集状态，工作目录 /Users/coco/agent-twitter：
1. 列出 03_kol_media/MoonX_YouTube_KOL名单_*.xlsx，找最新文件，统计总行数、A/B/C/D分级数量、有邮箱数量
2. 读取 03_kol_media/logs/youtube_kol.log 最后50行，提取最近运行时间、收集数量、使用策略级别、有无报错
3. 列出最新的 Twitter KOL 名单文件并统计行数
4. 用中文输出简洁日报，格式：
📊 KOL日报 [日期]
YouTube: 今日新增X个 / 累计X个 / 有邮箱X个 / A级X B级X C级X D级X / 使用策略: XXX
Twitter KOL: 最新名单X条
状态: 正常 或 ⚠️异常（说明原因）" \
  > "$OUTPUT_FILE" 2>&1

cat "$OUTPUT_FILE"

# 可选：发送到 Lark（取消注释启用）
# LARK_URL=$(grep LARK_LEAD /Users/coco/agent-twitter/.env | cut -d= -f2)
# if [ -n "$LARK_URL" ]; then
#   CONTENT=$(cat "$OUTPUT_FILE")
#   python3 -c "
# import json, urllib.request
# payload = json.dumps({'msg_type':'text','content':{'text':'''$CONTENT'''}}).encode()
# req = urllib.request.Request('$LARK_URL', data=payload, headers={'Content-Type':'application/json'})
# urllib.request.urlopen(req, timeout=8)
# print('Lark 已发送')
# "
# fi
