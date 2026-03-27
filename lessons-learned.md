# Lessons Learned — agent-twitter

每次踩坑后追加，格式：`日期 | 问题 | 解决方案`

---

## Twitter API

- **Bearer Token + OAuth 1.0a 双认证** — `get_user()` 查询用 Bearer Token，DM 发送用 OAuth 1.0a。两者都要配置到 `tweepy.Client`，缺一不可。
- **load_dotenv 缓存问题** — 必须加 `override=True`，否则读取旧缓存 env var 导致 401。
- **DM 防封号策略** — 每天上限 10 条，间隔 90~200 秒随机，5 个模版随机轮换。
- **大 KOL 普遍关闭 DM** — Ansem、Kaleo、DegenSpartan 等 DM 关闭，只能邮件或中间人联系。
- **虚构 KOL handle** — AI 生成的 KOL 名单中存在不存在的账号，需人工核实后再使用。

## 开发环境

- **`code` 命令指向 Cursor** — 修复：`sudo rm /usr/local/bin/code` + 重新链接 VSCode。用 `open -a "Visual Studio Code"` 更可靠。
- **VSCode 语言设置** — 写 `locale.json` 后必须 `Cmd+Q` 完全退出再重开才生效，关窗口不够。
- **VSCode Claude Code 扩展报错 exit code 1** — 原因：终端 session 有 `CLAUDECODE` 环境变量，嵌套启动冲突。临时方案：`env -u CLAUDECODE open -a "Visual Studio Code"`。

## Excel 操作

- **Mac 没有 Office** — 用 `openpyxl`（xlsx 读写）+ `python-docx`（Word）替代。
- **状态列更新** — 用 `openpyxl` 按行遍历，匹配 handle/email 后写入"已发送"。

## 工作区

- **多项目工作区** — `/Users/coco/workspace.code-workspace`，同时包含 agent-twitter + predict-gmgn。
