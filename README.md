# AI 模型监控 · AI Model Monitor

一个追踪主流 AI 厂商最新旗舰模型及其 benchmark 得分的静态网页，每日由 GitHub Actions 自动抓取更新。

A static web tool that tracks the latest flagship models from major AI vendors along with their benchmark scores. Data is refreshed daily by GitHub Actions.

**追踪厂商 / Vendors**: OpenAI · Anthropic · Google Gemini · DeepSeek · 智谱 GLM · 小米 MiMo · MiniMax

## 架构 Architecture

```
GitHub Actions (daily cron)
        │
        ▼
scripts/update_models.py
        │  fetch vendor blogs / model docs / leaderboards
        ▼
Xiaomi MiMo API (mimo-v2-pro, OpenAI-compatible)  ← structured extraction via function calling
        │
        ▼
data/models.json  ──git commit──▶  main branch  ──▶  Vercel 自动重部署
                                                          │
                                                          ▼
                                                  index.html (static)
```

- **前端**: 纯 vanilla HTML/CSS/JS，读取 `data/models.json` 渲染卡片
- **抓取**: Python 脚本 fetch 原始 HTML，通过 OpenAI SDK 调用小米 MiMo API (`mimo-v2-pro`) 做结构化抽取（function calling 强约束 JSON schema）
- **更新触发**: GitHub Actions 每日 02:00 UTC 运行，有变动则 commit 到 `main`

## 本地开发 Local development

```bash
# 静态预览 (任一方式)
python3 -m http.server 8000
open http://localhost:8000

# 运行抓取脚本 (需要 MIMO_API_KEY)
export MIMO_API_KEY=...          # 从 https://platform.xiaomimimo.com 获取
# 可选：切换模型或 base_url
# export MIMO_MODEL=mimo-v2-pro
# export MIMO_BASE_URL=https://api.xiaomimimo.com/v1
pip install -r scripts/requirements.txt
python scripts/update_models.py
```

## 部署 Deployment

1. Push 仓库到 GitHub
2. GitHub → Settings → Secrets and variables → Actions → 新建 `MIMO_API_KEY`
3. 在 [Vercel](https://vercel.com/) import 该仓库
   - Framework preset: **Other**
   - Build command: 留空
   - Output directory: `.` (仓库根目录)
4. 第一次手动触发 workflow：Actions → "Update AI models" → Run workflow
5. 之后每天 02:00 UTC 自动更新；有变动则 push `data/models.json`，Vercel 自动重部署

## 文件结构 Project structure

```
.
├── index.html              # 主页面
├── assets/
│   ├── style.css           # 样式 (暗/亮主题)
│   └── app.js              # 渲染 + 语言切换
├── data/
│   └── models.json         # 抓取产物 (自动更新)
├── scripts/
│   ├── sources.py          # 厂商 & 榜单 URL 配置
│   ├── extractor.py        # Claude 结构化抽取
│   ├── update_models.py    # 入口脚本
│   └── requirements.txt
├── .github/workflows/
│   └── update.yml          # 每日抓取工作流
├── vercel.json
└── README.md
```

## 添加新厂商 Adding a vendor

编辑 `scripts/sources.py`，在 `VENDORS` 列表追加一个 `Vendor(...)` 条目即可。前端无需改动——`data/models.json` 一更新卡片就会出现。

Edit `scripts/sources.py`, append a new `Vendor(...)` entry, and that's it — the frontend picks it up automatically from the next JSON refresh.

## 数据来源 Data sources

- **官方得分**: 厂商发布会 / 技术报告 / Blog / 文档页
- **第三方榜单**: [LMArena](https://lmarena.ai/leaderboard) · [LiveBench](https://livebench.ai/) · [Artificial Analysis](https://artificialanalysis.ai/models)

抓取器只采用页面上明确显示的数字，不会杜撰或估算分数。某家厂商抓取失败时，该卡片会保留上一次的数据并显示"更新失败"状态，其他厂商不受影响。
