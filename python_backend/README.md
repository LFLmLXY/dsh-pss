# dsh-pss · Python 侧

PSS 的「学习」和「改写意图」两件事放在 Python 里做，因为 LoRA 生态在这边。主进程（TypeScript）只负责状态、检索、可视化和调度。

```
dsh-pss (TS)                                   python_backend (FastAPI)
────────────                                   ────────────────────────
session/disposed ──POST /compile ───────────►  后台 LoRA 训练
                                               loras/{sessionId}/
retrieve()  ──POST /retrieve-intent ─────────►  加载 adapter → 改写检索意图
            ◄──────── {intentQuery, keywords}  （没 adapter / 没装 torch 时降级启发式）
```

## 三步跑起来

```bash
# 1) 只跑服务，验证契约（不需要 GPU，不需要 torch）
cd python_backend
python -m venv .venv && . .venv/Scripts/activate      # Windows
# source .venv/bin/activate                            # macOS / Linux
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8077
```

```bash
# 2) 冒烟：健康检查 + 提一个编译任务 + 查进度
curl -s http://127.0.0.1:8077/health | jq
curl -s -X POST http://127.0.0.1:8077/compile \
  -H 'content-type: application/json' \
  -d '{"sessionId":"demo","history":[{"role":"user","content":"PSS 的存储为什么是 O(1)？"},{"role":"assistant","content":"因为它只保留一个定长参数向量，历史随编译被压掉。"}]}' | jq
curl -s http://127.0.0.1:8077/jobs/<jobId> | jq
```

```bash
# 3) 真训练（可选，会拉 torch）
pip install -r requirements-train.txt
PSS_DRY_RUN=0 PSS_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct uvicorn main:app --port 8077
```

## 端点

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET  | `/health` | 健康检查、是否 dry-run、已有 adapter 列表 |
| POST | `/compile` | 提交编译任务，立刻返回 `jobId`（`force: true` 可强制重训） |
| GET  | `/jobs/{jobId}` | 训练进度：`status` / `step` / `loss` / `adapterPath` |
| GET  | `/jobs` | 最近 50 个任务 |
| POST | `/retrieve-intent` | `{sessionId, query, keywords, centroid, turns}` → `{intentQuery, keywords, source}` |
| GET  | `/adapters` | 所有已就绪 adapter 的 manifest |

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `PSS_HOST` | `127.0.0.1` | 监听地址 |
| `PSS_PORT` | `8077` | 端口 |
| `PSS_LORA_DIR` | `./loras` | adapter 输出目录 |
| `PSS_BASE_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | 基座模型 |
| `PSS_MAX_STEPS` | `120` | 每个 adapter 的训练步数 |
| `PSS_MAX_SEQ_LEN` | `1024` | 训练序列长度 |
| `PSS_LORA_R` | `8` | LoRA rank |
| `PSS_DRY_RUN` | `auto` | `auto` = 缺 torch/transformers/peft 时自动模拟；`0` 强制真训；`1` 强制模拟 |

## 为什么要有 dry-run

PSS 的价值在「状态怎么流转」，不在「谁的 GPU 更快」。默认 dry-run 让任何人都能在 3 分钟内跑通完整链路——前端脑图会真的长出来，`/compile` 会真的产出 `loras/{sessionId}/adapter_manifest.json`，只是权重是占位的。等你要认真训的时候换 `PSS_DRY_RUN=0` 即可，接口契约一个字都不用改。

## 产出目录

```
loras/
├── _jobs/<jobId>.json                # 任务镜像，进程重启后还能查
└── <sessionId>/
    ├── corpus.txt                    # 训练语料（会话压缩成意图摘要）
    ├── adapter_manifest.json         # 插件侧只读这个判断 ready
    ├── adapter_model.safetensors     # dry-run 时是占位文件
    └── checkpoints/                  # 真训练的 checkpoint
```
