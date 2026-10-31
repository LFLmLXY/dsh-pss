"""
dsh-pss · Python 训练后端
=========================

这一侧只干两件事，都是「离线、可中断、可降级」的：

1. ``POST /compile``         —— 接一段对话历史，异步训一个 LoRA adapter，落到 ``loras/{sessionId}/``。
2. ``POST /retrieve-intent`` —— 拿已经训好的 adapter，把一句模糊的用户提问改写成
   「参数化检索意图」（关键词 + 改写后的 query），再交给 dsh-pss 主进程去检索。

设计要点
--------
* **重依赖全部惰性 import**：没装 torch / peft 时服务照样能起，只是 ``dry_run`` 模式，
  用来跑通整条链路（前端、插件、HTTP 契约都能测）。真正训练才需要 ``requirements-train.txt``。
* **训练是后台任务**：``/compile`` 立刻返回 ``jobId``，进度用 ``GET /jobs/{id}`` 轮询。
* **adapter 目录即状态**：每个会话一个目录，里面 ``adapter_manifest.json`` 记录训练元数据，
  插件侧读这个文件就知道 adapter 是否 ready。
* **单进程够用**：这是本地工具，不是 SaaS。多 worker 会让后台任务和内存 LRU 失效，别开。

启动
----
    pip install -r requirements.txt          # 只跑服务/契约测试
    pip install -r requirements-train.txt    # 真正训练 LoRA
    PSS_DRY_RUN=1 uvicorn main:app --port 8077

环境变量
--------
PSS_HOST            监听地址，默认 127.0.0.1
PSS_PORT            端口，默认 8077
PSS_LORA_DIR        adapter 输出目录，默认 ./loras
PSS_BASE_MODEL      基座模型，默认 deepseek-ai/DeepSeek-V2-Lite 之外的任意 HF 因果 LM
PSS_MAX_STEPS       每个 adapter 的训练步数，默认 120
PSS_MAX_SEQ_LEN     训练序列长度，默认 1024
PSS_DRY_RUN         1 = 不碰 GPU，模拟训练（默认在缺少依赖时自动开启）
PSS_LORA_R          LoRA rank，默认 8
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

HOST = os.environ.get("PSS_HOST", "127.0.0.1")
PORT = int(os.environ.get("PSS_PORT", "8077"))
LORA_DIR = Path(os.environ.get("PSS_LORA_DIR", "./loras")).expanduser()
BASE_MODEL = os.environ.get("PSS_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_STEPS = int(os.environ.get("PSS_MAX_STEPS", "120"))
MAX_SEQ_LEN = int(os.environ.get("PSS_MAX_SEQ_LEN", "1024"))
LORA_R = int(os.environ.get("PSS_LORA_R", "8"))

logging.basicConfig(
    level=os.environ.get("PSS_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-5s %(message)s",
)
LOG = logging.getLogger("pss.trainer")


def _have_training_stack() -> bool:
    """torch + transformers + peft 都在才算能真训。"""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except Exception:  # noqa: BLE001 - 缺什么不重要，重要的是「能不能训」
        return False
    return True


#: 显式 DRY_RUN=1 或缺依赖 → 模拟训练
DRY_RUN = os.environ.get("PSS_DRY_RUN", "auto")
if DRY_RUN == "auto":
    DRY_RUN = not _have_training_stack()
else:
    DRY_RUN = DRY_RUN not in ("0", "false", "False", "")

LOG.info("dsh-pss trainer 启动：dry_run=%s base_model=%s lora_dir=%s", DRY_RUN, BASE_MODEL, LORA_DIR)


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------


class Turn(BaseModel):
    role: str = Field(description="user / assistant / system")
    content: str


class CompileRequest(BaseModel):
    sessionId: str = Field(description="会话 id，同时是 adapter 目录名")
    history: List[Turn] = Field(default_factory=list, description="对话历史，越长越好")
    baseModel: Optional[str] = Field(default=None, description="覆盖默认基座模型")
    maxSteps: Optional[int] = Field(default=None, ge=1, le=2000)
    force: bool = Field(default=False, description="已有 ready adapter 时是否重训")


class CompileResponse(BaseModel):
    jobId: str
    sessionId: str
    status: str
    dryRun: bool
    adapterPath: str


class JobView(BaseModel):
    jobId: str
    sessionId: str
    status: str
    step: int = 0
    totalSteps: int = 0
    loss: Optional[float] = None
    startedAt: str = ""
    finishedAt: Optional[str] = None
    error: Optional[str] = None
    adapterPath: str = ""
    dryRun: bool = True
    turns: int = 0


class IntentRequest(BaseModel):
    sessionId: str
    query: str
    keywords: List[str] = Field(default_factory=list)
    centroid: List[float] = Field(default_factory=list)
    turns: int = 0


class IntentResponse(BaseModel):
    intentQuery: str
    keywords: List[str]
    source: str


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

STOPWORDS = {
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "和", "就", "都", "也", "很",
    "把", "被", "让", "给", "对", "吗", "呢", "吧", "啊", "这", "那", "有", "个",
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_session_id(session_id: str) -> str:
    """sessionId 直接当目录名用，必须先净化，否则就是路径穿越。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", session_id).strip("._-")
    return cleaned[:96] or "session"


def adapter_dir(session_id: str) -> Path:
    return LORA_DIR / safe_session_id(session_id)


def extract_keywords(text: str, limit: int = 12) -> List[str]:
    """中英混排的关键词抽取：拉丁词 + 中文二元组，纯启发式，够用。"""
    text = text.lower()
    latin = re.findall(r"[a-z0-9_]{2,}", text)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    grams: List[str] = []
    for run in cjk_runs:
        if len(run) <= 2:
            grams.append(run)
            continue
        grams.extend(run[i : i + 2] for i in range(len(run) - 1))
    counts: Dict[str, int] = {}
    for token in latin + grams:
        if token in STOPWORDS or token.isdigit():
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [term for term, _ in ranked[:limit]]


def build_training_text(turns: Iterable[Turn]) -> str:
    """把历史拼成一段「会话摘要风格」的训练语料。

    PSS 要让 LoRA 学会的不是「怎么回答」，而是「这场会话在谈什么、用什么词谈」，
    所以训练文本是 `用户：… / 助手：…` 的对话形式 + 结尾的意图摘要指令。
    """
    lines: List[str] = []
    for turn in turns:
        if not turn.content.strip():
            continue
        who = {"user": "用户", "assistant": "助手", "system": "系统"}.get(turn.role, turn.role)
        lines.append(f"{who}：{turn.content.strip()}")
    body = "\n".join(lines)
    keywords = "、".join(extract_keywords(body, 16)) or "（无）"
    return (
        f"{body}\n\n"
        f"### 指令\n"
        f"把上面这场会话压缩成一句参数化检索意图，并列出最相关的关键词。\n"
        f"### 意图\n"
        f"围绕「{keywords}」的会话上下文\n"
        f"### 关键词\n"
        f"{keywords}\n"
    )


# ---------------------------------------------------------------------------
# 任务表（内存 + 落盘镜像）
# ---------------------------------------------------------------------------


@dataclass
class Job:
    jobId: str
    sessionId: str
    status: str = "queued"          # queued | running | done | failed
    step: int = 0
    totalSteps: int = MAX_STEPS
    loss: Optional[float] = None
    startedAt: str = field(default_factory=now_iso)
    finishedAt: Optional[str] = None
    error: Optional[str] = None
    adapterPath: str = ""
    dryRun: bool = True
    turns: int = 0

    def view(self) -> JobView:
        return JobView(**asdict(self))


JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def _job_path(job_id: str) -> Path:
    return LORA_DIR / "_jobs" / f"{job_id}.json"


def save_job(job: Job) -> None:
    path = _job_path(job.jobId)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(job), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 训练实现
# ---------------------------------------------------------------------------


def train_job(job: Job, turns: List[Turn], base_model: str) -> None:
    """后台任务主体：模拟训练 or 真训练，两条路都写同样的产物。"""
    out_dir = adapter_dir(job.sessionId)
    out_dir.mkdir(parents=True, exist_ok=True)
    job.status = "running"
    job.startedAt = now_iso()
    save_job(job)

    corpus = build_training_text(turns)
    (out_dir / "corpus.txt").write_text(corpus, encoding="utf-8")

    try:
        if job.dryRun:
            _dry_run_train(job, out_dir)
        else:
            _peft_train(job, out_dir, corpus, base_model)

        manifest = {
            "sessionId": job.sessionId,
            "adapterId": f"{safe_session_id(job.sessionId)}-{job.jobId[:8]}",
            "baseModel": base_model,
            "status": "ready",
            "trainedAt": now_iso(),
            "turns": job.turns,
            "steps": job.step,
            "finalLoss": job.loss,
            "dryRun": job.dryRun,
            "loraR": LORA_R,
            "dir": str(out_dir),
        }
        (out_dir / "adapter_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        job.status = "done"
        job.finishedAt = now_iso()
        job.adapterPath = str(out_dir)
        LOG.info("训练完成 session=%s steps=%s dry_run=%s", job.sessionId, job.step, job.dryRun)
    except Exception as error:  # noqa: BLE001 - 任何训练失败都要落成可读状态
        job.status = "failed"
        job.error = f"{type(error).__name__}: {error}"
        job.finishedAt = now_iso()
        LOG.exception("训练失败 session=%s", job.sessionId)
    finally:
        save_job(job)


def _dry_run_train(job: Job, out_dir: Path) -> None:
    """模拟训练：按步推进，loss 单调下降，产出和真训练同构的目录结构。

    存在的意义：没有 GPU 的机器也能把「编译 → 检索」整条链路跑通、跑测试。
    """
    total = job.totalSteps
    loss = 2.83
    for step in range(1, total + 1):
        time.sleep(0.02)                       # 别让接口瞬间返回，模拟真实耗时
        loss = max(0.42, loss * 0.972 + (0.02 if step % 7 == 0 else 0.0))
        job.step = step
        job.loss = round(loss, 4)
        if step % 10 == 0 or step == total:
            save_job(job)

    # 一个「看起来就是 LoRA」的占位权重文件，插件只读 manifest，但这个文件让目录更可信
    (out_dir / "adapter_model.safetensors").write_bytes(
        b"PSS-DRY-RUN-ADAPTER\n" + json.dumps({"r": LORA_R, "steps": total}).encode() + b"\n"
    )
    (out_dir / "train.log").write_text(
        f"dry-run mode\ntotal_steps={total}\nfinal_loss={job.loss}\n", encoding="utf-8"
    )


def _peft_train(job: Job, out_dir: Path, corpus: str, base_model: str) -> None:
    """真正的 LoRA 训练：transformers + peft，单卡，够小够快。"""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    LOG.info("加载基座模型 %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 语料很短，切成固定长度的块就够；真实场景该做 packing
    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    dataset = Dataset.from_dict({"text": [corpus] * 8}).map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=job.totalSteps,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=0,
        report_to=[],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    class ProgressCallback:
        """把 Trainer 的 loss 抄进 job，前端 / 插件才有进度可看。"""

        def on_log(self, _args, state, _control, logs=None, **_kwargs):  # noqa: ANN001
            if logs and "loss" in logs:
                job.step = int(state.global_step)
                job.loss = round(float(logs["loss"]), 4)
                save_job(job)

    from transformers import TrainerCallback

    class _Cb(TrainerCallback, ProgressCallback):
        pass

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[_Cb()],
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    job.step = job.totalSteps


# ---------------------------------------------------------------------------
# 意图服务：有 adapter 就用 adapter，没有就退回启发式
# ---------------------------------------------------------------------------

#: 最多同时驻留几个 adapter（每个几百 MB，别贪）
INTENT_CACHE: "OrderedDict[str, Any]" = OrderedDict()
INTENT_CACHE_MAX = 2


def _load_intent_model(session_id: str):
    """按需加载某会话的 adapter；失败返回 None（调用方自动降级）。"""
    if DRY_RUN:
        return None
    directory = adapter_dir(session_id)
    manifest = directory / "adapter_manifest.json"
    if not manifest.exists():
        return None

    if session_id in INTENT_CACHE:
        INTENT_CACHE.move_to_end(session_id)
        return INTENT_CACHE[session_id]

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        meta = json.loads(manifest.read_text(encoding="utf-8"))
        base = meta.get("baseModel", BASE_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(directory))
        model.eval()
        INTENT_CACHE[session_id] = (model, tokenizer)
        if len(INTENT_CACHE) > INTENT_CACHE_MAX:
            INTENT_CACHE.popitem(last=False)
        return INTENT_CACHE[session_id]
    except Exception:  # noqa: BLE001
        LOG.exception("加载 adapter 失败 session=%s，降级启发式", session_id)
        return None


def heuristic_intent(request: IntentRequest) -> IntentResponse:
    """没有 LoRA 时的兜底：把已有关键词 + 本次提问的关键词合并成检索意图。"""
    fresh = extract_keywords(request.query, 8)
    merged: List[str] = []
    for term in fresh + [k for k in request.keywords if k]:
        if term not in merged:
            merged.append(term)
    intent = " ".join(merged[:10]) or request.query
    return IntentResponse(intentQuery=intent, keywords=merged[:12], source="heuristic")


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="dsh-pss trainer",
    version="0.1.0",
    description="参数化会话状态（PSS）的离线 LoRA 训练与意图改写服务",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    """健康检查：插件和 Dashboard 都靠它判断后端在不在。"""
    adapters = [
        p.name for p in LORA_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")
    ] if LORA_DIR.exists() else []
    return {
        "ok": True,
        "service": "dsh-pss-trainer",
        "dryRun": DRY_RUN,
        "baseModel": BASE_MODEL,
        "loraDir": str(LORA_DIR.resolve()),
        "adapters": adapters,
        "jobs": len(JOBS),
        "trainingStack": _have_training_stack(),
    }


@app.post("/compile", response_model=CompileResponse, status_code=202)
def compile_session(request: CompileRequest, background: BackgroundTasks) -> CompileResponse:
    """提交一次编译：立刻返回 jobId，训练在后台跑。"""
    if not request.sessionId.strip():
        raise HTTPException(status_code=400, detail="sessionId is required")

    directory = adapter_dir(request.sessionId)
    manifest = directory / "adapter_manifest.json"
    if manifest.exists() and not request.force:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if meta.get("status") == "ready":
            # 已经有可用 adapter：幂等返回，不重复烧算力
            job = Job(
                jobId=f"cached-{uuid.uuid4().hex[:8]}",
                sessionId=request.sessionId,
                status="done",
                totalSteps=int(meta.get("steps") or 0),
                step=int(meta.get("steps") or 0),
                loss=meta.get("finalLoss"),
                adapterPath=str(directory),
                dryRun=bool(meta.get("dryRun", True)),
                turns=int(meta.get("turns") or 0),
                finishedAt=now_iso(),
            )
            with JOBS_LOCK:
                JOBS[job.jobId] = job
            return CompileResponse(
                jobId=job.jobId,
                sessionId=job.sessionId,
                status="done",
                dryRun=job.dryRun,
                adapterPath=job.adapterPath,
            )

    turns = [t for t in request.history if t.content.strip()]
    if not turns:
        raise HTTPException(status_code=400, detail="history is empty; nothing to learn from")

    job = Job(
        jobId=uuid.uuid4().hex[:12],
        sessionId=request.sessionId,
        totalSteps=request.maxSteps or MAX_STEPS,
        dryRun=bool(DRY_RUN),
        turns=len(turns),
        adapterPath=str(directory),
    )
    with JOBS_LOCK:
        JOBS[job.jobId] = job
    save_job(job)

    background.add_task(train_job, job, turns, request.baseModel or BASE_MODEL)
    LOG.info("收到编译任务 session=%s turns=%s job=%s", request.sessionId, len(turns), job.jobId)
    return CompileResponse(
        jobId=job.jobId,
        sessionId=job.sessionId,
        status="queued",
        dryRun=job.dryRun,
        adapterPath=str(directory),
    )


@app.get("/jobs/{job_id}", response_model=JobView)
def get_job(job_id: str) -> JobView:
    """查训练进度。"""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is not None:
        return job.view()

    # 进程重启过？从磁盘镜像里捞
    path = _job_path(job_id)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return JobView(**data)
    raise HTTPException(status_code=404, detail=f"unknown job {job_id}")


@app.get("/jobs")
def list_jobs() -> Dict[str, Any]:
    """最近的任务列表（调试用）。"""
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j.startedAt, reverse=True)
    return {"jobs": [job.view() for job in jobs[:50]]}


@app.post("/retrieve-intent", response_model=IntentResponse)
def retrieve_intent(request: IntentRequest) -> IntentResponse:
    """把提问改写成参数化检索意图：有 adapter 用 adapter，没有就启发式。"""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    loaded = _load_intent_model(request.sessionId)
    if loaded is None:
        return heuristic_intent(request)

    model, tokenizer = loaded
    try:
        import torch

        prompt = (
            "### 会话关键词\n"
            f"{'、'.join(request.keywords[:16]) or '（无）'}\n\n"
            f"### 用户提问\n{request.query}\n\n"
            "### 指令\n把提问改写成一句参数化检索意图，输出关键词，每行一个。\n### 意图\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        first_line = next((line for line in text.splitlines() if line.strip()), request.query)
        keywords: List[str] = []
        for line in text.splitlines()[1:]:
            term = line.strip(" -·•*")
            if term and term not in keywords:
                keywords.append(term)
        if not keywords:
            keywords = extract_keywords(first_line + " " + request.query, 10)
        return IntentResponse(intentQuery=first_line.strip(), keywords=keywords[:12], source="lora")
    except Exception:  # noqa: BLE001
        LOG.exception("adapter 推理失败，降级启发式")
        return heuristic_intent(request)


@app.get("/adapters")
def list_adapters() -> Dict[str, Any]:
    """列出所有已就绪的 adapter（插件可用它决定会话状态是否已有 LoRA）。"""
    items: List[Dict[str, Any]] = []
    if LORA_DIR.exists():
        for directory in sorted(LORA_DIR.iterdir()):
            if not directory.is_dir() or directory.name.startswith("_"):
                continue
            manifest = directory / "adapter_manifest.json"
            if manifest.exists():
                try:
                    items.append(json.loads(manifest.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    continue
    return {"adapters": items}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
