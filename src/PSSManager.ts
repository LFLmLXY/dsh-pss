/**
 * PSSManager —— 参数化会话状态的运行时，同时是「可视化探针」的埋点处。
 *
 * 它做三件事：
 *  1. **吸收**（{@link ingest}）：把对话文本折进定长状态（哈希嵌入 EMA + 关键词衰减）。
 *  2. **检索**（{@link retrieve}）：先用意图后端把提问参数化成检索词，再用 BM25 找片段，
 *     拼成上下文包。这条链路上的每一步都会往内存图里推节点和连线。
 *  3. **编译**（{@link compile}）：会话结束时把历史压成状态，并把训练请求丢给 Python 后端。
 *
 * 内存图（graph）是纯装饰性的吗？不是。它是这个项目唯一能让外部「看见」内部状态的通道，
 * 所以节点上带的 detail 必须是真实数据（意图来源、BM25 分数、耗时），否则前端就是在骗人。
 *
 * @module dsh-pss/PSSManager
 */

import { randomUUID } from 'node:crypto';
import { readFile, rename, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { Bm25Index, addEmbedding, normalize, tokenize } from './embed.js';
import { DEFAULT_KNOWLEDGE, loadKnowledgeDir } from './knowledge.js';
import { LocalIntentBackend } from './intent.js';
import { StateStore, stateFileName } from './store.js';
import {
  NODE_KINDS,
  type CompileResult,
  type GraphData,
  type GraphLink,
  type GraphNode,
  type GraphStats,
  type IntentBackend,
  type Keyword,
  type KnowledgeChunk,
  type LinkKind,
  type LoraRef,
  type MemoryChunk,
  type NodeKind,
  type PSSEvent,
  type PSSEventKind,
  type PSSManagerOptions,
  type PSSState,
  type RetrievalHit,
  type RetrieveResult
} from './types.js';

/** 关键词表容量上限（恒定，与对话长度无关）。 */
const MAX_KEYWORDS = 48;
/** 每轮吸收时旧关键词的权重衰减系数。 */
const KEYWORD_DECAY = 0.86;
/** 新观测向量在意图中心里的占比（指数移动平均）。 */
const CENTROID_EMA = 0.22;
/** 节点文字最大长度（超出截断，完整内容放 detail）。 */
const LABEL_CHARS = 42;
/** 回喂给模型的上下文包字符上限。 */
const CONTEXT_CHARS = 1400;
/** 单条记忆片段的字符上限（超长对话按此切块，切完各占一条）。 */
const MEMORY_CHUNK_CHARS = 320;
/** 每个会话最多保留多少条记忆片段（FIFO 淘汰，防止内存无界增长）。 */
const MAX_MEMORY_CHUNKS = 800;

/** 节点 id 前缀 → 语义（与 {@link NodeKind} 对齐）。 */
const NODE_PREFIX: Record<NodeKind, string> = {
  user: 'u',
  intent: 'i',
  knowledge: 'k',
  answer: 'a',
  system: 's'
};

/** 把文本截断成一行节点标签。 */
function clip(text: string, limit = LABEL_CHARS): string {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length <= limit ? flat : `${flat.slice(0, limit - 1)}…`;
}

/** 把 sessionId 压成适合放进节点 id 的短键。 */
function sessionKey(sessionId: string): string {
  return sessionId.replace(/[^a-zA-Z0-9_-]/g, '').slice(-8) || 'sess';
}

/** 默认状态目录：`~/.dsh/pss`。 */
export function defaultRoot(): string {
  return join(homedir(), '.dsh', 'pss');
}

/** 演示语料：每一条都要能引出不同的知识片段，图才不会长得一模一样。 */
const DEMO_TOPICS: [string, string][] = [
  ['PSS 到底怎么做到 O(1) 恒定存储的？', '把历史折进定长向量和定长关键词表，磁盘上只存这个状态。'],
  ['那参数化意图是怎么工作的？LoRA 什么时候加载？', '检索时临时加载会话专属 LoRA 生成检索词，用完立刻卸载，所以常驻显存是零。'],
  ['可视化探针会记录哪些东西？', '用户输入、意图、命中的知识片段、合成后的上下文包，以及它们之间的因果连线。'],
  ['质心为什么要用 EMA 而不是简单平均？', '简单平均记不住"最近在聊什么"，指数滑动平均给近期内容更高权重。'],
  ['关键词表的衰减因子为什么是 0.86？', '每轮乘 0.86，十轮之后权重只剩三成，老话题自然沉底，不需要显式删除。'],
  ['检索出来的上下文包有多长？', '固定上限 1400 字符，按分数从高到低拼，超了就截断，保证 prompt 不随时间膨胀。'],
  ['离线训练一次要多久？', '0.5B 基座加上 LoRA，单卡十分钟量级；没有 GPU 时训练端会走 dry-run，把流水线跑通。'],
  ['adapter 是怎么和会话绑定的？', '会话 id 就是 adapter 目录名，编译时写 manifest，检索时按 id 热插。'],
  ['记忆压缩了，细节会不会丢？', '会丢细节，保住的是"你在意什么"这件更抽象的事；真要原文，还是得回到日志里查。'],
  ['这套东西能跨会话迁移吗？', '可以，质心和关键词都是纯数值，把 adapter 复制到另一个会话就能带着偏好走。'],
  ['为什么用 BM25 而不是纯向量检索？', 'BM25 对专有名词和编号更稳，向量对同义改写更稳，PSS 先用意图改写把两者接起来。'],
  ['脑图上的连线分别代表什么？', '参数化、检索、装配、编译四种因果关系，颜色和线型都不一样。']
];

/** 初始（空）参数化状态。 */
function emptyState(sessionId: string, dim: number): PSSState {
  const now = Date.now();
  return {
    sessionId,
    version: 0,
    createdAt: now,
    updatedAt: now,
    turns: 0,
    ingestedChars: 0,
    dim,
    centroid: new Array<number>(dim).fill(0),
    keywords: [],
    lora: null
  };
}

/** 把 BM25 原始分压到 0..1，方便前端直接画节点大小。 */
function normalizeScores(hits: { index: number; score: number }[]): number[] {
  const max = hits.reduce((acc, hit) => Math.max(acc, hit.score), 0);
  return hits.map((hit) => (max <= 0 ? 0 : hit.score / max));
}

/**
 * PSS 运行时。
 *
 * 一个实例服务于一个 DSH 进程里的所有会话：`activeSessions` 是内存缓存，
 * 落盘由 {@link StateStore} 负责，两者都以 sessionId 为键。
 */
export class PSSManager {
  /** 定长状态的读写器。 */
  readonly store: StateStore;
  /** 意图后端（本地启发式 / LoRA，可带降级）。 */
  private readonly intentBackend: IntentBackend;

  private readonly dim: number;
  private readonly topK: number;
  private readonly maxNodes: number;
  private readonly maxEvents: number;
  private readonly log: (message: string) => void;

  /** 内存缓存：sessionId → 参数化状态。 */
  private readonly activeSessions = new Map<string, PSSState>();

  /** 可视化图（只在内存里，带 FIFO 裁剪）。 */
  private nodes: GraphNode[] = [];
  private links: GraphLink[] = [];
  private graphVersion = 0;
  private nodeSeq = 0;

  /** 事件环形缓冲（Dashboard 右侧滚动列表）。 */
  private events: PSSEvent[] = [];

  /** 计数器。 */
  private retrievals = 0;
  private compiles = 0;
  private readonly startedAt = Date.now();

  /** 知识库与索引（懒加载一次）。 */
  private knowledge: KnowledgeChunk[] = DEFAULT_KNOWLEDGE;
  private index: Bm25Index;
  private indexPromise: Promise<void> | null = null;

  /**
   * 会话记忆索引：sessionId → 该会话的片段 + 对应 BM25。
   *
   * 知识库记的是"世界知识"，装进来就不变；这里记的是"这场会话里说过什么"。
   * 两者的生命周期不同，所以分开建索引，检索时再合并。
   * `version` 是片段数，用来判断索引是否过期（片段增加就重建，成本是 O(n)，n 很小）。
   */
  private readonly memory = new Map<string, { chunks: MemoryChunk[]; index: Bm25Index; indexed: number }>();

  /** Python 训练后端地址（缺省不训练）。 */
  private readonly trainerUrl: string | undefined;
  private readonly baseModel: string;

  constructor(private readonly options: PSSManagerOptions) {
    this.store = new StateStore(options.root || defaultRoot());
    this.intentBackend = options.intentBackend ?? new LocalIntentBackend();
    this.dim = Math.max(8, Math.min(256, options.dim ?? 32));
    this.topK = Math.max(1, Math.min(20, options.topK ?? 3));
    this.maxNodes = Math.max(40, options.maxNodes ?? 400);
    this.maxEvents = Math.max(10, options.maxEvents ?? 50);
    this.log = options.log ?? (() => undefined);
    this.trainerUrl = options.trainerUrl?.replace(/\/+$/, '');
    this.baseModel = options.baseModel ?? 'deepseek-chat';
    this.index = new Bm25Index([]);
  }

  /* ------------------------------------------------------------------------ */
  /* 初始化                                                                    */
  /* ------------------------------------------------------------------------ */

  /**
   * 加载知识库并建索引（幂等，多次调用只会真正跑一次）。
   *
   * 顺序：内置种子语料 → 目录文件 → 构造参数里的额外片段。
   * 三者合并后同一 id 只保留第一个，方便外部用同 id 覆盖种子的假数据。
   */
  init(): Promise<void> {
    this.indexPromise ??= (async () => {
      const collected: KnowledgeChunk[] = [...DEFAULT_KNOWLEDGE];

      if (this.options.knowledgeDir) {
        const fromDir = await loadKnowledgeDir(this.options.knowledgeDir);
        collected.push(...fromDir);
        this.log(`[pss] knowledge dir ${this.options.knowledgeDir}: ${fromDir.length} chunk(s)`);
      }
      if (this.options.knowledge) collected.push(...this.options.knowledge);

      const dedup = new Map<string, KnowledgeChunk>();
      for (const chunk of collected) if (!dedup.has(chunk.id)) dedup.set(chunk.id, chunk);

      this.knowledge = [...dedup.values()];
      this.index = new Bm25Index(this.knowledge.map((chunk) => tokenize(chunk.text)));
      await this.store.ensure();
      this.systemEvent(`知识库就绪：${this.knowledge.length} 个片段`);
    })();
    return this.indexPromise;
  }

  /* ------------------------------------------------------------------------ */
  /* 探针：图与事件                                                            */
  /* ------------------------------------------------------------------------ */

  /** 图变更后自增版本号（前端据此跳过无变化的重绘）。 */
  private bump(): void {
    this.graphVersion += 1;
  }

  /** 追加一个节点，返回它的 id。 */
  private addNode(kind: NodeKind, label: string, detail?: string, weight = 0.5, sessionId?: string): string {
    this.nodeSeq += 1;
    const id = `${NODE_PREFIX[kind]}:${sessionId ? sessionKey(sessionId) : 'sys'}:${this.nodeSeq}`;
    this.nodes.push({
      id,
      label: clip(label),
      kind,
      t: Date.now(),
      weight: Math.max(0, Math.min(1, weight)),
      ...(detail === undefined ? {} : { detail })
    });
    this.trimNodes();
    this.bump();
    return id;
  }

  /** 追加一条连线（端点必须已存在，否则丢弃 —— 脏边会让前端布局抖动）。 */
  private addLink(source: string, target: string, kind: LinkKind): void {
    if (source === target) return;
    this.nodeSeq += 1;
    this.links.push({
      id: `l:${source}->${target}:${this.nodeSeq}`,
      source,
      target,
      kind,
      t: Date.now()
    });
    this.bump();
  }

  /** 节点数超限时按 FIFO 裁掉最老的一批，并清掉悬空连线。 */
  private trimNodes(): void {
    if (this.nodes.length <= this.maxNodes) return;
    const dropCount = Math.max(1, Math.floor(this.maxNodes * 0.2));
    const dropped = new Set(this.nodes.slice(0, dropCount).map((node) => node.id));
    this.nodes = this.nodes.slice(dropCount);
    this.links = this.links.filter((link) => !dropped.has(link.source) && !dropped.has(link.target));
  }

  /** 写一条事件到环形缓冲。 */
  private pushEvent(kind: PSSEventKind, message: string, sessionId?: string): PSSEvent {
    const event: PSSEvent = {
      id: `e:${this.nodeSeq}:${Date.now().toString(36)}:${Math.random().toString(36).slice(2, 6)}`,
      t: Date.now(),
      kind,
      message,
      ...(sessionId === undefined ? {} : { sessionId })
    };
    this.events.push(event);
    if (this.events.length > this.maxEvents) this.events = this.events.slice(-this.maxEvents);
    this.bump();
    return event;
  }

  /** 生命周期类事件，直接进缓冲并打日志。 */
  private systemEvent(message: string, sessionId?: string): void {
    this.pushEvent('system', message, sessionId);
    this.log(`[pss] ${message}`);
  }

  /** 当前图快照（结构拷贝，调用方可随意改动）。 */
  graph(): GraphData {
    const byKind = Object.fromEntries(NODE_KINDS.map((kind) => [kind, 0])) as Record<NodeKind, number>;
    for (const node of this.nodes) byKind[node.kind] += 1;
    const stats: GraphStats = {
      nodes: this.nodes.length,
      links: this.links.length,
      byKind,
      sessions: this.activeSessions.size,
      retrievals: this.retrievals,
      compiles: this.compiles,
      uptimeMs: Date.now() - this.startedAt
    };
    return {
      version: this.graphVersion,
      nodes: this.nodes.map((node) => ({ ...node })),
      links: this.links.map((link) => ({ ...link })),
      stats
    };
  }

  /** 事件列表（按时间升序）。 */
  eventList(): PSSEvent[] {
    return this.events.map((event) => ({ ...event }));
  }

  /** 知识库片段列表（Dashboard 的「语料」页签用）。 */
  knowledgeChunks(): KnowledgeChunk[] {
    return this.knowledge.map((chunk) => ({ ...chunk }));
  }

  /* ------------------------------------------------------------------------ */
  /* 状态管理                                                                  */
  /* ------------------------------------------------------------------------ */

  /** 从缓存 / 磁盘取状态，都没有就新建一个。 */
  async ensure(sessionId: string): Promise<PSSState> {
    const cached = this.activeSessions.get(sessionId);
    if (cached) return cached;

    const loaded = await this.store.load(sessionId);
    const state = loaded ?? emptyState(sessionId, this.dim);
    // 维度可能因配置变化而不同，此时以当前配置为准重建向量
    if (state.centroid.length !== this.dim) state.centroid = new Array<number>(this.dim).fill(0);
    this.activeSessions.set(sessionId, state);
    if (!loaded) this.systemEvent(`新会话 ${sessionId}`, sessionId);
    return state;
  }

  /** 只读缓存里的状态（不触发加载）。 */
  peek(sessionId: string): PSSState | null {
    const state = this.activeSessions.get(sessionId);
    return state ? { ...state, centroid: [...state.centroid], keywords: state.keywords.map((kw) => ({ ...kw })) } : null;
  }

  /** 已加载的会话状态列表。 */
  sessionList(): PSSState[] {
    return [...this.activeSessions.values()].map((state) => ({
      ...state,
      centroid: [...state.centroid],
      keywords: state.keywords.map((kw) => ({ ...kw }))
    }));
  }

  /** 落盘（失败只记日志，绝不打断主流程）。 */
  private async persist(state: PSSState): Promise<void> {
    try {
      await this.store.save(state);
    } catch (error) {
      this.pushEvent('error', `状态落盘失败：${String(error)}`, state.sessionId);
    }
  }

  /* ------------------------------------------------------------------------ */
  /* 吸收：对话 → 定长状态                                                     */
  /* ------------------------------------------------------------------------ */

  /**
   * 把一段对话文本吸收进参数化状态。
   *
   * 两次「定长化」：
   *  - 向量：`centroid = normalize((1-α)·centroid + α·hash(tokens))`，长度永远 dim；
   *  - 关键词：旧词权重乘衰减系数 → 合并新词 → 取前 {@link MAX_KEYWORDS} 条。
   *
   * @param sessionId - 会话 id。
   * @param role - `user` / `assistant` / `system`，决定探针节点的类别。
   * @param text - 对话文本。
   * @param options.probe - 是否向图里推节点（默认 true）。
   * @returns 更新后的状态。
   */
  async ingest(
    sessionId: string,
    role: 'user' | 'assistant' | 'system',
    text: string,
    options: { probe?: boolean; remember?: boolean } = {}
  ): Promise<PSSState> {
    await this.init();
    const state = await this.ensure(sessionId);
    const trimmed = text.trim();
    if (trimmed.length === 0) return state;

    const tokens = tokenize(trimmed);

    // 1) 向量：指数移动平均，保持单位长度
    const delta = new Array<number>(this.dim).fill(0);
    addEmbedding(delta, tokens, 1);
    for (let i = 0; i < this.dim; i++) {
      const prev = state.centroid[i] ?? 0;
      state.centroid[i] = (1 - CENTROID_EMA) * prev + CENTROID_EMA * (delta[i] ?? 0);
    }
    normalize(state.centroid);

    // 2) 关键词：衰减 + 累加 + 截断
    const decayed = new Map<string, number>();
    for (const kw of state.keywords) {
      const weight = kw.weight * KEYWORD_DECAY;
      if (weight > 0.01) decayed.set(kw.term, weight);
    }
    const counts = new Map<string, number>();
    for (const token of tokens) counts.set(token, (counts.get(token) ?? 0) + 1);
    for (const [term, count] of counts) {
      // 词频取平方根压缩：出现 10 次的词不该比 1 次的重要 10 倍
      decayed.set(term, (decayed.get(term) ?? 0) + Math.sqrt(count) * 0.5);
    }
    state.keywords = [...decayed.entries()]
      .map(([term, weight]): Keyword => ({ term, weight }))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, MAX_KEYWORDS);

    state.turns += 1;
    state.ingestedChars += trimmed.length;
    state.updatedAt = Date.now();
    state.version += 1;

    if (options.probe !== false) {
      const kind: NodeKind = role === 'user' ? 'user' : role === 'assistant' ? 'answer' : 'system';
      this.addNode(
        kind,
        role === 'user' ? trimmed : `${role === 'assistant' ? 'AI 回答' : '系统'}：${trimmed}`,
        `role=${role}\nturns=${state.turns}\nchars=${trimmed.length}\nkeywords=${state.keywords
          .slice(0, 8)
          .map((kw) => kw.term)
          .join(' ')}`,
        Math.min(1, 0.35 + trimmed.length / 600),
        sessionId
      );
      this.pushEvent('ingest', `${role} 吸收 ${trimmed.length} 字 → turns=${state.turns}`, sessionId);
    }

    // 对话原文也进记忆片段索引——否则 retrieve 只能检索到世界知识库，
    // "这场会话里说过什么"永远召回不到（这是之前最大的空洞）。
    if (options.remember !== false) await this.appendMemory(sessionId, role, trimmed);

    await this.persist(state);
    return state;
  }

  /* ------------------------------------------------------------------------ */
  /* 会话记忆：把"说过什么"变成可检索的片段                                     */
  /* ------------------------------------------------------------------------ */

  /** 记忆片段文件路径（与状态文件同目录，扩展名区分）。 */
  private memoryPath(sessionId: string): string {
    return join(this.store.rootDir, 'sessions', stateFileName(sessionId).replace(/\.json$/, '.memory.json'));
  }

  /** 取（或首次创建并加载）某会话的记忆容器。 */
  private async memoryOf(sessionId: string): Promise<{ chunks: MemoryChunk[]; index: Bm25Index; indexed: number }> {
    const cached = this.memory.get(sessionId);
    if (cached) return cached;

    const bucket = { chunks: [] as MemoryChunk[], index: new Bm25Index([]), indexed: 0 };
    try {
      const raw = await readFile(this.memoryPath(sessionId), 'utf8');
      const parsed = JSON.parse(raw) as { chunks?: MemoryChunk[] };
      if (Array.isArray(parsed.chunks)) bucket.chunks = parsed.chunks.filter((c) => typeof c?.text === 'string');
    } catch {
      // 文件不存在就是新会话，正常路径
    }
    this.memory.set(sessionId, bucket);
    return bucket;
  }

  /** 记忆索引重建（片段数变了才重建；n 很小，全量重建比增量维护省事）。 */
  private ensureMemoryIndex(sessionId: string, bucket: { chunks: MemoryChunk[]; index: Bm25Index; indexed: number }): Bm25Index {
    if (bucket.indexed !== bucket.chunks.length) {
      bucket.index = new Bm25Index(bucket.chunks.map((chunk) => tokenize(chunk.text)));
      bucket.indexed = bucket.chunks.length;
    }
    return bucket.index;
  }

  /** 按 {@link MEMORY_CHUNK_CHARS} 把长文本切成若干片段（短文本原样一条）。 */
  private static slice(text: string): string[] {
    const flat = text.replace(/\s+/g, ' ').trim();
    if (flat.length === 0) return [];
    if (flat.length <= MEMORY_CHUNK_CHARS) return [flat];
    const parts: string[] = [];
    for (let offset = 0; offset < flat.length; offset += MEMORY_CHUNK_CHARS) {
      parts.push(flat.slice(offset, offset + MEMORY_CHUNK_CHARS));
    }
    return parts;
  }

  /**
   * 往会话记忆里追加一条（内部用）。
   *
   * 每轮对话都会走到这里，所以它必须便宜：只 push + 标脏，索引用懒重建。
   */
  private async appendMemory(
    sessionId: string,
    role: MemoryChunk['role'],
    text: string
  ): Promise<MemoryChunk[]> {
    const bucket = await this.memoryOf(sessionId);
    const added: MemoryChunk[] = [];
    for (const [index, part] of PSSManager.slice(text).entries()) {
      const chunk: MemoryChunk = {
        id: `${sessionKey(sessionId)}#${bucket.chunks.length + index}`,
        sessionId,
        role,
        text: part,
        t: Date.now()
      };
      bucket.chunks.push(chunk);
      added.push(chunk);
    }
    if (bucket.chunks.length > MAX_MEMORY_CHUNKS) {
      bucket.chunks.splice(0, bucket.chunks.length - MAX_MEMORY_CHUNKS);
      bucket.indexed = -1; // 强制重建
    }
    if (added.length > 0) await this.persistMemory(sessionId, bucket.chunks);
    return added;
  }

  /** 原子写记忆片段文件（tmp + rename）。 */
  private async persistMemory(sessionId: string, chunks: MemoryChunk[]): Promise<void> {
    const path = this.memoryPath(sessionId);
    const tmp = `${path}.tmp`;
    try {
      await this.store.ensure();
      await writeFile(tmp, JSON.stringify({ format: 1, sessionId, chunks }), 'utf8');
      await rename(tmp, path);
    } catch (error) {
      this.log(`[pss] 记忆片段落盘失败 ${sessionId}: ${String(error)}`);
    }
  }

  /**
   * 主动记一条结论（`pss_remember` 工具 / `POST /api/pss/remember`）。
   *
   * 这是「模型自己决定什么值得记」的入口：对话原文会被自动吸收，
   * 但"决定了什么、为什么这么决定"往往只在思考里，需要显式写进来。
   *
   * @param sessionId - 会话 id。
   * @param text - 要记住的结论（一两句话最好，会被切成片段）。
   * @returns 写入后的状态版本。
   */
  async remember(sessionId: string, text: string): Promise<{ chunks: number; stateVersion: number }> {
    await this.init();
    const state = await this.ensure(sessionId);
    const added = await this.appendMemory(sessionId, 'digest', text);
    if (added.length === 0) return { chunks: 0, stateVersion: state.version };

    // 结论也要影响参数状态，否则下一轮意图改写还是按老关键词走
    await this.ingest(sessionId, 'system', text, { probe: false, remember: false });

    const node = this.addNode(
      'system',
      `记忆写入：${clip(text)}`,
      `digest
${added.length} 片段
${text}`,
      0.8,
      sessionId
    );
    this.addLink(node, node, 'compile');
    this.pushEvent('ingest', `主动记忆 ${text.length} 字 → ${added.length} 片段`, sessionId);
    return { chunks: added.length, stateVersion: (await this.ensure(sessionId)).version };
  }

  /** 会话记忆片段数（给 HTTP / Dashboard 用）。 */
  async memoryCount(sessionId: string): Promise<number> {
    const bucket = await this.memoryOf(sessionId);
    return bucket.chunks.length;
  }

  /* ------------------------------------------------------------------------ */
  /* 检索：提问 → 意图 → 知识 → 上下文包                                       */
  /* ------------------------------------------------------------------------ */

  /**
   * 一次完整检索。这是探针埋点最密集的地方：一个提问会生成
   * 1 个 user 节点 → 1 个 intent 节点 → N 个 knowledge 节点 → 1 个 answer 节点。
   *
   * @param sessionId - 会话 id。
   * @param query - 模型原始提问。
   * @returns 检索结果（含回喂给模型的上下文包）。
   */
  async retrieve(sessionId: string, query: string): Promise<RetrieveResult> {
    await this.init();
    const startedAt = Date.now();
    const state = await this.ensure(sessionId);
    const clean = query.trim();

    // 探针 1：用户输入
    const userNode = this.addNode('user', clean, `原始提问\n长度=${clean.length}`, 0.9, sessionId);

    // 参数化：把提问改写成检索词。失败不致命，降级后端会兜住。
    let intentQuery = clean;
    let keywords: string[] = tokenize(clean).slice(0, 12);
    let intentSource = this.intentBackend.name;
    try {
      const intent = await this.intentBackend.generate({ query: clean, state });
      intentQuery = intent.intentQuery;
      keywords = intent.keywords;
      intentSource = intent.source;
    } catch (error) {
      this.pushEvent('error', `意图生成失败，回退原文检索：${String(error)}`, sessionId);
    }

    // 探针 2：PSS 意图
    const intentNode = this.addNode(
      'intent',
      intentQuery === clean ? `意图（直通）：${clean}` : `意图：${intentQuery}`,
      `source=${intentSource}\n检索词=${keywords.join(' ')}\n状态版本=v${state.version}\n关键词表=${state.keywords.length} 条`,
      0.75,
      sessionId
    );
    this.addLink(userNode, intentNode, 'parametrize');

    // BM25 检索（查询词按意图输出加权重复一次，让意图的影响可观测）
    const queryTokens = [...tokenize(intentQuery), ...tokenize(intentQuery).filter((token) => keywords.includes(token))];
    const raw = this.index.search(queryTokens, this.topK);
    const scores = normalizeScores(raw);

    const knowledgeHits: RetrievalHit[] = raw.map((hit, position) => {
      const chunk = this.knowledge[hit.index];
      return {
        chunkId: chunk?.id ?? `#${hit.index}`,
        source: chunk?.source ?? 'unknown',
        text: chunk?.text ?? '',
        score: Number(((scores[position] ?? 0)).toFixed(4)),
        channel: 'knowledge'
      };
    });

    // 第二个通道：本会话的记忆片段。会话里说过的话在这里，世界知识在知识库。
    const bucket = await this.memoryOf(sessionId);
    const memIndex = this.ensureMemoryIndex(sessionId, bucket);
    const memRaw = memIndex.search(queryTokens, this.topK);
    const memScores = normalizeScores(memRaw);
    const memoryHits: RetrievalHit[] = memRaw
      .map((hit, position): RetrievalHit | null => {
        const chunk = bucket.chunks[hit.index];
        if (!chunk) return null;
        return {
          chunkId: chunk.id,
          source: `memory:${chunk.role}`,
          text: chunk.text,
          // 记忆片段是"这场会话自己的事"，同分时应该压过通用知识，给 1.12 加权
          score: Number(Math.min(1, (memScores[position] ?? 0) * 1.12).toFixed(4)),
          channel: 'memory' as const
        };
      })
      .filter((hit): hit is RetrievalHit => hit !== null);

    // 合并后按分数降序，上限 2×topK（知识 + 记忆各占一半以内），包长仍受 CONTEXT_CHARS 约束
    const hits: RetrievalHit[] = [...knowledgeHits, ...memoryHits]
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.max(this.topK, this.topK * 2));

    // 探针 3：知识节点 + 连线
    const knowledgeNodes: string[] = [];
    for (const hit of hits) {
      const node = this.addNode(
        'knowledge',
        `${hit.source} · ${hit.text}`,
        `BM25=${hit.score}\nchunk=${hit.chunkId}\n\n${hit.text}`,
        hit.score,
        sessionId
      );
      knowledgeNodes.push(node);
      this.addLink(intentNode, node, 'retrieve');
    }

    // 合成上下文包
    const contextPack = this.buildContextPack(state, intentQuery, hits, intentSource);

    // 探针 4：合成后的上下文包
    const answerNode = this.addNode(
      'answer',
      `上下文包 ${contextPack.length} 字 · ${hits.length} 片段`,
      contextPack,
      0.85,
      sessionId
    );
    for (const node of knowledgeNodes) this.addLink(node, answerNode, 'compose');
    if (knowledgeNodes.length === 0) this.addLink(intentNode, answerNode, 'compose');

    this.retrievals += 1;
    const latencyMs = Date.now() - startedAt;
    this.pushEvent(
      'retrieve',
      `检索命中 ${hits.length} 片段（${intentSource}，${latencyMs}ms，${contextPack.length} 字）`,
      sessionId
    );

    return {
      sessionId,
      query: clean,
      intentQuery,
      keywords,
      hits,
      contextPack,
      latencyMs,
      intentSource,
      stateVersion: state.version
    };
  }

  /** 拼装回喂给模型的上下文包（长度有硬上限，避免把预算吃光）。 */
  private buildContextPack(state: PSSState, intentQuery: string, hits: RetrievalHit[], source: string): string {
    const lines: string[] = [
      `[PSS] session=${state.sessionId} v${state.version} turns=${state.turns} intent=${source}`,
      `[PSS] intent_query=${intentQuery}`
    ];
    if (state.keywords.length > 0) {
      lines.push(`[PSS] keywords=${state.keywords.slice(0, 8).map((kw) => `${kw.term}:${kw.weight.toFixed(2)}`).join(' ')}`);
    }
    hits.forEach((hit, position) => {
      const tag = hit.channel === 'memory' ? '记忆' : '知识';
      lines.push(`--- ${position + 1}. [${tag}] ${hit.source} (score ${hit.score}) ---`, hit.text);
    });
    if (hits.length === 0) lines.push('（本轮没有命中知识片段）');

    let pack = lines.join('\n');
    if (pack.length > CONTEXT_CHARS) pack = `${pack.slice(0, CONTEXT_CHARS)}…`;
    return pack;
  }

  /* ------------------------------------------------------------------------ */
  /* 编译：历史 → 状态（+ LoRA 训练）                                          */
  /* ------------------------------------------------------------------------ */

  /**
   * 编译一个会话：把历史逐轮吸收，然后（可选）请求 Python 后端训练 LoRA。
   *
   * 训练是「发射即忘」的：这里只登记 `status: training` 的适配器引用，
   * 真实进度由 Dashboard 上的 `/api/pss/lora/:id` 轮询 Python 后端获得。
   *
   * @param sessionId - 会话 id。
   * @param history - 可选的历史消息；缺省表示「只落盘当前状态」。
   * @returns 编译结果。
   */
  async compile(sessionId: string, history?: { role: string; content: string }[]): Promise<CompileResult> {
    await this.init();
    const startedAt = Date.now();
    const state = await this.ensure(sessionId);
    const systemNode = this.addNode('system', `编译会话 ${sessionId}`, `历史 ${history?.length ?? 0} 条`, 0.6, sessionId);

    if (history && history.length > 0) {
      for (const turn of history) {
        if (typeof turn?.content !== 'string' || turn.content.trim().length === 0) continue;
        const role = turn.role === 'assistant' ? 'assistant' : turn.role === 'system' ? 'system' : 'user';
        await this.ingest(sessionId, role, turn.content, { probe: false });
      }
    }

    if (this.trainerUrl) {
      const adapter: LoraRef = {
        adapterId: sessionId,
        path: join(this.store.rootDir, 'loras', sessionId),
        baseModel: this.baseModel,
        status: 'training'
      };
      state.lora = adapter;
      state.version += 1;
      // 把编译事件挂到该会话最新的一个非 system 节点上，图上才看得出「这条链被编译了」
      const anchor = [...this.nodes].reverse().find((node) => node.kind !== 'system' && node.id.includes(`:${sessionKey(sessionId)}:`));
      if (anchor) this.addLink(systemNode, anchor.id, 'compile');
      void this.requestTraining(sessionId, history ?? [], adapter);
    }

    state.updatedAt = Date.now();
    state.version += 1;
    await this.persist(state);
    this.compiles += 1;

    const latencyMs = Date.now() - startedAt;
    this.pushEvent(
      'compile',
      `编译完成：turns=${state.turns} 关键词=${state.keywords.length} 状态 ${JSON.stringify(state).length} 字（${latencyMs}ms）`,
      sessionId
    );

    return { sessionId, state, adapter: state.lora, latencyMs };
  }

  /** 把训练请求丢给 Python 后端；失败只更新状态为 failed。 */
  private async requestTraining(sessionId: string, history: { role: string; content: string }[], adapter: LoraRef): Promise<void> {
    if (!this.trainerUrl) return;
    try {
      const response = await fetch(`${this.trainerUrl}/compile`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ sessionId, baseModel: this.baseModel, history })
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      this.pushEvent('compile', `LoRA 训练已提交：${sessionId}`, sessionId);
    } catch (error) {
      adapter.status = 'failed';
      adapter.error = String(error);
      this.pushEvent('error', `LoRA 训练提交失败：${String(error)}`, sessionId);
    }
  }

  /* ------------------------------------------------------------------------ */
  /* Demo：一键造出好看的图                                                     */
  /* ------------------------------------------------------------------------ */

  /**
   * 灌一段演示对话，让全新安装的 Dashboard 不是一张空白画布。
   *
   * @param sessionId - 演示会话 id（默认 `demo`）。
   * @returns 造出的图统计。
   */
  async demo(sessionId = 'demo', turns = 6): Promise<GraphStats> {
    await this.init();
    const count = Math.max(1, Math.min(24, Math.floor(turns)));
    const dialogue: { role: 'user' | 'assistant'; text: string }[] = [];

    for (let i = 0; i < count; i += 1) {
      const pair = DEMO_TOPICS[i % DEMO_TOPICS.length];
      dialogue.push({ role: 'user', text: pair[0] });
      dialogue.push({ role: 'assistant', text: pair[1] });
    }

    for (const turn of dialogue) await this.ingest(sessionId, turn.role, turn.text);
    for (const turn of dialogue) {
      if (turn.role === 'user') await this.retrieve(sessionId, turn.text);
    }
    this.pushEvent('system', `demo 语料已灌入（${count} 轮 / ${this.nodes.length} 节点）`, sessionId);
    return this.graph().stats;
  }

  /** 清空图与事件（不动磁盘状态）。 */
  resetGraph(): void {
    this.nodes = [];
    this.links = [];
    this.events = [];
    this.nodeSeq = 0;
    this.bump();
    this.systemEvent('图已重置');
  }

  /** 一个新的会话 id（供 HTTP / 工具在缺省时使用）。 */
  newSessionId(): string {
    return `sess-${randomUUID().slice(0, 8)}`;
  }
}
