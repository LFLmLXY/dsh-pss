/**
 * dsh-pss —— 核心类型定义
 *
 * 这个文件是整个项目的「契约层」：后端运行时（PSSManager）、DSH 插件入口（index.ts）、
 * HTTP 路由（http.ts）和前端 Dashboard（dashboard/index.html）共用同一套结构。
 *
 * 设计上刻意把「状态」和「探针」分开：
 *  - {@link PSSState}   —— 会被落盘的东西。**长度恒定**，与对话轮数无关（O(1)）。
 *  - {@link GraphData}  —— 只活在内存里的可视化数据。会随检索次数增长，但有上限裁剪。
 *
 * @module dsh-pss/types
 */

/* -------------------------------------------------------------------------- */
/* 可视化图结构（探针的输出）                                                    */
/* -------------------------------------------------------------------------- */

/**
 * 图节点类别。Dashboard 用类别决定颜色，所以这里是「视觉语义」的唯一来源：
 * - `user`      用户输入 / 提问 —— 红
 * - `intent`    PSS 生成的参数化检索意图 —— 青
 * - `knowledge` RAG 命中到的知识片段 —— 黄
 * - `answer`    合成后回喂给模型的上下文包 —— 蓝
 * - `system`    生命周期事件（会话创建、编译） —— 灰
 */
export type NodeKind = 'user' | 'intent' | 'knowledge' | 'answer' | 'system';

/** 全部节点类别的稳定顺序（前端做图例、后端做统计都按它来）。 */
export const NODE_KINDS: readonly NodeKind[] = ['user', 'intent', 'knowledge', 'answer', 'system'] as const;

/** 连线语义。区分它是「参数化」「检索」「合成」还是「编译」产生的关系。 */
export type LinkKind = 'parametrize' | 'retrieve' | 'compose' | 'compile';

/** 脑图节点。id 由后端按 `前缀:会话:序号` 生成，稳定且可预测（前端靠它做增量合并）。 */
export interface GraphNode {
  /** 全局唯一 id，如 `i:sess-42:7`。 */
  id: string;
  /** 节点上显示的文字（已截断）。 */
  label: string;
  /** 视觉类别。 */
  kind: NodeKind;
  /** 创建时间戳（ms），前端用它做「新节点脉冲」动画。 */
  t: number;
  /** 0..1 的视觉权重：影响节点大小 / 发光强度（检索得分、关键词权重）。 */
  weight: number;
  /** 悬浮或侧栏里的详细信息（未经截断的原文、来源文件名、得分等）。 */
  detail?: string;
}

/** 脑图连线。 */
export interface GraphLink {
  id: string;
  source: string;
  target: string;
  kind: LinkKind;
  t: number;
}

/** 前端 HUD 面板需要的全局统计。 */
export interface GraphStats {
  nodes: number;
  links: number;
  byKind: Record<NodeKind, number>;
  /** 已登记的参数化会话数。 */
  sessions: number;
  /** 累计检索次数。 */
  retrievals: number;
  /** 累计编译次数（会话 → 参数状态）。 */
  compiles: number;
  /** 进程启动至今的毫秒数，前端显示 uptime。 */
  uptimeMs: number;
}

/** `GET /api/pss/graph` 的响应体。 */
export interface GraphData {
  /** 每次图变更自增；前端可据此跳过无变化的渲染。 */
  version: number;
  nodes: GraphNode[];
  links: GraphLink[];
  stats: GraphStats;
}

/* -------------------------------------------------------------------------- */
/* 参数化会话状态（会落盘的东西）                                                */
/* -------------------------------------------------------------------------- */

/** 关键词及其权重（随轮次衰减，保证集合大小恒定）。 */
export interface Keyword {
  term: string;
  weight: number;
}

/** LoRA 适配器的引用信息（可能只存在于 Python 训练后端）。 */
export interface LoraRef {
  /** 适配器标识，通常等于 sessionId。 */
  adapterId: string;
  /** 权重目录（相对或绝对路径）。 */
  path: string;
  /** 训练基座模型名。 */
  baseModel: string;
  status: 'training' | 'ready' | 'failed';
  trainedAt?: number;
  /** status 为 failed 时的错误摘要。 */
  error?: string;
}

/**
 * 参数化会话状态 —— PSS 的「唯一真相」。
 *
 * 关键性质：**序列化后的字节数与对话长度无关**。
 * - `centroid` 是固定 `dim` 维的哈希嵌入向量（默认 32 维）；
 * - `keywords` 最多 `MAX_KEYWORDS` 条；
 * - 其余全是标量计数。
 *
 * 因此内存里放多少个会话、磁盘上存多少轮对话，占用都是常数级。
 */
export interface PSSState {
  sessionId: string;
  /** 状态版本号，每次 ingest/compile 自增，用于前端判断新鲜度。 */
  version: number;
  createdAt: number;
  updatedAt: number;
  /** 累计吸收的对话轮数。 */
  turns: number;
  /** 累计吸收的字符数（用于展示「历史多长，状态多短」的压缩比）。 */
  ingestedChars: number;
  /** 嵌入维度。 */
  dim: number;
  /** 归一化的意图中心向量，长度恒为 `dim`。 */
  centroid: number[];
  keywords: Keyword[];
  /** 当前绑定的 LoRA 适配器（没有则为 null）。 */
  lora: LoraRef | null;
}

/** 知识库片段（RAG 语料的最小单位）。 */
export interface KnowledgeChunk {
  id: string;
  /** 来源标签：文件名或 `session:<id>`。 */
  source: string;
  text: string;
}

/**
 * 会话记忆片段。
 *
 * 知识库（{@link KnowledgeChunk}）是"世界知识"，装进来就不变；
 * 记忆片段是"这场会话里说过什么、定过什么"，随对话增长。
 * 两者分开存、分开建索引，检索时再合并——因为它们的生命周期完全不同：
 * 知识库跟着包走，记忆片段跟着会话落盘。
 */
export interface MemoryChunk {
  id: string;
  sessionId: string;
  /** 说话的角色；`digest` 表示模型主动记下的一条结论。 */
  role: 'user' | 'assistant' | 'system' | 'digest';
  text: string;
  /** 写入时间戳（ms）。 */
  t: number;
}

/** 一条检索命中。 */
export interface RetrievalHit {
  chunkId: string;
  source: string;
  text: string;
  /** BM25 得分，已归一化到 0..1 供前端画大小。 */
  score: number;
  /** 命中来自哪个通道：世界知识库，还是本会话的记忆片段。 */
  channel: 'knowledge' | 'memory';
}

/** `pss_retrieve` 工具 / `POST /api/pss/retrieve` 的返回。 */
export interface RetrieveResult {
  sessionId: string;
  /** 模型原始提问。 */
  query: string;
  /** PSS 参数化后真正拿去找知识的检索词。 */
  intentQuery: string;
  /** 参与检索的关键词（带权重，按权重降序）。 */
  keywords: string[];
  hits: RetrievalHit[];
  /** 拼好准备回喂给模型的上下文文本。 */
  contextPack: string;
  /** 本次检索耗时（ms）。 */
  latencyMs: number;
  /** 意图生成走的是哪条路：`local` 还是 `lora:<url>`。 */
  intentSource: string;
  /** 生成这份结果时，会话状态的版本号。 */
  stateVersion: number;
}

/** 编译（会话历史 → 参数状态）的结果。 */
export interface CompileResult {
  sessionId: string;
  state: PSSState;
  /** 本次编译是否真的产出了 LoRA 适配器。 */
  adapter: LoraRef | null;
  /** 编译耗时（ms）。 */
  latencyMs: number;
}

/* -------------------------------------------------------------------------- */
/* 事件流（HUD 侧边栏）                                                         */
/* -------------------------------------------------------------------------- */

export type PSSEventKind = 'session' | 'ingest' | 'retrieve' | 'compile' | 'error' | 'system';

/** 环形缓冲里的一条事件，前端右侧滚动列表直接渲染。 */
export interface PSSEvent {
  id: string;
  t: number;
  kind: PSSEventKind;
  message: string;
  sessionId?: string;
}

/* -------------------------------------------------------------------------- */
/* 可插拔的意图后端                                                             */
/* -------------------------------------------------------------------------- */

/** 意图生成的输入：当前提问 + 参数化状态。 */
export interface IntentInput {
  query: string;
  state: PSSState;
}

/** 意图生成的输出。 */
export interface IntentOutput {
  /** 真正用来检索的句子。 */
  intentQuery: string;
  keywords: string[];
  /** 来源标识，用于在 Dashboard 上区分「本地启发式」和「LoRA 参数化」。 */
  source: string;
}

/**
 * 意图后端接口。
 *
 * 默认实现（{@link LocalIntentBackend}，见 PSSManager.ts）是纯本地的关键词扩展，
 * 保证 `npm i` 之后不装 Python、不装 GPU 也能跑通全链路。
 * 把 `intentBackend` 配置成 Python 服务的 URL，就会切到真正的 LoRA 参数化路径。
 */
export interface IntentBackend {
  readonly name: string;
  generate(input: IntentInput): Promise<IntentOutput>;
}

/* -------------------------------------------------------------------------- */
/* 运行时配置                                                                   */
/* -------------------------------------------------------------------------- */

/** {@link PSSManager} 的构造参数。 */
export interface PSSManagerOptions {
  /** 状态落盘根目录（默认 `~/.dsh/pss`）。 */
  root: string;
  /** 嵌入维度，默认 32。 */
  dim?: number;
  /** 检索返回条数，默认 3。 */
  topK?: number;
  /** 内存图节点上限，默认 400（超出后按 FIFO 裁剪，防止长跑会话吃内存）。 */
  maxNodes?: number;
  /** 事件环形缓冲长度，默认 50。 */
  maxEvents?: number;
  /** 知识库目录（`*.md` / `*.txt`），可选。 */
  knowledgeDir?: string;
  /** 额外注入的知识片段（便于测试和外部喂数据）。 */
  knowledge?: KnowledgeChunk[];
  /** 意图后端；缺省用本地启发式。 */
  intentBackend?: IntentBackend;
  /** Python 训练后端地址（如 `http://127.0.0.1:8077`）；缺省则不训练 LoRA。 */
  trainerUrl?: string;
  /** 训练用基座模型名，透传给 Python 后端，默认 `deepseek-chat`。 */
  baseModel?: string;
  /** 日志出口，默认静默（DSH 里由 index.ts 注入 ctx.logger）。 */
  log?: (message: string) => void;
}
