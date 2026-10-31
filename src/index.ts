/**
 * dsh-pss —— DSH 插件入口。
 *
 * 这个文件只做四件事，业务逻辑全在 {@link PSSManager} 里：
 *  1. 按配置装配运行时（意图后端：本地 / LoRA / 带降级）；
 *  2. 注册模型工具 `pss_retrieve`；
 *  3. 挂 `session/created`、`session/event`、`session/disposed`，把真实对话喂进 PSS；
 *  4. 用 `ctx.effect` 起停内置 Dashboard HTTP 服务（插件卸载时自动关闭）。
 *
 * 所有对 DSH 内部结构的访问都写了防御性判断：DSH 版本升级导致事件字段变化时，
 * 插件最多是「探针不工作」，不会把宿主会话搞崩。
 *
 * @module dsh-pss
 */

import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import z from '@deepseek-ai/schemastery';
import { defineTool } from '@deepseek-ai/dsh-tools';
import type { Context } from '@deepseek-ai/cordis';
import { PSSManager, defaultRoot } from './PSSManager.js';
import { DashboardServer } from './http.js';
import { FallbackIntentBackend, LocalIntentBackend, LoraIntentBackend } from './intent.js';
import type { IntentBackend } from './types.js';

/** 插件在 Loader 里的稳定标识。 */
export const name = 'dsh-pss';

/** 依赖的服务：只依赖工具注册表，其余全部可选。 */
export const inject = ['tools'];

/**
 * 包内自带的语料目录。
 *
 * 装到 profile 之后，包目录就是 profile 的 `node_modules/dsh-pss`，
 * 所以「自带的 knowledge/」必须相对模块自身解析，而不是相对 cwd。
 * 解析不到（例如被 tsx 直接从 src/ 跑）就退回内置种子语料。
 */
function packageKnowledgeDir(): string | undefined {
  const dir = fileURLToPath(new URL('../knowledge', import.meta.url));
  return existsSync(dir) ? dir : undefined;
}

/** 插件配置。 */
export interface Config {
  /** 总开关。 */
  enabled: boolean;
  /** 参数化状态的落盘根目录。 */
  root: string;
  /** Dashboard 端口（被占用时自动顺延）。 */
  port: number;
  /** 是否随插件启动 Dashboard。 */
  autoStart: boolean;
  /** Dashboard 静态目录（默认用包内 `dashboard/`）。 */
  dashboardDir: string;
  /** 嵌入维度（定长向量的长度）。 */
  dim: number;
  /** 每次检索回填的知识片段条数。 */
  topK: number;
  /** 内存图节点上限。 */
  maxNodes: number;
  /** 知识库目录（`*.md` / `*.txt`）。 */
  knowledgeDir: string;
  /** 意图后端类型：`local` 直接可用，`lora` 需要 Python 服务。 */
  intentBackend: 'local' | 'lora';
  /** LoRA 意图服务地址。 */
  intentBackendUrl: string;
  /** Python 训练服务地址；留空则不训练。 */
  trainerUrl: string;
  /** 训练基座模型名。 */
  baseModel: string;
  /** 是否把宿主会话的真实对话自动吸收进 PSS。 */
  autoIngest: boolean;
  /** 会话销毁时是否自动编译（落盘 + 触发训练）。 */
  compileOnDispose: boolean;
  /** 启动时是否灌一份演示语料（让 Dashboard 第一眼就有图）。 */
  seedDemo: boolean;
}

/** 校验后的配置（Loader 依据这份 schema 渲染设置面板）。 */
export const Config: z<Config> = z.object({
  enabled: z.boolean().default(true).description('总开关'),
  root: z.string().default(defaultRoot()).description('参数状态落盘目录'),
  port: z.number().step(1).min(1024).max(65535).default(8964).description('Dashboard 端口'),
  autoStart: z.boolean().default(true).description('随插件启动 Dashboard'),
  dashboardDir: z.string().default('').description('Dashboard 静态目录（留空用包内 dashboard/）'),
  dim: z.number().step(1).min(8).max(256).default(32).description('嵌入维度'),
  topK: z.number().step(1).min(1).max(20).default(3).description('回填片段数'),
  maxNodes: z.number().step(1).min(40).max(4000).default(400).description('内存图节点上限'),
  knowledgeDir: z.string().default('').description('知识库目录（*.md / *.txt）'),
  intentBackend: z.union([z.const('local'), z.const('lora')]).default('local').description('意图后端'),
  intentBackendUrl: z.string().default('http://127.0.0.1:8077').description('LoRA 意图服务地址'),
  trainerUrl: z.string().default('').description('Python 训练服务地址（留空不训练）'),
  baseModel: z.string().default('deepseek-chat').description('训练基座模型'),
  autoIngest: z.boolean().default(true).description('自动吸收宿主会话对话'),
  compileOnDispose: z.boolean().default(true).description('会话销毁时自动编译'),
  seedDemo: z.boolean().default(true).description('启动时灌演示语料')
});

/* -------------------------------------------------------------------------- */
/* 事件与内容的防御性解析                                                       */
/* -------------------------------------------------------------------------- */

/** 从模型消息的 `content` 里抽出纯文本（兼容字符串与 content block 数组）。 */
function extractText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (typeof block === 'string') {
      parts.push(block);
      continue;
    }
    if (typeof block !== 'object' || block === null) continue;
    const record = block as Record<string, unknown>;
    if (record.type === 'text' && typeof record.text === 'string') parts.push(record.text);
  }
  return parts.join('\n');
}

/** 宽松的会话事件视图：只声明我们真正会读的字段。 */
interface LooseSessionEvent {
  type?: string;
  data?: {
    content?: unknown;
    message?: { content?: unknown };
  };
}

/** 宽松的会话视图。 */
interface LooseSession {
  id?: string;
  snapshotEvents?: () => readonly LooseSessionEvent[];
}

/** 把会话事件日志翻成交给编译器的历史（只取最后 40 条消息，避免灌爆内存）。 */
function historyOf(session: LooseSession): { role: string; content: string }[] {
  const events = typeof session.snapshotEvents === 'function' ? session.snapshotEvents() : [];
  const history: { role: string; content: string }[] = [];
  for (const event of events) {
    if (event.type === 'user/message') {
      const content = extractText(event.data?.content).trim();
      if (content.length > 0) history.push({ role: 'user', content });
    } else if (event.type === 'assistant/message') {
      const content = extractText(event.data?.message?.content).trim();
      if (content.length > 0) history.push({ role: 'assistant', content });
    }
  }
  return history.slice(-40);
}

/* -------------------------------------------------------------------------- */
/* apply                                                                       */
/* -------------------------------------------------------------------------- */

/**
 * 插件装配入口。
 * @param ctx - 插件上下文（`inject: ['tools']` 保证 `ctx.tools` 可用）。
 * @param config - 校验后的配置。
 */
export function apply(ctx: Context, config: Config): void {
  if (!config.enabled) return;

  // DSH 的日志服务不是硬依赖，拿不到就静默（插件不能因为没日志就罢工）
  const logger = (ctx as unknown as { logger?: { info?: (message: string) => void; warn?: (message: string) => void } }).logger;
  const log = (message: string): void => {
    try {
      logger?.info?.(message);
    } catch {
      // 日志失败无关紧要
    }
  };

  /* --- 意图后端：local / lora（带本地降级） ------------------------------- */
  let intentBackend: IntentBackend = new LocalIntentBackend();
  if (config.intentBackend === 'lora') {
    intentBackend = new FallbackIntentBackend(
      new LoraIntentBackend(config.intentBackendUrl),
      new LocalIntentBackend(),
      (error) => log(`[pss] LoRA 意图后端不可用，降级本地启发式：${String(error)}`)
    );
  }

  const manager = new PSSManager({
    root: config.root,
    dim: config.dim,
    topK: config.topK,
    maxNodes: config.maxNodes,
    knowledgeDir: config.knowledgeDir || packageKnowledgeDir(),
    trainerUrl: config.trainerUrl || undefined,
    baseModel: config.baseModel,
    intentBackend,
    log
  });

  /* --- 工具：pss_retrieve ------------------------------------------------- */
  ctx.tools.register(
    defineTool({
      name: 'pss_retrieve',
      description:
        'Parametric Session State retrieval. Rewrites the question into a parametric retrieval intent with this session\'s compiled state, then returns the small context pack that should be folded back into the conversation. Use it on long sessions to recover facts that scrolled out of the context window instead of re-reading the history.',
      parameters: {
        query: {
          type: 'string',
          required: true,
          description: 'The question or topic to recall from this session and the knowledge base.'
        },
        sessionId: {
          type: 'string',
          description: 'Session whose compiled state to use. Defaults to the calling session.'
        }
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            sessionId: { type: 'string', required: true },
            query: { type: 'string', required: true },
            intentQuery: { type: 'string', required: true },
            keywords: { type: 'array', required: true, items: { type: 'string' } },
            hits: {
              type: 'array',
              required: true,
              items: {
                type: 'object',
                additionalProperties: false,
                properties: {
                  chunkId: { type: 'string', required: true },
                  source: { type: 'string', required: true },
                  text: { type: 'string', required: true },
                  score: { type: 'number', required: true }
                }
              }
            },
            contextPack: { type: 'string', required: true },
            latencyMs: { type: 'number', required: true },
            intentSource: { type: 'string', required: true },
            stateVersion: { type: 'number', required: true }
          }
        },
        render: (_args, value) => [
          {
            type: 'text',
            text: `${value.contextPack}\n\n[PSS] ${value.hits.length} hit(s) via ${value.intentSource} in ${value.latencyMs}ms`
          }
        ]
      },
      async execute(args, exec) {
        const callingSession = (exec as unknown as { agent?: { session?: LooseSession } }).agent?.session?.id;
        const sessionId = args.sessionId?.trim() || callingSession || manager.newSessionId();
        const result = await manager.retrieve(sessionId, args.query);
        return {
          sessionId: result.sessionId,
          query: result.query,
          intentQuery: result.intentQuery,
          keywords: result.keywords,
          hits: result.hits.map((hit) => ({
            chunkId: hit.chunkId,
            source: hit.source,
            text: hit.text,
            score: hit.score
          })),
          contextPack: result.contextPack,
          latencyMs: result.latencyMs,
          intentSource: result.intentSource,
          stateVersion: result.stateVersion
        };
      }
    })
  );

  /* --- Dashboard --------------------------------------------------------- */
  ctx.effect(() => {
    const server = new DashboardServer({
      manager,
      port: config.port,
      log,
      ...(config.dashboardDir ? { dashboardDir: config.dashboardDir } : {})
    });

    if (config.autoStart) {
      server
        .start()
        .then((url) => {
          log(`[pss] 可视化脑图：${url}`);
          if (config.seedDemo) {
            void manager.demo().then((stats) => log(`[pss] 演示语料就绪：${stats.nodes} 节点 / ${stats.links} 连线`));
          }
        })
        .catch((error: unknown) => log(`[pss] Dashboard 启动失败：${String(error)}`));
    } else {
      void manager.init();
    }

    return () => {
      void server.stop();
    };
  });

  /* --- 会话钩子：真实对话 → 参数状态 ------------------------------------- */
  if (config.autoIngest) {
    ctx.effect(() =>
      ctx.on('session/event', (session, event) => {
        try {
          const loose = session as unknown as LooseSession;
          const sessionId = loose.id;
          if (!sessionId) return;
          const looseEvent = event as unknown as LooseSessionEvent;
          if (looseEvent.type === 'user/message') {
            const text = extractText(looseEvent.data?.content).trim();
            if (text.length > 0 && text.length < 20000) void manager.ingest(sessionId, 'user', text);
          } else if (looseEvent.type === 'assistant/message') {
            const text = extractText(looseEvent.data?.message?.content).trim();
            if (text.length > 0 && text.length < 20000) void manager.ingest(sessionId, 'assistant', text);
          }
        } catch (error) {
          log(`[pss] 吸收会话事件失败：${String(error)}`);
        }
      })
    );
  }

  if (config.compileOnDispose) {
    ctx.effect(() =>
      ctx.on('session/disposed', (session) => {
        const loose = session as unknown as LooseSession;
        const sessionId = loose.id;
        if (!sessionId) return;
        void manager
          .compile(sessionId, historyOf(loose))
          .then((result) => log(`[pss] 会话 ${sessionId} 已编译：turns=${result.state.turns}`))
          .catch((error: unknown) => log(`[pss] 编译失败：${String(error)}`));
      })
    );
  }
}
