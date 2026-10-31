/**
 * 内置轻量 HTTP 服务：托管 Dashboard，并把 PSS 的图 / 事件 / 检索能力暴露成 REST。
 *
 * 为什么自己起一个 server 而不是挂进 DSH 的内部路由？
 * 因为这个项目要能被单独跑起来（`npm run dev` 就能看脑图），也要能被 DSH 当插件加载。
 * 用自己的 127.0.0.1 端口，这两种用法都是同一份代码，而且不依赖 DSH 版本里路由 API 的细节。
 *
 * 端点：
 *   GET  /                           → Dashboard 页面
 *   GET  /api/pss/health             → 健康检查
 *   GET  /api/pss/graph?since=N      → 图数据（since 命中当前版本时返回 unchanged）
 *   GET  /api/pss/events?after=ID     → 事件环形缓冲
 *   GET  /api/pss/sessions           → 已加载的会话状态列表
 *   GET  /api/pss/knowledge          → 知识库片段
 *   POST /api/pss/ingest             → 灌一段对话进状态
 *   POST /api/pss/retrieve           → 跑一次检索
 *   POST /api/pss/compile            → 编译会话（可选触发 LoRA 训练）
 *   POST /api/pss/demo               → 灌演示语料
 *   POST /api/pss/reset              → 清空图
 *
 * @module dsh-pss/http
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { PSSManager } from './PSSManager.js';

/** 本模块所在目录（用于定位 `../dashboard`）。 */
const HERE = fileURLToPath(new URL('.', import.meta.url));

/** 静态资源 MIME 表（够用即可，不引第三方）。 */
const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2'
};

/** 请求体大小上限（1MB），够 dashboard 用，也挡住了无脑灌数据。 */
const BODY_LIMIT = 1024 * 1024;

export interface DashboardServerOptions {
  /** PSS 运行时。 */
  manager: PSSManager;
  /** 首选端口，默认 8964；被占用时自动往后试 10 个。 */
  port?: number;
  /** 监听地址，默认 127.0.0.1（不暴露到局域网）。 */
  host?: string;
  /** Dashboard 静态目录，默认 `包根/dashboard`。 */
  dashboardDir?: string;
  /** 日志出口。 */
  log?: (message: string) => void;
}

/** 简单 JSON 响应。 */
function json(res: ServerResponse, status: number, payload: unknown): void {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'access-control-allow-origin': '*',
    'content-length': Buffer.byteLength(body)
  });
  res.end(body);
}

/** 读取并解析 JSON 请求体。 */
async function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = chunk as Buffer;
    size += buffer.length;
    if (size > BODY_LIMIT) throw new Error('request body too large');
    chunks.push(buffer);
  }
  if (chunks.length === 0) return {};
  const text = Buffer.concat(chunks).toString('utf8');
  if (text.trim().length === 0) return {};
  const parsed: unknown = JSON.parse(text);
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) throw new Error('body must be a JSON object');
  return parsed as Record<string, unknown>;
}

/** 取字符串参数（body 字段或 query 参数），空串按缺省处理。 */
function str(source: Record<string, unknown>, key: string): string | undefined {
  const value = source[key];
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : undefined;
}

/**
 * Dashboard HTTP 服务。
 *
 * 生命周期由调用方管理：插件里通过 `ctx.effect` 起停，`npm run dev` 里由 dev-server 起停。
 */
export class DashboardServer {
  private readonly manager: PSSManager;
  private readonly host: string;
  private readonly preferredPort: number;
  private readonly dashboardDir: string;
  private readonly log: (message: string) => void;
  private server: Server | null = null;
  private actualPort = 0;

  constructor(options: DashboardServerOptions) {
    this.manager = options.manager;
    this.host = options.host ?? '127.0.0.1';
    this.preferredPort = options.port ?? 8964;
    this.dashboardDir = options.dashboardDir ?? resolve(HERE, '..', 'dashboard');
    this.log = options.log ?? (() => undefined);
  }

  /** 服务地址；未启动时为空串。 */
  get url(): string {
    return this.server ? `http://${this.host}:${this.actualPort}` : '';
  }

  /** 实际监听端口。 */
  get port(): number {
    return this.actualPort;
  }

  /**
   * 启动服务。首选端口被占用时自动顺延，最多试 10 个。
   * @returns 可访问的地址。
   */
  async start(): Promise<string> {
    if (this.server) return this.url;

    const server = createServer((req, res) => {
      this.handle(req, res).catch((error: unknown) => {
        this.log(`[pss] 请求失败 ${req.method ?? '?'} ${req.url ?? '?'}: ${String(error)}`);
        if (!res.headersSent) json(res, 500, { error: String(error) });
        else res.end();
      });
    });
    server.on('clientError', (_error, socket) => socket.destroy());

    let lastError: unknown;
    for (let offset = 0; offset < 10; offset++) {
      const attempt = this.preferredPort + offset;
      try {
        await new Promise<void>((resolveListen, rejectListen) => {
          const onError = (error: NodeJS.ErrnoException) => {
            server.off('listening', onListening);
            rejectListen(error);
          };
          const onListening = () => {
            server.off('error', onError);
            resolveListen();
          };
          server.once('error', onError);
          server.once('listening', onListening);
          server.listen(attempt, this.host);
        });
        this.server = server;
        this.actualPort = attempt;
        this.log(`[pss] Dashboard 已启动：${this.url}`);
        return this.url;
      } catch (error) {
        lastError = error;
        if ((error as NodeJS.ErrnoException).code !== 'EADDRINUSE') break;
      }
    }
    throw new Error(`Dashboard 端口不可用（${this.preferredPort}..${this.preferredPort + 9}）：${String(lastError)}`);
  }

  /** 关闭服务（幂等）。 */
  async stop(): Promise<void> {
    const server = this.server;
    if (!server) return;
    this.server = null;
    await new Promise<void>((resolveClose) => server.close(() => resolveClose()));
    this.log('[pss] Dashboard 已停止');
  }

  /* ------------------------------------------------------------------------ */
  /* 路由                                                                      */
  /* ------------------------------------------------------------------------ */

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const method = req.method ?? 'GET';
    const url = new URL(req.url ?? '/', `http://${this.host}:${this.actualPort || this.preferredPort}`);
    const path = url.pathname;

    if (method === 'OPTIONS') {
      res.writeHead(204, {
        'access-control-allow-origin': '*',
        'access-control-allow-methods': 'GET,POST,OPTIONS',
        'access-control-allow-headers': 'content-type'
      });
      res.end();
      return;
    }

    if (path.startsWith('/api/pss/')) {
      await this.routeApi(method, path, url, req, res);
      return;
    }

    if (method !== 'GET' && method !== 'HEAD') {
      json(res, 405, { error: 'method not allowed' });
      return;
    }
    await this.serveStatic(path, res);
  }

  private async routeApi(
    method: string,
    path: string,
    url: URL,
    req: IncomingMessage,
    res: ServerResponse
  ): Promise<void> {
    const manager = this.manager;
    const query = Object.fromEntries(url.searchParams.entries());

    if (method === 'GET' && path === '/api/pss/health') {
      json(res, 200, {
        ok: true,
        name: 'dsh-pss',
        graphVersion: manager.graph().version,
        knowledge: manager.knowledgeChunks().length,
        sessions: manager.sessionList().length
      });
      return;
    }

    if (method === 'GET' && path === '/api/pss/graph') {
      const graph = manager.graph();
      const since = Number(url.searchParams.get('since') ?? 'NaN');
      if (Number.isFinite(since) && since === graph.version && url.searchParams.get('meta') !== '1') {
        json(res, 200, { unchanged: true, version: graph.version, stats: graph.stats });
        return;
      }
      json(res, 200, graph);
      return;
    }

    if (method === 'GET' && path === '/api/pss/events') {
      json(res, 200, { events: manager.eventList() });
      return;
    }

    if (method === 'GET' && path === '/api/pss/sessions') {
      json(res, 200, { sessions: manager.sessionList() });
      return;
    }

    if (method === 'GET' && path === '/api/pss/knowledge') {
      json(res, 200, { chunks: manager.knowledgeChunks() });
      return;
    }

    if (method === 'POST' && path === '/api/pss/ingest') {
      const body = await readJson(req);
      const sessionId = str(body, 'sessionId') ?? str(query, 'sessionId') ?? 'http';
      const text = str(body, 'text');
      if (!text) {
        json(res, 400, { error: 'text is required' });
        return;
      }
      const role = str(body, 'role') === 'assistant' ? 'assistant' : str(body, 'role') === 'system' ? 'system' : 'user';
      const state = await manager.ingest(sessionId, role, text);
      json(res, 200, { state });
      return;
    }

    if (method === 'POST' && path === '/api/pss/retrieve') {
      const body = await readJson(req);
      const sessionId = str(body, 'sessionId') ?? str(query, 'sessionId') ?? 'http';
      const text = str(body, 'query') ?? str(body, 'q');
      if (!text) {
        json(res, 400, { error: 'query is required' });
        return;
      }
      const result = await manager.retrieve(sessionId, text);
      json(res, 200, result);
      return;
    }

    if (method === 'POST' && path === '/api/pss/compile') {
      const body = await readJson(req);
      const sessionId = str(body, 'sessionId') ?? str(query, 'sessionId') ?? 'http';
      const history = Array.isArray(body.history)
        ? body.history
            .filter((item): item is { role: string; content: string } => {
              if (typeof item !== 'object' || item === null) return false;
              const record = item as Record<string, unknown>;
              return typeof record.role === 'string' && typeof record.content === 'string';
            })
            .map((item) => ({ role: item.role, content: item.content }))
        : undefined;
      const result = await manager.compile(sessionId, history);
      json(res, 200, result);
      return;
    }

    if (method === 'POST' && path === '/api/pss/demo') {
      const body = await readJson(req);
      const sessionId = str(body, 'sessionId') ?? 'demo';
      // turns 决定灌多少轮：演示用 12 轮刚好长出一片"星云"，测试用默认值也够
      const turns = typeof body.turns === 'number' ? body.turns : undefined;
      if (body.reset === true) manager.resetGraph();
      const stats = await manager.demo(sessionId, turns);
      json(res, 200, { stats, graph: manager.graph() });
      return;
    }

    if (method === 'POST' && path === '/api/pss/reset') {
      manager.resetGraph();
      json(res, 200, { ok: true, graph: manager.graph() });
      return;
    }

    json(res, 404, { error: `unknown endpoint ${method} ${path}` });
  }

  /** 静态文件（只服务 dashboard 目录内的文件，防目录穿越）。 */
  private async serveStatic(path: string, res: ServerResponse): Promise<void> {
    const relative = path === '/' ? 'index.html' : decodeURIComponent(path).replace(/^\/+/, '');
    const target = resolve(this.dashboardDir, normalize(relative));
    if (!target.startsWith(resolve(this.dashboardDir))) {
      json(res, 403, { error: 'forbidden' });
      return;
    }

    let body: Buffer;
    try {
      body = await readFile(target);
    } catch {
      json(res, 404, { error: `not found: ${path}` });
      return;
    }

    res.writeHead(200, {
      'content-type': MIME[extname(target).toLowerCase()] ?? 'application/octet-stream',
      'cache-control': 'no-cache',
      'content-length': body.length
    });
    res.end(body);
  }
}

/** 便捷函数：起一个服务并返回实例。 */
export async function startDashboard(options: DashboardServerOptions): Promise<DashboardServer> {
  const server = new DashboardServer(options);
  await server.start();
  return server;
}

/** 供测试使用的路径工具：把相对路径解析到 dashboard 目录。 */
export function dashboardPath(dir: string, file: string): string {
  return join(dir, file);
}
