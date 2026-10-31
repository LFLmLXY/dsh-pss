#!/usr/bin/env node
/**
 * dsh-pss 自检脚本（`npm run smoke`）。
 *
 * 这里跑的是「整条链路」，不是玩具断言：
 *
 *   A. 运行时底座   —— 分词/嵌入、状态落盘、意图后端
 *   B. PSS 运行时   —— 吸收 → 检索 → 编译，以及探针画出来的图是否自洽
 *   C. 插件契约     —— 用 mock ctx 真正调用 `apply()`，注册工具、喂会话事件、触发编译
 *   D. HTTP 端到端  —— 打插件自己起的 Dashboard 服务，验证前端与 REST 都活着
 *
 * 依赖：只需要 `npm run build` 产物，不需要 DSH 进程、不需要 Python、不需要网络。
 * 退出码非 0 = 有断言挂了。
 */

import assert from 'node:assert/strict';
import { mkdtemp, readdir, readFile, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const PORT = 18964;   // 刻意选一个不常用的端口，避免撞上正在跑的 dev server

/* -------------------------------------------------------------------------- */
/* 迷你测试框架                                                                */
/* -------------------------------------------------------------------------- */

let passed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed += 1;
    console.log(`  \x1b[32m✓\x1b[0m ${name}`);
  } catch (error) {
    failures.push({ name, error });
    console.log(`  \x1b[31m✗\x1b[0m ${name}`);
    console.log(`    ${String(error?.message ?? error).split('\n').join('\n    ')}`);
  }
}

function section(title) {
  console.log(`\n\x1b[36m${title}\x1b[0m`);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitFor(check, { timeoutMs = 6000, intervalMs = 60 } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await check();
    if (value) return value;
    await sleep(intervalMs);
  }
  throw new Error(`waitFor 超时（${timeoutMs}ms）`);
}

/* -------------------------------------------------------------------------- */
/* 加载构建产物                                                                */
/* -------------------------------------------------------------------------- */

if (!existsSync(join(ROOT, 'dist', 'index.js'))) {
  console.error('找不到 dist/ —— 先跑 `npm run build`');
  process.exit(1);
}

const embed = await import(new URL('../dist/embed.js', import.meta.url).href);
const { StateStore } = await import(new URL('../dist/store.js', import.meta.url).href);
const intent = await import(new URL('../dist/intent.js', import.meta.url).href);
const { PSSManager } = await import(new URL('../dist/PSSManager.js', import.meta.url).href);
const { DashboardServer } = await import(new URL('../dist/http.js', import.meta.url).href);
const plugin = await import(new URL('../dist/index.js', import.meta.url).href);

const workDir = await mkdtemp(join(tmpdir(), 'dsh-pss-smoke-'));
const cleanups = [];
const cleanup = async () => {
  for (const fn of cleanups.reverse()) {
    try {
      await fn();
    } catch {
      // 清理失败不该让测试变红
    }
  }
  await rm(workDir, { recursive: true, force: true });
};

/* -------------------------------------------------------------------------- */
/* A. 底座                                                                     */
/* -------------------------------------------------------------------------- */

section('A · 运行时底座');

await test('tokenize 切出拉丁词 + 中文二元组', () => {
  const tokens = embed.tokenize('PSS 用 LoRA 生成检索词，保持 O(1) 存储');
  assert.ok(tokens.length > 4, `token 太少：${JSON.stringify(tokens)}`);
  assert.ok(tokens.includes('lora'), '缺少小写化的拉丁词');
  assert.ok(tokens.some((t) => t.length === 2 && /[\u4e00-\u9fff]/.test(t)), '缺少中文二元组');
});

await test('hashToken 稳定且落在 32 位无符号范围', () => {
  const a = embed.hashToken('参数化');
  const b = embed.hashToken('参数化');
  assert.equal(a, b);
  assert.ok(Number.isInteger(a) && a >= 0 && a <= 0xffffffff, `越界：${a}`);
});

await test('嵌入向量定长、归一化后自相似度为 1', () => {
  const vec = new Array(32).fill(0);
  embed.addEmbedding(vec, embed.tokenize('把对话折进定长向量'), 1);
  assert.equal(vec.length, 32);
  embed.normalize(vec);
  const norm = Math.sqrt(vec.reduce((acc, v) => acc + v * v, 0));
  assert.ok(Math.abs(norm - 1) < 1e-9, `模长不是 1：${norm}`);
  assert.ok(Math.abs(embed.cosine(vec, vec) - 1) < 1e-9);
});

await test('BM25 能按相关度排序并排除指定下标', () => {
  const docs = [
    embed.tokenize('PSS 用定长向量做恒定存储'),
    embed.tokenize('LoRA 训练在会话结束时进行'),
    embed.tokenize('可视化探针画出记忆流转')
  ];
  const index = new embed.Bm25Index(docs);
  const hits = index.search(embed.tokenize('定长向量存储'), 3);
  assert.ok(hits.length > 0, '没有命中');
  assert.equal(hits[0].index, 0, '最相关的应该是第 0 篇');
  const excluded = index.search(embed.tokenize('定长向量存储'), 3, new Set([0]));
  assert.ok(excluded.every((hit) => hit.index !== 0), 'exclude 没生效');
});

await test('StateStore 原子写 + 读回一致', async () => {
  const store = new StateStore(join(workDir, 'state'));
  await store.ensure();
  const state = {
    sessionId: 'unit',
    version: 3,
    createdAt: 1,
    updatedAt: 2,
    turns: 5,
    ingestedChars: 42,
    dim: 8,
    centroid: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    keywords: [{ term: 'pss', weight: 1.2 }],
    lora: null
  };
  await store.save(state);
  const loaded = await store.load('unit');
  // load 会补上格式版本号，其余字段必须一字不差
  assert.equal(loaded.format, 1, 'load 没有补 format 版本号');
  const { format: _format, ...rest } = loaded;
  assert.deepEqual(rest, state);
  const files = await readdir(join(workDir, 'state', 'sessions'));
  assert.ok(files.includes('unit.json'), `落盘文件名不对：${files.join(',')}`);
  const list = await store.list();
  assert.ok(list.includes('unit'), `list 里没有 unit：${list.join(',')}`);
});

await test('LocalIntentBackend 产出可用的检索意图', async () => {
  const backend = new intent.LocalIntentBackend();
  assert.equal(backend.name, 'local');
  const output = await backend.generate({
    query: 'PSS 怎么做到恒定存储？',
    state: { sessionId: 'unit', keywords: [{ term: '参数化', weight: 2 }], turns: 3 }
  });
  assert.ok(output.intentQuery.length > 0);
  assert.ok(Array.isArray(output.keywords) && output.keywords.length > 0);
});

await test('FallbackIntentBackend 在主后端失败时降级', async () => {
  const broken = { name: 'broken', generate: async () => { throw new Error('boom'); } };
  const backend = new intent.FallbackIntentBackend(broken, new intent.LocalIntentBackend());
  const output = await backend.generate({ query: '降级测试', state: { keywords: [], sessionId: 'x', turns: 0 } });
  assert.ok(output.source.includes('fallback'), `source 没标记降级：${output.source}`);
});

/* -------------------------------------------------------------------------- */
/* B. PSS 运行时                                                               */
/* -------------------------------------------------------------------------- */

section('B · PSS 运行时与探针');

const manager = new PSSManager({
  root: join(workDir, 'pss'),
  knowledgeDir: join(ROOT, 'knowledge'),
  dim: 32,
  topK: 3,
  maxNodes: 120
});
cleanups.push(() => manager.store);

await test('init 加载知识库（内置种子 + knowledge/ 目录）', async () => {
  await manager.init();
  const chunks = manager.knowledgeChunks();
  assert.ok(chunks.length >= 8, `知识片段太少：${chunks.length}`);
  assert.ok(chunks.every((chunk) => chunk.id && chunk.text), '知识片段缺字段');
});

await test('ingest 把对话折进定长状态', async () => {
  const before = manager.peek('smoke');
  assert.equal(before, null);
  const state = await manager.ingest('smoke', 'user', 'PSS 怎么做到 O(1) 存储？');
  assert.equal(state.turns, 1);
  assert.equal(state.centroid.length, 32, '状态向量长度必须等于 dim');
  assert.ok(state.keywords.length > 0, '关键词表是空的');
  assert.ok(state.version > 0);
  const again = await manager.ingest('smoke', 'assistant', '把历史折成定长向量，磁盘只存状态。');
  assert.equal(again.turns, 2);
});

await test('retrieve 返回意图 + 命中 + 上下文包', async () => {
  const result = await manager.retrieve('smoke', '参数化意图是怎么工作的？');
  assert.ok(result.intentQuery.length > 0);
  assert.ok(result.hits.length > 0, '没有命中任何知识片段');
  assert.ok(result.contextPack.includes('[PSS]'), '上下文包缺少头部标记');
  assert.ok(result.contextPack.length <= 1500, `上下文包超长：${result.contextPack.length}`);
  assert.ok(result.latencyMs >= 0);
  assert.ok(result.hits.every((hit) => hit.score >= 0 && hit.score <= 1), '分数没归一化');
});

await test('探针画出的图结构自洽（节点类别 / 连线端点 / 版本号）', async () => {
  const graph = manager.graph();
  assert.ok(graph.nodes.length >= 5, `节点太少：${graph.nodes.length}`);
  assert.ok(graph.links.length >= 3, `连线太少：${graph.links.length}`);

  const kinds = new Set(graph.nodes.map((node) => node.kind));
  for (const kind of ['user', 'intent', 'knowledge', 'answer']) {
    assert.ok(kinds.has(kind), `缺少 ${kind} 节点`);
  }

  const ids = new Set(graph.nodes.map((node) => node.id));
  for (const link of graph.links) {
    assert.ok(ids.has(link.source), `悬空边 source：${link.source}`);
    assert.ok(ids.has(link.target), `悬空边 target：${link.target}`);
  }

  const linkKinds = new Set(graph.links.map((link) => link.kind));
  assert.ok(linkKinds.has('parametrize'), '缺少「提问 → 意图」连线');
  assert.ok(linkKinds.has('retrieve'), '缺少「意图 → 知识」连线');
  assert.ok(linkKinds.has('compose'), '缺少「知识 → 上下文包」连线');

  assert.ok(graph.stats.nodes === graph.nodes.length);
  assert.ok(graph.stats.byKind.user > 0);
  assert.ok(graph.version > 0, '版本号没增长');
});

await test('compile 落盘状态并递增编译计数', async () => {
  const result = await manager.compile('smoke', [
    { role: 'user', content: '可视化探针记录什么？' },
    { role: 'assistant', content: '用户输入、意图、知识片段、上下文包，以及它们之间的因果连线。' }
  ]);
  assert.ok(result.state.turns >= 3, `编译后 turns 不对：${result.state.turns}`);
  const path = join(workDir, 'pss', 'sessions', 'smoke.json');
  assert.ok(existsSync(path), '状态没落盘');
  const saved = JSON.parse(await readFile(path, 'utf8'));
  assert.equal(saved.sessionId, 'smoke');
  assert.ok(manager.graph().stats.compiles >= 1);
});

await test('demo 能造出一张像样的图', async () => {
  const stats = await manager.demo('demo');
  assert.ok(stats.nodes >= 12, `演示图节点太少：${stats.nodes}`);
  assert.ok(stats.byKind.intent >= 1 && stats.byKind.knowledge >= 1);
});

await test('resetGraph 清空图但不影响磁盘状态', async () => {
  manager.resetGraph();
  const graph = manager.graph();
  assert.equal(graph.nodes.length, 0);
  assert.equal(graph.links.length, 0);
  assert.ok(manager.knowledgeChunks().length > 0, '知识库被误清空了');
});

/* -------------------------------------------------------------------------- */
/* C. 插件契约（mock ctx）                                                     */
/* -------------------------------------------------------------------------- */

section('C · 插件契约（mock ctx 真调 apply）');

/** 极简 ctx：只实现插件真正用到的那几个面。 */
function makeCtx() {
  const toolList = [];
  const handlers = new Map();
  const disposers = [];
  const ctx = {
    tools: {
      register(tool) {
        toolList.push(tool);
        return () => undefined;
      }
    },
    effect(fn) {
      const disposer = fn();
      if (typeof disposer === 'function') disposers.push(disposer);
      return () => undefined;
    },
    on(event, handler) {
      handlers.set(event, handler);
      return () => handlers.delete(event);
    },
    logger: {
      info: () => undefined,
      warn: () => undefined
    }
  };
  return { ctx, toolList, handlers, disposers };
}

const { ctx, toolList, handlers, disposers } = makeCtx();
const config = plugin.Config({
  root: join(workDir, 'plugin'),
  port: PORT,
  autoStart: true,
  seedDemo: false,
  knowledgeDir: join(ROOT, 'knowledge'),
  log: undefined
});
cleanups.push(async () => {
  for (const disposer of disposers) {
    try {
      await disposer();
    } catch {
      // 忽略
    }
  }
});

await test('导出契约：name / inject / Config / apply', () => {
  assert.equal(plugin.name, 'dsh-pss');
  assert.deepEqual(plugin.inject, ['tools']);
  assert.equal(typeof plugin.apply, 'function');
  assert.equal(typeof plugin.Config, 'function');
});

await test('Config 缺省值可被 schemastery 校验并通过', () => {
  assert.equal(config.enabled, true);
  assert.equal(config.port, PORT);
  assert.equal(config.dim, 32);
  assert.equal(config.topK, 3);
  assert.equal(config.intentBackend, 'local');
  assert.equal(config.autoIngest, true);
  assert.equal(config.compileOnDispose, true);
});

await test('apply() 注册 pss_retrieve 工具', () => {
  plugin.apply(ctx, config);
  assert.equal(toolList.length, 1, `注册了 ${toolList.length} 个工具`);
  const tool = toolList[0];
  assert.equal(tool.name, 'pss_retrieve');
  assert.ok(tool.description.length > 20, '工具描述太短，模型不会选它');
  assert.ok(tool.parameters.properties?.query, '缺少 query 参数');
  assert.ok(tool.parameters.required?.includes('query'), 'query 应该是必填');
  assert.ok(tool.output?.schema?.properties?.contextPack, '输出 schema 缺少 contextPack');
});

await test('Dashboard 服务随插件启动并可访问', async () => {
  const health = await waitFor(async () => {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/api/pss/health`);
      return res.ok ? res.json() : null;
    } catch {
      return null;
    }
  });
  assert.equal(health.ok, true);
  assert.equal(health.name, 'dsh-pss');
  assert.ok(health.knowledge >= 8, `知识片段数不对：${health.knowledge}`);
});

await test('会话事件被吸收进状态（user/message + assistant/message）', async () => {
  const sessionHandler = handlers.get('session/event');
  assert.equal(typeof sessionHandler, 'function', '没有监听 session/event');
  const session = { id: 'plugin-smoke' };
  sessionHandler(session, { type: 'user/message', data: { content: [{ type: 'text', text: 'PSS 的存储为什么是常数级？' }] } });
  sessionHandler(session, {
    type: 'assistant/message',
    data: { message: { content: [{ type: 'text', text: '因为它只保留定长向量与关键词表。' }] } }
  });
  // apply 里是 fire-and-forget，等图里出现节点
  const graph = await waitFor(async () => {
    const res = await fetch(`http://127.0.0.1:${PORT}/api/pss/graph`);
    const data = await res.json();
    return data.nodes?.length > 0 ? data : null;
  });
  assert.ok(graph.nodes.some((node) => node.kind === 'user'), '没有吸收到用户消息');
  assert.ok(graph.nodes.some((node) => node.kind === 'answer'), '没有吸收到助手消息');
});

await test('pss_retrieve 工具能跑通并回喂上下文包', async () => {
  const tool = toolList[0];
  const result = await tool.execute({ query: 'PSS 的意图是怎么生成的？' }, { agent: { session: { id: 'plugin-smoke' } } });
  assert.equal(result.sessionId, 'plugin-smoke', '没有用调用方会话 id');
  assert.ok(result.contextPack.includes('[PSS]'));
  assert.ok(Array.isArray(result.hits) && result.hits.length > 0);
  assert.equal(typeof result.intentSource, 'string');
  const rendered = tool.output.render({ query: 'x' }, result);
  assert.equal(rendered[0].type, 'text');
  assert.ok(rendered[0].text.includes('[PSS]'), 'render 没吐出上下文包');
});

await test('session/disposed 触发编译（含 history 回放）', async () => {
  const disposeHandler = handlers.get('session/disposed');
  assert.equal(typeof disposeHandler, 'function', '没有监听 session/disposed');
  const session = {
    id: 'plugin-smoke',
    snapshotEvents: () => [
      { type: 'user/message', data: { content: [{ type: 'text', text: '第一个问题' }] } },
      { type: 'assistant/message', data: { message: { content: [{ type: 'text', text: '第一个回答' }] } } }
    ]
  };
  disposeHandler(session);
  const events = await waitFor(async () => {
    const res = await fetch(`http://127.0.0.1:${PORT}/api/pss/events`);
    const data = await res.json();
    return (data.events ?? []).some((event) => event.kind === 'compile') ? data.events : null;
  });
  assert.ok(events.length > 0);
});

/* -------------------------------------------------------------------------- */
/* D. HTTP 端到端                                                              */
/* -------------------------------------------------------------------------- */

section('D · HTTP 端到端（Dashboard + REST）');

const base = `http://127.0.0.1:${PORT}`;
const getJson = async (path) => {
  const res = await fetch(base + path);
  assert.ok(res.ok, `GET ${path} → ${res.status}`);
  return res.json();
};
const postJson = async (path, body) => {
  const res = await fetch(base + path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body)
  });
  assert.ok(res.ok, `POST ${path} → ${res.status}`);
  return res.json();
};

await test('GET / 返回 Dashboard 页面（含渲染器与 HUD）', async () => {
  const res = await fetch(base + '/');
  assert.equal(res.status, 200);
  const html = await res.text();
  assert.ok(html.includes('PSS · Brain'), 'HUD 标题不在页面里');
  assert.ok(html.includes('cytoscape'), '没有引入 cytoscape');
  assert.ok(html.includes('/api/pss/graph'), '页面没有轮询图接口');
  assert.ok(html.includes('vendor/cytoscape.min.js'), '没有走本地 vendor 副本');
});

await test('GET /vendor/cytoscape.min.js 静态资源可下载', async () => {
  const res = await fetch(base + '/vendor/cytoscape.min.js');
  assert.equal(res.status, 200);
  assert.equal(res.headers.get('content-type'), 'text/javascript; charset=utf-8');
  const text = await res.text();
  assert.ok(text.length > 100000, `体积不对：${text.length}`);
});

await test('静态服务挡住目录穿越', async () => {
  const res = await fetch(base + '/../package.json');
  assert.ok(res.status === 404 || res.status === 403, `居然返回 ${res.status}`);
});

await test('POST /api/pss/ingest 吸收文本', async () => {
  const data = await postJson('/api/pss/ingest', { sessionId: 'http', role: 'user', text: 'HTTP 通道也能吸收对话。' });
  assert.equal(data.state.sessionId, 'http');
  assert.ok(data.state.turns >= 1);
});

await test('POST /api/pss/retrieve 返回检索结果', async () => {
  const data = await postJson('/api/pss/retrieve', { sessionId: 'http', query: '可视化探针记录哪些节点？' });
  assert.ok(data.contextPack.includes('[PSS]'));
  assert.ok(data.hits.length > 0);
});

await test('POST /api/pss/compile 编译会话', async () => {
  const data = await postJson('/api/pss/compile', {
    sessionId: 'http',
    history: [{ role: 'user', content: '编译会做什么？' }, { role: 'assistant', content: '压状态并触发训练。' }]
  });
  assert.ok(data.state.turns >= 2);
});

await test('GET /api/pss/graph 支持 since 短路', async () => {
  const full = await getJson('/api/pss/graph');
  const same = await getJson(`/api/pss/graph?since=${full.version}`);
  assert.equal(same.unchanged, true, 'since 命中时应该返回 unchanged');
  assert.ok(typeof same.stats.nodes === 'number');
});

await test('GET /api/pss/events 与 /sessions 与 /knowledge 都有数据', async () => {
  const events = await getJson('/api/pss/events');
  assert.ok(events.events.length > 0);
  assert.ok(events.events.every((event) => event.id && event.kind && event.message));

  const sessions = await getJson('/api/pss/sessions');
  assert.ok(sessions.sessions.length >= 1);

  const knowledge = await getJson('/api/pss/knowledge');
  assert.ok(knowledge.chunks.length >= 8);
});

await test('POST /api/pss/demo 一键出图', async () => {
  const data = await postJson('/api/pss/demo', { reset: true });
  assert.ok(data.stats.nodes >= 12, `demo 节点太少：${data.stats.nodes}`);
  assert.ok(data.graph.nodes.length === data.stats.nodes);
});

await test('POST /api/pss/reset 清空', async () => {
  const data = await postJson('/api/pss/reset', {});
  assert.equal(data.graph.nodes.length, 0);
  assert.equal(data.graph.links.length, 0);
});

await test('未知端点返回 404 而不是崩掉', async () => {
  const res = await fetch(base + '/api/pss/nope');
  assert.equal(res.status, 404);
  const body = await res.json();
  assert.ok(body.error.includes('unknown endpoint'));
});

/* -------------------------------------------------------------------------- */
/* 收尾                                                                        */
/* -------------------------------------------------------------------------- */

await cleanup();

console.log('');
if (failures.length === 0) {
  console.log(`\x1b[32m全部通过：${passed} 项\x1b[0m`);
  process.exit(0);
} else {
  console.log(`\x1b[31m失败 ${failures.length} 项 / 通过 ${passed} 项\x1b[0m`);
  for (const failure of failures) console.log(`  · ${failure.name}`);
  process.exit(1);
}
