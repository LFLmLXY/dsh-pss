#!/usr/bin/env node
/**
 * dsh-pss 本地开发服务器。
 *
 * 作用：不装 DSH、不加载插件，直接跑 PSS 运行时 + Dashboard HTTP 服务，
 * 方便改前端、改探针埋点、灌语料看效果。
 *
 *   npm run dev                    # 构建 + 起服务 + 灌演示语料
 *   npm run dev -- --no-demo       # 不灌演示语料
 *   npm run dev -- --port=9100     # 换端口
 *   PSS_TRAINER_URL=http://127.0.0.1:8077 npm run dev   # 接上 Python 训练后端
 *
 * 数据落在 `./.pss-demo`（已 gitignore），删掉就是干净的初始状态。
 */

import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { existsSync } from 'node:fs';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');

/** 命令行参数解析：够用即可，不引 minimist。 */
function arg(name, fallback) {
  const hit = process.argv.find((item) => item.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : fallback;
}

const flag = (name) => process.argv.includes(`--${name}`);

const distEntry = join(ROOT, 'dist', 'PSSManager.js');
if (!existsSync(distEntry)) {
  console.error('[dev] 找不到 dist/，先跑 `npm run build`（`npm run dev` 已经包含这一步）');
  process.exit(1);
}

// dist 是 ESM，直接 import；这些路径在构建产物里是相对 dist 解析的
const { PSSManager } = await import(new URL('../dist/PSSManager.js', import.meta.url).href);
const { DashboardServer } = await import(new URL('../dist/http.js', import.meta.url).href);

const port = Number(arg('port', process.env.PSS_PORT ?? '8964'));
const root = arg('root', join(ROOT, '.pss-demo'));
const trainerUrl = process.env.PSS_TRAINER_URL?.trim();

const manager = new PSSManager({
  root,
  knowledgeDir: join(ROOT, 'knowledge'),
  topK: 3,
  dim: 32,
  maxNodes: 400,
  ...(trainerUrl ? { trainerUrl } : {}),
  log: (message) => console.log(message)
});

const server = new DashboardServer({ manager, port, log: (message) => console.log(message) });
const url = await server.start();

await manager.init();
if (!flag('no-demo')) {
  const stats = await manager.demo('demo');
  console.log(`[dev] 演示语料：${stats.nodes} 节点 / ${stats.links} 连线 / ${manager.knowledgeChunks().length} 知识片段`);
}

/**
 * 终端里中文占两个字符宽度，padEnd 只数 code unit，直接 pad 会让右边框歪掉。
 * 这里按显示宽度补空格。
 */
function vpad(text, width) {
  let shown = 0;
  for (const ch of text) {
    const code = ch.codePointAt(0) ?? 0;
    shown += code >= 0x1100 && (
      code <= 0x115f ||            // 韩文字母
      code === 0x2329 || code === 0x232a ||
      (code >= 0x2e80 && code <= 0xa4cf) ||  // CJK 部首 ~ 彝文
      (code >= 0xac00 && code <= 0xd7a3) ||  // 韩文音节
      (code >= 0xf900 && code <= 0xfaff) ||  // CJK 兼容
      (code >= 0xfe30 && code <= 0xfe6f) ||  // CJK 兼容形式
      (code >= 0xff00 && code <= 0xff60) ||  // 全角
      (code >= 0xffe0 && code <= 0xffe6)
    ) ? 2 : 1;
  }
  return text + ' '.repeat(Math.max(0, width - shown));
}

console.log('');
console.log('  ┌──────────────────────────────────────────────────┐');
console.log('  │  PSS 脑图   →  ' + vpad(`${url}/`, 32) + '│');
console.log('  │  图数据     →  ' + vpad(`${url}/api/pss/graph`, 32) + '│');
console.log('  │  事件流     →  ' + vpad(`${url}/api/pss/events`, 32) + '│');
console.log('  └──────────────────────────────────────────────────┘');
console.log(`  状态目录: ${root}`);
console.log(`  训练后端: ${trainerUrl || '(未配置，编译时只落盘不训练)'}`);
console.log('  Ctrl+C 退出');
console.log('');

const shutdown = async () => {
  console.log('\n[dev] 正在关闭…');
  await server.stop();
  process.exit(0);
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
