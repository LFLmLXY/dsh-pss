/**
 * 知识库加载：把目录里的 `*.md` / `*.txt` 切成片段，喂给 BM25。
 *
 * 为什么不用向量库？
 * PSS 的目标是「可解释 + 零依赖」。BM25 的命中原因（哪些词、权重多少）能直接画到
 * Dashboard 上，而向量相似度只能给一个玄学分数。可视化优先，所以选 BM25。
 *
 * @module dsh-pss/knowledge
 */

import { readFile, readdir } from 'node:fs/promises';
import { extname, join } from 'node:path';
import type { KnowledgeChunk } from './types.js';

/** 单片段最大字符数（超出后按段落切分）。 */
const MAX_CHUNK_CHARS = 700;

/**
 * 内置种子语料：不配知识库也能立刻跑通 demo。
 *
 * 内容刻意写成「项目自身的自述」—— demo 时检索出来的节点看起来就像模型在读自己的
 * README，视觉上非常有说服力。
 */
export const DEFAULT_KNOWLEDGE: KnowledgeChunk[] = [
  {
    id: 'seed:pss-core',
    source: 'seed/pss-core',
    text: 'PSS 即参数化会话状态（Parametric Session State）。它把一段任意长的对话压缩成一个定长向量加一张定长关键词表，因此序列化后的字节数与对话轮数无关，实现 O(1) 恒定存储。'
  },
  {
    id: 'seed:intent',
    source: 'seed/intent',
    text: '参数化意图：会话结束时用该会话的对话历史微调一个 LoRA 适配器，检索时临时加载这个适配器把用户提问改写成检索词，用完立即卸载。这样历史知识以权重形式存在，而不是以 token 形式塞回上下文。'
  },
  {
    id: 'seed:probe',
    source: 'seed/probe',
    text: '可视化探针：在检索的每个阶段向内存图推入节点与连线，用户输入用红色、PSS 意图用青色、RAG 知识用黄色、合成上下文用蓝色，前端用 Cytoscape 力导向布局渲染成神经元一样的动态脑图。'
  },
  {
    id: 'seed:budget',
    source: 'seed/budget',
    text: '上下文预算：长对话的真正瓶颈是上下文窗口而不是磁盘。PSS 把历史折叠进参数里，每轮只需要回填被检索中的少量知识片段，因此 token 成本随轮数增长趋近于零。'
  },
  {
    id: 'seed:decay',
    source: 'seed/decay',
    text: '关键词衰减：每吸收一轮对话，旧关键词权重乘以衰减系数，新词按词频累加，最后只保留权重最高的固定条数。这样关键词表大小恒定，又能自动遗忘不再重要的话题。'
  },
  {
    id: 'seed:hash',
    source: 'seed/hash',
    text: '哈希嵌入：用 FNV-1a 把 token 映射到固定维度的向量空间，并用哈希最高位做符号位降低碰撞偏置。不需要词表，不需要训练，内存恒定，这就是「零依赖」的代价与收益。'
  },
  {
    id: 'seed:lora',
    source: 'seed/lora',
    text: 'LoRA 训练后端用 FastAPI 加 HuggingFace peft 实现，接收会话历史，异步训练低秩适配器并存到 loras/{sessionId}。加载时只读几十 MB 的适配器权重，不触碰基座模型。'
  },
  {
    id: 'seed:dashboard',
    source: 'seed/dashboard',
    text: 'Dashboard 每三秒轮询一次 /api/pss/graph，按 version 字段做增量合并，只把新增的节点和连线塞进 Cytoscape，然后跑一次平滑的力导向重排，新节点带一次脉冲发光动画。'
  }
];

/** 按空行 / 二级标题把长文档切成片段。 */
export function splitChunks(text: string): string[] {
  const blocks = text
    .split(/\r?\n\s*\r?\n/)
    .map((block) => block.trim())
    .filter((block) => block.length > 0);

  const out: string[] = [];
  let buffer = '';
  for (const block of blocks) {
    if (buffer.length > 0 && buffer.length + block.length > MAX_CHUNK_CHARS) {
      out.push(buffer);
      buffer = block;
    } else {
      buffer = buffer.length === 0 ? block : `${buffer}\n${block}`;
    }
  }
  if (buffer.length > 0) out.push(buffer);
  return out;
}

/**
 * 读取知识库目录下的所有 `.md` / `.txt`。
 *
 * 目录不存在时返回空数组而不是抛错 —— 插件不应该因为用户没建目录就拒绝启动。
 *
 * @param dir - 知识库目录。
 * @returns 切片后的知识片段（id 为 `文件名#序号`，稳定可复现）。
 */
export async function loadKnowledgeDir(dir: string): Promise<KnowledgeChunk[]> {
  let entries: string[];
  try {
    const dirents = await readdir(dir, { withFileTypes: true });
    entries = dirents
      .filter((entry) => entry.isFile() && ['.md', '.txt', '.markdown'].includes(extname(entry.name).toLowerCase()))
      .map((entry) => entry.name)
      .sort();
  } catch {
    return [];
  }

  const chunks: KnowledgeChunk[] = [];
  for (const name of entries) {
    let raw: string;
    try {
      raw = await readFile(join(dir, name), 'utf8');
    } catch {
      continue;
    }
    const parts = splitChunks(raw);
    parts.forEach((part, index) => {
      chunks.push({
        id: `${name}#${index}`,
        source: name,
        text: part
      });
    });
  }
  return chunks;
}
