/**
 * 状态持久化：每个会话一个固定大小的小 JSON 文件。
 *
 * 这是「O(1) 恒定存储」的落地点 —— 写入的是 {@link PSSState}（定长向量 + 定长关键词表），
 * 不是聊天记录。所以：
 *  - 100 轮对话和 10000 轮对话，磁盘占用一样；
 *  - 加载是 O(1)，不需要回放历史；
 *  - 原子写（tmp + rename），进程被杀也不会留下半个文件。
 *
 * @module dsh-pss/store
 */

import { mkdir, readFile, readdir, rename, writeFile, unlink } from 'node:fs/promises';
import { join } from 'node:path';
import type { PSSState } from './types.js';

/** 状态文件格式版本。将来结构变更时可据此迁移。 */
const STATE_FORMAT = 1;

interface StoredState extends PSSState {
  format: number;
}

/** 把任意 sessionId 变成安全文件名（保留可读性的白名单替换）。 */
export function stateFileName(sessionId: string): string {
  const safe = sessionId.replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 96);
  return `${safe === '' ? 'session' : safe}.json`;
}

/**
 * 定长状态的读写器。
 */
export class StateStore {
  /** `root/sessions`：所有会话状态文件。 */
  private readonly dir: string;

  constructor(private readonly root: string) {
    this.dir = join(root, 'sessions');
  }

  /** 该会话状态文件的绝对路径。 */
  pathFor(sessionId: string): string {
    return join(this.dir, stateFileName(sessionId));
  }

  /** 确保目录存在（幂等）。 */
  async ensure(): Promise<void> {
    await mkdir(this.dir, { recursive: true });
  }

  /** 读取一个会话状态；不存在或损坏时返回 null（损坏不阻塞主流程）。 */
  async load(sessionId: string): Promise<PSSState | null> {
    try {
      const raw = await readFile(this.pathFor(sessionId), 'utf8');
      const parsed = JSON.parse(raw) as StoredState;
      if (parsed.format !== STATE_FORMAT || !Array.isArray(parsed.centroid)) return null;
      return parsed;
    } catch {
      return null;
    }
  }

  /**
   * 原子写入。先写 `xxx.json.tmp` 再 rename —— 同目录 rename 在 POSIX/NTFS 上都是原子的。
   * @param state - 要落盘的状态（大小与对话长度无关）。
   */
  async save(state: PSSState): Promise<void> {
    await this.ensure();
    const target = this.pathFor(state.sessionId);
    const tmp = `${target}.${process.pid}.tmp`;
    const payload: StoredState = { format: STATE_FORMAT, ...state };
    await writeFile(tmp, JSON.stringify(payload), 'utf8');
    try {
      await rename(tmp, target);
    } catch (error) {
      // rename 失败时尽力清理临时文件，避免脏数据堆积
      void unlink(tmp).catch(() => undefined);
      throw error;
    }
  }

  /** 已落盘的全部会话 id（不含扩展名）。 */
  async list(): Promise<string[]> {
    try {
      const entries = await readdir(this.dir, { withFileTypes: true });
      return entries
        .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
        .map((entry) => entry.name.slice(0, -'.json'.length));
    } catch {
      return [];
    }
  }

  /** 根目录（便于外部展示数据位置）。 */
  get rootDir(): string {
    return this.root;
  }
}
