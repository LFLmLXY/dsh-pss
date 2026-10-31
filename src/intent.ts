/**
 * 意图后端（Intent Backend）—— PSS 的「参数化」那一步。
 *
 * 同一个接口两种实现：
 *  1. {@link LocalIntentBackend}  —— 纯本地启发式：用参数状态里的关键词扩展提问。
 *     零依赖、毫秒级，保证 `npm i` 之后立刻可用（也是 LoRA 后端挂掉时的降级路径）。
 *  2. {@link LoraIntentBackend}   —— 调用 Python 训练后端，加载会话专属 LoRA 生成检索词。
 *     这是论文里那个「参数化意图」，也是 PSS 真正的卖点。
 *
 * 两者用 {@link FallbackIntentBackend} 串起来：优先 LoRA，出错自动降级，可视化探针会把
 * 实际走的路径记在节点 detail 里（Dashboard 上能看到 `source: lora` 还是 `source: local`）。
 *
 * @module dsh-pss/intent
 */

import { tokenize } from './embed.js';
import type { IntentBackend, IntentInput, IntentOutput, PSSState } from './types.js';

/** 从状态里取「与当前提问相关」的历史关键词，最多 6 个。 */
function pickExpansion(state: PSSState, queryTokens: readonly string[], limit = 6): string[] {
  const seen = new Set(queryTokens);
  return state.keywords
    .filter((kw) => kw.weight > 0.25 && kw.term.length > 1 && !seen.has(kw.term))
    .slice(0, limit)
    .map((kw) => kw.term);
}

/** 本地启发式意图后端。 */
export class LocalIntentBackend implements IntentBackend {
  readonly name = 'local';

  /** 同步完成，但保持 Promise 接口以便和 LoRA 后端互换。 */
  generate(input: IntentInput): Promise<IntentOutput> {
    const queryTokens = tokenize(input.query);
    const expansion = pickExpansion(input.state, queryTokens);
    // 关键词直接拼到原句后面：对 BM25 来说，词袋才是有效信号，语序无所谓。
    const intentQuery = expansion.length === 0 ? input.query : `${input.query} ${expansion.join(' ')}`;
    return Promise.resolve({
      intentQuery,
      keywords: [...new Set([...queryTokens.filter((t) => t.length > 1), ...expansion])].slice(0, 12),
      source: 'local'
    });
  }
}

/** Python 训练后端的意图生成实现（POST /retrieve-intent）。 */
export class LoraIntentBackend implements IntentBackend {
  readonly name = 'lora';

  /**
   * @param baseUrl - FastAPI 地址，例如 `http://127.0.0.1:8077`。
   * @param timeoutMs - 超时（毫秒）。LoRA 加载慢，默认给 8s。
   */
  constructor(
    private readonly baseUrl: string,
    private readonly timeoutMs = 8000
  ) {}

  async generate(input: IntentInput): Promise<IntentOutput> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await fetch(`${this.baseUrl.replace(/\/+$/, '')}/retrieve-intent`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          sessionId: input.state.sessionId,
          query: input.query,
          keywords: input.state.keywords.map((kw) => kw.term),
          centroid: input.state.centroid,
          turns: input.state.turns
        })
      });
      if (!response.ok) throw new Error(`lora backend HTTP ${response.status}`);
      const payload = (await response.json()) as { intentQuery?: unknown; keywords?: unknown };
      const intentQuery = typeof payload.intentQuery === 'string' ? payload.intentQuery : input.query;
      const keywords = Array.isArray(payload.keywords)
        ? payload.keywords.filter((k): k is string => typeof k === 'string')
        : tokenize(intentQuery);
      return { intentQuery, keywords, source: `lora:${this.baseUrl}` };
    } finally {
      clearTimeout(timer);
    }
  }
}

/**
 * 带降级的意图后端：主后端抛错时自动切到备用，并回调 `onFallback` 让上层记日志 / 打点。
 */
export class FallbackIntentBackend implements IntentBackend {
  readonly name: string;

  constructor(
    private readonly primary: IntentBackend,
    private readonly fallback: IntentBackend,
    private readonly onFallback?: (error: unknown) => void
  ) {
    this.name = `${primary.name}|${fallback.name}`;
  }

  async generate(input: IntentInput): Promise<IntentOutput> {
    try {
      return await this.primary.generate(input);
    } catch (error) {
      this.onFallback?.(error);
      const result = await this.fallback.generate(input);
      // 标记降级来源，前端一眼能看出「LoRA 没跑起来」
      return { ...result, source: `${result.source}(fallback from ${this.primary.name})` };
    }
  }
}
