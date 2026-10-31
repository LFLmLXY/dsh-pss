/**
 * 轻量文本工具：分词、哈希嵌入、BM25。
 *
 * 为什么不用 embedding 模型？
 * 因为 PSS 的卖点是「零依赖、零 GPU 也能跑通」，而且我们需要的不是语义精度，
 * 而是**固定的参数化表示**：把任意长的历史压成一个定长向量 + 定长关键词表。
 * 哈希嵌入（feature hashing）恰好满足这点 —— 不存词表、不存索引，内存恒定。
 *
 * @module dsh-pss/embed
 */

/** 英文 + 中文的常见停用词（只做粗过滤，宁可漏杀也不要误杀）。 */
const STOPWORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'for', 'of', 'to', 'in', 'on', 'at',
  'by', 'with', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does', 'did',
  'can', 'could', 'should', 'would', 'will', 'shall', 'may', 'might', 'must', 'have', 'has', 'had',
  'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you', 'we', 'they', 'he', 'she', 'them',
  'my', 'your', 'our', 'their', 'me', 'us', 'not', 'no', 'yes', 'so', 'than', 'too', 'very',
  '的', '了', '是', '在', '我', '你', '他', '她', '它', '们', '和', '与', '就', '都', '也', '很',
  '把', '被', '给', '对', '这', '那', '有', '没', '不', '要', '会', '能', '着', '过', '吗', '呢',
  '吧', '啊', '呀', '一个', '什么', '怎么', '可以', '这个', '那个', '我们', '你们', '他们'
]);

/** 拉丁词 / 数字 / 常见技术符号组成的一个 token。 */
const LATIN_RE = /[a-z0-9][a-z0-9_+#.-]*/g;
/** CJK 连续段（含扩展 A 区与兼容区）。 */
const CJK_RE = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+/g;

/**
 * 把一段文本切成 token。
 *
 * 中文没有分词器，这里用「单字 + 相邻二元组」近似：
 * `记忆模型` → `记` `忆` `模` `型` `记忆` `忆模` `模型`。
 * 二元组自带大部分词边界信息，对 BM25 检索来说效果足够，且不需要任何依赖。
 *
 * @param text - 任意文本。
 * @returns token 数组（可能含重复，BM25/嵌入各自处理词频）。
 */
export function tokenize(text: string): string[] {
  const out: string[] = [];
  if (!text) return out;
  const lower = text.toLowerCase();

  for (const word of lower.match(LATIN_RE) ?? []) {
    if (word.length > 1 && !STOPWORDS.has(word)) out.push(word);
  }

  for (const run of lower.match(CJK_RE) ?? []) {
    for (let i = 0; i < run.length; i++) {
      const ch = run.charAt(i);
      if (!STOPWORDS.has(ch)) out.push(ch);
      if (i + 1 < run.length) {
        const bi = run.slice(i, i + 2);
        if (!STOPWORDS.has(bi)) out.push(bi);
      }
    }
  }

  return out;
}

/** FNV-1a 32 位哈希：快、够散、可重复（同一 token 永远落到同一维度）。 */
export function hashToken(token: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < token.length; i++) {
    h ^= token.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/**
 * 把 token 累加进一个定长向量（feature hashing）。
 *
 * 用哈希最高位做符号位（signed hashing），可以显著降低碰撞带来的系统性偏置 ——
 * 这是 Vowpal Wabbit / sklearn `HashingVectorizer` 的经典做法。
 *
 * @param acc - 累加目标，长度即嵌入维度。
 * @param tokens - 已分好的 token。
 * @param weight - 每条 token 的权重（默认 1）。
 */
export function addEmbedding(acc: number[], tokens: readonly string[], weight = 1): void {
  const dim = acc.length;
  if (dim === 0) return;
  for (const token of tokens) {
    const h = hashToken(token);
    const idx = h % dim;
    const sign = (h >>> 31) === 0 ? 1 : -1;
    acc[idx] = (acc[idx] ?? 0) + sign * weight;
  }
}

/** 原地归一化（L2）。零向量保持不变。 */
export function normalize(vec: number[]): number[] {
  let sum = 0;
  for (const v of vec) sum += v * v;
  const len = Math.sqrt(sum);
  if (len === 0) return vec;
  for (let i = 0; i < vec.length; i++) vec[i] = (vec[i] ?? 0) / len;
  return vec;
}

/** 余弦相似度（输入已归一化时就是点积）。 */
export function cosine(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/** 一个 BM25 命中结果（按 doc 下标返回，调用方自己映射回原文）。 */
export interface Bm25Hit {
  index: number;
  score: number;
}

/**
 * 极简 BM25 索引。语料在构造时固定（会话运行期不变），因此不需要增量维护。
 *
 * 公式：`idf(t) * tf*(k1+1) / (tf + k1*(1-b+b*|d|/avgdl))`，`idf = ln(1 + (N-df+0.5)/(df+0.5))`。
 */
export class Bm25Index {
  private readonly postings = new Map<string, Map<number, number>>();
  private readonly lengths: number[] = [];
  private readonly avgLength: number;
  private readonly docCount: number;

  constructor(
    docs: readonly (readonly string[])[],
    private readonly k1 = 1.2,
    private readonly b = 0.75
  ) {
    this.docCount = docs.length;
    let total = 0;

    for (let i = 0; i < docs.length; i++) {
      const tokens = docs[i] ?? [];
      this.lengths.push(tokens.length);
      total += tokens.length;
      const tf = new Map<string, number>();
      for (const token of tokens) tf.set(token, (tf.get(token) ?? 0) + 1);
      for (const [token, count] of tf) {
        let byDoc = this.postings.get(token);
        if (!byDoc) {
          byDoc = new Map<number, number>();
          this.postings.set(token, byDoc);
        }
        byDoc.set(i, count);
      }
    }

    this.avgLength = this.docCount === 0 ? 0 : total / this.docCount;
  }

  /** 语料条数。 */
  get size(): number {
    return this.docCount;
  }

  /**
   * 检索。
   * @param tokens - 查询 token（通常已去重或带权重，这里只按词频计）。
   * @param limit - 最多返回多少条。
   * @param exclude - 需要排除的 doc 下标（例如已在本轮用过的）。
   */
  search(tokens: readonly string[], limit: number, exclude?: ReadonlySet<number>): Bm25Hit[] {
    if (this.docCount === 0 || tokens.length === 0) return [];

    const queryTf = new Map<string, number>();
    for (const token of tokens) queryTf.set(token, (queryTf.get(token) ?? 0) + 1);

    const scores = new Map<number, number>();
    const avg = this.avgLength === 0 ? 1 : this.avgLength;

    for (const [token, qtf] of queryTf) {
      const byDoc = this.postings.get(token);
      if (!byDoc || byDoc.size === 0) continue;
      const df = byDoc.size;
      const idf = Math.log(1 + (this.docCount - df + 0.5) / (df + 0.5));
      for (const [docIndex, tf] of byDoc) {
        if (exclude?.has(docIndex)) continue;
        const dl = this.lengths[docIndex] ?? 0;
        const denom = tf + this.k1 * (1 - this.b + (this.b * dl) / avg);
        const score = idf * ((tf * (this.k1 + 1)) / (denom === 0 ? 1 : denom)) * qtf;
        scores.set(docIndex, (scores.get(docIndex) ?? 0) + score);
      }
    }

    return [...scores.entries()]
      .map(([index, score]) => ({ index, score }))
      .sort((x, y) => y.score - x.score)
      .slice(0, Math.max(0, limit));
  }
}
