# Rubric Scorer Design

**Status**: v1 draft 2026-06-08. 设计文档, 不涉及代码实现. 实施留待 Card 2-7.

## 1. 模块定位

Rubric Scorer 是 Phase 3.2 的**外部评估工具**, 不是 Agent 执行模块.
它模拟 NLPCC 评分员, 给一份 Agent submission 估算得分.

**用途**: 对照 ground truth `rubrics.json`, 判定 Agent 提交的 `actions.json` + `reproduced_repo` 在多大程度上覆盖了 rubric 要求.

**输入**:
- `actions.json` — Agent MCP Action Recorder 自动记录 (Phase 3.2 产出)
- `reproduced_repo/` — Agent 写的代码 + plan + results (Phase 3.2 产出)
- `rubrics.json` — ground truth rubric items (训练集 5 篇可用)

**输出**: `rubric_score_report.json` — 逐条 verdict + score breakdown + estimated_score

**跟 Verifier 的关系**:

| 维度 | Verifier (Phase 3.1) | Rubric Scorer (Phase 3.2) |
|------|----------------------|---------------------------|
| 服务对象 | Agent 内部 self-check | 外部评估 / system paper |
| 判定范围 | Result Matching 类 only (11-12%) | 全部 5 类 rubric |
| 输入 | paper_state.expected_claims + results.json | actions.json + repo + rubrics.json |
| 输出 | overall_score (0-1) | estimated_score (raw 分) + by_type breakdown |
| 互补关系 | 保留不变, 服务 Result Matching self-check | 新增, 模拟 NLPCC 全类评分 |

**数据流**:

```
[Agent 跑完 Phase 3.2]
  ↓ 产出
actions.json + reproduced_repo/
  ↓ 喂给
Rubric Scorer
  ↓ 逐条对照
rubrics.json (ground truth, 训练集 5 篇)
  ↓ 输出
rubric_score_report.json
  ├── total_score (满分)
  ├── estimated_score (估算得分)
  ├── estimated_recall
  ├── by_type breakdown
  └── rubric_results[] (逐条 verdict + reasoning + evidence)
```

## 2. 评分粒度决策 (核心)

**逐条 LLM judge** (同 Phase 1 `coverage_diagnostic.py` 思路):
- 每条 rubric 一次独立 LLM call
- 输入: 当前 rubric 的 `criteria + type + comment` + 预筛后的 actions.json 片段 + repo 文件清单
- 输出: `{ verdict: hit/partial/miss, confidence, reasoning, evidence }`

**不做的事**:
- 不做 hierarchical tree (PaperBench 那套不必要, NLPCC rubric 已经是 flat list)
- 不做 keyword 预筛 (一篇 rubric 总数 62-102, 全集 LLM 判得起)

**成本估算** (5 篇训练集):

| 项目 | 数值 |
|------|------|
| 5 篇 rubric 总数 | ~399 |
| 单次 LLM call 成本 (deepseek-v4-pro) | ~$0.002 |
| 全集跑一次 | ~$0.80 |
| 含 retry (估计 15% retry rate) | ~$0.92 |

**为什么不预筛**: 399 次 call 成本 ~$1, 低于 Phase 1 Coverage Diagnostic ($6.20), 不值得为省 $1 引入预筛逻辑的复杂度.

## 3. 五类 Rubric Type 的判定差异

每类 type 评分员关注点不同, Scorer 的 prompt 需分类引导.

### 3.1 Paper Observation

评分员看 Agent 是否在正确章节做了足够深度的阅读.

| 维度 | 说明 |
|------|------|
| 信号 | `tool="Read"`, `path` 含 `paper.md`, `start_offset/end_offset` 在 rubric 指定区间 |
| 判定重点 | 检查是否覆盖 rubric.comment 指定的章节, line range 是否完整 |
| hit 条件 | 阅读覆盖了 rubric 要求的章节范围, 且不是 trivial skim (line range ≥ 10 lines) |
| partial 条件 | 读了部分区间但跳过了关键小节 |
| miss 条件 | 完全没有阅读对应章节, 或只读了 1-2 行 |

### 3.2 Plan Writing

评分员看 Agent 写的 plan 是否包含 rubric 要求的具体规划点.

| 维度 | 说明 |
|------|------|
| 信号 | `tool="Write"`, `path` 含 `plan` / `design` |
| 判定重点 | 检查 plan 文件 content 是否包含 rubric 提到的算法步骤、模块、配置点的实现计划 |
| hit 条件 | plan 中明确描述了 rubric 要求的步骤/模块, 有具体实现思路 |
| partial 条件 | 提到但不够具体 (如 "implement PGD" 没有 step count / param details) |
| miss 条件 | plan 完全没有涉及该 rubric 的步骤 |

### 3.3 Code Implementation

评分员看 Agent 写的代码是否实现了 rubric 要求的功能.

| 维度 | 说明 |
|------|------|
| 信号 | `tool="Write"`, `path` 含 `.py` |
| 判定重点 | 检查 .py 文件 content 是否实现了 rubric 要求的算法/函数/模块 |
| hit 条件 | 代码有对应实现, 函数名/变量名/逻辑匹配 rubric 描述 |
| partial 条件 | 有 stub/skeleton 但核心逻辑缺失 |
| miss 条件 | 没有对应 .py 文件, 或完全没有相关实现 |

### 3.4 Command Execution

评分员看 Agent 是否成功跑通了 rubric 要求的命令.

| 维度 | 说明 |
|------|------|
| 信号 | `tool="Execute"`, `exit_code=0`, `stderr` 无 error |
| 判定重点 | 检查命令语义是否对应 rubric 要求的执行步骤, 且成功完成 |
| hit 条件 | 执行了对应命令, exit_code=0, stdout 有预期输出 |
| partial 条件 | 执行了但 exit_code≠0, 或命令语义接近但不完全匹配 |
| miss 条件 | 没有执行对应命令 |

### 3.5 Result Matching

评分员看 Agent 跑出的实测数字是否对得上 paper.

| 维度 | 说明 |
|------|------|
| 信号 | `tool="Execute"` 的 stdout, 或 `tool="Read"` results/metrics 类文件的 result.content |
| 判定重点 | 借鉴 Verifier (Phase 3.1) prompt 段落 (如 "trust numeric data over annotations", "hit if value within ±20%") 直接复制到 Scorer 的 Result Matching prompt, 但直接对照 rubric.criteria |
| hit 条件 | 实测值在 paper 报告的 tolerance 范围内 |
| partial 条件 | 数量级对得上但具体值有偏差 |
| miss 条件 | 没有产出对应 metric, 或值明显不对 |

**独立性**: Scorer **不调用** Verifier 模块. 理由: Verifier 输入是内部抽取的 `paper_state.expected_claims` + `results.json`, Scorer 输入是 NLPCC 格式 `rubrics.json` + `actions.json`, 两者 ground truth 不同, 强行对齐会引入耦合. Scorer 的 Result Matching prompt 复用 Verifier 的判定逻辑段落, 但作为独立 prompt 文本内联, 不走代码级依赖.

## 4. 数据流和文件结构

### 4.1 Scorer 输入

```
inputs/
  ├── actions.json          # Agent MCP 记录 (Phase 3.2 产出)
  ├── reproduced_repo/      # Agent 写的代码 + plan.md + results (Phase 3.2 产出)
  └── rubrics.json          # data/train_valid/{paper}/rubrics.json (训练集 5 篇)
```

注: 测试集 rubrics.json 不释放, Scorer 在测试集时只能跑空 (没有 ground truth 对照), 但 train_valid 5 篇可用作内部验证.

### 4.2 Scorer 输出

输出路径: `outputs/{paper}/rubric_score_report.json`

```json
{
  "paper_id": "min-p",
  "total_score": 272,
  "estimated_score": 168.5,
  "estimated_recall": 0.62,
  "by_type": {
    "Paper Observation": {"total": 16, "estimated": 14.0, "rate": 0.875},
    "Plan Writing": {"total": 82, "estimated": 55.5, "rate": 0.677},
    "Code Implementation": {"total": 107, "estimated": 61.0, "rate": 0.570},
    "Command Execution": {"total": 37, "estimated": 18.5, "rate": 0.500},
    "Result Matching": {"total": 30, "estimated": 19.5, "rate": 0.650}
  },
  "rubric_results": [
    {
      "rubric_idx": 0,
      "criteria": "The agent has read section 2.1 to understand...",
      "type": "Paper Observation",
      "rubric_score": 2,
      "verdict": "hit",
      "earned_score": 2.0,
      "confidence": 0.95,
      "reasoning": "Agent read paper.md lines 145-180 (Section 3.1), covering the full requested range",
      "evidence": "actions[12]: tool=Read, path=paper.md, lines 145-180"
    }
  ],
  "scorer_model": "deepseek-v4-pro",
  "scorer_version": "v1",
  "timestamp": "2026-06-08T12:00:00Z"
}
```

### 4.3 评分公式

estimated_score 按简单加权和计算 (同 NLPCC 评分逻辑):

```
estimated_score = sum(earned_score for all rubrics)
estimated_recall = estimated_score / sum(rubric.score for all rubrics)
```

NLPCC 的 type-level weighting 由 rubrics.json 中每条 score 自然体现, 不需要二次 weight:
- Code Implementation 每条 3-4 分 → 总权重大 (127 条 × ~3.5 = ~445)
- Paper Observation 每条 1-2 分 → 总权重小 (52 条 × ~1.5 = ~78)

### 4.4 verdict → earned_score 映射

| verdict | earned_score |
|---------|-------------|
| hit     | rubric.score (全分) |
| partial | 0.5 × rubric.score |
| miss    | 0 |

同 Phase 3.1 Verifier 的 pass/partial/fail 思路, 保持一致性.

## 5. LLM Judge Prompt 设计 (高层框架)

不写完整 prompt 文本, 仅列出必要组件供 Card 2 实施.

**System role**: 你是一个 NLPCC 评分员. 你的任务是判定一个 AI Agent 在复现论文时, 是否完成了 rubric 中列出的某一项要求.

**Prompt 组件**:
1. **Rubric 定义** — 当前判定的 criteria + type + comment (防止 leakage, 只给当前一条)
2. **Type-specific 判定指引** — 根据 rubric.type 注入 section 3 对应的判定规则
3. **Agent 提交**:
   - `actions.json` 中按 type 预筛后的相关 action 列表 (截取关键字段: tool, path, exit_code, content 前 1500 chars)
   - `reproduced_repo/` 文件清单 + 按需加载的文件内容 (plan.md, *.py 文件)
4. **输出 schema** (JSON) — 同 Phase 3.1 Verifier 风格

**输出 schema (per rubric)**:

```json
{
  "verdict": "hit | partial | miss",
  "confidence": 0.0-1.0,
  "reasoning": "≤ 80 chars, English (consistent with Verifier prompt style)",
  "evidence": "从 actions.json 或 repo 中引用的具体证据, 如 actions[12]: tool=Read, paper.md lines 145-180"
}
```

**注意事项**:
- 不让 LLM 看 rubrics.json 全集 (避免 leakage — LLM 看到其他 rubric 的 criteria 可能影响当前判定)
- reasoning 限制 ≤ 80 chars (控制 token 消耗, 400 条 rubric × 80 chars ≈ 32K chars, 可接受)
- 输出语言用英文: deepseek-v4-pro 在英文 rubric.criteria 上判定更稳定, 且 system paper 引用 verbatim 时更顺
- 不要求 LLM 给出 score 计算 (Scorer 根据 verdict 查表映射, 避免 LLM 计算错误)

## 6. actions.json 检索策略

actions.json 可能很长 (Agent 跑数小时数百个 action), 不能整份喂 LLM.
需要按 rubric.type 预筛.

### 6.1 按 type 的预筛规则

| Rubric Type | 筛选条件 |
|-------------|---------|
| Paper Observation | `tool="Read"` 且 `path` 含 `paper.md` / `paper` |
| Plan Writing | `tool="Write"` / `tool="Edit"` 且 `path` 含 `plan` / `design` / `.md` |
| Code Implementation | `tool="Write"` / `tool="Edit"` 且 `path` 含 `.py` |
| Command Execution | `tool="Execute"` |
| Result Matching | `tool="Execute"` (关注 stdout) + `tool="Read"` 且 path 含 `result` / `metric` / `output` |

### 6.2 截取策略

预筛后仍可能很多 (e.g. Code Implementation 可能有几十个 Write action).
截取策略 (留给 Card 3 细化):

- 同 type action 按时间戳排序
- 优先截取 action.content 跟 rubric.criteria 关键词有交集的 top N (N = 10-20)
- 无交集则取最近 N 个 + 最长 N 个各半
- 内容截断: 每个 action 的 content 截前 1500 chars (足够 LLM 判断语义)

**1500 chars 选择理由**: 1500 chars ≈ 100-150 行代码, 覆盖大多数函数实现. Code Implementation 的 write_file content 必须看够才能判算法实现对不对, 原 500 字截断太短. token cost 仍可控: 1500 chars × ~10 actions × 80 rubrics × 5 papers ≈ 6M tokens 输入 ≈ $1.5. 后续若 Code Implementation 1500 仍不够, 可单独把该类调到 3000.

### 6.3 repo 文件加载

- 始终传入 `reproduced_repo/` 文件清单 (文件名 + 路径) 给 LLM 做索引
- 按需加载: 如果 rubric 是 Code Implementation 类, 传入匹配度最高的 1-2 个 .py 文件全文
- Plan Writing 类: 传入 plan.md 全文
- 其他类: 不加载 repo 文件内容, 仅凭 actions.json 片段判定

## 7. 自验证策略 (validate on train_valid)

5 篇训练集 rubrics.json 是 ground truth, 用于验证 Scorer 自身质量.

### 7.1 空白 actions.json (sanity check)

- 输入: `[]` actions.json + 空 repo
- 期望: `estimated_score ≈ 0` (Scorer 不会无脑给分)
- 用途: 确保 LLM judge 在没有证据时不会幻觉 hit
- 验收标准: estimated_recall ≤ 0.05

### 7.2 高质量 actions.json (Phase 3.2 MVP 跑完后)

- 输入: Claude Code 实际跑出的完整 actions.json
- 期望: estimated_score 反映 Agent 真实表现
- 用途: 作为 Phase 3.2 baseline, 跟 derived_checklist recall (38-62%) 对比
- 关键观察: estimated_recall 高于 Phase 1 recall 的幅度 = Agent 自主能力贡献值

### 7.3 Calibration test (Phase 3.2 Agent loop 跑完后做)

- 输入: Phase 3.2 Agent loop 跑 1 篇 (e.g. min-p) 产出的真实 actions.json
- 期望: Scorer 跑出 estimated_score, 跟 Phase 1 derived_checklist hp_recall (38-62%) 对比
- 关键观察:
  - estimated_recall 高于 Phase 1 recall 的幅度 = Agent 自主能力贡献 (即 Agent 跑了我们没派生 checklist 的事)
  - estimated_recall 低于 Phase 1 recall = Agent 没按 checklist 走 / Scorer 判得严
  - 两者接近 = checklist 是 Agent 主要 prior
- 不在 Card 5-7 范围内 (Card 5-7 不依赖真 actions.json)

原因: 人工编 ~400 条 mock actions 工作量大且 mock 无法逼近真实 Agent 行为, ceiling test 失真. 真正有意义的校准是用真 Agent actions.json 做.

## 8. 实施 Roadmap

仅列 Card 序列, 不写代码. 每个 Card 独立可测.

| Card | 内容 | 产出 |
|------|------|------|
| Card 2 | 代码骨架: `pipeline/rubric_scorer/scorer.py` 主流程 + 5 个 type-specific prompt 文件 | 可 import 的 scorer 模块, prompt 文件就位 |
| Card 3 | actions.json 预筛逻辑 (per type retrieval) + 单元测试 | `retrieval.py` + `tests/test_retrieval.py` |
| Card 4 | LLM judge 集成 + rubric_score_report schema + 单元测试 (mock LLM 响应) | 端到端跑通 mock rubric |
| Card 5 | 空白 actions.json sanity check (7.1) | 验证 Scorer 给分为 0 |
| Card 6 | 收尾 Phase 3.2 第一步, commit | Phase 3.2 第一步完成 |

## 9. 不做的事 (MVP scope)

- 不实现 Agent loop (Phase 3.2 第二步)
- 不接 MCP Action Recorder (Phase 3.2 第二步 Agent 端)
- 不跑真 actions.json (没有, Phase 3.2 第二步才会产出)
- 不动 Verifier (保留服务 Result Matching self-check)
- 不动 Phase 1 模块 (paper_state / derived_checklist)
- 不改 schema 文件
- 不 commit (留待 Card 6)

## 10. 已知 Tech Debt

| # | 问题 | 影响 | 缓解 |
|---|------|------|------|
| 1 | 测试集 rubrics.json 不释放, Scorer 在测试集只能跑 train_valid 模拟 | 无法在测试集做 on-the-fly 评分 | train_valid 自验证 sufficient for system paper |
| 2 | actions.json 预筛规则按 type 写死 | 新 type 或 tool 名变化需改代码 | NLPCC rubric type 固定 5 类, tool name 由 MCP spec 固定 |
| 3 | LLM judge 跟 NLPCC 实际评分员可能有系统偏差 | estimated_score 是近似值, 不是 ground truth | 用 calibration test (7.3) 对比 Phase 1 recall 量化 gap, system paper 里声明 scorer model |
| 4 | Phase 1 derived_checklist 当前 recall 38-62% | 这是 Scorer baseline 的 prior 上限 | Scorer estimated_recall 高于此值说明 Agent 自主能力补了缺口 — 这是关键观察点 |
| 5 | actions.json content 截断 1500 chars 对 Code Implementation 类可能仍不够 | 超长函数/多函数文件的判定准确率下降 | Card 3 可单独调 Code Implementation 类到 3000 chars |

## 11. 跟现有模块对齐

| 对齐项 | 来源 | 说明 |
|--------|------|------|
| LLM client | `pipeline/paper_observer/llm_summarizer.py` | 复用 `_make_client()` + `_model_name()` 模式 |
| Retry 机制 | `pipeline/verifier/verifier.py` | 复用 combined prompt + complete schema template, max 3 retry, malformed JSON → fail |
| 输出 schema 风格 | `pipeline/schemas/verification_report.schema.json` | verdict 枚举 + confidence + reasoning + evidence |
| max_tokens | Verifier | 4000 (同一模型, 相似输入规模) |
| 模型 | Phase 1/2/3.1 统一 | `deepseek-v4-pro` |
| 训练/测试区分 | Phase 1 模式 | 训练集有 rubrics.json → Scorer 跑全量; 测试集无 rubrics.json → Scorer 仅跑空 |

