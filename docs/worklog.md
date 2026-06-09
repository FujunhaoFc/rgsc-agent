# RGSC-Agent Worklog

记录每次代码改动、跑的命令、关键数据点。

格式：日期 | 任务 | 改了什么 | 关键结果
---
2026-05-12 | Phase 1 closeout | derive_from_state.py 实现 + coverage_diagnostic.py 5 轮迭代 | overall recall 49.2% baseline
2026-05-27 | Stage Planner 规则微调 | planner.py Steps 1-4 + 大 stage 拆分 + preprocessing 类型 | 5 篇平均 5.6 stages, coverage 100%, min-p has_training=False 正确, Beyond-Ngram baseline-training(60 items) 按 type 拆为 plan/impl/exec 三个子 stage
2026-05-29 | Stage Planner Step 5 LLM refine 收尾 | planner.py: _llm_refine_stages() + _debug 落盘 + max_tokens 4000→8000; prompt stage_refinement.txt; tests/test_stage_planner.py (10 tests × 5 fixtures = 46 passed) | max_tokens=8000 决策依据: 4000 下 Beyond-Ngram (7 stages, 17k prompt) 100% 失败回退 rule-based, 8000 下 5/5 全过 (4/5 一次过, Beyond-Ngram 需 2-3 attempts). atomic_step_assignment 字段 LLM 实际输出为空, 分配信息隐式在 actions 里. 5 篇最终成本 ~$0.10 (含重跑). Phase 2 Stage Planner 总成本 ~$0.10, Phase 1 Paper Observer ~$0.80 + Coverage Diagnostic ~$6.20

2026-06-02 | Verifier Phase 3.1 | verifier.py 主流程 + placeholder mock + 2 schema + tests | 5 篇 mock 全 skipped, 0 LLM call, 14/14 test pass; claim_judge prompt 待 web Claude 提供
2026-06-02 | Verifier 加固 | LLM malformed JSON 3 retry → fail (不再 skipped); 删 mock notes 字段; claim_judge.txt 加 trust-numeric 规则(rule 8) | AMUN broken overall=0.8 (4P/1F), real overall=1.0 (5P, 3rd run), pytest 14/14 pass. Known: DeepSeek V4 偶尔返回 {} / 残缺 JSON (claim-2 两次 non-deterministic failure, 3rd run pass), retry prompt 只让 LLM 修提到的错误而漏掉其他 required fields

技术债务记录:
- pipeline/stage_planner/planner.py 重复实现了 LLM client (复用了 paper_observer/llm_summarizer.py 的设计但没复用代码). Phase 3 启动时统一抽到 pipeline/common/llm.py
- pipeline/verifier/verifier.py 同样重复实现了 _call_llm (遵循 planner.py 模式). Phase 3 后期 Executor 时统一抽取.
2026-06-08 | Phase 3.2 Card 6 五篇空输入 sanity | 5 篇空 actions/repo 跑 Scorer, recall 全部 0.0000 (AMUN 77M / Beyond-Ngram 62M / I0T 102M / INCLINE 76M / min-p 82M): Scorer 在空输入下正确判全 miss, 不幻觉. 399 LLM 调用合计 token 178k in / 80k out, cost ~$0.14. 中途 ~20+ 次 API timeout/connection error 自动 retry 恢复, 无 crash. RUUBRIC SCORER PHASE 3.2 STEP 1 收尾 — 等 jerry commit. Phase 3.2 step 2 (Agent loop + actions.json 真生成) 是下个里程碑. 70 tests pass. git status clean (_debug 被 gitignore 正确排除)
2026-06-05 | Phase 3.1 Verifier 5 篇 mock 端到端完成 | 5 篇 real mock 全部跑通: AMUN/Beyond-Ngram/I0T/INCLINE overall=1.0, min-p overall=0.929 (6P/1Q). Patches: Beyond-Ngram claim-3 二次收窄, I0T figure-5 summary 化 + claim-6 patch, min-p claim-1/5/6 patch + table-4 summary 化. Tech debt 暴露: (1) expected_claims schema 只支持单 evidence, 跨表 claim 处理需 array; (2) results schema oneOf table/figure 互斥, mixed entry 不便 → workaround 用 summary-only; (3) Phase 1 expected_claims 抽取 over-generalization 是稳定模式 (5 篇里 3 篇出现, 论文 narrative-data mismatch 案例): Beyond-Ngram (COMET highest 描述过宽), I0T claim-6 (MCSIE reduces gap 与数据矛盾), min-p claim-1/5/6 (笛卡尔积 over-coverage). Verifier 反向作为 Paper Observer 质检工具的价值是 system paper key insight
