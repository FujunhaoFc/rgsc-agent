# RGSC-Agent 项目工作约定

## 项目背景

NLPCC 2026 Shared Task 11 — agent 实验复现。整体设计文档：
- docs/paper_state_design.md （schema 决策）
- outputs/AMUN/_review_notes.md （已知问题）

## 工作分工

我（Jerry）会在 web Claude 那边讨论设计、决定改动方向，然后给你"任务卡片"。
你的职责是：
- 严格按任务卡片改代码、跑命令
- 不要自作主张扩展任务范围
- 不要修改 schema、prompt、generator 的设计逻辑
- 修复 bug 时优先 root cause，不打补丁
- 改完代码后跑测试 / 诊断脚本验证

## 代码风格

- Python 3.11，类型注解
- 函数级文档字符串解释 why（不是 what）
- 修改前先 read 文件理解结构，不要盲改
- 编辑后跑相关测试（tests/test_*.py）确保不破坏

## 不要做的事

- 不要重写已经稳定的模块（section_parser, anchor_parser, entity_extractor）
- 不要改 paper_state schema（pipeline/schemas/paper_state.schema.json）
- 不要改 prompt 文件（pipeline/paper_observer/prompts/）除非任务卡片明确要求
- 不要 commit（commit 留给 Jerry 手动做）

## 当前状态

- Phase 1：paper_state 抽取 + derived checklist + coverage diagnostic 已跑通
- 当前 baseline：49.2% 总体 recall（v5）
- 正在做：加 LLM retry 修评测器 bug（v6）
- 下一阶段：Stage Planner（Phase 2 第一个模块）

## 工作日志位置

每次完成任务在 docs/worklog.md 末尾追加一行：
- 日期 / 任务 / 改了哪些文件 / 关键数据点
