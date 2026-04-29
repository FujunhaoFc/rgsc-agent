# 数据获取说明

本项目使用 NLPCC 2026 Shared Task 11 的官方数据。出于版权和上游同步考虑，本仓库不直接包含数据文件，请按以下方式获取：

1. 浏览器打开 https://github.com/KOU-199024/NLPCC-2026-Shared-Task-11
2. 点 Code → Download ZIP，解压
3. 把解压出的 data/train_valid、example_repo、tools 复制到本项目的 data/ 下，
   分别重命名为 train_valid、example_repo、official_tools
4. README.md 复制为 data/OFFICIAL_README.md

## 数据集结构

- data/train_valid/ - 5 篇训练论文（AMUN, Beyond-Ngram, I0T, INCLINE, min-p）
  - 每篇含 paper.md, paper.pdf, rubrics.json, 图片资源
- data/example_repo/ - 官方示例 agent 输出仓库
- data/official_tools/ - 官方 MCP Action Recorder 实现
- data/OFFICIAL_README.md - 上游 repo 的 README

数据来源：https://github.com/KOU-199024/NLPCC-2026-Shared-Task-11
