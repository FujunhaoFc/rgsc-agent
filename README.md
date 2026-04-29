# NLPCC 2026 Task 11 - RGSC Agent

本项目是 NLPCC 2026 Shared Task 11（Reproducibility-oriented Generative Scientific Contribution，RGSC）的参赛 agent 实现。目标是通过 LLM-driven multi-agent 系统自动复现论文实验并生成审查报告。

## 项目结构

```
rgsc-agent/
├── data/                    # 官方数据（git ignored，需手动下载）
│   ├── README.md            # 数据获取说明
│   ├── train_valid/         # 5 篇训练论文
│   ├── example_repo/        # 官方示例 agent 输出
│   └── official_tools/      # 官方 MCP Action Recorder
├── pipeline/                # 核心 pipeline 模块
│   ├── paper_observer/      # LLM-based 论文观察器
│   ├── section_parser/      # 论文章节解析
│   ├── anchor_parser/       # 锚点定位解析
│   ├── rubric_checklist/    # Rubric checklist 派生
│   └── agent/               # Agent 主流程
├── eval/                    # 评估与诊断脚本
├── outputs/                 # agent 输出目录
├── tests/                   # 单元测试
├── requirements.txt         # Python 依赖
├── .env.example             # 环境变量模板
└── .gitignore
```

## 安装步骤

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 获取数据（详见 data/README.md）
# 手动从上游 repo 下载并复制数据文件

# 4. 配置 API key
cp .env.example .env
# 编辑 .env 填入实际的 API key
```

## 当前进度

- [x] 项目骨架初始化
- [x] section_parser 实现
- [ ] anchor_parser 实现
- [ ] LLM-based paper_state 构造
- [ ] rubric checklist 派生
- [ ] 5 篇论文上的覆盖率诊断
- [ ] agent 主流程
