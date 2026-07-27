# DocMind — 智能文档问答系统 🧠

> 一个面向新手学习、面试准备的 **RAG + Agent + Fine-tuning** 全栈 LLM 项目

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 这个项目是什么？

DocMind 是一个 **智能文档问答系统**，你可以上传公司文档（PDF、Word、Markdown），然后像问 ChatGPT 一样问它文档里的问题。它会：

1. 🔍 **搜索**相关文档片段（RAG 检索）
2. 🤖 **调用大模型**生成准确回答（DeepSeek）
3. 🧠 **多步推理**复杂问题（Agent 模式）

**适合谁学**：LLM 开发新手、准备面试的 AI 工程师、想在公司内部搭建知识库问答的人。

---

## 📖 项目架构（一眼看懂整体设计）

```
用户浏览器 (Web 前端)
        ↓  HTTP 请求
  ┌─────────────────────────────┐
  │      FastAPI (后端)          │ ← 你在这里
  ├─────────────────────────────┤
  │  /api/v1/chat      对话接口  │
  │  /api/v1/documents  文档管理 │
  │  /api/v1/agent     Agent接口 │
  └─────────────────────────────┘
        ↓            ↓
  ┌──────────┐  ┌──────────────┐
  │ ChromaDB │  │ BM25 索引     │
  │(向量搜索)│  │(关键词匹配)   │
  └──────────┘  └──────────────┘
        ↓            ↓
  ┌───────────────────────────┐
  │   混合检索 (RRF 融合)      │
  └───────────────────────────┘
        ↓
  ┌───────────────────────────┐
  │   DeepSeek API (LLM)      │
  │   生成最终回答             │
  └───────────────────────────┘
```

---

## 🚀 快速开始（5 分钟跑起来）

### 第一步：克隆项目

```bash
git clone https://github.com/PeichengLiao/LLM-.git
cd LLM-
```

### 第二步：创建虚拟环境

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 第三步：安装依赖

```bash
pip install -r requirements.txt
```

### 第四步：配置 API Key

```bash
# 复制配置文件模板
cp .env.example .env

# 编辑 .env 文件，填入你的 DeepSeek API Key
# 去 https://platform.deepseek.com/api_keys 免费注册获取
```

> 💡 **新手提示**：DeepSeek 注册就送 500 万 tokens 免费额度，够你测试很久了。

### 第五步：导入示例文档

```bash
# 把示例文档（供应链术语表、库存月报）导入知识库
python scripts/ingest_docs.py --dir data/sample_docs/
```

这条命令做的事：
1. 解析 Markdown 文档 → 纯文本
2. 清洗（去噪、统一格式）
3. 分块（递归策略，每个块约 512 字符）
4. 生成嵌入向量（BGE-M3 模型）
5. 存到 ChromaDB + 构建 BM25 索引

### 第六步：启动 API 服务

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

打开浏览器访问 **http://localhost:8000/docs** → 你会看到自动生成的 API 文档（Swagger UI），可以直接在网页上测试所有接口！

### 第七步：打开 Web 前端（可选）

```bash
# 在浏览器中打开
open frontend/index.html
```

或者直接访问 **http://localhost:8000/docs** 用 Swagger UI 测试 API。

---

## 📂 项目目录结构（帮你快速定位文件）

```
llm项目开发/
├── src/                        # ← 核心代码（面试问最多的部分）
│   ├── config.py              #   配置管理（环境变量/API Key）
│   ├── document_processing/   #   文档处理（解析→清洗→分块）
│   │   ├── parser.py          #     PDF/Word/MD/TXT 多格式解析
│   │   ├── chunker.py         #     分块策略（固定/递归/语义）
│   │   └── preprocessor.py    #     文本清洗+术语保护
│   ├── embeddings/            #   嵌入模型封装
│   │   └── embedder.py        #     BGE-M3 嵌入（文本→向量）
│   ├── retrieval/             #   检索模块（面试最爱问！）
│   │   ├── vector_store.py    #     ChromaDB 向量数据库
│   │   ├── bm25_retriever.py  #     BM25 关键词检索
│   │   └── hybrid_retriever.py#     混合检索（RRF 融合）
│   ├── generation/            #   LLM 生成
│   │   ├── llm.py             #     DeepSeek API 调用
│   │   └── prompts.py         #     Prompt 模板管理
│   ├── agent/                 #   AI Agent（ReAct 模式）
│   │   ├── agent.py           #     Agent 主逻辑
│   │   ├── tools.py           #     工具（搜索/计算器）
│   │   └── prompts.py         #     Agent 专用提示词
│   └── api/                   #   Web API
│       ├── main.py            #     FastAPI 入口
│       └── routes/            #     路由（chat/documents/agent）
├── scripts/                   # 脚本工具
│   ├── ingest_docs.py         #   文档批量导入
│   └── finetune_lora.py       #   LoRA 微调脚本
├── notebooks/                 # Jupyter Notebook（评测实验）
│   └── eval_ragas.ipynb       #   RAGAS 系统评测
├── tests/                     # 单元测试 & 集成测试
│   ├── test_parser.py         #   解析器测试
│   ├── test_chunker.py        #   分块器测试
│   ├── test_preprocessor.py   #   预处理器测试
│   ├── test_bm25.py           #   BM25 检索测试
│   ├── test_tools.py          #   Agent 工具测试
│   ├── test_rag_pipeline.py   #   RAG 管道集成测试
│   └── test_agent.py          #   Agent 解析测试
├── data/
│   ├── sample_docs/           #   示例文档（先导入这些来测试）
│   └── finetune_data.jsonl    #   微调示例数据
├── docs/                      # 文档
├── finetuning/                # 微调相关脚本
├── frontend/                  # Web 前端（纯 HTML/CSS/JS）
├── requirements.txt           # Python 依赖清单
├── Makefile                   # 常用命令快捷方式
└── README.md                  # ← 你正在看的文件
```

---

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 只运行某一个测试文件
pytest tests/test_chunker.py -v

# 带覆盖率报告
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 📊 系统评测（RAGAS）

```bash
# 启动 Jupyter
jupyter notebook notebooks/eval_ragas.ipynb

# 或者用 VS Code 打开 notebook 文件直接运行
```

评测维度：**忠实度、答案相关性、上下文召回率、上下文精确率**

---

## 🔬 微调模型（LoRA）

```bash
# 用示例数据训练
python scripts/finetune_lora.py \
  --data data/finetune_data.jsonl \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 3

# 用你自己的数据训练
python scripts/finetune_lora.py --data your_data.jsonl --epochs 5
```

---

## 🎓 面试考点速查表

这个项目覆盖了 LLM 面试最常见的考点：

| 面试问题 | 对应代码 | 要说的要点 |
|----------|----------|------------|
| "你做过RAG吗？怎么做的？" | `retrieval/`, `generation/` | 解析→分块→嵌入→检索→生成，完整管道 |
| "混合检索是什么？" | `hybrid_retriever.py` | BM25+向量+RRF融合，各取所长 |
| "分块策略怎么选？" | `chunker.py` | 三种策略的trade-off，最终选递归 |
| "Agent和RAG区别？" | `agent/agent.py` | RAG是"单跳"，Agent是"多跳+工具调用" |
| "怎么防幻觉？" | `prompts.py` | Prompt约束+来源标注+检索质量 |
| "评测过你的系统吗？" | `eval_ragas.ipynb` | RAGAS四维评测+分块策略对比实验 |

---

## 📝 开发日志（面试时讲你的迭代过程）

1. **V1（基础RAG）**：文档解析 + 向量检索 + LLM 生成
2. **V2（混合检索）**：加入 BM25 关键词匹配，RRF 融合 → 精确率提升 15%
3. **V3（Agent 模式）**：ReAct Agent + 工具调用 → 支持多步推理
4. **V4（系统评测）**：RAGAS 评测 → 发现召回率偏低 → 调优分块策略
5. **V5（微调实验）**：LoRA 微调 → 针对供应链领域的术语理解提升

---

## ⚙️ 技术栈

| 层级 | 技术 | 为什么选它 |
|------|------|------------|
| 嵌入模型 | BGE-M3 (BAAI) | 中英双语、本地运行、免费 |
| 向量数据库 | ChromaDB | 轻量级、Python 原生、持久化 |
| 关键词检索 | BM25 (rank-bm25) | 经典算法、精确匹配好 |
| LLM | DeepSeek API | 中文强、便宜（GPT-4 的 1/20）、OpenAI 兼容 |
| Web 框架 | FastAPI | 异步、自动文档、高性能 |
| 评测框架 | RAGAS | RAG 评测标准工具 |

---

## ⚠️ 常见问题

**Q: 启动时报 "No module named 'src'"？**
A: 确保在项目根目录（`llm项目开发/`）下运行命令。

**Q: 嵌入模型下载很慢？**
A: 首次运行会自动下载 BGE-M3 模型（约 2GB），需要几分钟。也可以设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q: 不想用 DeepSeek？**
A: 修改 `.env` 中的 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`，支持任何 OpenAI 兼容的 API（如 vLLM 本地部署）。

---

## 📄 License

MIT — 随意使用、修改、分发。Star ⭐ 是对我最大的鼓励！
